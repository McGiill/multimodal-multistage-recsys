"""검색 가중치 탐색 (valid 쿼리만 사용 → test는 최종 보고용으로 남겨 둔다).

    python -m src.evaluation.tune_search
    python -m src.evaluation.tune_search --model patrickjohncyh/fashion-clip

- text_query.image_weight : 텍스트 검색에서 (텍스트→상품 이미지) 비중
- hybrid.text_weight      : 하이브리드 검색에서 텍스트 경로 비중 (text_query.image_weight는 위에서 고른 값으로 고정)
쿼리는 한 번만 인코딩하고 가중치만 바꿔 다시 채점한다. 결과: artifacts/metrics/tune_<모델>.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.common.config import ROOT, load_config
from src.evaluation.eval_search import score
from src.search.catalog import model_slug, with_overrides
from src.search.searcher import Searcher

GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def encode_queries(searcher: Searcher, df: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None]:
    bs = searcher.cfg["clip"]["batch_size"]
    q_text = searcher.encoder.encode_texts(df["query_text"].tolist(), bs) if df["query_text"].notna().all() else None
    q_image = None
    if df["query_image_path"].notna().all():
        root = ROOT / searcher.cfg["paths"]["image_root"]
        q_image = np.concatenate(list(searcher.encoder.encode_image_files([root / p for p in df["query_image_path"]], bs)))
    return q_text, q_image


def sweep(
    searcher: Searcher, df: pd.DataFrame, q_text: np.ndarray | None, q_image: np.ndarray | None, section: str, key: str
) -> list[dict[str, float]]:
    ev = searcher.cfg["eval"]
    rows = []
    for w in GRID:
        searcher.cfg[section][key] = w
        ranked = [
            searcher.search_vectors(
                q_text[i] if q_text is not None else None, q_image[i] if q_image is not None else None, ev["k"]
            ).product_ids
            for i in range(len(df))
        ]
        r = score(ranked, df, ev["k"], ev["mrr_min_grade"])
        rows.append({"weight": w, "mrr": r["mrr"], "ndcg": r[f"ndcg@{ev['k']}"]})
    return rows


def best(rows: list[dict[str, float]]) -> float:
    """NDCG 최우선, 동률이면 MRR (두 기준 중 NDCG가 더 어려워서)."""
    return max(rows, key=lambda r: (r["ndcg"], r["mrr"]))["weight"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None, help="기본: clip.index_device")
    args = parser.parse_args()

    cfg = with_overrides(load_config("search"), model=args.model)
    searcher = Searcher(cfg, device=args.device or cfg["clip"]["index_device"])
    valid = pd.read_parquet(ROOT / cfg["paths"]["eval_set"]).query("split == 'valid'")

    text_df = valid[valid["query_type"] == "text"].reset_index(drop=True)
    hybrid_df = valid[valid["query_type"] == "hybrid"].reset_index(drop=True)

    text_rows = sweep(searcher, text_df, *encode_queries(searcher, text_df), "text_query", "image_weight")
    searcher.cfg["text_query"]["image_weight"] = best_text = best(text_rows)
    hybrid_rows = sweep(searcher, hybrid_df, *encode_queries(searcher, hybrid_df), "hybrid", "text_weight")
    best_hybrid = best(hybrid_rows)

    model = cfg["clip"]["model_name"]
    print(f"[tune] {model} (valid: text {len(text_df)}, hybrid {len(hybrid_df)})")
    print("| w | text MRR | text NDCG | hybrid MRR | hybrid NDCG |\n|---|---|---|---|---|")
    for t, h in zip(text_rows, hybrid_rows):
        print(f"| {t['weight']:.1f} | {t['mrr']:.3f} | {t['ndcg']:.3f} | {h['mrr']:.3f} | {h['ndcg']:.3f} |")
    print(f"\n→ --text-image-weight {best_text} --hybrid-text-weight {best_hybrid}")

    out = ROOT / cfg["paths"]["metrics_dir"] / f"tune_{model_slug(model)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"model_name": model, "text_image_weight": best_text, "hybrid_text_weight": best_hybrid,
             "text_sweep": text_rows, "hybrid_sweep": hybrid_rows},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
