"""검색 가중치 탐색 (valid 쿼리만 사용 → test는 최종 보고용으로 남겨 둔다).

    python -m src.evaluation.tune_search
    python -m src.evaluation.tune_search --model openai/clip-vit-base-patch32

순서대로 정한다 (앞에서 고른 값은 뒤 탐색에서 고정)
1. text   : text_query.image_weight × text_query.bm25_weight   (텍스트 쿼리)
2. image  : image_query.color_weight × image_query.category_weight (이미지 쿼리, 색·소분류 분류기가 있을 때)
3. hybrid : hybrid.text_weight                                   (하이브리드 쿼리)
쿼리는 한 번만 인코딩하고 가중치만 바꿔 다시 채점한다. 결과: artifacts/metrics/tune_<모델>[_<정답 기준>].json
(정답 기준 eval.relevance가 기존 기준(design)이 아니면 파일 이름 끝에 붙여서 기존 결과를 덮어쓰지 않는다)
→ 고른 값을 configs/search.yaml에 옮겨 적는다.
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd

from src.common.config import ROOT, load_config
from src.evaluation.eval_search import relevance_tag, score
from src.search.catalog import model_slug, with_overrides
from src.search.searcher import Searcher

GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BONUS_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]  # 색·소분류 가점 (유사도는 0~1)


def encode_queries(searcher: Searcher, df: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None]:
    bs = searcher.cfg["clip"]["batch_size"]
    q_text = searcher.encoder.encode_texts(df["query_text"].tolist(), bs) if df["query_text"].notna().all() else None
    q_image = None
    if df["query_image_path"].notna().all():
        root = ROOT / searcher.cfg["paths"]["image_root"]
        q_image = np.concatenate(list(searcher.encoder.encode_image_files([root / p for p in df["query_image_path"]], bs)))
    return q_text, q_image


def sweep(
    searcher: Searcher, df: pd.DataFrame, q_text: np.ndarray | None, q_image: np.ndarray | None,
    grid: dict[tuple[str, str], list[float]],
) -> list[dict[str, float]]:
    """grid: (설정 섹션, 키) → 후보 값. 모든 조합을 채점한다."""
    ev = searcher.cfg["eval"]
    texts = df["query_text"].tolist()
    rows = []
    for values in itertools.product(*grid.values()):
        for (section, key), value in zip(grid, values):
            searcher.cfg.setdefault(section, {})[key] = value
        ranked = [
            searcher.search_vectors(
                q_text[i] if q_text is not None else None, q_image[i] if q_image is not None else None, ev["k"],
                texts[i] if q_text is not None else None,
            ).product_ids
            for i in range(len(df))
        ]
        r = score(ranked, df, ev["k"], ev["mrr_min_grade"])
        metrics = {"mrr": r["mrr"], "ndcg": r[f"ndcg@{ev['k']}"], "hit": r[f"hit_rate@{ev['k']}"]}
        rows.append({f"{s}.{k}": v for (s, k), v in zip(grid, values)} | metrics)
    return rows


def choose(searcher: Searcher, rows: list[dict[str, float]]) -> dict[str, float]:
    """NDCG 최우선, 동률이면 MRR (두 기준 중 NDCG가 더 어려워서). 고른 값을 검색기 설정에 반영.

    단 HitRate가 탐색 값이 모두 0인 설정보다 낮아지는 조합은 뺀다. 색·소분류 가점을 세게 주면
    NDCG는 조금 더 오르지만, 종류를 잘못 추정한 사진은 top-10이 통째로 엉뚱한 종류로 채워진다
    (valid 이미지: 가점 1 / 2 → NDCG 0.394, HitRate 0.884 → 0.838).
    """
    names = [name for name in rows[0] if name not in ("mrr", "ndcg", "hit")]
    floor = next(r["hit"] for r in rows if all(r[n] == 0 for n in names))
    best = max((r for r in rows if r["hit"] >= floor), key=lambda r: (r["ndcg"], r["mrr"]))
    params = {name: best[name] for name in names}
    for name, value in params.items():
        section, key = name.split(".")
        searcher.cfg[section][key] = value
    print(f"  → {params}  (valid MRR {best['mrr']:.3f} / NDCG {best['ndcg']:.3f} / HitRate {best['hit']:.3f})")
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None, help="기본: clip.index_device")
    args = parser.parse_args()

    cfg = with_overrides(load_config("search"), model=args.model)
    searcher = Searcher(cfg, device=args.device or cfg["clip"]["index_device"])
    valid = pd.read_parquet(ROOT / cfg["paths"]["eval_set"]).query("split == 'valid'")
    frames = {t: valid[valid["query_type"] == t].reset_index(drop=True) for t in ("text", "image", "hybrid")}
    model = cfg["clip"]["model_name"]
    print(f"[tune] {model} (valid: " + ", ".join(f"{t} {len(df)}" for t, df in frames.items()) + ")")

    sweeps, chosen = {}, {}
    print("1. text: image_weight × bm25_weight")
    sweeps["text"] = sweep(searcher, frames["text"], *encode_queries(searcher, frames["text"]),
                           {("text_query", "image_weight"): GRID, ("text_query", "bm25_weight"): GRID})
    chosen |= choose(searcher, sweeps["text"])

    if searcher.attributes is not None:
        print("2. image: color_weight × category_weight")
        sweeps["image"] = sweep(searcher, frames["image"], *encode_queries(searcher, frames["image"]),
                                {("image_query", "color_weight"): BONUS_GRID,
                                 ("image_query", "category_weight"): BONUS_GRID})
        chosen |= choose(searcher, sweeps["image"])
    else:
        print("2. image: 색·소분류 분류기 없음 → 건너뜀 (python -m src.search.attributes)")

    print("3. hybrid: text_weight")
    sweeps["hybrid"] = sweep(searcher, frames["hybrid"], *encode_queries(searcher, frames["hybrid"]),
                             {("hybrid", "text_weight"): GRID})
    chosen |= choose(searcher, sweeps["hybrid"])

    out = ROOT / cfg["paths"]["metrics_dir"] / f"tune_{model_slug(model)}{relevance_tag(cfg)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"model_name": model, "relevance": cfg["eval"].get("relevance", "design"),
              "eval_set": cfg["paths"]["eval_set"], "chosen": chosen, "sweeps": sweeps}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[tune] configs/search.yaml에 옮겨 적을 값: {chosen}\n저장 → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
