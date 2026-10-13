"""검색 평가: CLIP(text / image / hybrid) vs BM25 베이스라인 — MRR, NDCG@10, HitRate@10.

    python -m src.evaluation.build_search_eval   # 평가셋이 없으면 먼저 생성
    python -m src.evaluation.eval_search
    python -m src.evaluation.eval_search --model patrickjohncyh/fashion-clip --text-image-weight 0.3
        (모델 실험: 가중치는 python -m src.evaluation.tune_search 로 valid에서 먼저 정한다)

결과: artifacts/metrics/search_<모델>_<인덱스>.json + 콘솔 표 (docs/report_search.md 에 옮겨 적기)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import ROOT, load_config
from src.evaluation.metrics import hit_rate_at_k, mrr, ndcg_at_k
from src.search.bm25 import BM25Searcher
from src.search.catalog import load_catalog, model_slug, with_overrides
from src.search.searcher import Searcher


def score(ranked: list[list[str]], eval_df: pd.DataFrame, k: int, mrr_min_grade: float) -> dict[str, float]:
    relevant = [dict(zip(ids, rel)) for ids, rel in zip(eval_df["relevant_ids"], eval_df["relevance"])]
    return {
        "n_queries": len(ranked),
        "mrr": round(mrr(ranked, relevant, min_grade=mrr_min_grade), 4),
        f"ndcg@{k}": round(ndcg_at_k(ranked, relevant, k), 4),
        f"hit_rate@{k}": round(hit_rate_at_k(ranked, relevant, k), 4),
    }


def run_clip(searcher: Searcher, df: pd.DataFrame, k: int, batch_size: int) -> list[list[str]]:
    image_root = ROOT / searcher.cfg["paths"]["image_root"]
    enc = searcher.encoder
    q_text = enc.encode_texts(df["query_text"].tolist(), batch_size) if df["query_text"].notna().all() else None
    q_image = None
    if df["query_image_path"].notna().all():
        paths = [image_root / p for p in df["query_image_path"]]
        q_image = np.concatenate(list(enc.encode_image_files(paths, batch_size)))
    return [
        searcher.search_vectors(
            q_text[i] if q_text is not None else None, q_image[i] if q_image is not None else None, k
        ).product_ids
        for i in range(len(df))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None, help="평가 시 CLIP 장치 (기본: clip.index_device)")
    parser.add_argument("--split", choices=["test", "valid", "all"], default="test", help="보고용은 test")
    parser.add_argument("--model", default=None, help="모델 이름 (기본: configs/search.yaml)")
    parser.add_argument("--text-image-weight", type=float, default=None, help="text_query.image_weight 덮어쓰기")
    parser.add_argument("--hybrid-text-weight", type=float, default=None, help="hybrid.text_weight 덮어쓰기")
    args = parser.parse_args()

    cfg = with_overrides(load_config("search"), args.model, args.text_image_weight, args.hybrid_text_weight)
    ev = cfg["eval"]
    k = ev["k"]
    eval_set = pd.read_parquet(ROOT / cfg["paths"]["eval_set"])
    if args.split != "all":
        eval_set = eval_set[eval_set["split"] == args.split]
    searcher = Searcher(cfg, device=args.device or cfg["clip"]["index_device"])

    results: dict[str, Any] = {}
    ranked_by_type: dict[str, list[list[str]]] = {}
    frames: dict[str, pd.DataFrame] = {}
    for qtype in ("text", "image", "hybrid"):
        df = eval_set[eval_set["query_type"] == qtype].reset_index(drop=True)
        if len(df):
            frames[qtype] = df
            ranked_by_type[qtype] = run_clip(searcher, df, k, cfg["clip"]["batch_size"])
            results[f"clip_{qtype}"] = score(ranked_by_type[qtype], df, k, ev["mrr_min_grade"])

    ranked_all = [r for t in frames for r in ranked_by_type[t]]
    results["clip_all"] = score(ranked_all, pd.concat(frames.values()), k, ev["mrr_min_grade"])

    if "text" in frames:
        bm25 = BM25Searcher(load_catalog(cfg))
        ranked = [bm25.search(q, k)[0] for q in frames["text"]["query_text"]]
        results["bm25_text"] = score(ranked, frames["text"], k, ev["mrr_min_grade"])

    report = {
        "model_name": cfg["clip"]["model_name"],
        "index": cfg["index"],
        "weights": {"text_query.image_weight": cfg["text_query"]["image_weight"], "hybrid.text_weight": cfg["hybrid"]["text_weight"]},
        "k": k,
        "split": args.split,
        "mrr_min_grade": ev["mrr_min_grade"],
        "seed": cfg["seed"],
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }
    out = ROOT / cfg["paths"]["metrics_dir"] / f"search_{model_slug(cfg['clip']['model_name'])}_{cfg['index']['type']}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n| 방법 | 쿼리 수 | MRR (≥0.55) | NDCG@{k} (≥0.50) | HitRate@{k} |")
    print("|---|---|---|---|---|")
    for name, r in results.items():
        print(f"| {name} | {r['n_queries']:,} | {r['mrr']:.4f} | {r[f'ndcg@{k}']:.4f} | {r[f'hit_rate@{k}']:.4f} |")
    print(f"\n[eval-search] 저장 → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
