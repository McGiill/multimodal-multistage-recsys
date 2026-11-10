"""FAISS 인덱스 Trade-off: 인덱스 종류와 탐색 파라미터별 속도 / Recall / 크기 / 검색 품질.

    python -m src.evaluation.index_tradeoff
    python -m src.evaluation.index_tradeoff --model patrickjohncyh/fashion-clip

인덱서가 저장한 상품 임베딩으로 인덱스를 새로 만들어 비교만 한다 (기존 인덱스 파일과 설정은 그대로).
- flat      : 전부 비교. Recall의 기준(정답)
- hnsw      : 그래프 탐색. efSearch(탐색 중 유지하는 후보 수)를 바꿔 가며
- ivfflat   : 클러스터링만. nprobe(살펴볼 클러스터 수, 전체 nlist개 중)를 바꿔 가며 → 클러스터링 때문에 놓치는 양
- ivfpq_m16 : 클러스터링 + 곱 양자화(벡터 하나 2,048바이트 → 16바이트). nprobe를 바꿔 가며 → 압축 때문에 더 놓치는 양
- ivfpq_m64 : 같은 방식, 64바이트로 압축 (덜 압축하면 Recall이 얼마나 돌아오는지)

측정 항목
- recall@200   : flat의 상위 200개 중 인덱스도 찾은 비율. 200 = 검색기가 인덱스에서 가져오는 후보 수(retrieval.candidate_k)
- top10_in_200 : flat의 상위 10개가 인덱스 후보 200개 안에 들어온 비율. 검색기는 후보를 원래 임베딩으로 다시 채점하므로
                 이 값이 1이면 유사도만으로 매긴 최종 상위 10개는 flat과 같다
  쿼리 = 검색 평가셋 전체 (정답 라벨은 쓰지 않음). 검색기가 인덱스를 쓰는 세 가지 조합을 따로 잰다:
  사진 → 상품 사진 인덱스 / 문장 → 상품 사진 인덱스 / 문장 → 상품 텍스트 인덱스
- latency      : 인덱스 검색 1회 (사진 → 상품 사진 인덱스, 쿼리 1개, 후보 200개). warm-up 10회 후 100회의 p50 / p95
- size_mb / build_sec : 인덱스 크기(직렬화)와 만드는 시간. 상품 사진·텍스트 인덱스 두 개 합
- valid        : 인덱스만 바꾸고 검색기 전체(재채점, BM25, 색·소분류 가점 포함)를 valid 쿼리로 평가한 MRR / NDCG@10
                 (+ HitRate@10) → 인덱스가 놓친 후보가 최종 검색 품질에 주는 영향 (test는 보고용이라 쓰지 않음)

결과: artifacts/metrics/index_tradeoff_<모델>[_<정답 기준>].json + 콘솔 표
Docker 4코어 측정 (응답시간 기준과 같은 환경):
    docker run --rm --cpus 4 --memory 4g -e OMP_NUM_THREADS=4 -e HF_HOME=/app/artifacts/models \\
      -v "$PWD/artifacts:/app/artifacts" -v "$PWD/data:/app/data:ro" multi-stage-tools \\
      python -m src.evaluation.index_tradeoff
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime
from typing import Any

import faiss
import numpy as np
import pandas as pd

from src.common.config import ROOT, load_config
from src.common.faiss_utils import build_index, set_search_params
from src.evaluation.eval_search import relevance_tag, score
from src.evaluation.latency_bench import machine_info, measure
from src.search.catalog import model_slug, with_overrides
from src.search.searcher import Searcher

# (이름, 인덱스 설정 덮어쓰기, 바꿔 가며 잴 탐색 파라미터, 값)
VARIANTS: list[tuple[str, dict[str, Any], str | None, list[int | None]]] = [
    ("flat", {"type": "flat"}, None, [None]),
    ("hnsw", {"type": "hnsw"}, "ef_search", [16, 32, 64, 128, 256, 512]),
    ("ivfflat", {"type": "ivfflat"}, "nprobe", [1, 4, 16, 64, 256]),
    ("ivfpq_m16", {"type": "ivfpq", "m": 16}, "nprobe", [1, 4, 16, 64, 256]),
    ("ivfpq_m64", {"type": "ivfpq", "m": 64}, "nprobe", [1, 4, 16, 64, 256]),
]
PAIRS = (("image→image", "image", "image"), ("text→image", "text", "image"), ("text→text", "text", "text"))


def overlap(found: np.ndarray, truth: np.ndarray) -> float:
    """행마다 truth 중 found에 있는 비율의 평균."""
    return float(np.mean([np.isin(t, f).mean() for f, t in zip(found, truth)]))


def encode_all(searcher: Searcher, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """평가셋 전체 쿼리를 한 번만 인코딩 (행 순서 = df, 문장·사진이 없는 행은 0 벡터)."""
    enc, bs = searcher.encoder, searcher.cfg["clip"]["batch_size"]
    dim = searcher.image_emb.shape[1]
    q_text, q_image = np.zeros((len(df), dim), "float32"), np.zeros((len(df), dim), "float32")
    has_text, has_image = df["query_text"].notna().to_numpy(), df["query_image_path"].notna().to_numpy()
    q_text[has_text] = enc.encode_texts(df.loc[has_text, "query_text"].tolist(), bs)
    root = ROOT / searcher.cfg["paths"]["image_root"]
    paths = [root / p for p in df.loc[has_image, "query_image_path"]]
    q_image[has_image] = np.concatenate(list(enc.encode_image_files(paths, bs)))
    return q_text, q_image


def valid_quality(searcher: Searcher, df: pd.DataFrame, q_text: np.ndarray, q_image: np.ndarray) -> dict[str, Any]:
    """검색기의 인덱스만 바꾼 상태로 valid 쿼리를 평가 (eval_search와 같은 채점)."""
    ev = searcher.cfg["eval"]
    k, min_grade = ev["k"], ev["mrr_min_grade"]
    valid = df[df["split"] == "valid"]
    out, ranked_all, frames = {}, [], []
    for qtype in ("text", "image", "hybrid"):
        g = valid[valid["query_type"] == qtype]
        ranked = []
        for i, row in g.iterrows():
            text = row["query_text"] if pd.notna(row["query_text"]) else None
            image = q_image[i] if pd.notna(row["query_image_path"]) else None
            ranked.append(searcher.search_vectors(q_text[i] if text else None, image, k, text).product_ids)
        r = score(ranked, g.reset_index(drop=True), k, min_grade)
        out[qtype] = {"mrr": r["mrr"], "ndcg": r[f"ndcg@{k}"], "hit": r[f"hit_rate@{k}"]}
        ranked_all += ranked
        frames.append(g)
    r = score(ranked_all, pd.concat(frames).reset_index(drop=True), k, min_grade)
    out["all"] = {"mrr": r["mrr"], "ndcg": r[f"ndcg@{k}"], "hit": r[f"hit_rate@{k}"]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="모델 이름 (기본: configs/search.yaml)")
    parser.add_argument("--device", default=None, help="쿼리 인코딩 장치 (기본: clip.index_device)")
    parser.add_argument("--out", default=None, help="저장 경로 (기본: artifacts/metrics/index_tradeoff_<모델>[_<정답 기준>].json)")
    args = parser.parse_args()

    cfg = with_overrides(load_config("search"), model=args.model)
    searcher = Searcher(cfg, device=args.device or cfg["clip"]["index_device"])
    df = pd.read_parquet(ROOT / cfg["paths"]["eval_set"]).reset_index(drop=True)
    q_text, q_image = encode_all(searcher, df)
    queries = {"text": q_text[df["query_text"].notna().to_numpy()],
               "image": q_image[df["query_image_path"].notna().to_numpy()]}
    data = {"image": searcher.image_emb, "text": searcher.text_emb}
    k = cfg["retrieval"]["candidate_k"]
    print(f"[index-tradeoff] 상품 {len(searcher.product_ids):,}개, 쿼리 문장 {len(queries['text']):,} / "
          f"사진 {len(queries['image']):,}, 후보 {k}개, FAISS 스레드 {faiss.omp_get_max_threads()}")

    rows: list[dict[str, Any]] = []
    truth: dict[str, np.ndarray] = {}
    for name, override, param, values in VARIANTS:
        index_cfg = copy.deepcopy(cfg["index"]) | override
        t0 = time.perf_counter()
        indexes = {kind: build_index(data[kind], index_cfg) for kind in ("image", "text")}
        build_sec = time.perf_counter() - t0
        size_mb = sum(faiss.serialize_index(ix).nbytes for ix in indexes.values()) / 2**20
        for value in values:
            if param:
                index_cfg[param] = value
                for ix in indexes.values():
                    set_search_params(ix, index_cfg)
            found = {pair: indexes[ix].search(queries[q], k)[1] for pair, q, ix in PAIRS}
            if name == "flat":
                truth = found
            img = queries["image"]
            latency = measure(lambda i: indexes["image"].search(img[i % len(img)][None], k))  # noqa: B023
            searcher.image_index, searcher.text_index = indexes["image"], indexes["text"]
            row = {
                "index": name, "param": param, "value": value,
                "current": index_cfg["type"] == cfg["index"]["type"] and (not param or value == cfg["index"][param])
                and all(index_cfg[key] == cfg["index"][key] for key in override),
                "size_mb": round(size_mb, 1), "build_sec": round(build_sec, 1),
                "latency_ms": {"p50": latency["p50_ms"], "p95": latency["p95_ms"]},
                "recall@200": {pair: round(overlap(found[pair], truth[pair]), 4) for pair, _, _ in PAIRS},
                "top10_in_200": {pair: round(overlap(found[pair], truth[pair][:, :10]), 4) for pair, _, _ in PAIRS},
                "valid": valid_quality(searcher, df, q_text, q_image),
            }
            rows.append(row)
            print(f"  {name:9s} {param or '':9s} {value or '':>4} | {row['size_mb']:6.1f}MB | "
                  f"p95 {row['latency_ms']['p95']:6.2f}ms | recall@200 {row['recall@200']['image→image']:.3f} | "
                  f"valid NDCG {row['valid']['all']['ndcg']:.3f}", flush=True)

    report = {
        "model_name": cfg["clip"]["model_name"],
        "n_products": len(searcher.product_ids),
        "candidate_k": k,
        "index_cfg": cfg["index"],
        "relevance": cfg["eval"].get("relevance", "design"),
        "eval_set": cfg["paths"]["eval_set"],
        "n_queries": {kind: len(q) for kind, q in queries.items()},
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        **machine_info(),
        "faiss_threads": faiss.omp_get_max_threads(),
        "rows": rows,
    }
    name = f"index_tradeoff_{model_slug(cfg['clip']['model_name'])}{relevance_tag(cfg)}.json"
    out = ROOT / (args.out or f"{cfg['paths']['metrics_dir']}/{name}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n| 인덱스 | 파라미터 | 크기(MB) | 구축(s) | p50 / p95 (ms) | Recall@200 사진→사진 / 문장→사진 / 문장→텍스트 "
          "| 상위10 포함률 (같은 순서) | valid MRR / NDCG@10 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        mark = " (현재)" if r["current"] else ""
        param = f"{r['param']}={r['value']}" if r["param"] else "-"
        recall = " / ".join(f"{v:.3f}" for v in r["recall@200"].values())
        top10 = " / ".join(f"{v:.3f}" for v in r["top10_in_200"].values())
        v = r["valid"]["all"]
        print(f"| {r['index']}{mark} | {param} | {r['size_mb']:.1f} | {r['build_sec']:.1f} | "
              f"{r['latency_ms']['p50']:.2f} / {r['latency_ms']['p95']:.2f} | {recall} | {top10} | "
              f"{v['mrr']:.3f} / {v['ndcg']:.3f} |")
    print(f"\n[index-tradeoff] 저장 → {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
