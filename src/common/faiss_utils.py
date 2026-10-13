"""FAISS 인덱스 생성/저장/로드 공통 유틸 (검색·추천 양쪽에서 사용).

벡터는 L2 정규화된 상태로 넣고 내적(METRIC_INNER_PRODUCT)으로 검색한다 → 점수 = 코사인 유사도.

지원 타입 (configs/*.yaml 의 index.type)
- flat    : 전수 비교. 정확도 기준선
- hnsw    : 그래프 기반 ANN. 학습 불필요, 높은 Recall
- ivfflat : 클러스터링(IVF)만 적용
- ivfpq   : 클러스터링 + 곱 양자화(PQ). 메모리↓ 속도↑, 대신 Recall↓
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype="float32")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def build_index(vectors: np.ndarray, cfg: dict[str, Any]) -> faiss.Index:
    """정규화된 벡터로 인덱스를 만든다. cfg = configs/search.yaml 의 index 섹션."""
    vectors = np.ascontiguousarray(vectors, dtype="float32")
    n, dim = vectors.shape
    index_type = cfg["type"]
    ip = faiss.METRIC_INNER_PRODUCT

    if index_type == "flat":
        index = faiss.IndexFlatIP(dim)
    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, cfg["hnsw_m"], ip)
        index.hnsw.efConstruction = cfg["ef_construction"]
    elif index_type in ("ivfflat", "ivfpq"):
        # 클러스터 수는 데이터가 적을 때(개발용 소규모) 자동으로 줄인다
        nlist = min(cfg["nlist"], max(1, n // 39))
        quantizer = faiss.IndexFlatIP(dim)
        if index_type == "ivfflat":
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, ip)
        else:
            index = faiss.IndexIVFPQ(quantizer, dim, nlist, cfg["m"], cfg["nbits"], ip)
        index.train(vectors)
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    index.add(vectors)
    set_search_params(index, cfg)
    return index


def set_search_params(index: faiss.Index, cfg: dict[str, Any]) -> None:
    """검색 시 속도-정확도 파라미터 (IVF: nprobe, HNSW: efSearch)."""
    try:
        faiss.extract_index_ivf(index).nprobe = cfg["nprobe"]
        return
    except RuntimeError:  # IVF 계열이 아님
        pass
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = cfg["ef_search"]


def save_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path, cfg: dict[str, Any] | None = None) -> faiss.Index:
    index = faiss.read_index(str(path))
    if cfg is not None:
        set_search_params(index, cfg)
    return index
