"""검색에서 사전 계산한 이미지 특징을 추천 상품 ID 순서로 정렬한다.

학습 시에만 검색 산출물을 읽는다. 정렬된 벡터는 TwoTower checkpoint buffer에
포함되므로 서빙·재로딩 시 검색 인덱스 폴더가 없어도 같은 입력을 사용한다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import ROOT


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_image_features(item_ids: list[str], config: dict) -> tuple[np.ndarray | None, dict | None]:
    """반환 벡터의 0행은 padding, 이후 행은 item_ids와 정확히 대응한다.

    검색 카탈로그의 여분 상품은 허용하지만 누락·중복 ID와 잘못된 벡터는
    조용히 0으로 대체하지 않는다. 이미지 특징을 끈 기존 모델은 파일을 읽지 않는다.
    """
    if not config.get("enabled", False):
        return None, None
    if not item_ids or len(item_ids) != len(set(item_ids)):
        raise ValueError("Recommendation image features need unique nonempty item IDs")
    directory = config.get("embeddings_dir")
    if not directory:
        raise ValueError("candidate.image_features.embeddings_dir is required when enabled")
    folder = (ROOT / directory).resolve()
    ids_path, vectors_path = folder / "product_ids.parquet", folder / "image_emb.npy"
    ids = pd.read_parquet(ids_path, columns=["product_id"])["product_id"]
    if ids.isna().any() or ids.duplicated().any() or not ids.map(lambda x: isinstance(x, str)).all():
        raise ValueError("Image feature product IDs must be non-null unique strings")
    source = np.load(vectors_path, allow_pickle=False, mmap_mode="r")
    if (source.ndim != 2 or source.shape[0] != len(ids) or source.shape[1] == 0
            or not np.issubdtype(source.dtype, np.floating)):
        raise ValueError("Image feature matrix must be a floating 2D array aligned with product_ids")
    positions = pd.Index(ids).get_indexer(item_ids)
    if (positions < 0).any():
        missing = [item_ids[i] for i in np.flatnonzero(positions < 0)[:5]]
        raise ValueError(f"Image features missing recommendation products: {missing}")
    selected = np.asarray(source[positions], dtype=np.float32)
    norm = np.linalg.norm(selected, axis=1, keepdims=True)
    if not np.isfinite(selected).all() or not np.isfinite(norm).all() or (norm <= 1e-8).any():
        raise ValueError("Image features must contain finite, nonzero vectors")
    vectors = np.zeros((len(item_ids) + 1, source.shape[1]), dtype=np.float32)
    vectors[1:] = selected / norm
    metadata = {
        "source_dir": str(folder),
        "source_items": len(ids),
        "aligned_items": len(item_ids),
        "input_dim": source.shape[1],
        "normalization": "l2; padding row zero",
        "image_emb_sha256": file_sha256(vectors_path),
        "product_ids_sha256": file_sha256(ids_path),
        "aligned_ids_sha256": hashlib.sha256(json.dumps(item_ids).encode()).hexdigest(),
    }
    return vectors, metadata
