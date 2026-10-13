"""text / image / hybrid 검색.

검색 흐름
1. 쿼리를 CLIP으로 인코딩 (텍스트 → q_t, 이미지 → q_i)
2. FAISS 인덱스에서 후보를 넉넉히(candidate_k) 가져온다
3. 후보를 저장된 임베딩으로 정밀 재채점해서 top_k 반환

점수 (configs/search.yaml)
- image  : 이미지→상품 이미지 코사인 유사도
- text   : w * (텍스트→상품 이미지) + (1-w) * (텍스트→상품 텍스트), w = text_query.image_weight
- hybrid : w * 텍스트 경로 + (1-w) * 이미지 경로,                 w = hybrid.text_weight
두 점수를 섞을 때는 척도가 달라서(텍스트↔이미지 0.2~0.35, 텍스트↔텍스트 0.7~0.9)
후보 안에서 0~1로 min-max 정규화한 뒤 가중합한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from src.common.config import load_config
from src.common.faiss_utils import load_index
from src.common.schemas import SearchType
from src.search.catalog import index_dir, load_catalog
from src.search.encoder import BaseEncoder, build_encoder


@dataclass
class SearchResult:
    search_type: SearchType
    product_ids: list[str]
    scores: list[float]
    total_count: int                      # 재채점한 후보 풀의 크기
    rows: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, dtype=int))
    items: list[dict[str, Any]] = field(default_factory=list)


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.ones_like(x)


class Searcher:
    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        encoder: BaseEncoder | None = None,
        device: str | None = None,
    ) -> None:
        self.cfg = cfg or load_config("search")
        d = index_dir(self.cfg)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if meta["model_name"] != self.cfg["clip"]["model_name"]:
            raise ValueError(f"인덱스 모델({meta['model_name']})과 설정 모델이 다릅니다")

        ids = pd.read_parquet(d / "product_ids.parquet")["product_id"]
        catalog = load_catalog(self.cfg, columns=["product_id", "name", "price"]).set_index("product_id")
        catalog = catalog.loc[ids].reset_index()
        self.product_ids = catalog["product_id"].to_numpy()
        self.names = catalog["name"].to_numpy()
        self.prices = catalog["price"].to_numpy()

        self.image_emb = np.load(d / "image_emb.npy")
        self.text_emb = np.load(d / "text_emb.npy")
        self.image_index = load_index(d / "image.index", self.cfg["index"])
        self.text_index = load_index(d / "text.index", self.cfg["index"])

        clip = self.cfg["clip"]
        self.encoder = encoder or build_encoder(
            clip["model_name"], device=device or clip["serve_device"], pad_square=meta["pad_to_square"]
        )

    # ------------------------------------------------------------------ 후보 + 점수
    def _candidates(self, index: Any, q: np.ndarray) -> np.ndarray:
        _, idx = index.search(q.reshape(1, -1), self.cfg["retrieval"]["candidate_k"])
        return idx[0][idx[0] >= 0]

    def _text_route(self, q_t: np.ndarray, cand: np.ndarray) -> np.ndarray:
        w = self.cfg["text_query"]["image_weight"]
        return w * _minmax(self.image_emb[cand] @ q_t) + (1 - w) * _minmax(self.text_emb[cand] @ q_t)

    def search_vectors(
        self, q_text: np.ndarray | None, q_image: np.ndarray | None, top_k: int
    ) -> SearchResult:
        """이미 인코딩된 쿼리 벡터로 검색 (평가 스크립트가 배치 인코딩 후 호출)."""
        if q_text is not None and q_image is not None:
            search_type = SearchType.HYBRID
            cand = np.union1d(
                np.union1d(self._candidates(self.image_index, q_text), self._candidates(self.text_index, q_text)),
                self._candidates(self.image_index, q_image),
            )
            w = self.cfg["hybrid"]["text_weight"]
            scores = w * _minmax(self._text_route(q_text, cand)) + (1 - w) * _minmax(self.image_emb[cand] @ q_image)
        elif q_text is not None:
            search_type = SearchType.TEXT
            cand = np.union1d(self._candidates(self.image_index, q_text), self._candidates(self.text_index, q_text))
            scores = self._text_route(q_text, cand)
        elif q_image is not None:
            search_type = SearchType.IMAGE
            cand = self._candidates(self.image_index, q_image)
            scores = self.image_emb[cand] @ q_image
        else:
            raise ValueError("query_text 또는 image 중 하나는 필요합니다")

        order = np.argsort(-scores)[:top_k]
        rows = cand[order]
        return SearchResult(
            search_type=search_type,
            product_ids=self.product_ids[rows].tolist(),
            scores=[round(float(s), 4) for s in scores[order]],
            total_count=len(cand),
            rows=rows,
        )

    # ------------------------------------------------------------------ API 진입점
    def search(self, query_text: str | None, image: Image.Image | None, top_k: int = 10) -> SearchResult:
        q_text = self.encoder.encode_texts([query_text])[0] if query_text else None
        q_image = self.encoder.encode_images([image])[0] if image is not None else None
        result = self.search_vectors(q_text, q_image, top_k)
        result.items = [
            {"product_id": pid, "name": str(self.names[r]), "score": score, "price": float(self.prices[r])}
            for pid, score, r in zip(result.product_ids, result.scores, result.rows)
        ]
        return result
