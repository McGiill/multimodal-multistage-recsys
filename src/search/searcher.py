"""text / image / hybrid 검색.

검색 흐름
1. 쿼리를 CLIP으로 인코딩 (텍스트 → q_t, 이미지 → q_i)
2. FAISS 인덱스에서 후보를 넉넉히(candidate_k) 가져온다 (텍스트 쿼리는 BM25 상위 candidate_k개도)
3. 후보를 저장된 임베딩으로 정밀 재채점해서 top_k 반환

점수 (configs/search.yaml)
- text   : CLIP = w * (텍스트→상품 이미지) + (1-w) * (텍스트→상품 텍스트),  w = text_query.image_weight
           → (1-l) * CLIP + l * BM25,                                      l = text_query.bm25_weight
           (상품명처럼 CLIP이 약한 고유 단어는 BM25가, 비슷한 말·표현은 CLIP이 잡는다)
- image  : 이미지→상품 이미지 유사도 + 색·소분류 가점 (사진의 색·종류 추정 확률, src/search/attributes.py)
           가점 비중 = image_query.color_weight / category_weight (둘 다 0이면 코사인 유사도 그대로)
- hybrid : w * 텍스트 경로 + (1-w) * 이미지 경로,                            w = hybrid.text_weight
점수를 섞을 때는 척도가 달라서(텍스트↔이미지 0.2~0.35, 텍스트↔텍스트 0.7~0.9, BM25 0~수십)
후보 안에서 0~1로 min-max 정규화한 뒤 가중합한다.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from src.common.config import load_config
from src.common.faiss_utils import load_index
from src.common.schemas import SearchType
from src.search.attributes import AttributeProbe
from src.search.bm25 import BM25Index, tokenize
from src.search.catalog import index_dir, load_catalog, product_texts
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
        columns = ["product_id", "name", "price", "category_l1", "category_l3", "color"]
        catalog = load_catalog(self.cfg, columns=columns).set_index("product_id")
        catalog = catalog.loc[ids].reset_index()
        self.product_ids = catalog["product_id"].to_numpy()
        self.names = catalog["name"].to_numpy()
        self.prices = catalog["price"].to_numpy()
        self.category_l1 = catalog["category_l1"].to_numpy()
        self.category_l3 = catalog["category_l3"].to_numpy()
        self.colors = catalog["color"].to_numpy()

        self.image_emb = np.load(d / "image_emb.npy")
        self.text_emb = np.load(d / "text_emb.npy")
        self.image_index = load_index(d / "image.index", self.cfg["index"])
        self.text_index = load_index(d / "text.index", self.cfg["index"])

        clip = self.cfg["clip"]
        self.encoder = encoder or build_encoder(
            clip["model_name"], device=device or clip["serve_device"], pad_square=meta["pad_to_square"]
        )
        self.bm25 = self._build_bm25() if self.cfg["text_query"].get("bm25_weight", 0) else None
        self.attributes = self._load_attributes(d / "attributes.npz")

    def _build_bm25(self) -> BM25Index:
        columns = ["product_id", "name", "color", "pattern", "category_l3", "description"]
        catalog = load_catalog(self.cfg, columns=columns).set_index("product_id").loc[self.product_ids].reset_index()
        return BM25Index([tokenize(t) for t in product_texts(catalog)])

    def _load_attributes(self, path: Any) -> AttributeProbe | None:
        if path.exists():
            return AttributeProbe(path, {"color": self.colors, "category": self.category_l3})
        if any(self._attribute_weights().values()):
            print(f"[searcher] {path.name} 없음 → 이미지 검색에 색·소분류 가점 없이 동작 (python -m src.search.attributes)")
        return None

    def _attribute_weights(self) -> dict[str, float]:
        image_query = self.cfg.get("image_query", {})
        return {"color": image_query.get("color_weight", 0), "category": image_query.get("category_weight", 0)}

    # ------------------------------------------------------------------ 후보 + 점수
    def _candidates(self, index: Any, q: np.ndarray) -> np.ndarray:
        _, idx = index.search(q.reshape(1, -1), self.cfg["retrieval"]["candidate_k"])
        return idx[0][idx[0] >= 0]

    def _lexical(self, query_text: str | None) -> np.ndarray | None:
        """전체 상품의 BM25 점수. bm25_weight가 0이거나 쿼리 문장이 없으면 None."""
        if not query_text or not self.cfg["text_query"].get("bm25_weight", 0):
            return None
        if self.bm25 is None:  # 가중치 탐색처럼 실행 중에 비중을 바꾼 경우 처음 쓸 때 만든다
            self.bm25 = self._build_bm25()
        return self.bm25.scores(query_text)

    def _lexical_candidates(self, lexical: np.ndarray | None) -> list[np.ndarray]:
        if lexical is None:
            return []
        k = min(self.cfg["retrieval"]["candidate_k"], len(lexical) - 1)
        top = np.argpartition(-lexical, k)[:k]
        return [top[lexical[top] > 0]]

    def _text_route(self, q_t: np.ndarray, cand: np.ndarray, lexical: np.ndarray | None = None) -> np.ndarray:
        w = self.cfg["text_query"]["image_weight"]
        clip = w * _minmax(self.image_emb[cand] @ q_t) + (1 - w) * _minmax(self.text_emb[cand] @ q_t)
        if lexical is None:
            return clip
        lw = self.cfg["text_query"]["bm25_weight"]
        return (1 - lw) * _minmax(clip) + lw * _minmax(lexical[cand])

    def _image_route(self, q_i: np.ndarray, cand: np.ndarray) -> np.ndarray:
        similarity = self.image_emb[cand] @ q_i
        weights = self._attribute_weights()
        if self.attributes is None or not any(weights.values()):
            return similarity
        return _minmax(similarity) + self.attributes.bonus(q_i, cand, weights)

    def search_vectors(
        self, q_text: np.ndarray | None, q_image: np.ndarray | None, top_k: int, query_text: str | None = None
    ) -> SearchResult:
        """이미 인코딩된 쿼리 벡터로 검색 (평가 스크립트가 배치 인코딩 후 호출). BM25는 query_text가 있을 때만."""
        lexical = self._lexical(query_text) if q_text is not None else None
        if q_text is not None and q_image is not None:
            search_type = SearchType.HYBRID
            cand = np.unique(np.concatenate([
                self._candidates(self.image_index, q_text), self._candidates(self.text_index, q_text),
                self._candidates(self.image_index, q_image), *self._lexical_candidates(lexical),
            ]))
            w = self.cfg["hybrid"]["text_weight"]
            scores = w * _minmax(self._text_route(q_text, cand, lexical)) + (1 - w) * _minmax(
                self._image_route(q_image, cand)
            )
        elif q_text is not None:
            search_type = SearchType.TEXT
            cand = np.unique(np.concatenate([
                self._candidates(self.image_index, q_text), self._candidates(self.text_index, q_text),
                *self._lexical_candidates(lexical),
            ]))
            scores = self._text_route(q_text, cand, lexical)
        elif q_image is not None:
            search_type = SearchType.IMAGE
            cand = self._candidates(self.image_index, q_image)
            scores = self._image_route(q_image, cand)
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

    # ------------------------------------------------------------------ 조건 안에서 검색 (스타일 사진 검색용)
    def select_rows(self, l3: Iterable[str] | None = None, l1: Iterable[str] | None = None,
                    colors: Iterable[str] | None = None) -> np.ndarray:
        """소분류 / 고객층 / 색 조건을 모두 만족하는 상품의 행 번호. None인 조건은 걸지 않는다."""
        mask = np.ones(len(self.product_ids), dtype=bool)
        for values, column in ((l3, self.category_l3), (l1, self.category_l1), (colors, self.colors)):
            if values is not None:
                mask &= np.isin(column, list(values))
        return np.flatnonzero(mask)

    def rank_rows(
        self, q_image: np.ndarray, rows: np.ndarray, top_k: int,
        q_text: np.ndarray | None = None, query_text: str | None = None,
    ) -> SearchResult:
        """주어진 상품들 안에서 이미지 유사도순. 조건으로 좁힌 후보는 수천 개라 인덱스 없이 전부 계산한다.
        문장(q_text)이 있으면 텍스트 경로와 이미지 유사도를 hybrid.text_weight로 가중합한다."""
        scores = self.image_emb[rows] @ q_image
        use_text = q_text is not None and len(rows) > 0
        if use_text:
            w = self.cfg["hybrid"]["text_weight"]
            text_scores = self._text_route(q_text, rows, self._lexical(query_text))
            scores = w * _minmax(text_scores) + (1 - w) * _minmax(scores)
        order = np.argsort(-scores)[:top_k]
        result = SearchResult(
            search_type=SearchType.HYBRID if use_text else SearchType.IMAGE,
            product_ids=self.product_ids[rows[order]].tolist(),
            scores=[round(float(s), 4) for s in scores[order]],
            total_count=len(rows),
            rows=rows[order],
        )
        result.items = self._items(result)
        return result

    # ------------------------------------------------------------------ API 진입점
    def search(self, query_text: str | None, image: Image.Image | None, top_k: int = 10) -> SearchResult:
        q_text = self.encoder.encode_texts([query_text])[0] if query_text else None
        q_image = self.encoder.encode_images([image])[0] if image is not None else None
        return self.search_encoded(q_text, q_image, top_k, query_text)

    def search_encoded(
        self, q_text: np.ndarray | None, q_image: np.ndarray | None, top_k: int, query_text: str | None = None
    ) -> SearchResult:
        """search_vectors + 결과의 상품 정보 (사진을 이미 인코딩한 스타일 사진 검색에서 호출)."""
        result = self.search_vectors(q_text, q_image, top_k, query_text)
        result.items = self._items(result)
        return result

    def _items(self, result: SearchResult) -> list[dict[str, Any]]:
        return [
            {"product_id": pid, "name": str(self.names[r]), "score": score, "price": float(self.prices[r])}
            for pid, score, r in zip(result.product_ids, result.scores, result.rows)
        ]
