"""Stage 1 검색 점수 보정: Two-Tower 내적 + 계수 × (최근 7일 인기, 이력의 소분류·중분류 비율, 최근 가중 소분류 비율).

상품 벡터 끝에 [인기, 소분류 one-hot, 중분류 one-hot]을, 사용자 벡터 끝에
[β, γ × 소분류 비율 + ρ × 최근 가중 소분류 비율, δ × 중분류 비율]을 붙이면 내적 하나가 같은 점수가 된다
→ FAISS 내적 검색을 그대로 쓴다. 최근 가중은 같은 소분류 열을 쓰므로 색인은 그대로다.
최근 가중: 이력에서 k개 전 조회의 무게 = 0.5 ** (k / recent_half_life) (가장 최근 = 1).
인기는 기록 기준 시각(frozen_split)마다 달라서 기준 일마다 색인을 한 번 만든다 (5만 개 × 약 200차원, 정확 검색).
계수는 valid 기간 사용자 단위 Recall@300으로 골랐다 (configs/recommend.yaml candidate.retrieval_prior).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np

L2_COL, L3_COL = 1, 2  # store.cats 열 순서 = ranking.features.ITEM_FIELDS (category_l1, category_l2, category_l3, ...)


@dataclass(frozen=True)
class RetrievalPrior:
    popularity: float = 0.0
    category_l3: float = 0.0
    category_l2: float = 0.0
    recent_l3: float = 0.0            # 최근 가중 소분류 비율 계수 (0이면 이전 동작)
    recent_half_life: float = 5.0     # 최근 가중의 반감 (이력 조회 개수)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> RetrievalPrior:
        raw = ((cfg or {}).get("candidate") or {}).get("retrieval_prior") or {}
        weights = {k: float(raw.get(k, 0.0)) for k in ("popularity", "category_l3", "category_l2", "recent_l3")}
        return cls(**weights, recent_half_life=float(raw.get("recent_half_life", 5.0)))

    @property
    def active(self) -> bool:
        return any((self.popularity, self.category_l3, self.category_l2, self.recent_l3))


class PriorIndex:
    """기록 기준 일별 보정 색인과 사용자 쪽 확장 벡터."""

    def __init__(self, vectors: np.ndarray, store: Any, prior: RetrievalPrior, max_cached: int = 2) -> None:
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)  # 상품 번호 1..n → 행 0..n-1
        if len(self.vectors) != len(store.item_ids):
            raise ValueError("Prior index vectors must match the catalog")
        self.store, self.prior, self.max_cached = store, prior, max_cached
        self.n2 = int(store.cats[:, L2_COL].max()) + 1
        self.n3 = int(store.cats[:, L3_COL].max()) + 1
        self._cache: dict[int, faiss.Index] = {}

    def _day(self, at: float) -> int:
        return int(np.floor(self.store.frozen_time(np.array([float(at)]))[0] / 86400))

    def index(self, at: float) -> faiss.Index:
        day = self._day(at)
        if day not in self._cache:
            n, d = self.vectors.shape
            rows = np.arange(n)
            aug = np.zeros((n, d + 1 + self.n3 + self.n2), np.float32)
            aug[:, :d] = self.vectors
            aug[:, d] = self.store.popularity(rows + 1, np.full(n, float(at)))
            aug[rows, d + 1 + self.store.cats[1:, L3_COL]] = 1
            aug[rows, d + 1 + self.n3 + self.store.cats[1:, L2_COL]] = 1
            index = faiss.IndexFlatIP(aug.shape[1])
            index.add(aug)
            if len(self._cache) >= self.max_cached:
                self._cache.pop(next(iter(self._cache)))
            self._cache[day] = index
        return self._cache[day]

    def query(self, q: np.ndarray, history: np.ndarray) -> np.ndarray:
        """q: User Tower 벡터 (1, d), history: 상품 번호 이력 (0 = 빈칸)."""
        h = np.asarray(history)
        h = h[h > 0]
        s3 = np.zeros(self.n3, np.float32)
        s2 = np.zeros(self.n2, np.float32)
        p = self.prior
        user3 = np.zeros(self.n3, np.float32)
        if len(h):
            np.add.at(s3, self.store.cats[h, L3_COL], 1 / len(h))
            np.add.at(s2, self.store.cats[h, L2_COL], 1 / len(h))
            user3 = p.category_l3 * s3
            if p.recent_l3:
                w = 0.5 ** (np.arange(len(h))[::-1] / p.recent_half_life)  # 이력은 오래된 것 → 최근 순
                r3 = np.zeros(self.n3, np.float32)
                np.add.at(r3, self.store.cats[h, L3_COL], w / w.sum())
                user3 = user3 + p.recent_l3 * r3
        return np.concatenate([np.ravel(q), [p.popularity], user3, p.category_l2 * s2]).astype(np.float32)[None]
