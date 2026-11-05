"""cold-start용 인기·트렌딩 목록. 요청 시각 이전 기록만 쓴다 (test 기간 정보가 섞이지 않게).

인기   = 기준 시각 전 window_days일 구매 수 순 (동률이면 조회 수 순, 조회도 구매도 없으면 제외)
트렌딩 = 최근 trending_recent_days일의 하루 평균 조회 / 그 전 window_days일의 하루 평균 조회 (둘 다 +1 보정)
         높은 순. 최근 조회가 trending_min_views 미만인 상품은 제외 (조회 1~2번짜리 우연 급상승 방지)
score  = 목록 안 최댓값을 1로 맞춘 0~1 값 (인기: 구매 수, 트렌딩: 증가 배율)

상품 번호는 Stage 2와 같은 1부터 시작하는 내부 번호 (store.item_ids[id - 1] = product_id).
담당: D
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.common.config import load_config

DAY = 86400.0


@dataclass
class PopularityConfig:
    window_days: int = 7
    trending_recent_days: int = 1
    trending_min_views: int = 3
    list_size: int = 100

    def __post_init__(self) -> None:
        if min(self.window_days, self.trending_recent_days, self.list_size) < 1:
            raise ValueError("window_days, trending_recent_days, list_size는 1 이상이어야 함")
        if self.trending_min_views < 0:
            raise ValueError("trending_min_views는 0 이상이어야 함")

    @classmethod
    def from_recommend_config(cls, raw: Mapping[str, Any] | None = None) -> PopularityConfig:
        """configs/recommend.yaml 의 reranking.popularity 항목을 읽는다."""
        if raw is None:
            raw = load_config("recommend")
        p = raw["reranking"]["popularity"]
        return cls(**{k: p[k] for k in cls.__dataclass_fields__ if k in p})


class PopularityLists:
    """조회·구매 기록(시간순 정렬) → 기준 시각별 인기·트렌딩 목록.

    view_times / view_items: 조회 이벤트의 unix 초와 상품 번호 (시간 오름차순)
    purchase_times / purchase_items: 구매 이벤트 (시간 오름차순)
    n_items: 상품 수 (번호는 1..n_items)
    """

    def __init__(self, view_times: np.ndarray, view_items: np.ndarray,
                 purchase_times: np.ndarray, purchase_items: np.ndarray,
                 n_items: int, config: PopularityConfig | None = None) -> None:
        for t in (view_times, purchase_times):
            if len(t) > 1 and np.any(np.diff(t) < 0):
                raise ValueError("기록은 시간 오름차순이어야 함")
        self.views = (np.asarray(view_times, dtype=float), np.asarray(view_items, dtype=np.int64))
        self.purchases = (np.asarray(purchase_times, dtype=float), np.asarray(purchase_items, dtype=np.int64))
        self.n_items = n_items
        self.config = config or PopularityConfig.from_recommend_config()

    def _counts(self, log: tuple[np.ndarray, np.ndarray], start: float, end: float) -> np.ndarray:
        """[start, end) 구간의 상품 번호별 이벤트 수 (길이 n_items + 1, 0번은 비움)."""
        times, items = log
        lo, hi = np.searchsorted(times, [start, end], side="left")
        return np.bincount(items[lo:hi], minlength=self.n_items + 1)

    def _top(self, keys: tuple[np.ndarray, ...], valid: np.ndarray, score: np.ndarray,
             k: int) -> tuple[np.ndarray, np.ndarray]:
        """valid인 번호 중 keys 사전순 내림차순 상위 k개와 0~1 score."""
        ids = np.flatnonzero(valid)
        order = np.lexsort(tuple(-key[ids] for key in reversed(keys)))[:k]
        ids = ids[order]
        s = score[ids].astype(np.float32)
        return ids, s / s.max() if len(s) and s.max() > 0 else s

    def popular(self, at: float, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        start = at - cfg.window_days * DAY
        buys, views = self._counts(self.purchases, start, at), self._counts(self.views, start, at)
        valid = (buys + views) > 0
        valid[0] = False
        return self._top((buys, views), valid, buys if buys.max() > 0 else views, k or cfg.list_size)

    def trending(self, at: float, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        split = at - cfg.trending_recent_days * DAY
        recent = self._counts(self.views, split, at)
        before = self._counts(self.views, split - cfg.window_days * DAY, split)
        growth = (recent / cfg.trending_recent_days + 1) / (before / cfg.window_days + 1)
        valid = (recent >= max(cfg.trending_min_views, 1))
        valid[0] = False
        return self._top((growth, recent), valid, growth, k or cfg.list_size)
