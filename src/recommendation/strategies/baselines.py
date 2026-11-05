"""베이스라인 전략: @register('two_tower_only'), @register('popularity'), @register('recency').

configs/recommend.yaml 의 strategy 값만 바꾸면 multistage 대신 쓰인다 (동료평가: "유사도순 → 최신순" 교체).
- two_tower_only : Stage 1 후보를 User Tower 유사도순 그대로 (명세 베이스라인 "Ranking 없이 후보만")
- popularity     : 최근 window_days일 구매 수 순 (cold-start 인기 목록과 같은 기준, reranking/popularity.py). A/B Control
- recency        : 요청 시각까지 등록된 상품 중 등록일 최신순. 같은 디자인 색상 변형(product_code)은 1개만
세션·랭킹·탐색 칸이 없으므로 session_context는 비우고, 처리 시간은 candidate_ms에 모두 넣는다.

담당: D
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np

from src.common.schemas import PipelineLatency, RecommendationItem, RecommendResponse, SessionContext
from src.recommendation.ranking.features import stamp
from src.recommendation.reranking.stage2 import POPULAR_REASON, get_rerank_service
from src.recommendation.strategies.base import RecommendStrategy, register
from src.serving.model_loader import get_recommendation_models

TWO_TOWER_REASON = "two_tower"
RECENCY_REASON = "recency"


def _response(user_id: str, items: list[RecommendationItem], begin: float) -> RecommendResponse:
    ms = (time.perf_counter() - begin) * 1000
    return RecommendResponse(
        user_id=user_id,
        recommendations=items,
        pipeline_latency=PipelineLatency(candidate_ms=ms, ranking_ms=0.0, reranking_ms=0.0, total_ms=ms),
        session_context=SessionContext(),
    )


@register("two_tower_only")
class TwoTowerOnlyStrategy(RecommendStrategy):
    def recommend(self, user_id: str, top_n: int, at: datetime | None = None) -> RecommendResponse:
        begin = time.perf_counter()
        models = get_recommendation_models()
        now = stamp(at or datetime.now(timezone.utc))
        if now < models.store.metadata["selection_cutoff"]:
            raise ValueError("Prediction time precedes model validation cutoff")
        ids, scores, _ = models.retriever.retrieve(user_id, top_n, now)
        items = [
            RecommendationItem(product_id=models.store.item_ids[i - 1], score=float(s), reason=TWO_TOWER_REASON)
            for i, s in zip(ids, scores)
        ]
        return _response(user_id, items, begin)


@register("popularity")
class PopularityStrategy(RecommendStrategy):
    def recommend(self, user_id: str, top_n: int, at: datetime | None = None) -> RecommendResponse:
        begin = time.perf_counter()
        rerank = get_rerank_service(get_recommendation_models())
        if rerank.popularity is None:
            raise ValueError("Popularity lists unavailable")
        now = stamp(at or datetime.now(timezone.utc))
        ids, scores = rerank.popularity.popular(rerank.history_time(now), top_n)
        items = [
            RecommendationItem(product_id=rerank.item_ids[i - 1], score=float(s), reason=POPULAR_REASON)
            for i, s in zip(ids, scores)
        ]
        return _response(user_id, items, begin)


@register("recency")
class RecencyStrategy(RecommendStrategy):
    def recommend(self, user_id: str, top_n: int, at: datetime | None = None) -> RecommendResponse:
        begin = time.perf_counter()
        rerank = get_rerank_service(get_recommendation_models())
        now = stamp(at or datetime.now(timezone.utc))
        created = rerank.created_seconds
        order = np.argsort(-np.where(created <= now, created, -np.inf), kind="stable")
        codes = rerank.catalog["product_code"].to_numpy()
        items, seen = [], set()
        for j in order:
            if created[j] > now or len(items) >= top_n:
                break
            key = codes[j] if isinstance(codes[j], str) else rerank.item_ids[j]  # 코드 없으면 상품 단위
            if key in seen:
                continue
            seen.add(key)
            age_days = (now - created[j]) / 86400
            items.append(RecommendationItem(product_id=rerank.item_ids[j], score=float(1 / (1 + age_days)),
                                            reason=RECENCY_REASON))
        return _response(user_id, items, begin)
