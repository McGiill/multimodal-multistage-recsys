"""Multi-Stage 파이프라인: Stage1 Candidate → Stage2 Ranking → Stage3 Re-ranking.

각 Stage 처리 시간을 측정해 pipeline_latency 로 반환한다.
담당: C(Stage1/2 연결) + D(Stage3, 최종 조립)
"""

from __future__ import annotations

import time

from src.common.schemas import (
    PipelineLatency,
    RecommendationItem,
    RecommendResponse,
    SessionContext,
)
from src.recommendation.strategies.base import RecommendStrategy, register


@register("multistage")
class MultiStageStrategy(RecommendStrategy):
    def recommend(self, user_id: str, top_n: int) -> RecommendResponse:
        t0 = time.perf_counter()
        # TODO(C): candidates = retriever.retrieve(user_id, n=300)
        t1 = time.perf_counter()
        # TODO(C): ranked = ranker.score(user_id, candidates)
        t2 = time.perf_counter()
        # TODO(D): final = reranker.apply(user_id, ranked, top_n)  # diversity / cold start / new item / MAB
        items: list[RecommendationItem] = [
            RecommendationItem(product_id="P00000", score=0.0, reason="stub", is_exploration=False)
        ]
        t3 = time.perf_counter()

        return RecommendResponse(
            user_id=user_id,
            recommendations=items[:top_n],
            pipeline_latency=PipelineLatency(
                candidate_ms=(t1 - t0) * 1000,
                ranking_ms=(t2 - t1) * 1000,
                reranking_ms=(t3 - t2) * 1000,
                total_ms=(t3 - t0) * 1000,
            ),
            session_context=SessionContext(),  # TODO(D): Redis 세션 피처
        )
