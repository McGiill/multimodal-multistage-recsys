"""Stage1 후보 → Stage2 CVR. Stage3는 미연결이며 is_exploration=False."""

import time
from datetime import UTC, datetime

import numpy as np

from src.common.schemas import (
    PipelineLatency,
    RecommendationItem,
    RecommendResponse,
    SessionContext,
)
from src.recommendation.ranking.features import stamp
from src.recommendation.strategies.base import RecommendStrategy, register
from src.serving.model_loader import get_recommendation_models


@register("multistage")
class MultiStageStrategy(RecommendStrategy):
    def recommend(self, user_id, top_n, at=None):
        begin = time.perf_counter()
        models = get_recommendation_models()
        now = stamp(at or datetime.now(UTC))
        if now < models.store.metadata["selection_cutoff"]:
            raise ValueError("Prediction time precedes model validation cutoff")
        ids, _, state = models.retriever.retrieve(
            user_id, models.cfg["candidate"]["n_candidates"], now
        )
        candidate = time.perf_counter()
        scores = models.score(ids, state, now)
        order = np.argsort(-scores, kind="stable")[:top_n]
        ranked = time.perf_counter()
        items = [
            RecommendationItem(
                product_id=models.store.item_ids[ids[i] - 1],
                score=float(scores[i]),
                reason="cvr",
                is_exploration=False,
            )
            for i in order
        ]
        return RecommendResponse(
            user_id=user_id,
            recommendations=items,
            pipeline_latency=PipelineLatency(
                candidate_ms=(candidate - begin) * 1000,
                ranking_ms=(ranked - candidate) * 1000,
                reranking_ms=0,
                total_ms=(ranked - begin) * 1000,
            ),
            session_context=SessionContext(
                recent_clicks=[models.store.item_ids[i - 1] for i in state[0] if i > 0]
            ),
        )
