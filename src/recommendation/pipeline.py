"""Stage1 후보 (+ 세션 단기 관심) → Stage2 CVR → Stage3 Re-ranking (다양성·신규 상품·MAB 탐색 슬롯)."""

import time
from datetime import datetime, timezone

import numpy as np

from src.common.schemas import (
    PipelineLatency,
    RecommendationItem,
    RecommendResponse,
    SessionContext,
)
from src.recommendation.ranking.features import stamp
from src.recommendation.reranking.stage2 import get_rerank_service
from src.recommendation.session.encoder import (
    SessionConfig,
    build_session_context,
    get_session_model,
)
from src.recommendation.strategies.base import RecommendStrategy, register
from src.serving.feature_store import get_feature_store
from src.serving.model_loader import get_recommendation_models


@register("multistage")
class MultiStageStrategy(RecommendStrategy):
    def recommend(self, user_id, top_n, at=None, session_id=None, session_clicks=None):
        """session_clicks: 세션 클릭을 직접 넘길 때 (오프라인 평가·시뮬레이터). 없으면 session_id로 Feature Store 조회."""
        begin = time.perf_counter()
        models = get_recommendation_models()
        now = stamp(at or datetime.now(timezone.utc))
        if now < models.store.metadata["selection_cutoff"]:
            raise ValueError("Prediction time precedes model validation cutoff")
        ids, long_scores, state = models.retriever.retrieve(
            user_id, models.cfg["candidate"]["n_candidates"], now
        )
        # 세션: 장기(User Tower) + 단기(현재 세션 클릭) 결합, 세션 관심 상품을 후보에 추가
        if session_clicks is not None:
            clicks = list(session_clicks)
        else:
            clicks = get_feature_store().get_session_clicks(session_id) if session_id else []
        session = get_session_model() if clicks else None
        if session is not None:
            store = models.store
            allowed = session.allowed_mask(store.item_ids, store.created[1:] <= now)
            pids, _ = session.merge_candidates(
                clicks, [store.item_ids[i - 1] for i in ids], long_scores, allowed
            )
            ids = np.array([store.pmap[p] for p in pids if p in store.pmap], dtype=np.int64)
        candidate = time.perf_counter()
        scores = models.score(ids, state, now)
        ranked = time.perf_counter()
        # Stage 3: top_n으로 자르기 전 후보 전체를 넘긴다 (탐색 칸은 상위 top_n 밖에서 고름)
        rerank = get_rerank_service(models)
        new_ids = rerank.new_item_ids(now)
        result = rerank.rerank(
            ids, scores, state, now, top_n,
            new_ids=new_ids, new_scores=models.score(new_ids, state, now),
        )
        rerank.record_shown(user_id, result)  # MAB 탐색 칸 노출 (클릭은 POST /api/events)
        reranked = time.perf_counter()
        items = [
            RecommendationItem(
                product_id=r.product_id,
                score=r.score,
                reason=r.reason,
                is_exploration=r.is_exploration,
            )
            for r in result.items
        ]
        if clicks:
            cfg = SessionConfig.from_recommend_config()
            context = build_session_context(clicks, rerank.catalog[cfg.interest_field], cfg.max_len)
        else:
            context = SessionContext(
                recent_clicks=[models.store.item_ids[i - 1] for i in state[0] if i > 0]
            )
        return RecommendResponse(
            user_id=user_id,
            recommendations=items,
            pipeline_latency=PipelineLatency(
                candidate_ms=(candidate - begin) * 1000,
                ranking_ms=(ranked - candidate) * 1000,
                reranking_ms=(reranked - ranked) * 1000,
                total_ms=(reranked - begin) * 1000,
            ),
            session_context=context,
        )
