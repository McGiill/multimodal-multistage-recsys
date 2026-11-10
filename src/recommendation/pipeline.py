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
        """session_clicks: 세션 클릭을 직접 넘길 때 (오프라인 평가·시뮬레이터). 없으면 session_id로 Feature Store 조회.

        최근 행동 반영 (configs/recommend.yaml session)
        - live_history: 저장된 이력 뒤에 최근 조회를 붙인다 → User Tower 후보, DeepFM 피처, cold-start 판단이 모두 본다.
          서빙은 Feature Store의 사용자 최근 조회(이번 방문 + 이전 방문, 세션이 끝나도 남음),
          오프라인은 넘겨받은 세션 클릭 (그 시점까지 같은 방문에서 본 상품).
        - blend_candidates: 세션 결합 점수 = alpha * 장기 + (1 - alpha) * 세션 (merge_candidates).
          이 점수 상위 n_candidates개로 후보를 자르고, 최종 순서는 DeepFM이 정한다.
        """
        begin = time.perf_counter()
        models = get_recommendation_models()
        store = models.store
        scfg = SessionConfig.from_recommend_config()
        now = stamp(at or datetime.now(timezone.utc))
        if now < store.metadata["selection_cutoff"]:
            raise ValueError("Prediction time precedes model validation cutoff")
        if session_clicks is not None:
            clicks = recent = list(session_clicks)
        else:
            features = get_feature_store()
            clicks = features.get_session_clicks(session_id) if session_id else []
            recent = features.get_recent_views(user_id) if scfg.live_history else []
        live = [store.pmap[p] for p in recent if p in store.pmap] if scfg.live_history else []
        n_candidates = models.cfg["candidate"]["n_candidates"]
        ids, long_scores, state = models.retriever.retrieve(user_id, n_candidates, now, recent=live)
        # 세션: 장기(User Tower) + 단기(현재 세션 클릭) 결합, 세션 관심 상품을 후보에 추가
        session = get_session_model() if clicks else None
        if session is not None:
            allowed = session.allowed_mask(store.item_ids, store.created[1:] <= now)
            pids, _ = session.merge_candidates(  # 결합 점수 내림차순
                clicks, [store.item_ids[i - 1] for i in ids], long_scores, allowed
            )
            if scfg.blend_candidates:
                pids = pids[:n_candidates]
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
            context = build_session_context(clicks, rerank.catalog[scfg.interest_field], scfg.max_len)
        else:
            context = SessionContext(
                recent_clicks=[store.item_ids[i - 1] for i in state[0] if i > 0]
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
