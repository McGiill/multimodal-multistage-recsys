"""Stage1 후보 (+ 세션 단기 관심) → Stage2 CVR → Stage3 Re-ranking (다양성·신규 상품·MAB 탐색 슬롯)."""

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.common.schemas import (
    PipelineLatency,
    RecommendationItem,
    RecommendResponse,
    SessionContext,
)
from src.recommendation.audience import audience_mask
from src.recommendation.final_rank import score_pool
from src.recommendation.ranking.features import stamp
from src.recommendation.reranking.aggregate import DiversityConfig, diversify
from src.recommendation.reranking.stage2 import get_rerank_service
from src.recommendation.runtime_config import resolve_runtime_config
from src.recommendation.session.blending import blend_final_scores
from src.recommendation.session.encoder import (
    SessionConfig,
    build_session_context,
    get_session_model,
    model_path,
)
from src.recommendation.strategies.base import RecommendStrategy, register
from src.serving.feature_store import get_feature_store
from src.serving.model_loader import get_recommendation_models


@register("multistage")
class MultiStageStrategy(RecommendStrategy):
    def recommend(self, user_id, top_n, at=None, session_id=None, session_clicks=None,
                  audience=None, session_weight=None, recent_clicks=None):
        """session_clicks: 세션 클릭을 직접 넘길 때 (오프라인 평가·시뮬레이터). 없으면 session_id로 Feature Store 조회.

        최근 행동 반영 (configs/recommend.yaml session)
        - live_history: 저장된 이력 뒤에 최근 조회를 붙인다 → User Tower 후보, DeepFM 피처, cold-start 판단이 모두 본다.
          서빙은 Feature Store의 사용자 최근 조회(이번 방문 + 이전 방문, 세션이 끝나도 남음),
          오프라인은 넘겨받은 세션 클릭 (그 시점까지 같은 방문에서 본 상품).
        - final_score_weight: DeepFM과 세션 점수를 percentile로 정규화하여 최종 정렬에도 반영.
        - 오프라인은 session_clicks=[]를 포함하여 명시적으로 전달하며 Redis를 읽지 않는다.
        - 실제 노출은 클라이언트가 /api/impressions로 확정한다. 추천 계산만으로 MAB 통계를 바꾸지 않는다.
        """
        begin = time.perf_counter()
        models = get_recommendation_models()
        store = models.store
        runtime_cfg = resolve_runtime_config(getattr(models, "cfg", None))
        scfg = SessionConfig.from_recommend_config(runtime_cfg)
        now = stamp(at or datetime.now(timezone.utc))
        if now < store.metadata["selection_cutoff"]:
            raise ValueError("Prediction time precedes model validation cutoff")
        if session_clicks is not None:
            clicks = recent = list(session_clicks)
        else:
            features = get_feature_store()
            clicks = features.get_session_clicks(session_id) if session_id else []
            recent = features.get_recent_views(user_id) if scfg.live_history else []
        if recent_clicks is not None:
            recent = list(recent_clicks)
        live = [store.pmap[p] for p in recent if p in store.pmap] if scfg.live_history else []
        n_candidates = models.cfg["candidate"]["n_candidates"]
        allowed_items = audience_mask(store, audience)
        if getattr(models, "final_ranker", None) is not None:
            return self._final_rank(models, runtime_cfg, scfg, user_id, top_n, now, clicks, live,
                                    allowed_items, session_weight, begin)
        options = {"allowed": allowed_items} if allowed_items is not None else {}
        ids, long_scores, state = models.retriever.retrieve(user_id, n_candidates, now, recent=live, **options)
        # 세션: 장기(User Tower) + 단기(현재 세션 클릭) 결합, 세션 관심 상품을 후보에 추가
        session = _session_model(models, runtime_cfg) if clicks else None
        if session is not None:
            available = store.created[1:] <= now
            if allowed_items is not None:
                available &= allowed_items[1:]
            allowed = session.allowed_mask(store.item_ids, available)
            pids, _ = session.merge_candidates(  # 결합 점수 내림차순
                clicks, [store.item_ids[i - 1] for i in ids], long_scores, allowed
            )
            if scfg.blend_candidates:
                pids = pids[:n_candidates]
            ids = np.array([store.pmap[p] for p in pids if p in store.pmap], dtype=np.int64)
        candidate = time.perf_counter()
        rerank = get_rerank_service(models)
        new_ids = rerank.new_item_ids(now)
        if allowed_items is not None:
            new_ids = new_ids[allowed_items[new_ids]]
        all_ids = np.unique(np.r_[ids, new_ids]).astype(np.int64)
        all_scores = models.score(all_ids, state, now)
        weight = scfg.final_score_weight if session_weight is None else session_weight
        if not 0 <= weight <= 1:
            raise ValueError("session_weight must be between 0 and 1")
        mixed = session is not None and weight > 0 and session.encode(clicks) is not None
        if mixed:
            short_scores = session.score(clicks, [store.item_ids[i - 1] for i in all_ids])
            all_scores = blend_final_scores(all_scores, short_scores, weight)
        scores = all_scores[np.searchsorted(all_ids, ids)]
        ranked = time.perf_counter()
        # Stage 3: top_n으로 자르기 전 후보 전체를 넘긴다 (탐색 칸은 상위 top_n 밖에서 고름)
        rerank_options = {"allowed": allowed_items} if allowed_items is not None else {}
        result = rerank.rerank(
            ids, scores, state, now, top_n,
            new_ids=new_ids, new_scores=all_scores[np.searchsorted(all_ids, new_ids)], **rerank_options,
        )
        reranked = time.perf_counter()
        return _response(models, rerank, scfg, user_id, result, mixed, clicks, state, {},
                         (begin, candidate, ranked, reranked))

    def _final_rank(self, models, runtime_cfg, scfg, user_id, top_n, now, clicks, live, allowed_items,
                    session_weight, begin):
        """최종 순위 경로 (final_rank, 2026-11-19): Stage 1 상위 pool_size개 + 관심 카테고리 목록 후보
        → DeepFM 점수 포함 특징 → GBDT → (세션 클릭이 있으면 세션 점수 결합) → 다양성 칸 → Stage 3 규칙."""
        store, ctx = models.store, models.final_context
        q, state = models.retriever.user_query(user_id, now, recent=live)
        stage1, _ = models.retriever.search(q, state, now, ctx.pool_size, allowed=allowed_items)
        session = _session_model(models, runtime_cfg) if clicks else None
        extra = []
        if session is not None:  # 세션 관심 상품도 후보 풀에 넣는다 (기존 merge_candidates와 같은 수)
            available = store.created[1:] <= now
            if allowed_items is not None:
                available &= allowed_items[1:]
            pids, _ = session.top_k(clicks, scfg.n_candidates, session.allowed_mask(store.item_ids, available))
            extra = [store.pmap[p] for p in pids if p in store.pmap]
        candidate = time.perf_counter()
        pool, scores, _, _ = score_pool(models, ctx, models.final_ranker, user_id, now, q, state, stage1,
                                        allowed_items, extra)
        weight = scfg.final_score_weight if session_weight is None else session_weight
        if not 0 <= weight <= 1:
            raise ValueError("session_weight must be between 0 and 1")
        mixed = session is not None and weight > 0 and session.encode(clicks) is not None
        if mixed:
            short = session.score(clicks, [store.item_ids[i - 1] for i in pool])
            scores = blend_final_scores(scores, short, weight).astype(np.float32)
        ranked = time.perf_counter()
        rerank = get_rerank_service(models)
        cfg3 = rerank.config
        div = DiversityConfig.from_config((runtime_cfg.get("final_rank") or {}).get("diversity"))
        tail = div.tail(top_n)
        n_slots = cfg3.new_item_slots + (cfg3.exploration_slots if cfg3.exploration_enabled else 0)
        order = diversify(pool, scores, rerank.design_codes, rerank.exposure,
                          max(top_n - n_slots - tail, 0), tail, div.window, div.penalty)
        age = now - store.created[pool]
        fresh = (age >= 0) & (age < cfg3.new_item_max_age_days * 86400)
        new_ids, new_scores = pool[fresh], scores[fresh]
        if not len(new_ids):  # 풀에 신상품이 없으면 기존처럼 등록 7일 이내 상품 전체를 DeepFM으로
            new_ids = rerank.new_item_ids(now)
            if allowed_items is not None:
                new_ids = new_ids[allowed_items[new_ids]]
            new_scores = models.score(new_ids, state, now)
        result = rerank.rerank(pool, order, state, now, top_n, new_ids=new_ids, new_scores=new_scores,
                               **({"allowed": allowed_items} if allowed_items is not None else {}))
        rerank.record_exposure([r.product_id for r in result.items])
        reranked = time.perf_counter()
        # 정렬에 쓴 다양성 점수 대신 최종 순위 모델 점수를 응답에 싣는다 (cold-start 인기·트렌딩 목록은 그 점수 그대로)
        display = {} if result.is_cold_start else dict(zip((store.item_ids[i - 1] for i in pool), scores.tolist()))
        return _response(models, rerank, scfg, user_id, result, mixed, clicks, state, display,
                         (begin, candidate, ranked, reranked))


def _session_model(models, runtime_cfg):
    bundled = Path(models.folder) / "session_encoder.pt" if hasattr(models, "folder") else None
    if bundled is not None and bundled.exists():
        return get_session_model(str(bundled))
    return get_session_model(str(model_path(runtime_cfg)))


def _response(models, rerank, scfg, user_id, result, mixed, clicks, state, display, marks):
    begin, candidate, ranked, reranked = marks
    items = [
        RecommendationItem(
            product_id=r.product_id,
            score=float(display.get(r.product_id, r.score)),
            reason=f"{r.reason}_session" if mixed and r.reason == models.objective else r.reason,
            is_exploration=r.is_exploration,
        )
        for r in result.items
    ]
    if clicks:
        context = build_session_context(clicks, rerank.catalog[scfg.interest_field], scfg.max_len)
    else:
        context = SessionContext(recent_clicks=[models.store.item_ids[i - 1] for i in state[0] if i > 0])
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
