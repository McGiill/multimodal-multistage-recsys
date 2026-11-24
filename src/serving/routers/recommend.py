"""GET /api/recommend. at은 재현 가능한 오프라인 데모 시점(시간대 필수), session_id는 현재 방문."""

from datetime import datetime
import time
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Response

import src.recommendation.pipeline  # noqa: F401
import src.recommendation.strategies.baselines  # noqa: F401
from src.common.config import load_config
from src.common.schemas import RecommendResponse
from src.recommendation.reranking.stage2 import get_rerank_service
from src.recommendation.strategies.base import get_strategy
from src.serving.outfit_preferences import fetch_outfit_preferences
from src.serving.event_log import get_event_log
from src.serving.feedback import JournalBanditStats
from src.serving.model_loader import get_recommendation_models

router = APIRouter()


@router.get("/recommend", response_model=RecommendResponse)
def recommend(
    response: Response,
    user_id: str,
    top_n: int = Query(default=10, ge=1, le=100),
    at: datetime | None = None,
    session_id: str | None = None,
    use_outfit_preferences: bool = False,
    audience: Literal["all", "women", "men", "kids"] = "all",
) -> RecommendResponse:
    try:
        strategy = get_strategy(load_config("recommend")["strategy"])
        models = get_recommendation_models()
        service = get_rerank_service(models)
        log = get_event_log()
        service.reranker.explorer.store = JournalBanditStats(log, models.folder.name)
        kwargs = {"audience": audience}
        if at is not None:
            kwargs["at"] = at
        # 현재 세션 클릭(POST /api/events로 쌓인 view)을 추천에 반영
        if session_id is not None and load_config("recommend")["strategy"] == "multistage":
            kwargs["session_id"] = session_id
        preference_ms = 0.0
        if use_outfit_preferences:
            if load_config('recommend')['strategy'] != 'multistage':
                raise ValueError('Outfit preferences supported only for multistage')
            preference_begin = time.perf_counter()
            weights, status = fetch_outfit_preferences(user_id)
            preference_ms = (time.perf_counter() - preference_begin) * 1000
            kwargs['preference_weights'] = weights
            response.headers['X-Outfit-Preference-Source'] = status
        result = strategy.recommend(user_id, top_n, **kwargs)
        # Account for source lookup latency, including source failure.
        if preference_ms:
            result.pipeline_latency.candidate_ms += preference_ms
            result.pipeline_latency.total_ms += preference_ms
        arms = service.exploration_arms_of(result.recommendations)
        result.recommendation_id = uuid4().hex
        result.model_version = models.folder.name
        log.register_recommendation(
            result.recommendation_id, user_id, session_id,
            [item.model_dump() | {"arm": arms.get(item.product_id)} for item in result.recommendations],
            result.model_version,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail="Recommendation model missing; train candidate, build index and train CVR first",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
