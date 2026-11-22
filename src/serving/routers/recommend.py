"""GET /api/recommend. at은 재현 가능한 오프라인 데모 시점(시간대 필수), session_id는 현재 방문."""

from datetime import datetime
import time

from fastapi import APIRouter, HTTPException, Query, Response

import src.recommendation.pipeline  # noqa: F401
import src.recommendation.strategies.baselines  # noqa: F401
from src.common.config import load_config
from src.common.schemas import RecommendResponse
from src.recommendation.strategies.base import get_strategy
from src.serving.outfit_preferences import fetch_outfit_preferences

router = APIRouter()


@router.get("/recommend", response_model=RecommendResponse)
def recommend(
    response: Response,
    user_id: str,
    top_n: int = Query(default=10, ge=1, le=100),
    at: datetime | None = None,
    session_id: str | None = None,
    use_outfit_preferences: bool = False,
) -> RecommendResponse:
    try:
        strategy = get_strategy(load_config("recommend")["strategy"])
        kwargs = {}
        if at is not None:
            kwargs["at"] = at
        if session_id is not None:
            # 현재 세션 클릭(POST /api/events로 쌓인 view)을 추천에 반영
            if load_config("recommend")["strategy"] != "multistage":
                raise ValueError("session_id supported only for multistage")
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
        return result
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail="Recommendation model missing; train candidate, build index and train CVR first",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
