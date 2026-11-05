"""GET /api/recommend. at은 재현 가능한 오프라인 데모 시점(시간대 필수), session_id는 현재 방문."""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

import src.recommendation.pipeline  # noqa: F401
import src.recommendation.strategies.baselines  # noqa: F401
from src.common.config import load_config
from src.common.schemas import RecommendResponse
from src.recommendation.strategies.base import get_strategy

router = APIRouter()


@router.get("/recommend", response_model=RecommendResponse)
def recommend(
    user_id: str,
    top_n: int = Query(default=10, ge=1, le=100),
    at: datetime | None = None,
    session_id: str | None = None,
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
        return strategy.recommend(user_id, top_n, **kwargs)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail="Recommendation model missing; train candidate, build index and train CVR first",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
