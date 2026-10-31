"""GET /api/recommend. at은 재현 가능한 오프라인 데모 시점(시간대 필수)."""

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query

import src.recommendation.pipeline  # noqa: F401
from src.common.config import load_config
from src.common.schemas import RecommendResponse
from src.recommendation.strategies.base import get_strategy

router = APIRouter()


@router.get("/recommend", response_model=RecommendResponse)
def recommend(
    user_id: str,
    top_n: int = Query(default=10, ge=1, le=100),
    at: datetime | None = None,
) -> RecommendResponse:
    try:
        strategy = get_strategy(load_config("recommend")["strategy"])
        if at is not None:
            if load_config("recommend")["strategy"] != "multistage":
                raise ValueError("at supported only for multistage")
            return strategy.recommend(user_id, top_n, at=at)
        return strategy.recommend(user_id, top_n)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail="Recommendation model missing; train candidate, build index and train CVR first",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
