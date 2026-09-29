"""GET /api/recommend?user_id=U1234&top_n=10   담당: D"""

from __future__ import annotations

from fastapi import APIRouter, Query

import src.recommendation.pipeline  # noqa: F401  (전략 등록)
from src.common.config import load_config
from src.common.schemas import RecommendResponse
from src.recommendation.strategies.base import get_strategy

router = APIRouter()


@router.get("/recommend", response_model=RecommendResponse)
def recommend(user_id: str, top_n: int = Query(default=10, ge=1, le=100)) -> RecommendResponse:
    strategy = get_strategy(load_config("recommend")["strategy"])
    return strategy.recommend(user_id=user_id, top_n=top_n)
