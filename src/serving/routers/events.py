"""POST /api/events — 시뮬레이터/클라이언트 클릭 이벤트 수신 → Redis 실시간 피처 갱신 + MAB 클릭 기록.  담당: D"""

from __future__ import annotations

from fastapi import APIRouter

from src.common.config import load_config
from src.common.schemas import Event, EventType
from src.serving.feature_store import get_feature_store

router = APIRouter()


@router.post("/events")
def ingest_event(event: Event) -> dict[str, str]:
    get_feature_store().push_event(event)
    if event.event_type == EventType.VIEW and event.product_id:
        record_exploration_click(event.user_id, event.product_id)
    return {"status": "accepted"}


def record_exploration_click(user_id: str, product_id: str) -> None:
    """마지막 추천의 MAB 탐색 칸 상품을 클릭했으면 arm 통계에 클릭 1번 (multistage 전략만)."""
    if load_config("recommend")["strategy"] != "multistage":
        return
    from src.recommendation.reranking.stage2 import get_rerank_service
    from src.serving.model_loader import get_recommendation_models

    try:
        models = get_recommendation_models()
    except FileNotFoundError:
        return  # 추천 모델이 없으면 진열된 탐색 칸도 없다
    get_rerank_service(models).record_click(user_id, product_id)
