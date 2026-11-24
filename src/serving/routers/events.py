"""POST /api/events — 시뮬레이터/클라이언트 클릭 이벤트 수신 → Redis 실시간 피처 갱신 + MAB 클릭 기록.  담당: D"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.common.schemas import Event, EventType, ImpressionRequest
from src.serving.event_log import get_event_log
from src.serving.feature_store import get_feature_store

router = APIRouter()


@router.post("/events")
def ingest_event(event: Event) -> dict[str, str]:
    try:
        log = get_event_log()
        inserted = log.append_event(event)
        if log.feature_pending(event.event_id):
            get_feature_store().push_event(event)
            log.mark_feature_delivered(event.event_id)
        if event.event_type == EventType.VIEW and event.product_id:
            record_exploration_click(event)
        return {"status": "accepted" if inserted else "duplicate"}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/impressions")
def ingest_impression(impression: ImpressionRequest) -> dict[str, str]:
    """필터·중복 제거·화면 표시 이후의 목록만 노출로 확정한다."""
    log = get_event_log()
    try:
        inserted = log.append_impression(impression, request_id=impression.recommendation_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"status": "accepted" if inserted else "duplicate"}


def record_exploration_click(event: Event) -> None:
    """실제 노출 ID에 귀속된 클릭만 1회 보상한다. 다른 화면/중복 클릭은 제외."""
    get_event_log().claim_click(event)
