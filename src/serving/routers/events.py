"""POST /api/events — 시뮬레이터/클라이언트 클릭 이벤트 수신 → Redis 실시간 피처 갱신.  담당: D"""

from __future__ import annotations

from fastapi import APIRouter

from src.common.schemas import Event

router = APIRouter()


@router.post("/events")
def ingest_event(event: Event) -> dict[str, str]:
    # TODO(D): feature_store.push_event(event)
    return {"status": "accepted"}
