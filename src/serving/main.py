"""FastAPI 앱: uvicorn src.serving.main:app --port 8000

담당: D (라우터별 로직은 search=B, recommend=C/D)
"""

from __future__ import annotations

from fastapi import FastAPI

from src.serving.routers import events, recommend, search

app = FastAPI(title="Multimodal Search & Multi-Stage Recommender")

app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(recommend.router, prefix="/api", tags=["recommend"])
app.include_router(events.router, prefix="/api", tags=["events"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
