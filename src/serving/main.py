"""FastAPI 앱: uvicorn src.serving.main:app --port 8000

담당: D (라우터별 로직은 search=B, recommend=C/D)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.common.config import env
from src.serving.routers import events, recommend, search


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # 첫 요청이 느려지지 않도록 서버 시작 시 모델·인덱스를 미리 로딩 (PRELOAD_MODELS=0 이면 생략)
    if env("PRELOAD_MODELS", "1") != "0":
        try:
            s = search.get_searcher()
            print(f"[api-server] 검색 인덱스 로딩 완료 ({len(s.product_ids):,} 상품, {s.encoder.model_name})", flush=True)
        except FileNotFoundError as e:
            print(f"[api-server] 검색 인덱스 없음 → make catalog / make index 먼저 실행 ({e})", flush=True)
    if env("PRELOAD_MODELS", "1") != "0":
        from src.serving.model_loader import get_recommendation_models
        try:
            get_recommendation_models()
        except FileNotFoundError:
            print("[api-server] 추천 모델 없음: Stage 1·2 학습 필요", flush=True)
    yield


app = FastAPI(title="Multimodal Search & Multi-Stage Recommender", lifespan=lifespan)

app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(recommend.router, prefix="/api", tags=["recommend"])
app.include_router(events.router, prefix="/api", tags=["events"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
