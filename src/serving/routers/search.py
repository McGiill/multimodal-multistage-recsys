"""POST /api/search — text / image / hybrid 검색.

두 가지 요청 형식을 모두 받는다.
- JSON      : {"query_text": "black strap top", "image_base64": "<base64>", "top_k": 10}
- multipart : query_text=..., image=<파일>, top_k=10
텍스트만 있으면 text, 이미지만 있으면 image, 둘 다 있으면 hybrid.
"""

from __future__ import annotations

import base64
import binascii
import time
from functools import lru_cache
from io import BytesIO
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from src.common.config import load_config
from src.common.schemas import SearchResponse, SearchResultItem
from src.search.searcher import Searcher

router = APIRouter()


@lru_cache(maxsize=1)
def get_searcher() -> Searcher:
    """모델·인덱스는 무거워서 프로세스당 한 번만 로딩 (테스트에서는 dependency override)."""
    return Searcher()


async def _parse(request: Request) -> tuple[str | None, bytes | None, int]:
    default_k = load_config("search")["default_top_k"]
    if request.headers.get("content-type", "").startswith("application/json"):
        body = await request.json()
        image_b64 = body.get("image_base64")
        try:
            image_bytes = base64.b64decode(image_b64) if image_b64 else None
        except (binascii.Error, ValueError) as e:
            raise HTTPException(422, "image_base64 디코딩 실패") from e
        return body.get("query_text") or None, image_bytes, int(body.get("top_k") or default_k)

    form = await request.form()
    upload = form.get("image")
    image_bytes = await upload.read() if upload is not None and hasattr(upload, "read") else None
    return form.get("query_text") or None, image_bytes or None, int(form.get("top_k") or default_k)


@router.post("/search", response_model=SearchResponse)
async def search(request: Request, searcher: Annotated[Searcher, Depends(get_searcher)]) -> SearchResponse:
    t0 = time.perf_counter()
    query_text, image_bytes, top_k = await _parse(request)
    if not query_text and image_bytes is None:
        raise HTTPException(422, "query_text 또는 image 중 하나는 필요합니다")
    if not 1 <= top_k <= 100:
        raise HTTPException(422, "top_k는 1~100 사이여야 합니다")

    image = None
    if image_bytes is not None:
        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError as e:
            raise HTTPException(422, "이미지 파일을 읽을 수 없습니다") from e

    result = await run_in_threadpool(searcher.search, query_text, image, top_k)
    return SearchResponse(
        search_type=result.search_type,
        results=[SearchResultItem(**item) for item in result.items],
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        total_count=result.total_count,
    )
