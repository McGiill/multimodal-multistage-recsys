"""POST /api/search — text / image / hybrid (multipart: query_text, image, top_k).  담당: B"""

from __future__ import annotations

import time

from fastapi import APIRouter, File, Form, UploadFile

from src.common.schemas import SearchResponse, SearchType

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(
    query_text: str | None = Form(default=None),
    image: UploadFile | None = File(default=None),
    top_k: int = Form(default=10),
) -> SearchResponse:
    t0 = time.perf_counter()
    if query_text and image:
        search_type = SearchType.HYBRID
    elif image:
        search_type = SearchType.IMAGE
    else:
        search_type = SearchType.TEXT

    # TODO(B): searcher.search(query_text, image, top_k)
    results: list = []

    return SearchResponse(
        search_type=search_type,
        results=results,
        latency_ms=(time.perf_counter() - t0) * 1000,
        total_count=len(results),
    )
