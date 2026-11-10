"""POST /api/search — text / image / hybrid 검색.
POST /api/search/style — 스타일 사진(사람이 옷을 입은 사진) + 선택 텍스트 → 사진 속 옷별로 비슷한 상품.

두 가지 요청 형식을 모두 받는다.
- JSON      : {"query_text": "black strap top", "image_base64": "<base64>", "top_k": 10}
- multipart : query_text=..., image=<파일>, top_k=10
/api/search: 텍스트만 있으면 text, 이미지만 있으면 image, 둘 다 있으면 hybrid.
             audience(선택): all(기본) / women / men / kids → 그 고객층(category_l1) 상품만 (쇼핑몰 화면)
/api/search/style: 사진 필수. 텍스트 예 "이 옷이랑 어울리는 바지", "소매 더 짧게".
                   상품 사진 + 사진만 / 설명 문장은 /api/search와 같은 검색, 색 변경·종류·디테일 요청은 조건 + 유사도.
                   audience(선택): auto(기본) / all / women / men / kids
                   box(선택): 검색할 영역 [x1, y1, x2, y2] (사진 픽셀, multipart는 "x1,y1,x2,y2")
GET /api/products/{product_id} — 상품 이름·소분류·색·가격 (추천 결과처럼 ID만 있는 상품을 보여줄 때)
GET /api/products/{product_id}/image — 상품 사진 (데모 화면 등에서 결과를 보여줄 때)
GET /api/products/{product_id}/similar — 사진이 비슷한 다른 상품 (쇼핑몰 상품 상세, 같은 디자인 제외)
                                         top_k(기본 8), audience(기본: 기준 상품과 같은 고객층 / all / women / men / kids)
"""

from __future__ import annotations

import base64
import binascii
import time
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from src.common.config import ROOT, load_config
from src.common.schemas import (
    ProductInfo,
    SearchResponse,
    SearchResultItem,
    SimilarProductsResponse,
    StyleDetection,
    StyleSearchGroup,
    StyleSearchResponse,
    StyleSearchResultItem,
)
from src.search.catalog import load_catalog
from src.search.detector import Box, FashionDetector, GroundingDetector
from src.search.searcher import Searcher
from src.search.style_search import StyleSearcher

router = APIRouter()


@lru_cache(maxsize=1)
def get_searcher() -> Searcher:
    """모델·인덱스는 무거워서 프로세스당 한 번만 로딩 (테스트에서는 dependency override)."""
    return Searcher()


@lru_cache(maxsize=1)
def get_style_searcher() -> StyleSearcher:
    """검색기(상품 임베딩·CLIP)는 /api/search와 같이 쓰고, 옷 검출 모델은 첫 요청 때 로딩 (Grounding DINO는 처음 쓸 때)."""
    cfg = load_config("style_search")
    grounding = GroundingDetector(cfg) if cfg["grounding"]["mode"] != "off" else None
    return StyleSearcher(get_searcher(), FashionDetector(cfg), cfg, grounding)


@lru_cache(maxsize=1)
def get_product_images() -> dict[str, Path]:
    """상품 ID → 사진 파일. 경로는 카탈로그에서만 가져오므로 요청으로 다른 파일을 읽을 수 없다."""
    cfg = load_config("search")
    catalog = load_catalog(cfg, columns=["product_id", "image_path"])
    root = ROOT / cfg["paths"]["image_root"]
    return {pid: root / path for pid, path in zip(catalog["product_id"], catalog["image_path"])}


@lru_cache(maxsize=1)
def get_product_catalog() -> pd.DataFrame:
    """상품 ID → 화면 표시용 정보. 처음 요청할 때 한 번만 읽는다."""
    columns = ["product_id", "name", "category_l3", "color", "price"]
    return load_catalog(load_config("search"), columns=columns).set_index("product_id")


async def _parse(request: Request, default_k: int) -> tuple[str | None, bytes | None, int, dict]:
    """(query_text, 이미지 바이트, top_k, 나머지 필드)."""
    if request.headers.get("content-type", "").startswith("application/json"):
        body = await request.json()
        image_b64 = body.get("image_base64")
        try:
            image_bytes = base64.b64decode(image_b64) if image_b64 else None
        except (binascii.Error, ValueError) as e:
            raise HTTPException(422, "image_base64 디코딩 실패") from e
        return body.get("query_text") or None, image_bytes, int(body.get("top_k") or default_k), body

    form = await request.form()
    upload = form.get("image")
    image_bytes = await upload.read() if upload is not None and hasattr(upload, "read") else None
    return form.get("query_text") or None, image_bytes or None, int(form.get("top_k") or default_k), dict(form)


def _open_image(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as e:
        raise HTTPException(422, "이미지 파일을 읽을 수 없습니다") from e


def _parse_box(value: object, image_size: tuple[int, int]) -> Box | None:
    """[x1, y1, x2, y2] 또는 "x1,y1,x2,y2" → 사진 안으로 자른 박스."""
    if value in (None, "", []):
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in (value.split(",") if isinstance(value, str) else value))
    except (TypeError, ValueError) as e:
        raise HTTPException(422, "box는 [x1, y1, x2, y2] 형식이어야 합니다") from e
    w, h = image_size
    box = (round(max(0.0, x1)), round(max(0.0, y1)), round(min(w, x2)), round(min(h, y2)))
    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
        raise HTTPException(422, "box가 너무 작거나 사진 밖에 있습니다")
    return box


def _audience_l1(audience: object) -> list[str] | None:
    """고객층 이름 → category_l1 목록 (configs/style_search.yaml audience.groups). 없거나 all이면 None (제한 없음)."""
    if audience in (None, "", "all"):
        return None
    groups = load_config("style_search")["audience"]["groups"]
    if audience not in groups:
        raise HTTPException(422, f"audience는 all / {' / '.join(groups)} 중 하나")
    return list(groups[audience]["category_l1"])


def _same_audience_l1(category_l1: str) -> list[str] | None:
    """기준 상품과 같은 고객층: 그 category_l1이 들어 있는 고객층 묶음의 category_l1 전부 (예: Ladieswear → 여성복 전체).
    어느 묶음에도 없으면 None (제한 없음)."""
    groups = load_config("style_search")["audience"]["groups"].values()
    same = sorted({c for g in groups if category_l1 in g["category_l1"] for c in g["category_l1"]})
    return same or None


@router.post("/search", response_model=SearchResponse)
async def search(request: Request, searcher: Annotated[Searcher, Depends(get_searcher)]) -> SearchResponse:
    t0 = time.perf_counter()
    query_text, image_bytes, top_k, fields = await _parse(request, load_config("search")["default_top_k"])
    if not query_text and image_bytes is None:
        raise HTTPException(422, "query_text 또는 image 중 하나는 필요합니다")
    if not 1 <= top_k <= 100:
        raise HTTPException(422, "top_k는 1~100 사이여야 합니다")
    l1 = _audience_l1(fields.get("audience"))

    image = _open_image(image_bytes) if image_bytes is not None else None
    result = await run_in_threadpool(searcher.search, query_text, image, top_k, l1)
    return SearchResponse(
        search_type=result.search_type,
        results=[SearchResultItem(**item) for item in result.items],
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        total_count=result.total_count,
    )


@router.post("/search/style", response_model=StyleSearchResponse)
async def style_search(
    request: Request, style_searcher: Annotated[StyleSearcher, Depends(get_style_searcher)]
) -> StyleSearchResponse:
    t0 = time.perf_counter()
    cfg = load_config("style_search")
    query_text, image_bytes, top_k, fields = await _parse(request, cfg["default_top_k"])
    if image_bytes is None:
        raise HTTPException(422, "스타일 사진(image)이 필요합니다")
    if not 1 <= top_k <= 100:
        raise HTTPException(422, "top_k는 1~100 사이여야 합니다")
    audience = fields.get("audience") or None
    if audience not in (None, "auto", "all", *cfg["audience"]["groups"]):
        raise HTTPException(422, f"audience는 auto / all / {' / '.join(cfg['audience']['groups'])} 중 하나")

    image = _open_image(image_bytes)
    box = _parse_box(fields.get("box"), image.size)
    result = await run_in_threadpool(style_searcher.search, image, query_text, top_k, audience, box)
    slot_name = {**cfg["slots"], "all": "전체"}
    return StyleSearchResponse(
        intent=result.intent.value,
        photo_type="style" if result.is_person_photo else "product",
        audience=result.audience,
        groups=[
            StyleSearchGroup(
                slot=g.slot,
                slot_name=slot_name[g.slot],
                kind=g.kind,
                basis=g.basis,
                box=list(g.box) if g.box else None,
                box_source=g.box_source,
                detection_score=g.detection_score,
                filters=g.filters,
                results=[StyleSearchResultItem(**item) for item in g.result.items],
                total_count=g.result.total_count,
            )
            for g in result.groups
        ],
        detections=[
            StyleDetection(kind=d.kind, slot=d.slot, slot_name=slot_name[d.slot], score=d.score, box=list(d.box),
                           source=d.source)
            for d in result.detections
        ],
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@router.get("/products/{product_id}", response_model=ProductInfo)
def product_info(product_id: str, catalog: Annotated[pd.DataFrame, Depends(get_product_catalog)]) -> ProductInfo:
    if product_id not in catalog.index:
        raise HTTPException(404, "상품이 없습니다")
    row = catalog.loc[product_id]
    return ProductInfo(product_id=product_id, name=row["name"], category=row["category_l3"], color=row["color"],
                       price=row["price"])


@router.get("/products/{product_id}/similar", response_model=SimilarProductsResponse)
async def similar_products(
    product_id: str,
    searcher: Annotated[Searcher, Depends(get_searcher)],
    top_k: Annotated[int, Query(ge=1, le=50)] = 8,
    audience: str | None = None,
) -> SimilarProductsResponse:
    t0 = time.perf_counter()
    if product_id not in searcher.row_of:
        raise HTTPException(404, "상품이 없습니다")
    l1 = _same_audience_l1(searcher.category_l1[searcher.row_of[product_id]]) if audience is None else _audience_l1(audience)
    result = await run_in_threadpool(searcher.similar_products, product_id, top_k, l1)
    return SimilarProductsResponse(
        product_id=product_id,
        results=[StyleSearchResultItem(**item, category=str(searcher.category_l3[r]), color=str(searcher.colors[r]))
                 for item, r in zip(result.items, result.rows)],
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        total_count=result.total_count,
    )


@router.get("/products/{product_id}/image")
def product_image(product_id: str, images: Annotated[dict[str, Path], Depends(get_product_images)]) -> FileResponse:
    path = images.get(product_id)
    if path is None or not path.is_file():
        raise HTTPException(404, "상품 사진이 없습니다")
    return FileResponse(path, media_type="image/jpeg")
