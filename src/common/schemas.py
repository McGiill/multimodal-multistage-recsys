"""팀 공통 데이터 계약(Contract).

상품 / 사용자 / 행동 로그 / API 응답 스키마를 이 파일 한 곳에서만 정의한다.
모든 모듈(simulator, search, recommendation, serving, evaluation)은 여기서 import 해서 쓴다.

⚠️ 필드를 추가/변경할 때는 PR에 팀 전원 리뷰를 요청할 것 (병행 개발의 기준점).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 도메인 데이터 (시뮬레이터가 생성 → data/processed/*.parquet)
# ---------------------------------------------------------------------------
class EventType(str, Enum):
    SEARCH = "search"
    VIEW = "view"
    CART = "cart"
    PURCHASE = "purchase"


class Persona(str, Enum):
    TRENDSETTER = "trendsetter"          # 트렌드세터
    PRAGMATIST = "pragmatist"            # 실용주의자
    BARGAIN_HUNTER = "bargain_hunter"    # 가성비추구
    BRAND_LOYAL = "brand_loyal"          # 브랜드충성
    IMPULSE_BUYER = "impulse_buyer"      # 충동구매
    CAREFUL_EXPLORER = "careful_explorer"  # 신중탐색


class Product(BaseModel):
    product_id: str                      # "P00001"
    name: str
    description: str = ""
    category_l1: str                     # 대분류 (예: 상의)
    category_l2: str                     # 중분류 (예: 니트)
    category_l3: str                     # 소분류 (예: 라운드넥 니트)
    brand: str
    color: str
    price: float                         # 달러 (H&M 최빈 거래가 × 590)
    image_path: str | None = None        # data/raw/images/... 기준 상대 경로
    created_at: datetime                 # 신규 상품(7일 이내) 판단용
    popularity: float = 0.0


class User(BaseModel):
    """공개 고객 정보 (users.parquet) — 추천 모델 입력으로 써도 된다."""

    user_id: str                         # "U00001"
    age_group: str                       # "10s", "20s", ... "60s+"
    gender: str                          # "F" | "M"
    signup_at: datetime


class UserHidden(BaseModel):
    """숨김 성향 (users_hidden.parquet) — 행동 생성과 페르소나별 분석에만. 모델 입력 금지 (정답 유출)."""

    user_id: str
    persona: Persona
    main_segment: str                    # 주 쇼핑 고객층 (category_l1)
    preferred_l2: list[str]              # 선호 카테고리 (category_l2)
    preferred_brands: list[str]
    price_sensitivity: float = Field(ge=0.0, le=1.0)
    new_item_affinity: float = Field(ge=0.0, le=1.0)
    brand_loyalty: float = Field(ge=0.0, le=1.0)
    activity: float                      # 방문 수 배율 (평균 1)
    expected_sessions: float
    seed: int                            # 취향 위치를 빌려온 실제 고객(취향 시드) 번호


class Event(BaseModel):
    event_id: str
    user_id: str
    session_id: str
    event_type: EventType
    product_id: str | None = None        # search 이벤트는 None 가능
    query: str | None = None             # search 이벤트일 때만
    timestamp: datetime
    source: str | None = None            # 이벤트가 나온 화면: "search" | "browse" | "home" | "cart"(지난 장바구니 구매)
    impression_id: str | None = None     # view·cart·purchase가 나온 목록 노출 (cart 구매는 None)


class Impression(BaseModel):
    """화면에 보여준 상품 목록 (impressions.parquet) — 클릭 안 한 상품 = Ranking 음성 데이터, 클릭률 계산."""

    impression_id: str
    user_id: str
    session_id: str
    timestamp: datetime
    source: str                          # "search" | "browse" | "home"
    context: str                         # 검색어 / "고객층/소분류 (정렬)" / 홈의 고객층
    product_ids: list[str]               # 보여준 순서 그대로 (앞이 위쪽)
    clicked_ids: list[str]               # 클릭한 순서


# ---------------------------------------------------------------------------
# 검색 API  (POST /api/search) — 과제 필수 필드: search_type, results[], latency_ms, total_count
# ---------------------------------------------------------------------------
class SearchType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    HYBRID = "hybrid"


class SearchResultItem(BaseModel):
    product_id: str
    name: str
    score: float
    price: float                         # 달러


class SearchResponse(BaseModel):
    search_type: SearchType
    results: list[SearchResultItem]
    latency_ms: float
    total_count: int                     # FAISS 후보 풀에서 찾은 전체 결과 수 (≥ len(results))


# ---------------------------------------------------------------------------
# 스타일 사진 검색 API  (POST /api/search/style) — 사람이 옷을 입은 사진 (+ 선택 텍스트)
# ---------------------------------------------------------------------------
class StyleSearchResultItem(SearchResultItem):
    category: str                        # 소분류 (category_l3)
    color: str


class StyleSearchGroup(BaseModel):
    slot: str                            # 부위: outer / top / dress / bottom / shoes / ... (옷을 못 찾았으면 "all")
    slot_name: str                       # 화면 표시용 (아우터, 상의, ...)
    kind: str | None                     # 사진에서 찾은 옷 종류 (configs/style_search.yaml의 kinds)
    basis: str                           # photo_item: 사진 속 그 옷 기준 | photo_style: 사진 전체 분위기 기준 | whole_photo
    box: list[int] | None                # 사진 속 위치 [x1, y1, x2, y2]
    box_source: str | None               # detector: 검출 | low_score: 콕 집은 부위라 낮은 점수 검출도 사용
                                         # | grounding: Grounding DINO로 다시 찾음 | user: 사용자가 지정한 영역
    detection_score: float | None
    filters: list[str]                   # 적용한 조건 (예: "color:red", "sleeve:down")
    results: list[StyleSearchResultItem]
    total_count: int                     # 조건을 만족한 후보 수


class StyleDetection(BaseModel):
    kind: str | None
    slot: str
    slot_name: str
    score: float | None
    box: list[int]
    source: str                          # StyleSearchGroup.box_source와 같음


class StyleSearchResponse(BaseModel):
    intent: str                          # style: 사진 속 옷 전부 | find: 지정한 옷 | match: 어울리는 옷
    photo_type: str                      # style: 사람이 입은 사진 (옷 검출) | product: 옷만 찍은 사진 (사진 전체가 기준)
    audience: str | None                 # 제한한 고객층 (women / men / kids), 제한 안 했으면 None
    groups: list[StyleSearchGroup]
    detections: list[StyleDetection]     # 사진에서 찾은 옷 전부 (화면에서 영역을 골라 다시 검색할 때)
    latency_ms: float


# ---------------------------------------------------------------------------
# 추천 API  (GET /api/recommend) — 과제 필수 필드:
#   user_id, recommendations[], pipeline_latency, session_context
# ---------------------------------------------------------------------------
class RecommendationItem(BaseModel):
    product_id: str
    score: float
    reason: str                          # "recent_view" | "similar_users" | "mab_exploration" | "popular" | "new_item" ...
    is_exploration: bool = False


class PipelineLatency(BaseModel):
    candidate_ms: float
    ranking_ms: float
    reranking_ms: float
    total_ms: float


class SessionContext(BaseModel):
    recent_clicks: list[str] = Field(default_factory=list)
    session_interest: str | None = None


class RecommendResponse(BaseModel):
    user_id: str
    recommendations: list[RecommendationItem]
    pipeline_latency: PipelineLatency
    session_context: SessionContext
