"""Stage 3 공통 데이터 구조: 입력 후보(Candidate)와 출력(RerankedItem, RerankResult).

담당: D
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Iterable


@dataclass(frozen=True)
class Candidate:
    """Stage 2 출력 한 건 (또는 cold-start/신규 상품 목록의 한 건)."""

    product_id: str
    score: float
    category_l1: str
    category_l2: str
    category_l3: str
    created_at: datetime
    reason: str = "personalized"  # Stage 2가 채워 주면 그대로 유지
    product_code: str | None = None  # 같은 디자인의 색상 변형 묶음 (없으면 product_id로 대체)


@dataclass(frozen=True)
class RerankedItem:
    """추천 API recommendations[] 한 건 (명세 필수 필드)."""

    product_id: str
    score: float
    reason: str
    is_exploration: bool

    def to_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "score": round(self.score, 4),
            "reason": self.reason,
            "is_exploration": self.is_exploration,
        }


@dataclass
class RerankResult:
    items: list[RerankedItem]
    is_cold_start: bool
    reranking_ms: float
    # 후보가 부족해 다양성 규칙을 지키지 못한 위치 (로그·디버깅용)
    diversity_relaxed_positions: list[int] = field(default_factory=list)
    # MAB가 고른 칸의 product_id → arm. 클릭 피드백은 이 칸만 기록 (신규 상품 칸 제외)
    exploration_arms: dict[str, str] = field(default_factory=dict)


def with_reason(c: Candidate, reason: str) -> Candidate:
    return replace(c, reason=reason)


def dedupe_key(c: Candidate, dedupe_field: str | None) -> str:
    """중복 판단 기준. dedupe_field가 없거나 값이 비어 있으면 product_id."""
    if dedupe_field is None:
        return c.product_id
    return getattr(c, dedupe_field) or c.product_id


def dedupe(items: Iterable[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    out: list[Candidate] = []
    for c in items:
        if c.product_id not in seen:
            seen.add(c.product_id)
            out.append(c)
    return out
