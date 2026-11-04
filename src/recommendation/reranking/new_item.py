"""신규 상품(등록 7일 이내) 노출 보장 슬롯.

담당: D
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Sequence

from src.recommendation.reranking.items import Candidate, dedupe


def build_new_item_pool(
    candidates: Sequence[Candidate],
    fallback_pool: Sequence[Candidate],
    now: datetime,
    max_age_days: int,
) -> list[Candidate]:
    """후보 + 보조 목록에서 등록 max_age_days일 이내 상품을 점수 내림차순으로.

    now는 실제 오늘이 아니라 요청 시각(또는 카탈로그 기준일)을 넘길 수 있게 인자로 받는다.
    경계는 팀 카탈로그 기준(build_catalog.summarize)과 같게 now - max_age_days 초과.
    """
    cutoff = now - timedelta(days=max_age_days)
    pool = dedupe(c for c in [*candidates, *fallback_pool] if cutoff < c.created_at <= now)
    pool.sort(key=lambda c: c.score, reverse=True)
    return pool
