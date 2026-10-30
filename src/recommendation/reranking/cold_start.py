"""신규 사용자(이력 < 5) → 인기 + 트렌딩 상품 폴백.

담당: D
"""

from __future__ import annotations

from typing import Iterable, Sequence

from src.recommendation.reranking.items import Candidate, with_reason


def is_cold_start(history_len: int, min_history: int) -> bool:
    return history_len < min_history


def interleave(
    popular: Sequence[Candidate],
    trending: Sequence[Candidate],
    ratio: tuple[int, int] = (1, 1),
) -> list[Candidate]:
    """인기:트렌딩을 ratio 비율로 교차 배치 (중복 제거). reason은 popular / trending."""
    p_n, t_n = ratio
    p_iter, t_iter = iter(popular), iter(trending)
    out: list[Candidate] = []
    seen: set[str] = set()

    def take(it: Iterable[Candidate], k: int, reason: str) -> bool:
        got = False
        for _ in range(k):
            for c in it:
                if c.product_id not in seen:
                    seen.add(c.product_id)
                    out.append(with_reason(c, reason))
                    got = True
                    break
        return got

    while True:
        a = take(p_iter, p_n, "popular")
        b = take(t_iter, t_n, "trending")
        if not a and not b:
            return out
