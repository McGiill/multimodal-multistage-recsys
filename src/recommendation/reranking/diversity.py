"""동일 카테고리 연속 3개 이상 금지 (max_same_category_run). (보너스) MMR.

왼쪽부터 한 칸씩 채우면서 직전 max_consecutive개만 검사하면
슬롯 삽입 후에도 전체 리스트에서 규칙이 보장된다.
담당: D
"""

from __future__ import annotations

from typing import Sequence


def breaks_diversity(prev_categories: Sequence[str], category: str, max_consecutive: int) -> bool:
    """category를 다음 칸에 넣으면 같은 카테고리가 max_consecutive개를 넘게 이어지는지."""
    k = max_consecutive
    return len(prev_categories) >= k and all(c == category for c in prev_categories[-k:])


def max_run_length(categories: Sequence[str]) -> int:
    """연속 동일 카테고리 최대 길이 (테스트·평가용)."""
    best = run = 0
    prev = None
    for c in categories:
        run = run + 1 if c == prev else 1
        best = max(best, run)
        prev = c
    return best
