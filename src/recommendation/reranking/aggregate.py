"""전체 고객 단위 다양성 (aggregate diversity → Coverage).

목록의 끝 tail칸만 그 고객의 상위 window개 후보 중에서 '지금까지 다른 고객 목록에 덜 나온 상품'을 우선한다:
    정렬 점수 = 순위 점수(1 → 0) - penalty × log(1 + 노출 횟수)
앞쪽 head칸은 최종 점수 순서 그대로다. 노출 횟수는 이 모델 묶음이 돌려준 추천 목록에서 센다 (RerankService.exposure).
설정: configs/recommend.yaml final_rank.diversity. valid 기간에서 Coverage ≥ 0.21을 만족하는 설정 중 NDCG가 가장 높은
설정으로 골랐다 (50칸 중 끝 3칸, window 1200, penalty 1.0).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DiversityConfig:
    tail_share: float = 0.0   # 목록에서 다양성 칸 비율 (50칸 × 0.06 = 3칸)
    window: int = 1200        # 다양성 칸 후보 = 순위 상위 몇 개 안에서
    penalty: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.tail_share < 1 or self.window < 1 or self.penalty < 0:
            raise ValueError("diversity: 0 <= tail_share < 1, window >= 1, penalty >= 0")

    @classmethod
    def from_config(cls, raw: Mapping[str, Any] | None) -> DiversityConfig:
        raw = raw or {}
        return cls(**{k: type(getattr(cls, k))(raw[k]) for k in ("tail_share", "window", "penalty") if k in raw})

    def tail(self, top_n: int) -> int:
        return round(top_n * self.tail_share)


def diversify(ids: np.ndarray, scores: np.ndarray, codes: np.ndarray, counts: np.ndarray,
              head: int, tail: int, window: int, penalty: float) -> np.ndarray:
    """정렬용 점수 (이 점수 순서 = 최종 순서). ids: 상품 번호, codes/counts: 상품 번호별 디자인 코드·노출 횟수.

    같은 디자인(색상 변형)은 점수가 가장 높은 하나만 head·tail 자리를 차지한다 (Stage 3 중복 제거와 같은 기준).
    나머지는 원래 점수 순서로 뒤에 둔다 (Stage 3의 탐색 칸·빈칸 채우기용).
    """
    ids, scores = np.asarray(ids), np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    out = -1e6 - np.arange(len(ids), dtype=np.float64)[np.argsort(order)]  # 기본: 원래 순서
    if tail <= 0 or not len(ids):
        return out
    _, first = np.unique(codes[ids[order]], return_index=True)
    ranked = order[np.sort(first)]
    head_idx, cand = ranked[:head], ranked[head:max(window, head)]
    out[head_idx] = 1e6 - np.arange(len(head_idx))
    if len(cand):
        out[cand] = np.linspace(1, 0, len(cand)) - penalty * np.log1p(counts[ids[cand]])
    return out
