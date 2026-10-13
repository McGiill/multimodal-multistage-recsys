"""평가 지표 (과제 지정 경로: src/evaluation/metrics.py).

Offline top-k 기준 — 검색 k=10, 추천 k=50

relevant 인자는 쿼리(또는 사용자)마다 하나씩이며 두 형태를 모두 받는다.
- set[str]            : 이진 관련도 (들어 있으면 관련도 1)
- dict[str, float]    : 등급 관련도 (예: 정확히 일치 2, 부분 일치 1)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet

import numpy as np

Relevance = Mapping[str, float] | AbstractSet[str]


def _grades(relevant: Relevance) -> Mapping[str, float]:
    return relevant if isinstance(relevant, Mapping) else {item: 1.0 for item in relevant}


def mrr(ranked: Sequence[Sequence[str]], relevant: Sequence[Relevance], min_grade: float = 1.0) -> float:
    """Mean Reciprocal Rank: 관련도 min_grade 이상인 첫 결과 순위의 역수 평균. 목표(검색) ≥ 0.55"""
    total = 0.0
    for items, rel in zip(ranked, relevant, strict=True):
        grades = _grades(rel)
        for rank, item in enumerate(items, start=1):
            if grades.get(item, 0.0) >= min_grade:
                total += 1.0 / rank
                break
    return total / len(ranked) if ranked else 0.0


def ndcg_at_k(ranked: Sequence[Sequence[str]], relevant: Sequence[Relevance], k: int) -> float:
    """NDCG@k (gain = 2^grade - 1). 목표: 검색 @10 ≥ 0.50 / 추천 @50 ≥ 0.08"""
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    scores = []
    for items, rel in zip(ranked, relevant, strict=True):
        grades = _grades(rel)
        gains = np.array([2.0 ** grades.get(item, 0.0) - 1.0 for item in items[:k]])
        dcg = float((gains * discounts[: len(gains)]).sum())
        ideal = np.sort(2.0 ** np.array(list(grades.values()), dtype=float) - 1.0)[::-1][:k]
        idcg = float((ideal * discounts[: len(ideal)]).sum())
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def hit_rate_at_k(ranked: Sequence[Sequence[str]], relevant: Sequence[Relevance], k: int) -> float:
    """HitRate@k: 상위 k개에 관련 항목이 하나라도 있는 비율. 목표(추천) @50 ≥ 0.20"""
    hits = [any(_grades(rel).get(item, 0.0) > 0 for item in items[:k]) for items, rel in zip(ranked, relevant, strict=True)]
    return float(np.mean(hits)) if hits else 0.0


def recall_at_k(ranked: Sequence[Sequence[str]], relevant: Sequence[Relevance], k: int) -> float:
    """Recall@k: 관련 항목 중 상위 k개에 들어온 비율의 평균. 목표(Candidate) @300 ≥ 0.30"""
    recalls = []
    for items, rel in zip(ranked, relevant, strict=True):
        positives = {item for item, g in _grades(rel).items() if g > 0}
        if positives:
            recalls.append(len(positives.intersection(items[:k])) / len(positives))
    return float(np.mean(recalls)) if recalls else 0.0


def coverage(ranked: Sequence[Sequence[str]], n_total_items: int) -> float:
    """추천된 고유 상품 수 / 전체 상품 수. 목표 ≥ 0.20"""
    unique = {item for items in ranked for item in items}
    return len(unique) / n_total_items if n_total_items else 0.0


def auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """ROC-AUC (순위 기반, 동점은 평균 순위). Ranking 모델 목표 ≥ 0.70"""
    y = np.asarray(y_true)
    s = np.asarray(y_score, dtype=float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):  # 동점 구간에 평균 순위 부여
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
