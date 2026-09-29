"""평가 지표 (과제 지정 경로: src/evaluation/metrics.py).  담당: A

Offline top-k 기준 — 검색 k=10, 추천 k=50
"""

from __future__ import annotations

from collections.abc import Sequence


def mrr(ranked: Sequence[Sequence[str]], relevant: Sequence[set[str]]) -> float:
    """Mean Reciprocal Rank. 목표(검색) ≥ 0.55"""
    raise NotImplementedError


def ndcg_at_k(ranked: Sequence[Sequence[str]], relevant: Sequence[set[str]], k: int) -> float:
    """NDCG@k. 목표: 검색 @10 ≥ 0.50 / 추천 @50 ≥ 0.08"""
    raise NotImplementedError


def hit_rate_at_k(ranked: Sequence[Sequence[str]], relevant: Sequence[set[str]], k: int) -> float:
    """HitRate@k. 목표(추천) @50 ≥ 0.20"""
    raise NotImplementedError


def recall_at_k(ranked: Sequence[Sequence[str]], relevant: Sequence[set[str]], k: int) -> float:
    """Recall@k. 목표(Candidate) @300 ≥ 0.30"""
    raise NotImplementedError


def coverage(ranked: Sequence[Sequence[str]], n_total_items: int) -> float:
    """추천된 고유 상품 수 / 전체 상품 수. 목표 ≥ 0.20"""
    raise NotImplementedError


def auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Ranking 모델 AUC. 목표 ≥ 0.70"""
    raise NotImplementedError
