import math

import pytest

from src.evaluation.metrics import (
    auc,
    coverage,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
)


def test_mrr_binary_and_graded() -> None:
    ranked = [["a", "b", "c"], ["x", "y", "z"]]
    assert mrr(ranked, [{"b"}, {"z"}]) == pytest.approx((1 / 2 + 1 / 3) / 2)
    # min_grade=2 이면 관련도 1인 'a'는 무시하고 관련도 2인 'c'(3위)를 찾는다
    assert mrr([["a", "b", "c"]], [{"a": 1, "c": 2}], min_grade=2) == pytest.approx(1 / 3)
    assert mrr([["a"]], [{"q"}]) == 0.0


def test_ndcg_matches_hand_calculation() -> None:
    # 순위: [rel=1, rel=0, rel=2] / 이상적 순서: [2, 1]
    dcg = (2**1 - 1) / math.log2(2) + (2**2 - 1) / math.log2(4)
    idcg = (2**2 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3)
    assert ndcg_at_k([["a", "b", "c"]], [{"a": 1, "c": 2}], k=3) == pytest.approx(dcg / idcg)
    assert ndcg_at_k([["c", "a"]], [{"a": 1, "c": 2}], k=2) == pytest.approx(1.0)


def test_hit_recall_coverage() -> None:
    ranked = [["a", "b"], ["c", "d"]]
    relevant = [{"b", "e"}, {"x"}]
    assert hit_rate_at_k(ranked, relevant, k=2) == 0.5
    assert recall_at_k(ranked, relevant, k=2) == pytest.approx((1 / 2 + 0) / 2)
    assert coverage(ranked, n_total_items=8) == 0.5


def test_auc() -> None:
    assert auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
