"""A/B 통계 검정 (src/abtest/stats.py). 기준값은 scipy.stats.chi2_contingency / statsmodels로 미리 계산한 값."""

from __future__ import annotations

import pytest

from src.abtest.stats import GroupStats, compare, format_result

CONTROL, TREATMENT = GroupStats(160, 5000), GroupStats(205, 5000)  # 3.2% vs 4.1%


def test_z_test_matches_reference() -> None:
    r = compare(CONTROL, TREATMENT, "z_test", 0.05)
    assert r.statistic == pytest.approx(2.3996075629, rel=1e-9)   # statsmodels proportions_ztest
    assert r.p_value == pytest.approx(0.0164126570, rel=1e-8)
    assert r.significant and r.lift == pytest.approx(0.28125)
    assert r.diff == pytest.approx(0.009)


def test_chi_square_equals_z_squared() -> None:
    z = compare(CONTROL, TREATMENT, "z_test", 0.05)
    c = compare(CONTROL, TREATMENT, "chi_square", 0.05)
    assert c.statistic == pytest.approx(5.7581164561, rel=1e-9)   # chi2_contingency(correction=False)
    assert c.statistic == pytest.approx(z.statistic ** 2)
    assert c.p_value == pytest.approx(z.p_value)


def test_confidence_interval_matches_wald() -> None:
    r = compare(CONTROL, TREATMENT, "z_test", 0.05)
    assert (r.ci_low, r.ci_high) == pytest.approx((0.0016510498, 0.0163489502), rel=1e-7)  # statsmodels wald
    assert r.ci_low < r.diff < r.ci_high
    wider = compare(CONTROL, TREATMENT, "z_test", 0.01)
    assert wider.ci_low < r.ci_low and wider.ci_high > r.ci_high


def test_no_difference_not_significant() -> None:
    r = compare(GroupStats(100, 5000), GroupStats(100, 5000), "z_test", 0.05)
    assert r.statistic == 0 and r.p_value == pytest.approx(1.0) and not r.significant
    assert r.lift == 0


def test_defaults_from_config_and_validation() -> None:
    r = compare(CONTROL, TREATMENT)  # configs/abtest.yaml
    assert r.test in {"z_test", "chi_square"} and 0 < r.alpha < 1
    with pytest.raises(ValueError):
        compare(CONTROL, TREATMENT, "t_test", 0.05)
    with pytest.raises(ValueError):
        GroupStats(10, 5)
    assert compare(GroupStats(0, 100), GroupStats(5, 100), "z_test", 0.05).lift is None


def test_format_result_has_required_fields() -> None:
    r = compare(CONTROL, TREATMENT, "z_test", 0.05)
    text = format_result(r, CONTROL, TREATMENT)
    for s in ("3.20%", "4.10%", "+28%", "p-value: 0.0164", "유의함", "95% CI: [0.17%, 1.63%]"):
        assert s in text
