"""Z-test / Chi-square, p-value, 95% 신뢰구간, Lift.

두 그룹(Control / Treatment)의 전환율(전환 수 / 표본 수) 차이를 검정한다. 설정: configs/abtest.yaml (test, alpha)
- z_test     : 두 비율 Z 검정 (양측). 검정통계량은 합친 비율(pooled)의 표준오차로 계산
- chi_square : 2x2 분할표 카이제곱 검정 (자유도 1, 연속성 보정 없음). z_test와 p-value가 같다 (χ² = z²)
- 신뢰구간  : 전환율 차이(Treatment - Control)의 Wald 구간, 그룹별 표준오차(unpooled)
- Lift      : Treatment / Control - 1
scipy 없이 math.erfc로 계산한다 (정규분포·자유도 1 카이제곱은 erfc로 정확히 표현됨).

담당: D
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

from src.common.config import load_config

TESTS = {"z_test", "chi_square"}


@dataclass(frozen=True)
class GroupStats:
    conversions: int
    n: int

    def __post_init__(self) -> None:
        if self.n <= 0 or not 0 <= self.conversions <= self.n:
            raise ValueError("0 <= conversions <= n, n > 0 이어야 함")

    @property
    def rate(self) -> float:
        return self.conversions / self.n


@dataclass(frozen=True)
class TestResult:
    test: str
    alpha: float
    control_rate: float
    treatment_rate: float
    diff: float              # treatment - control (비율 차이)
    lift: float | None       # treatment / control - 1 (control 0이면 None)
    statistic: float         # z 또는 χ²
    p_value: float
    ci_low: float            # 차이의 (1 - alpha) 신뢰구간
    ci_high: float
    significant: bool        # p_value < alpha

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _two_sided_p_from_z(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2))


def z_statistic(control: GroupStats, treatment: GroupStats) -> float:
    pooled = (control.conversions + treatment.conversions) / (control.n + treatment.n)
    se = math.sqrt(pooled * (1 - pooled) * (1 / control.n + 1 / treatment.n))
    return (treatment.rate - control.rate) / se if se > 0 else 0.0


def chi_square_statistic(control: GroupStats, treatment: GroupStats) -> float:
    """2x2 분할표 (전환 / 미전환 × 그룹) 카이제곱 통계량, 연속성 보정 없음."""
    table = [[control.conversions, control.n - control.conversions],
             [treatment.conversions, treatment.n - treatment.conversions]]
    total = control.n + treatment.n
    cols = [table[0][j] + table[1][j] for j in range(2)]
    stat = 0.0
    for row, n in zip(table, (control.n, treatment.n)):
        for j in range(2):
            expected = n * cols[j] / total
            if expected > 0:
                stat += (row[j] - expected) ** 2 / expected
    return stat


def diff_confidence_interval(control: GroupStats, treatment: GroupStats,
                             alpha: float = 0.05) -> tuple[float, float]:
    """전환율 차이의 Wald 신뢰구간 (그룹별 표준오차)."""
    diff = treatment.rate - control.rate
    se = math.sqrt(control.rate * (1 - control.rate) / control.n
                   + treatment.rate * (1 - treatment.rate) / treatment.n)
    z = NormalDist().inv_cdf(1 - alpha / 2)
    return diff - z * se, diff + z * se


def compare(control: GroupStats, treatment: GroupStats, test: str | None = None,
            alpha: float | None = None) -> TestResult:
    """두 그룹 전환율 비교. test / alpha가 없으면 configs/abtest.yaml 값."""
    if test is None or alpha is None:
        cfg = load_config("abtest")
        test = cfg["test"] if test is None else test
        alpha = cfg["alpha"] if alpha is None else alpha
    if test not in TESTS:
        raise ValueError(f"지원하지 않는 검정: {test} (가능: {sorted(TESTS)})")
    if not 0 < alpha < 1:
        raise ValueError("alpha는 0~1 사이여야 함")

    if test == "z_test":
        stat = z_statistic(control, treatment)
        p = _two_sided_p_from_z(stat)
    else:
        stat = chi_square_statistic(control, treatment)
        p = math.erfc(math.sqrt(stat / 2))  # 자유도 1 카이제곱의 상위 꼬리 확률
    low, high = diff_confidence_interval(control, treatment, alpha)
    return TestResult(
        test=test, alpha=alpha,
        control_rate=control.rate, treatment_rate=treatment.rate,
        diff=treatment.rate - control.rate,
        lift=treatment.rate / control.rate - 1 if control.conversions else None,
        statistic=stat, p_value=p, ci_low=low, ci_high=high, significant=p < alpha,
    )


def format_result(result: TestResult, control: GroupStats, treatment: GroupStats,
                  metric: str = "CVR") -> str:
    """명세 예시 형식의 텍스트 (터미널 출력·리포트용)."""
    lift = f"{result.lift:+.0%}" if result.lift is not None else "-"
    verdict = "유의함" if result.significant else "유의하지 않음"
    level = f"{1 - result.alpha:.0%}"
    return "\n".join([
        "========== A/B 테스트 결과 ==========",
        f"샘플 수: Control {control.n:,}, Treatment {treatment.n:,}",
        f"{'Metric':<10} {'Control':>9} {'Treatment':>10} {'Lift':>7}",
        f"{metric:<10} {result.control_rate:>9.2%} {result.treatment_rate:>10.2%} {lift:>7}",
        f"통계적 유의성 ({result.test}):",
        f"  - p-value: {result.p_value:.4f} ({'<' if result.significant else '>='} {result.alpha}, {verdict})",
        f"  - {level} CI: [{result.ci_low:.2%}, {result.ci_high:.2%}]",
    ])
