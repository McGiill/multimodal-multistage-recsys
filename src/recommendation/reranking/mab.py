"""MAB 탐색 슬롯 1~2개: Epsilon-Greedy / UCB. is_exploration=True

탐색 선택기는 pick(candidates, n) 인터페이스만 맞추면 Reranker 수정 없이 교체된다.

MAB 구조
  arm    = category_l2 (상품 단위는 클릭이 너무 드물어 학습이 안 됨)
  보상   = 탐색 슬롯에 노출된 뒤 클릭 여부
  통계   = arm별 (노출 수, 클릭 수), 전체 사용자 공용. StatsStore로 분리 → 지금은 메모리, 나중에 Redis
  노출 수는 pick 시점이 아니라 실제 진열 후 record_feedback()으로 센다
  (pick은 다양성 규칙 대비 여유분까지 돌려주므로 pick 시점에 세면 과대 집계됨)
담당: D
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Any, Mapping, Protocol, Sequence

from src.common.config import load_config
from src.recommendation.reranking.items import Candidate

ALGORITHMS = {"epsilon_greedy", "ucb"}


class ExplorationPicker(Protocol):
    """탐색 슬롯에 넣을 후보를 우선순위 순서로 돌려준다."""

    def pick(self, candidates: Sequence[Candidate], n: int) -> list[Candidate]: ...


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

@dataclass
class MabConfig:
    algorithm: str = "epsilon_greedy"
    epsilon: float = 0.1
    ctr_prior: tuple[int, int] = (1, 2)  # Epsilon-Greedy 클릭률 보정 (클릭, 노출)
    ucb_c: float = 2.0
    arm_field: str = "category_l2"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"지원하지 않는 MAB 알고리즘: {self.algorithm} (가능: {sorted(ALGORITHMS)})")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon은 0~1 사이여야 함")
        if self.ctr_prior[0] < 0 or self.ctr_prior[1] <= self.ctr_prior[0]:
            raise ValueError("ctr_prior는 0 <= 클릭 < 노출 이어야 함")
        if self.ucb_c < 0:
            raise ValueError("ucb_c는 0 이상이어야 함")
        if self.arm_field not in {"category_l1", "category_l2", "category_l3"}:
            raise ValueError(f"arm_field 값이 잘못됨: {self.arm_field}")

    @classmethod
    def from_recommend_config(cls, raw: Mapping[str, Any] | None = None) -> MabConfig:
        """configs/recommend.yaml 의 reranking.mab 항목을 읽는다."""
        if raw is None:
            raw = load_config("recommend")
        m = raw["reranking"]["mab"]
        return cls(
            algorithm=m["algorithm"],
            epsilon=m["epsilon"],
            ctr_prior=tuple(m.get("ctr_prior", [1, 2])),
            ucb_c=m["ucb_c"],
            arm_field=m.get("arm_field", "category_l2"),
            seed=raw.get("seed", 42),
        )


# ---------------------------------------------------------------------------
# arm 통계 저장소
# ---------------------------------------------------------------------------

@dataclass
class ArmStats:
    shown: int = 0
    clicks: int = 0

    @property
    def ctr(self) -> float:
        return self.clicks / self.shown if self.shown else 0.0


class StatsStore(Protocol):
    """arm별 노출·클릭 통계. Redis 구현은 같은 메서드만 맞추면 교체된다."""

    def get(self, arms: Sequence[str]) -> dict[str, ArmStats]: ...

    def record(self, arm: str, clicked: bool) -> None: ...


class InMemoryStatsStore:
    """프로세스 메모리 저장소 (개발·테스트·오프라인 실험용)."""

    def __init__(self) -> None:
        self._stats: dict[str, ArmStats] = {}

    def get(self, arms: Sequence[str]) -> dict[str, ArmStats]:
        return {a: replace(self._stats.get(a, ArmStats())) for a in arms}

    def record(self, arm: str, clicked: bool) -> None:
        s = self._stats.setdefault(arm, ArmStats())
        s.shown += 1
        s.clicks += int(clicked)


# ---------------------------------------------------------------------------
# 탐색 선택기
# ---------------------------------------------------------------------------

class BanditExplorer:
    """MAB 공통 뼈대: 후보를 arm으로 묶고, arm 순위대로 돌아가며 arm별 최고 점수 상품을 꺼낸다.

    한 요청 안에서는 arm을 중복 없이 순위 매긴다. 같은 arm만 연달아 고르면
    탐색 칸 2개가 같은 상품군이 되고, 다양성 규칙 대비 여유분도 쓸모가 없어지기 때문.
    한 번도 노출되지 않은 arm은 항상 먼저 시도한다.
    """

    def __init__(self, store: StatsStore | None = None, arm_field: str = "category_l2",
                 seed: int = 42) -> None:
        self.store = store or InMemoryStatsStore()
        self.arm_field = arm_field
        self.rng = random.Random(seed)

    def arm_of(self, c: Candidate) -> str:
        return getattr(c, self.arm_field)

    def record_feedback(self, arm: str, clicked: bool) -> None:
        """탐색 슬롯에 실제로 진열된 상품의 arm과 클릭 여부를 기록한다."""
        self.store.record(arm, clicked)

    def pick(self, candidates: Sequence[Candidate], n: int) -> list[Candidate]:
        by_arm: dict[str, list[Candidate]] = {}
        for c in candidates:
            by_arm.setdefault(self.arm_of(c), []).append(c)
        for items in by_arm.values():
            items.sort(key=lambda c: c.score, reverse=True)

        order = self._rank_arms(sorted(by_arm), self.store.get(list(by_arm)))
        picked: list[Candidate] = []
        depth = 0
        while len(picked) < n and any(len(by_arm[a]) > depth for a in order):
            for arm in order:
                if len(picked) >= n:
                    break
                if len(by_arm[arm]) > depth:
                    picked.append(by_arm[arm][depth])
            depth += 1
        return picked

    def _rank_arms(self, arms: list[str], stats: dict[str, ArmStats]) -> list[str]:
        untried = [a for a in arms if stats[a].shown == 0]
        self.rng.shuffle(untried)
        return untried + self._rank_tried([a for a in arms if stats[a].shown > 0], stats)

    def _rank_tried(self, arms: list[str], stats: dict[str, ArmStats]) -> list[str]:
        raise NotImplementedError


class EpsilonGreedyExplorer(BanditExplorer):
    """매 순위마다 확률 epsilon으로 무작위 arm, 나머지는 보정 클릭률이 가장 높은 arm.

    보정 클릭률 = (클릭 + prior_clicks) / (노출 + prior_shown).
    클릭이 드물면 초반에 우연히 클릭 0번인 좋은 arm이 0%로 굳어 밀려난다.
    사전값을 섞으면 표본이 적은 arm은 "아직 모름" 쪽으로 평가돼 충분히 시험된다.
    (seed 100개 실험: 최고 arm 찾은 비율 82% → 100%)
    """

    def __init__(self, epsilon: float = 0.1, ctr_prior: tuple[int, int] = (1, 2),
                 **kw: Any) -> None:
        super().__init__(**kw)
        self.epsilon = epsilon
        self.prior_clicks, self.prior_shown = ctr_prior

    def smoothed_ctr(self, s: ArmStats) -> float:
        return (s.clicks + self.prior_clicks) / (s.shown + self.prior_shown)

    def _rank_tried(self, arms: list[str], stats: dict[str, ArmStats]) -> list[str]:
        remaining = list(arms)
        out: list[str] = []
        while remaining:
            if self.rng.random() < self.epsilon:
                arm = self.rng.choice(remaining)
            else:
                arm = max(remaining, key=lambda a: self.smoothed_ctr(stats[a]))  # 동률이면 이름순
            remaining.remove(arm)
            out.append(arm)
        return out


class UCBExplorer(BanditExplorer):
    """UCB1: 클릭률 + c * sqrt(ln 전체노출 / arm노출). 덜 시험한 arm일수록 가산점."""

    def __init__(self, c: float = 2.0, **kw: Any) -> None:
        super().__init__(**kw)
        self.c = c

    def ucb_score(self, s: ArmStats, total: int) -> float:
        return s.ctr + self.c * math.sqrt(math.log(total) / s.shown)

    def _rank_tried(self, arms: list[str], stats: dict[str, ArmStats]) -> list[str]:
        total = max(sum(stats[a].shown for a in arms), 1)
        return sorted(arms, key=lambda a: self.ucb_score(stats[a], total), reverse=True)


class RandomCategoryExplorer:
    """무작위 기준선: 카테고리를 무작위로 고르고 그 안에서 점수 최고 상품 (MAB 비교용)."""

    def __init__(self, category_field: str = "category_l2", seed: int = 42) -> None:
        self.category_field = category_field
        self.rng = random.Random(seed)

    def pick(self, candidates: Sequence[Candidate], n: int) -> list[Candidate]:
        by_cat: dict[str, list[Candidate]] = {}
        for c in candidates:
            by_cat.setdefault(getattr(c, self.category_field), []).append(c)
        for items in by_cat.values():
            items.sort(key=lambda c: c.score, reverse=True)

        picked: list[Candidate] = []
        cats = list(by_cat)
        while cats and len(picked) < n:
            cat = self.rng.choice(cats)
            picked.append(by_cat[cat].pop(0))
            if not by_cat[cat]:
                cats.remove(cat)
        return picked


def build_explorer(cfg: MabConfig | None = None, store: StatsStore | None = None) -> BanditExplorer:
    """설정의 algorithm에 맞는 탐색 선택기를 만든다. cfg가 없으면 recommend.yaml에서 읽는다."""
    cfg = cfg or MabConfig.from_recommend_config()
    common = dict(store=store, arm_field=cfg.arm_field, seed=cfg.seed)
    if cfg.algorithm == "epsilon_greedy":
        return EpsilonGreedyExplorer(epsilon=cfg.epsilon, ctr_prior=cfg.ctr_prior, **common)
    return UCBExplorer(c=cfg.ucb_c, **common)
