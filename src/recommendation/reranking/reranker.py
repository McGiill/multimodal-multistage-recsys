"""Stage 3: Re-ranking (비즈니스 로직 + 탐색 슬롯).

Stage 2가 점수를 매긴 후보를 받아 최종 Top-N을 만든다.

적용 순서
  1. cold-start 판단: 이력이 min_history 미만이면 메인 목록을 인기+트렌딩으로 교체
  2. 메인 목록 정렬 (Stage 2 점수 내림차순)
  3. 슬롯 위치 계산: 설정에는 개수만 있고, 위치는 Top-N을 균등하게 나눈 지점 (slot_positions)
  4. 위치별 채우기: 슬롯 위치(신규 상품, 탐색)는 해당 풀에서 먼저 고르고,
     나머지 위치는 메인 목록에서 고른다.
  5. 모든 위치에서 다양성 규칙을 함께 검사한다.

외부 의존(Redis, MAB, Stage 2)은 모두 인자로 주입받아 단독 테스트가 가능하다.
담당: D
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from typing import Any, Mapping, Sequence

from src.common.config import load_config
from src.recommendation.reranking.cold_start import interleave, is_cold_start
from src.recommendation.reranking.diversity import breaks_diversity
from src.recommendation.reranking.items import Candidate, RerankedItem, RerankResult, dedupe_key
from src.recommendation.reranking.mab import ExplorationPicker, RandomCategoryExplorer
from src.recommendation.reranking.new_item import build_new_item_pool

# 위치 → (풀, reason, is_exploration)
SlotPools = dict[int, tuple[list[Candidate], str, bool]]


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

@dataclass
class RerankConfig:
    top_n: int = 10

    diversity_enabled: bool = True
    category_field: str = "category_l2"
    max_consecutive: int = 2
    dedupe_field: str | None = "product_code"  # 같은 디자인 색상 변형은 1개만. None이면 product_id만

    min_history: int = 5
    popular_trending_ratio: tuple[int, int] = (1, 1)

    new_item_max_age_days: int = 7
    new_item_slots: int = 1           # 개수. 0이면 비활성

    exploration_enabled: bool = True  # A/B 테스트 Control은 False
    exploration_slots: int = 2        # 개수 (명세: 1~2개)

    def __post_init__(self) -> None:
        if self.category_field not in {"category_l1", "category_l2", "category_l3"}:
            raise ValueError(f"category_field 값이 잘못됨: {self.category_field}")
        if self.dedupe_field not in {None, "product_code"}:
            raise ValueError(f"dedupe_field 값이 잘못됨: {self.dedupe_field}")
        if self.max_consecutive < 1:
            raise ValueError("max_consecutive는 1 이상이어야 함")
        if self.new_item_slots < 0 or self.exploration_slots < 0:
            raise ValueError("슬롯 개수는 0 이상이어야 함")

    @classmethod
    def from_recommend_config(cls, raw: Mapping[str, Any] | None = None) -> RerankConfig:
        """configs/recommend.yaml 의 reranking 항목을 읽는다. raw를 주면 그 dict를 쓴다."""
        if raw is None:
            raw = load_config("recommend")
        r = raw["reranking"]
        mab = r.get("mab", {})
        return cls(
            top_n=raw.get("default_top_n", 10),
            category_field=r.get("category_field", "category_l2"),
            max_consecutive=r["max_same_category_run"],
            dedupe_field=r.get("dedupe_field", "product_code"),
            min_history=r["cold_start_min_events"],
            popular_trending_ratio=tuple(r.get("popular_trending_ratio", [1, 1])),
            new_item_max_age_days=r["new_item_days"],
            new_item_slots=r["new_item_min_slots"],
            exploration_enabled=mab.get("enabled", True),
            exploration_slots=mab["exploration_slots"],
        )


def slot_positions(top_n: int, n_new: int, n_exp: int) -> tuple[list[int], list[int]]:
    """슬롯 개수 → 0-based 위치. Top-N을 (개수+1) 등분한 지점에 배치한다.

    예) top_n=10, 신규 1개 → [5], 탐색 2개 → [3, 7].
    신규 상품을 먼저 놓고, 탐색 위치가 겹치면 다음 빈 칸(없으면 앞쪽 빈 칸)으로 민다.
    빈 칸이 모자라면 남는 슬롯은 버린다.
    """
    taken: set[int] = set()

    def spread(k: int) -> list[int]:
        return [min(top_n - 1, int(top_n * (i + 1) / (k + 1) + 0.5)) for i in range(k)]

    def place(targets: list[int]) -> list[int]:
        out: list[int] = []
        for p in targets:
            for q in chain(range(p, top_n), range(p - 1, -1, -1)):
                if q not in taken:
                    taken.add(q)
                    out.append(q)
                    break
        return out

    if top_n <= 0:
        return [], []
    new_pos = place(spread(n_new))
    exp_pos = place(spread(n_exp))
    return new_pos, exp_pos


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class Reranker:
    def __init__(self, config: RerankConfig, explorer: ExplorationPicker | None = None) -> None:
        self.config = config
        self.explorer = explorer or RandomCategoryExplorer()

    # ---- public ----------------------------------------------------------

    def rerank(
        self,
        candidates: Sequence[Candidate],
        history_len: int,
        now: datetime,
        popular: Sequence[Candidate] = (),
        trending: Sequence[Candidate] = (),
        new_item_pool: Sequence[Candidate] = (),
        top_n: int | None = None,
    ) -> RerankResult:
        """최종 Top-N을 만든다.

        Args:
            candidates: Stage 2 출력 (cold-start면 비어 있어도 됨)
            history_len: 사용자 행동 이력 수 (Redis에서 조회)
            now: 신규 상품 판단 기준 시각 (요청 시각). 후보 created_at과 시간대 형식이
                같아야 한다 (카탈로그 후보는 UTC → catalog.to_utc(now))
            popular, trending: cold-start용 사전 집계 목록
            new_item_pool: 후보에 신규 상품이 없을 때 쓸 보조 목록
            top_n: API 요청의 top_n. None이면 config.top_n
        """
        start = time.perf_counter()
        cfg = self.config
        n = cfg.top_n if top_n is None else top_n
        is_cold = is_cold_start(history_len, cfg.min_history)

        if is_cold:
            main = interleave(popular, trending, cfg.popular_trending_ratio)
        else:
            main = sorted(candidates, key=lambda c: c.score, reverse=True)

        slot_pools = self._build_slot_pools(candidates, main, now, new_item_pool, n)

        items, relaxed = self._fill(main, slot_pools, n)
        elapsed = (time.perf_counter() - start) * 1000
        return RerankResult(items, is_cold, elapsed, relaxed)

    # ---- 풀 구성 ---------------------------------------------------------

    def _build_slot_pools(
        self,
        candidates: Sequence[Candidate],
        main: Sequence[Candidate],
        now: datetime,
        new_item_pool: Sequence[Candidate],
        top_n: int,
    ) -> SlotPools:
        cfg = self.config
        n_exp = cfg.exploration_slots if cfg.exploration_enabled else 0
        new_pos, exp_pos = slot_positions(top_n, cfg.new_item_slots, n_exp)
        pools: SlotPools = {}

        if new_pos:
            new_items = build_new_item_pool(candidates, new_item_pool, now,
                                            cfg.new_item_max_age_days)
            for pos in new_pos:
                pools[pos] = (new_items, "new_item", False)

        if exp_pos:
            # 메인 상위 top_n 밖에서 탐색해야 의미가 있다
            outside = list(main[top_n:]) or list(main)
            explored = self.explorer.pick(outside, len(exp_pos) * 3)  # 다양성 때문에 여유분 확보
            for pos in exp_pos:
                pools[pos] = (explored, "mab_exploration", True)

        return pools

    # ---- 채우기 ----------------------------------------------------------

    def _fill(
        self,
        main: Sequence[Candidate],
        slot_pools: SlotPools,
        top_n: int,
    ) -> tuple[list[RerankedItem], list[int]]:
        cfg = self.config
        used: set[str] = set()
        cats: list[str] = []
        out: list[RerankedItem] = []
        relaxed: list[int] = []

        for pos in range(top_n):
            # 시도 순서: 슬롯 풀 → 메인
            attempts: list[tuple[Sequence[Candidate], str | None, bool]] = []
            if pos in slot_pools:
                attempts.append(slot_pools[pos])
            attempts.append((main, None, False))

            chosen = self._choose(attempts, used, cats, strict=True)
            if chosen is None:
                chosen = self._choose(attempts, used, cats, strict=False)
                if chosen is None:
                    break  # 후보 소진
                relaxed.append(pos)

            cand, reason, is_exp = chosen
            used.add(dedupe_key(cand, cfg.dedupe_field))
            cats.append(getattr(cand, cfg.category_field))
            out.append(RerankedItem(cand.product_id, cand.score, reason or cand.reason, is_exp))

        return out, relaxed

    def _choose(
        self,
        attempts: Sequence[tuple[Sequence[Candidate], str | None, bool]],
        used: set[str],
        cats: Sequence[str],
        strict: bool,
    ) -> tuple[Candidate, str | None, bool] | None:
        cfg = self.config
        for pool, reason, is_exp in attempts:
            for c in pool:
                if dedupe_key(c, cfg.dedupe_field) in used:
                    continue
                if (strict and cfg.diversity_enabled
                        and breaks_diversity(cats, getattr(c, cfg.category_field),
                                             cfg.max_consecutive)):
                    continue
                return c, reason, is_exp
        return None
