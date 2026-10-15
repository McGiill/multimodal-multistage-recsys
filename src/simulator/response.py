"""반응 함수: 상품 목록을 보여주면 가상 고객이 무엇을 누르고, 담고, 사는지 정한다.

같은 함수를 두 곳에서 쓴다.
- 오프라인 로그 생성 (behavior.py): 시뮬레이터가 만든 목록(검색 결과, 카테고리 둘러보기, 홈)
- 온라인 모드 (추후): 추천 API가 준 목록 → 클릭이 MAB 보상, A/B 테스트 클릭률이 된다

상품 점수 (configs/simulator.yaml → utility)
    고정 점수 = 취향 유사도 + 인기도 + 선호 카테고리 + 주 쇼핑 고객층 + 선호 브랜드 − 가격 부담 + 고정 랜덤 선호
              → 고객마다 전체 상품에서 표준화(z)
    보는 순간 점수 = 고정 점수 + 신상품 보너스(등록 7일 이내) + 이번 방문에서 본 상품과의 유사도 − 이미 본 상품
목록 반응
    "볼 확률"은 위쪽 순위일수록 높다 (exp(-decay × 순위)). 목록 안 상품과 "아무것도 안 누름" 중에서
    점수에 비례한 확률로 고른다 → 클릭하면 view, 일정 확률로 cart → purchase (또는 장바구니에 남김)
    담기·구매·이탈 확률은 페르소나 값 (Retailrocket 칸에 맞춰 보정됨, calibrate.py)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.simulator.catalog import World
from src.simulator.personas import PersonaParams


@dataclass
class Shopper:
    """한 고객의 시뮬레이션 상태 (방문을 넘어 유지)."""

    index: int
    persona: PersonaParams
    segment: int                     # 주 쇼핑 고객층 코드
    new_affinity: float
    base: np.ndarray                 # 상품별 고정 점수 z
    pool: np.ndarray                 # "찾는 상품" 후보 (고정 점수 상위)
    cart: dict[int, int] = field(default_factory=dict)   # 상품 → 담은 방문 번호 (방문을 넘어 유지)


@dataclass
class Visit:
    """한 번의 방문(세션) 상태."""

    day: int                         # 로그 시작일 기준 날짜
    number: int = 0
    viewed: list[int] = field(default_factory=list)
    vec_sum: np.ndarray | None = None


@dataclass
class Reaction:
    actions: list[tuple[str, int]]   # ("view" | "cart" | "purchase", 상품)
    clicked: list[int]
    next_screen: str                 # "search" | "browse" | "home" | "exit"


def base_scores(world: World, w: dict[str, float], taste: np.ndarray, pref_l2: np.ndarray, segment: int,
                pref_brands: np.ndarray, loyalty: float, price_sensitivity: float,
                rng: np.random.Generator) -> np.ndarray:
    s = (
        w["taste"] * (world.vectors @ taste)
        + w["popularity"] * world.popularity
        + w["category"] * np.isin(world.l2, pref_l2)
        + w["segment"] * (world.l1 == segment)
        + w["brand"] * loyalty * np.isin(world.brand, pref_brands)
        - w["price"] * price_sensitivity * world.rel_price
        + w["noise"] * rng.normal(size=world.n)
    )
    return ((s - s.mean()) / s.std()).astype(np.float32)


def _sample(rng: np.random.Generator, probs: dict[str, float], exclude: str | None = None) -> str:
    keys = [k for k in probs if k != exclude]
    p = np.array([probs[k] for k in keys], dtype=float)
    return str(rng.choice(keys, p=p / p.sum()))


class ResponseModel:
    def __init__(self, world: World, cfg: dict[str, Any]) -> None:
        self.world = world
        self.w = cfg["utility"]
        self.decay = cfg["lists"]["position_decay"]
        self.new_days = cfg["lists"]["new_days"]
        self.funnel = cfg["funnel"]
        self.flow = cfg["session_flow"]

    # ------------------------------------------------------------------ 점수
    def scores(self, shopper: Shopper, visit: Visit, items: np.ndarray) -> np.ndarray:
        world, w = self.world, self.w
        created = world.created_day[items]
        is_new = (created > visit.day - self.new_days) & (created <= visit.day)
        s = shopper.base[items] + w["new_item"] * shopper.new_affinity * is_new
        if visit.viewed:
            unit = visit.vec_sum / max(np.linalg.norm(visit.vec_sum), 1e-6)
            s = s + w["session"] * (world.vectors[items] @ unit) - 1.5 * np.isin(items, visit.viewed)
        return s

    def pick_target(self, shopper: Shopper, visit: Visit, rng: np.random.Generator) -> int:
        """고객이 "마음속으로 찾는 상품" (검색어·둘러볼 카테고리를 정함)."""
        world = self.world
        recent_new = world.new_items[
            (world.created_day[world.new_items] > visit.day - self.new_days)
            & (world.created_day[world.new_items] <= visit.day)
        ]
        cands = world.registered(np.union1d(shopper.pool, recent_new), visit.day)
        s = self.scores(shopper, visit, cands)
        return int(cands[np.argmax(s + rng.gumbel(size=len(s)))])

    # ------------------------------------------------------------------ 목록 반응
    def _after_view(self, persona: PersonaParams, rng: np.random.Generator, same_list: bool = True) -> str:
        """조회 다음 행동: exit_after_view 확률로 방문 종료, 아니면 session_flow.after_view 비율로 계속."""
        if rng.random() < persona.exit_after_view:
            return "exit"
        return _sample(rng, self.flow["after_view"], exclude=None if same_list else "same_list")

    def react(self, shopper: Shopper, visit: Visit, items: np.ndarray, rng: np.random.Generator) -> Reaction:
        p, f = shopper.persona, self.funnel
        actions: list[tuple[str, int]] = []
        clicked: list[int] = []
        exam = -self.decay * np.arange(len(items))
        open_ = np.ones(len(items), dtype=bool)
        cart_odds = np.log(p.cart_rate / (1 - p.cart_rate))

        while True:
            if len(clicked) >= p.max_clicks or not open_.any():
                return Reaction(actions, clicked, self._after_view(p, rng, same_list=False))
            z = self.scores(shopper, visit, items)
            logits = np.append(np.where(open_, z + exam, -np.inf), p.click_threshold + 0.5 * len(clicked))
            probs = np.exp(logits - logits.max())
            choice = int(rng.choice(len(probs), p=probs / probs.sum()))
            if choice == len(items):  # 아무것도 안 누름
                nxt = _sample(rng, self.flow["after_no_click"]) if not clicked else self._after_view(p, rng, False)
                return Reaction(actions, clicked, nxt)

            item = int(items[choice])
            open_[choice] = False
            clicked.append(item)
            actions.append(("view", item))
            visit.viewed.append(item)
            v = self.world.vectors[item]
            visit.vec_sum = v.copy() if visit.vec_sum is None else visit.vec_sum + v

            # 점수가 높은 상품일수록 잘 담는다 (페르소나 기본 확률 cart_rate 주변에서)
            p_cart = 1 / (1 + np.exp(-(cart_odds + f["cart_utility_effect"] * (z[choice] - f["cart_utility_ref"]))))
            if rng.random() < p_cart:
                actions.append(("cart", item))
                if rng.random() < p.purchase_in_session:
                    actions.append(("purchase", item))
                    shopper.base[item] -= 2.0  # 산 상품은 다시 잘 안 봄
                else:
                    shopper.cart[item] = visit.number

            nxt = self._after_view(p, rng)
            if nxt != "same_list":
                return Reaction(actions, clicked, nxt)

    def checkout_cart(self, shopper: Shopper, visit: Visit, rng: np.random.Generator) -> list[int]:
        """방문 시작 때 지난번 장바구니 처리: 일부는 구매, 일부는 포기, 나머지는 그대로."""
        f, bought = self.funnel, []
        for item in list(shopper.cart):
            r = rng.random()
            if r < shopper.persona.purchase_next_visit:
                bought.append(item)
                del shopper.cart[item]
                shopper.base[item] -= 2.0
            elif r > 1 - f["cart_forget"]:
                del shopper.cart[item]
        return bought
