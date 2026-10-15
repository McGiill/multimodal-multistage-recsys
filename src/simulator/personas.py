"""6개 페르소나 파라미터 로딩과 검증 (configs/simulator.yaml → persona_ratio, personas).

페르소나는 "어떻게 사는가"를 정한다.
- 전환·이탈 (담기 확률, 같은 방문 구매, 다음 방문 구매, 조회 후 이탈, 방문 횟수):
  Retailrocket에서 페르소나 정의에 맞는 칸(rr_group)의 전이 확률을 목표로 보정한 값 (calibrate.py)
- 가격 민감도, 브랜드 충성, 신상품 선호, 선호 카테고리, 화면 이용 습관: 설계값 (Retailrocket으로 측정 불가)
"무엇을 좋아하는가"(스타일)는 고객마다 빌려온 실제 H&M 고객의 취향 위치가 정한다 (users.py).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from src.common.schemas import Persona

SCREENS = ("search", "browse", "home")
BROWSE_SORTS = ("popular", "newest", "price_low")
RR_GROUPS = {f"{c}/{d}" for c in ("low", "mid", "high") for d in ("short", "long")}
FUNNEL_KEYS = ("cart_rate", "purchase_in_session", "purchase_next_visit", "exit_after_view", "sessions")


@dataclass(frozen=True)
class PersonaParams:
    name: str
    rr_group: str                      # Retailrocket 칸 ("high/short" 등)
    preferred_l2: tuple[str, ...]
    n_preferred: int
    price_sensitivity: tuple[float, float]
    new_item_affinity: tuple[float, float]
    brand_loyalty: tuple[float, float]
    screens: dict[str, float]
    browse_sort: dict[str, float]      # 카테고리 둘러보기 정렬 비율 (없으면 lists.browse_sort)
    click_threshold: float
    max_clicks: int
    # 보정 대상 (초기값 = funnel.initial, calibrate.py가 Retailrocket 칸에 맞춰 바꿈)
    cart_rate: float                   # 조회한 상품을 담을 기본 확률
    purchase_in_session: float         # 담은 직후 같은 방문에서 살 확률
    purchase_next_visit: float         # 장바구니에 남은 상품을 다음 방문 시작 때 살 확률
    exit_after_view: float             # 조회 직후 방문을 끝낼 확률
    sessions: float                    # 90일 평균 방문 수


def _check_probs(name: str, probs: dict[str, float], keys: tuple[str, ...] | None = None) -> None:
    if keys is not None and set(probs) != set(keys):
        raise ValueError(f"{name}: 키는 {keys} 이어야 합니다 ({sorted(probs)})")
    if abs(sum(probs.values()) - 1.0) > 1e-6:
        raise ValueError(f"{name}: 확률 합이 1이 아닙니다 ({sum(probs.values()):.3f})")


def load_personas(cfg: dict[str, Any]) -> dict[str, PersonaParams]:
    """설정 파일 기준 페르소나 (전환·이탈 값은 보정 전 초기값)."""
    ratio = cfg["persona_ratio"]
    names = {p.value for p in Persona}
    if set(ratio) != names or set(cfg["personas"]) != names:
        raise ValueError(f"페르소나는 {sorted(names)} 6개를 모두 정의해야 합니다")
    _check_probs("persona_ratio", ratio)
    for row, probs in cfg["session_flow"].items():
        if isinstance(probs, dict):
            _check_probs(f"session_flow.{row}", probs)

    init = cfg["funnel"]["initial"]
    out = {}
    for name, p in cfg["personas"].items():
        _check_probs(f"personas.{name}.screens", p["screens"], SCREENS)
        browse_sort = p.get("browse_sort", cfg["lists"]["browse_sort"])
        _check_probs(f"personas.{name}.browse_sort", browse_sort, BROWSE_SORTS)
        if p["rr_group"] not in RR_GROUPS:
            raise ValueError(f"personas.{name}.rr_group은 {sorted(RR_GROUPS)} 중 하나: {p['rr_group']}")
        out[name] = PersonaParams(
            name=name,
            rr_group=p["rr_group"],
            preferred_l2=tuple(p["preferred_l2"]),
            n_preferred=int(p["n_preferred"]),
            price_sensitivity=tuple(p["price_sensitivity"]),
            new_item_affinity=tuple(p["new_item_affinity"]),
            brand_loyalty=tuple(p["brand_loyalty"]),
            screens=dict(p["screens"]),
            browse_sort=dict(browse_sort),
            click_threshold=float(p["click_threshold"]),
            max_clicks=int(p["max_clicks"]),
            cart_rate=float(init["cart_rate"]),
            purchase_in_session=float(init["purchase_in_session"]),
            purchase_next_visit=float(init["purchase_next_visit"]),
            exit_after_view=float(init["exit_after_view"]),
            sessions=float(cfg["users"]["sessions_base"]),
        )
    return out


def with_funnel(personas: dict[str, PersonaParams], values: dict[str, dict[str, float]]) -> dict[str, PersonaParams]:
    """보정된 전환·이탈 값을 덮어쓴 페르소나."""
    return {
        name: dataclasses.replace(p, **{k: float(values[name][k]) for k in FUNNEL_KEYS if k in values.get(name, {})})
        for name, p in personas.items()
    }
