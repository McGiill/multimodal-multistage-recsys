"""페르소나 전환·이탈 값을 Retailrocket 전이 확률에 맞춘다: python -m src.simulator.calibrate

목표: 페르소나마다 rr_group 칸(rr_calibration.py)의 view → cart, 조회 후 이탈, 담은 상품의 같은 방문·나중 구매

파라미터는 두 종류다.
1. 넣은 값 = 로그에 나오는 비율 → 목표값을 그대로 넣고 보정하지 않는다
   purchase_in_session: 담은 직후 "이번 방문에 산다"를 이 확률로 정하고, 다른 경로가 없다.
   (소규모 생성으로 다시 재면 우연한 차이만 보이므로, 그걸 따라 고치면 오히려 목표에서 멀어진다)
2. 넣은 값 ≠ 로그에 나오는 비율 → 작게 생성해 보고 → 같은 지표(behavior_stats, Retailrocket과 같은 함수)로
   재고 → 차이만큼 조정하기를 반복한다.
   예: 조회 뒤 다른 화면으로 갔다가 아무것도 안 누르고 나가도 로그에는 "조회 → 이탈"로 보인다.
   cart_rate            × (목표 view_to_cart ÷ 측정값)         점수가 높은 상품일수록 잘 담아서 1:1이 아님
   exit_after_view      logit을 (목표 logit − 측정값 logit)만큼 이동
   purchase_next_visit  × (목표 cart_purchase_later ÷ 측정값)   방문마다 확률이라 남은 방문 수에 따라 달라짐
   조정 폭은 반복할수록 줄인다 (1, 1/2, 1/3, ...): 처음에는 구조 차이를 크게 메우고, 이후에는 우연한 차이에
   끌려다니지 않도록 여러 번의 측정을 평균하는 효과를 낸다.

방문 횟수는 페르소나와 상관없이 같다: users.sessions_base × 전체 배율 (전체 이벤트가 scale.target_events가 되도록).
Retailrocket의 칸별 방문 횟수 차이(activity_ratio)는 쓰지 않는다. 그 값을 쓰면 방문이 잦은 칸의 페르소나가 로그를
크게 차지해서, 추천 모델이 페르소나마다 고르게 배우지 못한다. 고객마다의 방문 수 차이는 users.activity_sigma로 준다.
결과: data/processed/simulator/persona_calibration.json — 설정이나 목표가 바뀌면 다시 보정한다.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import ROOT, load_config
from src.simulator import item_space
from src.simulator.catalog import World
from src.simulator.personas import (
    FUNNEL_KEYS,
    PersonaParams,
    load_personas,
    with_funnel,
)
from src.simulator.pipeline import Inputs, load_inputs, load_world, simulate
from src.simulator.rr_calibration import behavior_stats, load_targets

TARGET_KEYS = ("view_to_cart", "view_exit", "cart_purchase_same_visit", "cart_purchase_later")
CONFIG_SECTIONS = ("seed", "scale", "period", "users", "persona_ratio", "personas", "utility", "lists", "search",
                   "funnel", "session_flow", "time", "calibration", "retailrocket")


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return float(np.log(p / (1 - p)))


def _sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


def persona_targets(cfg: dict[str, Any], targets: dict[str, Any]) -> dict[str, dict[str, float]]:
    out = {}
    for name, p in cfg["personas"].items():
        cell = targets["cells"][p["rr_group"]]
        out[name] = {"rr_group": p["rr_group"], **{k: cell[k] for k in TARGET_KEYS}}
    return out


def _initial(cfg: dict[str, Any], goal: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """목표 비율을 그대로 시작값으로. 다음 방문 구매는 "방문마다 확률"로 바꿔 둔다."""
    forget = cfg["funnel"]["cart_forget"]
    values = {}
    for name, g in goal.items():
        later = g["cart_purchase_later"] / max(1 - g["cart_purchase_same_visit"], 1e-3)  # 남은 장바구니 중 나중에 산 비율
        values[name] = {
            "cart_rate": g["view_to_cart"],
            "purchase_in_session": g["cart_purchase_same_visit"],  # 1:1 → 이 값으로 고정
            "purchase_next_visit": later * forget / max(1 - later, 1e-3),  # 방문마다 p, 포기 q → 결국 살 확률 p/(p+q)
            "exit_after_view": g["view_exit"],
            "sessions": float(cfg["users"]["sessions_base"]),  # 모든 페르소나 같은 값
        }
    return values


def _measure(events: pd.DataFrame, pop_hidden: pd.DataFrame) -> dict[str, dict[str, float]]:
    persona = events["user_id"].map(pop_hidden.set_index("user_id")["persona"])
    return {name: behavior_stats(events[persona == name]) for name in sorted(pop_hidden["persona"].unique())}


def _update(values: dict[str, float], got: dict[str, float], goal: dict[str, float], step: float = 1.0) -> dict[str, float]:
    """측정값을 목표 쪽으로 step만큼 옮긴다 (step=1이면 차이를 한 번에 메움). purchase_in_session은 고정."""
    def ratio(key: str) -> float:
        r = goal[key] / got[key] if got[key] > 1e-4 else 2.0
        return float(np.clip(r, 0.5, 2.0)) ** step  # 한 번에 최대 2배까지만

    v = dict(values)
    v["cart_rate"] = float(np.clip(v["cart_rate"] * ratio("view_to_cart"), 1e-3, 0.6))
    v["exit_after_view"] = float(np.clip(
        _sigmoid(_logit(v["exit_after_view"]) + step * (_logit(goal["view_exit"]) - _logit(got["view_exit"]))),
        0.02, 0.95))
    v["purchase_next_visit"] = float(np.clip(v["purchase_next_visit"] * ratio("cart_purchase_later"), 0.0, 0.5))
    return v


def calibration_key(cfg: dict[str, Any], targets: dict[str, Any]) -> str:
    payload = {k: cfg[k] for k in CONFIG_SECTIONS} | {"targets": targets["cells"]}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def calibrate(cfg: dict[str, Any], world: World, inputs: Inputs, targets: dict[str, Any]) -> dict[str, Any]:
    cal = cfg["calibration"]
    n_cal = cal["users"]
    goal = persona_targets(cfg, targets)
    values = _initial(cfg, goal)
    base = load_personas(cfg)
    target_epu = cfg["scale"]["target_events"] / cfg["scale"]["n_users"]
    history = []

    for it in range(cal["iterations"] + 1):
        pop, events, _ = simulate(cfg, world, with_funnel(base, values), inputs, n_users=n_cal)
        got = _measure(events, pop.hidden)
        epu = len(events) / n_cal
        history.append({"iteration": it, "events_per_user": round(epu, 1),
                        "fit": {n: {k: [goal[n][k], got[n][k]] for k in TARGET_KEYS} for n in goal}})
        err = np.mean([abs(got[n][k] - goal[n][k]) / max(goal[n][k], 1e-3) for n in goal for k in TARGET_KEYS])
        print(f"  보정 {it}: 고객당 이벤트 {epu:.0f}, 목표 대비 평균 상대오차 {err:.1%}", flush=True)
        if it == cal["iterations"]:
            break
        values = {n: _update(values[n], got[n], goal[n], step=1 / (it + 1)) for n in goal}
        scale = target_epu / epu
        for n in values:
            values[n]["sessions"] *= scale

    return {"key": calibration_key(cfg, targets), "values": values, "targets": goal, "achieved": got,
            "events_per_user": round(len(events) / n_cal, 1), "history": history}


def result_path(cfg: dict[str, Any]):
    return item_space.space_dir(cfg) / "persona_calibration.json"


def load_or_calibrate(cfg: dict[str, Any], world: World, inputs: Inputs, force: bool = False) -> dict[str, Any]:
    targets = load_targets(cfg)
    path = result_path(cfg)
    if not force and path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("key") == calibration_key(cfg, targets):
            return saved
    print(f"[simulator] Retailrocket 전이 확률에 맞춰 페르소나 보정 ({cfg['calibration']['users']:,}명으로 반복 생성)")
    result = calibrate(cfg, world, inputs, targets)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def calibrated_personas(cfg: dict[str, Any], world: World, inputs: Inputs,
                        force: bool = False) -> tuple[dict[str, PersonaParams], dict[str, Any]]:
    result = load_or_calibrate(cfg, world, inputs, force=force)
    return with_funnel(load_personas(cfg), result["values"]), result


def main() -> None:
    cfg = copy.deepcopy(load_config("simulator"))
    world = load_world(cfg)
    _, result = calibrated_personas(cfg, world, load_inputs(cfg, load_config("catalog")), force=True)
    rows = {n: {**{f"{k} (목표/결과)": f"{result['targets'][n][k]:.3f} / {result['achieved'][n][k]:.3f}"
                   for k in TARGET_KEYS}, **{k: round(result['values'][n][k], 4) for k in FUNNEL_KEYS}}
            for n in result["values"]}
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(pd.DataFrame(rows).T.to_string())
    print(f"[simulator] 저장 → {result_path(cfg).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
