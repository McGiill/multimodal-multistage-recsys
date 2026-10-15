"""Retailrocket 행동 로그로 전이 확률 추정 (명세서: Retailrocket → Markov Chain 전이 확률 추정).

    python -m src.simulator.rr_calibration     # data/processed/simulator/rr_targets.json

분포 기반 배정
1. 방문을 30분 공백 기준으로 나누고, 활동 방문자(방문 2회 이상, 이벤트 5개 이상)만 쓴다.
   Retailrocket 방문의 대부분은 한 번 보고 나가는 유입이라 가입 고객인 우리 시뮬레이션 고객과 성격이 다르다.
2. 방문자마다 방문을 시간순으로 반 나눠, 앞쪽 절반으로 칸을 정하고 뒤쪽 절반으로 전이 확률을 잰다.
   (같은 기록으로 분류·측정하면 우연히 많이/적게 산 사람이 극단 칸에 몰려 차이가 과장된다)
3. 앞쪽 절반 기준으로 전환율(하 / 중 / 상) × 방문 깊이(짧음 / 김) 6칸으로 나눈다.
   - 전환율 = (담기 + 구매) ÷ 조회 (조회가 적은 방문자는 전체 평균 쪽으로 당겨서 계산)
   - 방문 깊이 = 방문당 평균 이벤트 수
4. 칸마다 전이 확률을 계산한다. 페르소나 ↔ 칸 대응은 configs/simulator.yaml의 personas.*.rr_group.
   수치는 모두 실제 데이터, 우리가 정하는 것은 "어느 페르소나에 어느 칸"뿐이다.
   (Retailrocket은 브랜드가 없고 가격이 해시라서 브랜드·가격 성향은 측정할 수 없다 → H&M과 설계값 사용)

behavior_stats()는 Retailrocket과 시뮬레이터 로그에 똑같이 적용해 둘을 같은 기준으로 비교한다.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import ROOT, load_config

CONVERSION_BINS = ("low", "mid", "high")
DEPTH_BINS = ("short", "long")


def load_rr_events(cfg: dict[str, Any]) -> pd.DataFrame:
    """Retailrocket events.csv → user_id, session_id, timestamp, event_type, product_id (방문 단위로 나눔)."""
    rr = cfg["retailrocket"]
    ev = pd.read_csv(ROOT / rr["events"], usecols=["timestamp", "visitorid", "event", "itemid"])
    ev = ev.sort_values(["visitorid", "timestamp"], kind="stable").reset_index(drop=True)
    gap = ev.groupby("visitorid")["timestamp"].diff().fillna(np.inf) > cfg["time"]["session_gap_min"] * 60 * 1000
    return pd.DataFrame({
        "user_id": ev["visitorid"].to_numpy(),
        "session_id": gap.cumsum().to_numpy(),
        "timestamp": pd.to_datetime(ev["timestamp"], unit="ms"),
        "event_type": ev["event"].map({"view": "view", "addtocart": "cart", "transaction": "purchase"}),
        "product_id": ev["itemid"].to_numpy(),
    })


def behavior_stats(events: pd.DataFrame) -> dict[str, float]:
    """이벤트 단위 전이 확률 + 담은 상품의 구매 시점. search 이벤트는 Retailrocket에 없으므로 제외하고 계산."""
    ev = events[events["event_type"] != "search"].sort_values(["session_id", "timestamp"], kind="stable")
    nxt = ev.groupby("session_id")["event_type"].shift(-1).fillna("exit")
    after_view = nxt[ev["event_type"] == "view"].value_counts(normalize=True)

    carts = ev.loc[ev["event_type"] == "cart", ["user_id", "session_id", "product_id", "timestamp"]].reset_index()
    buys = ev.loc[ev["event_type"] == "purchase", ["user_id", "session_id", "product_id", "timestamp"]]
    m = carts.merge(buys, on=["user_id", "product_id"], suffixes=("", "_buy"))
    m = m[m["timestamp_buy"] >= m["timestamp"]].sort_values("timestamp_buy").drop_duplicates("index")
    same = int((m["session_id_buy"] == m["session_id"]).sum())
    n_carts = max(len(carts), 1)
    sessions = ev.groupby("session_id")["user_id"].first()
    return {
        "view_to_view": round(float(after_view.get("view", 0.0)), 4),
        "view_to_cart": round(float(after_view.get("cart", 0.0) + after_view.get("purchase", 0.0)), 4),
        "view_exit": round(float(after_view.get("exit", 0.0)), 4),
        "cart_purchase_same_visit": round(same / n_carts, 4),
        "cart_purchase_later": round((len(m) - same) / n_carts, 4),
        "events_per_session": round(len(ev) / max(len(sessions), 1), 3),
        "sessions_per_user": round(len(sessions) / max(sessions.nunique(), 1), 3),
    }


def split_halves(events: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """활동 방문자만 남기고, 방문마다 앞쪽 절반(분류용) / 뒤쪽 절반(전이 확률 측정용)으로 나눈다.

    같은 기록으로 분류하고 측정하면 "우연히 많이 산 사람"이 높은 칸에 몰려 차이가 과장된다.
    """
    rr = cfg["retailrocket"]
    per = events.groupby("user_id").agg(sessions=("session_id", "nunique"), events=("event_type", "size"))
    active = per.index[(per["sessions"] >= rr["min_sessions"]) & (per["events"] >= rr["min_events"])]
    ev = events[events["user_id"].isin(active)]
    order = ev.groupby("user_id")["session_id"].rank(method="dense")
    n = ev["user_id"].map(per["sessions"])
    first = order <= np.ceil(n / 2)
    return ev[first], ev[~first]


def assign_groups(profile: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """분류용 기록으로 방문자 → (전환율 칸, 방문 깊이 칸)."""
    rr = cfg["retailrocket"]
    per = profile.groupby("user_id").agg(
        sessions=("session_id", "nunique"),
        events=("event_type", "size"),
        views=("event_type", lambda s: int((s == "view").sum())),
        converts=("event_type", lambda s: int(s.isin(["cart", "purchase"]).sum())),
    )
    p0 = per["converts"].sum() / max(per["views"].sum(), 1)
    k = rr["conversion_smoothing"]
    per["conversion"] = (per["converts"] + k * p0) / (per["views"] + k)
    per["depth"] = per["events"] / per["sessions"]
    per["conversion_bin"] = pd.qcut(per["conversion"].rank(method="first"), 3, labels=CONVERSION_BINS)
    depth_cut = per["depth"].quantile(rr["depth_split"])
    per["depth_bin"] = np.where(per["depth"] > depth_cut, "long", "short")
    return per


def estimate_targets(cfg: dict[str, Any]) -> dict[str, Any]:
    events = load_rr_events(cfg)
    profile, measure = split_halves(events, cfg)
    groups = assign_groups(profile, cfg)
    active_events = pd.concat([profile, measure])
    out: dict[str, Any] = {
        "source": cfg["retailrocket"]["events"],
        "all_visitors": {
            "visitors": int(events["user_id"].nunique()),
            "single_event_session_share": round(float((events.groupby("session_id").size() == 1).mean()), 4),
            **behavior_stats(events),
        },
        "active": {"visitors": len(groups), **behavior_stats(active_events)},
        "cells": {},
    }
    overall_spu = out["active"]["sessions_per_user"]
    for (conv, depth), g in groups.groupby(["conversion_bin", "depth_bin"], observed=True):
        stats = behavior_stats(measure[measure["user_id"].isin(g.index)])  # 측정은 뒤쪽 절반으로
        spu = active_events[active_events["user_id"].isin(g.index)].groupby("user_id")["session_id"].nunique().mean()
        out["cells"][f"{conv}/{depth}"] = {
            "visitors": len(g),
            "activity_ratio": round(float(spu) / overall_spu, 3),  # 방문 횟수 (활동 방문자 평균 대비)
            **stats,
        }
    return out


def targets_path(cfg: dict[str, Any]):
    return ROOT / cfg["paths"]["space_dir"] / "rr_targets.json"


def build_targets(cfg: dict[str, Any]) -> dict[str, Any]:
    targets = estimate_targets(cfg)
    path = targets_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(targets, ensure_ascii=False, indent=2), encoding="utf-8")
    return targets


def load_targets(cfg: dict[str, Any]) -> dict[str, Any]:
    path = targets_path(cfg)
    if not path.exists():
        print("[simulator] Retailrocket 전이 확률 추정")
        return build_targets(cfg)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    cfg = load_config("simulator")
    targets = build_targets(cfg)
    cells = pd.DataFrame(targets["cells"]).T
    print(f"[rr] 전체 방문자 {targets['all_visitors']['visitors']:,}명, 활동 방문자 {targets['active']['visitors']:,}명")
    print(f"  전체: {targets['all_visitors']}\n  활동: {targets['active']}")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(cells.to_string())
    print(f"[rr] 저장 → {targets_path(cfg).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
