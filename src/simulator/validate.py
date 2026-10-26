"""생성된 시뮬레이션 데이터 검증·요약: python -m src.simulator.validate

1. 명세서 기준: 고객 1만 명 이상, 행동 로그 100만 건 이상, 이벤트 4종, 페르소나 6종
2. 페르소나별 차이: 목록 클릭률, 전환율, 평균 구매가, 선호 카테고리 비율, 브랜드 집중도, 상대 가격, 신상품 비율
   + Retailrocket 목표(페르소나에 대응한 칸) vs 시뮬레이터 결과
3. 학습 가능성 점검: 공식 분할(src/evaluation/split.py, 방문 단위 8:1:1)의 test 기간 HitRate@50을 간단한 기준 모델로 계산
   (전체 인기순 / 고객층 인기순 / 자주 본 카테고리 인기순 / 최근 본 상품과 함께 본 상품)
   개인화 기준 모델이 전체 인기순보다 높으면 추천 모델이 배울 개인화 신호가 있다는 뜻

결과: artifacts/metrics/simulator_report.json + 콘솔 표
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import ROOT, load_config
from src.evaluation.split import boundaries, session_labels
from src.simulator.rr_calibration import behavior_stats

RR_KEYS = ("view_to_cart", "view_exit", "cart_purchase_same_visit", "cart_purchase_later")


def _persona_table(cfg: dict[str, Any], catalog: pd.DataFrame, hidden: pd.DataFrame, events: pd.DataFrame,
                   impressions: pd.DataFrame) -> pd.DataFrame:
    cat = catalog.set_index("product_id")
    l3_median = catalog.groupby("category_l3")["price"].transform("median")
    rel_price = pd.Series(np.log(catalog["price"].to_numpy() / l3_median.to_numpy()), index=catalog["product_id"])
    persona = hidden.set_index("user_id")["persona"]
    ev = events.assign(persona=events["user_id"].map(persona))
    views = ev[ev["event_type"] == "view"].copy()
    views["l2"] = views["product_id"].map(cat["category_l2"])
    views["brand"] = views["product_id"].map(cat["brand"])
    pref = hidden.set_index("user_id")["preferred_l2"].map(set)
    views["in_pref"] = [l2 in pref[u] for u, l2 in zip(views["user_id"], views["l2"])]

    start = pd.Timestamp(cfg["period"]["start_date"])
    last_week = start + pd.Timedelta(days=cfg["period"]["days"] - cfg["lists"]["new_days"])
    created = pd.to_datetime(views["product_id"].map(cat["created_at"]))
    recent_views = views[views["timestamp"] >= last_week]
    is_new = (created.loc[recent_views.index] >= last_week)

    brand_conc = (
        views.groupby(["user_id", "brand"]).size().groupby(level=0).agg(lambda s: s.max() / s.sum() if s.sum() >= 10 else np.nan)
    )
    imp = impressions.assign(persona=impressions["user_id"].map(persona), clicked_any=impressions["clicked_ids"].map(len) > 0)
    purchases = ev[ev["event_type"] == "purchase"]

    rows = []
    for name, users in hidden.groupby("persona"):
        e = ev[ev["persona"] == name]
        n = {t: int((e["event_type"] == t).sum()) for t in ("search", "view", "cart", "purchase")}
        v = views[views["persona"] == name]
        im = imp[imp["persona"] == name]
        rows.append({
            "persona": name,
            "users": len(users),
            "event_share": round(len(e) / max(len(ev), 1), 3),  # 전체 로그 중 이 페르소나 비중
            "events_per_user": round(len(e) / len(users), 1),
            "sessions_per_user": round(e["session_id"].nunique() / len(users), 1),
            "search_share": round(n["search"] / max(len(e), 1), 3),
            "list_click_rate": round(float(im["clicked_any"].mean()), 3),
            "view_to_cart": round(n["cart"] / max(n["view"], 1), 4),
            "cart_to_purchase": round(n["purchase"] / max(n["cart"], 1), 3),
            "avg_purchase_price": round(float(purchases[purchases["persona"] == name]["product_id"].map(cat["price"]).mean()), 2),
            "view_rel_price": round(float(v["product_id"].map(rel_price).mean()), 3),
            "pref_category_share": round(float(v["in_pref"].mean()), 3),
            "brand_concentration": round(float(brand_conc.reindex(users["user_id"]).mean()), 3),
            "new_item_share_last_week": round(float(is_new[recent_views["persona"] == name].mean()), 3),
            "top_categories": ", ".join(f"{k} {s:.0%}" for k, s in v["l2"].value_counts(normalize=True).head(3).items()),
        })
    return pd.DataFrame(rows).set_index("persona")


def _hit_rate(cfg: dict[str, Any], catalog: pd.DataFrame, events: pd.DataFrame, k: int = 50) -> dict[str, float]:
    split = events["session_id"].map(session_labels(events["session_id"], events["timestamp"], boundaries(cfg)))
    inter = events[events["event_type"].isin(["view", "cart", "purchase"])]
    train = inter[split[inter.index] == "train"]
    test = inter[split[inter.index] == "test"]
    users = sorted(set(train["user_id"]) & set(test["user_id"]))
    truth = test.groupby("user_id")["product_id"].agg(set)

    cat = catalog.set_index("product_id")
    counts = train["product_id"].value_counts()
    ranked = counts.index.to_numpy()
    l1_of = cat["category_l1"]
    l3_of = cat["category_l3"]
    by_l1 = {g: s.index.to_numpy() for g, s in counts.groupby(l1_of.reindex(counts.index).to_numpy())}
    by_l3 = {g: s.index.to_numpy() for g, s in counts.groupby(l3_of.reindex(counts.index).to_numpy())}
    tr = train.assign(l1=train["product_id"].map(l1_of), l3=train["product_id"].map(l3_of))
    user_l1 = tr.groupby("user_id")["l1"].agg(lambda s: s.value_counts().index[0])
    user_l3 = tr.groupby("user_id")["l3"].agg(lambda s: list(s.value_counts().index[:5]))

    # 같은 방문에서 함께 본 상품 (train 로그만 사용) → 최근 본 상품과 자주 같이 본 상품을 추천
    views = train[train["event_type"] == "view"][["session_id", "product_id"]].drop_duplicates()
    pairs = views.merge(views, on="session_id")
    pairs = pairs[pairs["product_id_x"] != pairs["product_id_y"]]
    covisit = pairs.groupby(["product_id_x", "product_id_y"]).size().rename("n").reset_index()
    covisit = covisit.sort_values(["product_id_x", "n"], ascending=[True, False]).groupby("product_id_x").head(20)
    neighbors = covisit.groupby("product_id_x")["product_id_y"].agg(list)
    recent = train.sort_values("timestamp").groupby("user_id")["product_id"].agg(lambda s: list(dict.fromkeys(s[::-1]))[:5])

    hits = {"popular": 0, "segment_popular": 0, "category_popular": 0, "covisit": 0}
    top = set(ranked[:k])
    for u in users:
        t = truth[u]
        hits["popular"] += bool(t & top)
        hits["segment_popular"] += bool(t & set(by_l1.get(user_l1[u], ranked)[:k]))
        cats = user_l3[u]
        per = int(np.ceil(k / len(cats)))
        recs = [p for c in cats for p in by_l3.get(c, [])[:per]][:k]
        hits["category_popular"] += bool(t & set(recs))
        co = list(dict.fromkeys(p for item in recent[u] for p in neighbors.get(item, [])[:10]))
        co += [p for p in ranked if p not in set(co)][: max(0, k - len(co))]
        hits["covisit"] += bool(t & set(co[:k]))
    n = max(len(users), 1)
    return {"n_test_users": len(users), **{f"hit_rate@{k}_{m}": round(h / n, 4) for m, h in hits.items()}}


def _rr_fit(hidden: pd.DataFrame, events: pd.DataFrame, rr_targets: dict[str, Any]) -> list[dict[str, Any]]:
    """페르소나별 Retailrocket 목표(해당 칸) vs 시뮬레이터 결과 — 같은 함수(behavior_stats)로 계산.

    carts: 구매 비율 두 개의 바탕이 되는 담기 수. 적을수록 우연한 차이가 크다 (나중 구매처럼 비율이 아주 작으면 특히).
    """
    persona = events["user_id"].map(hidden.set_index("user_id")["persona"])
    rows = []
    for name, goal in rr_targets.items():
        ev = events[persona == name]
        got = behavior_stats(ev)
        rows.append({"persona": name, "rr_group": goal["rr_group"],
                     **{k: f"{goal[k]:.3f} / {got[k]:.3f}" for k in RR_KEYS},
                     "carts": int((ev["event_type"] == "cart").sum())})
    return rows


def build_report(cfg: dict[str, Any], catalog: pd.DataFrame, users: pd.DataFrame, hidden: pd.DataFrame,
                 events: pd.DataFrame, impressions: pd.DataFrame,
                 rr_targets: dict[str, Any] | None = None) -> dict[str, Any]:
    by_type = events["event_type"].value_counts().to_dict()
    per_user = events.groupby("user_id").size().reindex(users["user_id"], fill_value=0)
    viewed = events.loc[events["event_type"] == "view", "product_id"].nunique()
    spec = {
        "users >= 10,000": len(users) >= 10_000,
        f"events >= {cfg['scale']['min_events']:,}": len(events) >= cfg["scale"]["min_events"],
        "event types = search/view/cart/purchase": set(by_type) == {"search", "view", "cart", "purchase"},
        "personas = 6": hidden["persona"].nunique() == 6,
    }
    return {
        "totals": {
            "users": len(users),
            "sessions": int(events["session_id"].nunique()),
            "events": len(events),
            "events_by_type": {k: int(v) for k, v in by_type.items()},
            "impressions": len(impressions),
            "catalog_items_viewed": f"{viewed:,} / {len(catalog):,} ({viewed / len(catalog):.1%})",
            "users_under_5_events": int((per_user < 5).sum()),
            "period": f"{events['timestamp'].min()} ~ {events['timestamp'].max()}",
        },
        "spec": spec,
        "personas": _persona_table(cfg, catalog, hidden, events, impressions).reset_index().to_dict("records"),
        "retailrocket_fit": _rr_fit(hidden, events, rr_targets) if rr_targets else [],
        "learnability": _hit_rate(cfg, catalog, events),
    }


def print_report(report: dict[str, Any]) -> None:
    t = report["totals"]
    print(f"\n[simulator] 고객 {t['users']:,} / 방문 {t['sessions']:,} / 이벤트 {t['events']:,} / 노출 {t['impressions']:,}")
    print(f"  이벤트 종류: {t['events_by_type']}")
    print(f"  조회된 상품: {t['catalog_items_viewed']}, 이벤트 5건 미만 고객: {t['users_under_5_events']:,}")
    for name, ok in report["spec"].items():
        print(f"  [{'OK' if ok else '미달'}] {name}")
    table = pd.DataFrame(report["personas"]).set_index("persona")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n" + table.to_string())
        if report["retailrocket_fit"]:
            print("\n  Retailrocket 목표 / 시뮬레이터 결과 (search 이벤트 제외, 같은 계산 방식)")
            print(pd.DataFrame(report["retailrocket_fit"]).set_index("persona").to_string())
    print(f"\n  학습 가능성 점검 (test 기간): {report['learnability']}")


def main() -> None:
    cfg = load_config("simulator")
    out = ROOT / cfg["paths"]["output_dir"]
    catalog = pd.read_parquet(ROOT / cfg["paths"]["catalog"])
    calibration = ROOT / cfg["paths"]["space_dir"] / "persona_calibration.json"
    rr_targets = json.loads(calibration.read_text(encoding="utf-8"))["targets"] if calibration.exists() else None
    report = build_report(
        cfg, catalog, pd.read_parquet(out / "users.parquet"), pd.read_parquet(out / "users_hidden.parquet"),
        pd.read_parquet(out / "events.parquet"), pd.read_parquet(out / "impressions.parquet"), rr_targets,
    )
    save_report(cfg, report)
    print_report(report)


def save_report(cfg: dict[str, Any], report: dict[str, Any]) -> None:
    path = ROOT / cfg["paths"]["report"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
