"""사용자 단위 무작위 A/B 실험: 추천 → 같은 반응 함수 → 클릭/구매 → 통계 검정.

python -m src.abtest.simulate --users-per-group 20 --days 2 --out /tmp/ab-smoke.json

실사용자 온라인 효과가 아닌 시뮬레이터 안에서의 효과다. 개별 노출을 독립 표본으로
취급하지 않고, 실험 기간 중 한 번 이상 클릭/구매한 사용자로 검정한다.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import heapq
import json
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import count
from pathlib import Path
from typing import Any

import numpy as np

from src.abtest.stats import GroupStats, compare
from src.common.config import artifact_dir, load_config
from src.simulator.online import OnlineResponseEngine, stable_seed, utc

SIMULATION_VERSION = 2


@dataclass
class UserOutcome:
    user_id: str
    impressions: int = 0
    shown_items: int = 0
    clicked_items: int = 0
    purchases: int = 0


@dataclass
class _Visit:
    user: str
    session_id: str
    next_at: datetime
    screen: int = 0
    clicks: list[str] = field(default_factory=list)


def assign_groups(user_ids: list[str], n: int, seed: int) -> dict[str, list[str]]:
    if n < 1 or len(set(user_ids)) != len(user_ids) or len(user_ids) < 2 * n:
        raise ValueError("A/B requires unique users and at least 2 * n_users_per_group users")
    users = np.random.default_rng(seed).permutation(sorted(user_ids)).tolist()
    return {"control": users[:n], "treatment": users[n:2 * n]}


def _schedule(engine, users: list[str], start_at: datetime, days: int, seed: int) -> list[tuple]:
    """전략/반응과 독립적으로 방문을 미리 생성한다. 사용자 재방문도 같은 그룹에 남는다."""
    end_at = start_at + timedelta(days=days)
    visits = []
    for user in users:
        available = max(start_at, engine.signup_at(user))
        span = (end_at - available).total_seconds()
        if span <= 0:
            continue
        rng = np.random.default_rng(stable_seed(seed, f"schedule:{user}"))
        count = int(rng.poisson(engine.expected_visits(user, days) * span / (days * 86400)))
        offsets = np.sort(rng.uniform(0, span, count))
        for j, offset in enumerate(offsets):
            visits.append((available + timedelta(seconds=float(offset)), user, j))
    return sorted(visits)


def _summarize(outcomes: dict[str, UserOutcome]) -> dict[str, Any]:
    exposed = [x for x in outcomes.values() if x.impressions]
    def total(key):
        return sum(getattr(x, key) for x in outcomes.values())
    clicked_users = sum(x.clicked_items > 0 for x in exposed)
    purchased_users = sum(x.purchases > 0 for x in exposed)
    shown = total("shown_items")
    return {
        "assigned_users": len(outcomes), "exposed_users": len(exposed), "clicked_users": clicked_users,
        "purchased_users": purchased_users, "impressions": total("impressions"), "shown_items": shown,
        "clicked_items": total("clicked_items"), "purchase_events": total("purchases"),
        "user_ctr": clicked_users / len(exposed) if exposed else None,
        "user_cvr": purchased_users / len(exposed) if exposed else None,
        "assigned_user_cvr": purchased_users / len(outcomes) if outcomes else None,
        "item_ctr": total("clicked_items") / shown if shown else None,
        "users": [asdict(x) for x in outcomes.values()],
    }


class StrategyProvider:
    """평가 프로세스 안에서 전략별 MAB를 초기화한다. Redis/실제 데모 기록을 읽거나 쓰지 않는다."""

    def __init__(self, strategy_name: str) -> None:
        from src.recommendation import pipeline  # noqa: F401 (registry)
        from src.recommendation.reranking.stage2 import get_rerank_service
        from src.recommendation.strategies import baselines  # noqa: F401 (registry)
        from src.recommendation.strategies.base import get_strategy
        from src.serving.model_loader import get_recommendation_models

        get_rerank_service.cache_clear()
        self.name = strategy_name
        self.strategy = get_strategy(strategy_name)
        self.models = get_recommendation_models()
        self.rerank = get_rerank_service(self.models)

    def recommend(self, user_id, top_n, at, session_clicks, recent_clicks):
        kw = {"session_clicks": session_clicks, "recent_clicks": recent_clicks} if self.name == "multistage" else {}
        return self.strategy.recommend(user_id, top_n, at=at, **kw).recommendations

    def record_exposure(self, items, at):
        if self.name == "multistage":
            record = getattr(self.rerank.reranker.explorer, "record_shown", None)
            if record is not None:
                for arm in self.rerank.exploration_arms_of(items).values():
                    record(arm)

    def record_click(self, items, product_id, at):
        if self.name == "multistage":
            record = getattr(self.rerank.reranker.explorer, "record_click", None)
            arm = self.rerank.exploration_arms_of(items).get(product_id)
            if record is not None and arm is not None:
                record(arm)


def _simulate_arm(engine, provider, users, cfg, start_at, fingerprint, name):
    """노출과 행동을 같은 시간 큐에서 처리하여 미래 클릭이 다음 추천으로 새지 않게 한다."""
    days, seed = int(cfg["days"]), int(cfg["seed"])
    top_n, max_screens = int(cfg.get("top_n", 10)), int(cfg.get("max_screens_per_visit", 3))
    end_at = start_at + timedelta(days=days)
    outcomes = {u: UserOutcome(u) for u in users}
    recent = {u: deque(maxlen=int(cfg.get("history_len", 20))) for u in users}
    daily: dict[str, dict[str, int]] = {}
    active: dict[str, _Visit] = {}
    queue = []
    sequence = count()
    # 같은 시각에서는 발생한 행동을 먼저 반영하고, 이전 방문을 종료한 뒤 새 방문을 시작한다.
    priority = {"action": 0, "release": 1, "screen": 2, "visit": 3}

    def enqueue(at, kind, payload):
        if start_at <= at < end_at:
            heapq.heappush(queue, (at, priority[kind], next(sequence), kind, payload))

    def day_counts(at):
        return daily.setdefault(at.date().isoformat(), {"impressions": 0, "shown_items": 0,
                                                       "clicked_items": 0, "purchases": 0})

    for at, user, visit in _schedule(engine, users, start_at, days, seed):
        enqueue(at, "visit", (user, visit))
    while queue:
        at, _, _, kind, payload = heapq.heappop(queue)
        if kind == "visit":
            user, number = payload
            if user in active:
                # 이전 방문의 반응/다음 화면이 끝나기 전에는 같은 엔진 사용자 상태를 바꾸지 않는다.
                enqueue(active[user].next_at + timedelta(microseconds=1), "visit", payload)
                continue
            visit = _Visit(user, f"ab-{fingerprint}-{name}-{user}-{number}", at)
            active[user] = visit
            enqueue(at, "screen", visit)
            continue
        if kind == "release":
            del active[payload.user]
            continue
        if kind == "action":
            visit, items, event, clicked = payload
            row, day = outcomes[visit.user], day_counts(at)
            if event.event_type == "view":
                if event.product_id not in clicked:
                    clicked.add(event.product_id)
                    row.clicked_items += 1
                    day["clicked_items"] += 1
                    provider.record_click(items, event.product_id, at)
                visit.clicks.append(event.product_id)
                recent[visit.user].append(event.product_id)
            elif event.event_type == "purchase":
                row.purchases += 1
                day["purchases"] += 1
            continue

        visit = payload
        items = provider.recommend(visit.user, top_n, at, visit.clicks.copy(), list(recent[visit.user]))
        pids = [r.product_id for r in items]
        if len(set(pids)) != len(pids):
            raise ValueError("Recommendation returned duplicate displayed products")
        if not pids:
            del active[visit.user]
            continue
        impression_id = f"{visit.session_id}-{visit.screen}"
        reaction = engine.respond(visit.user, pids, at, visit.session_id, impression_id=impression_id)
        events = sorted(reaction.events, key=lambda e: e.timestamp)
        viewed = {e.product_id for e in events if e.event_type == "view"}
        if not set(reaction.clicked_ids) <= set(pids) or viewed != set(reaction.clicked_ids):
            raise ValueError("Response clicks must match displayed view events")
        if any(e.timestamp < at or e.user_id != visit.user or e.session_id != visit.session_id for e in events):
            raise ValueError("Response actions must follow their exposure and share its user/session")
        provider.record_exposure(items, at)
        row, day = outcomes[visit.user], day_counts(at)
        row.impressions += 1
        row.shown_items += len(pids)
        day["impressions"] += 1
        day["shown_items"] += len(pids)
        clicked: set[str] = set()
        for event in events:
            # end_at과 그 이후의 클릭/구매는 집계, 추천 이력, MAB 보상 모두에서 제외한다.
            enqueue(event.timestamp, "action", (visit, items, event, clicked))
        last_at = events[-1].timestamp if events else at
        visit.screen += 1
        if reaction.next_screen == "home" and visit.screen < max_screens:
            visit.next_at = last_at + timedelta(seconds=1)
            enqueue(visit.next_at, "screen", visit)
        else:
            visit.next_at = last_at + timedelta(microseconds=1)
            enqueue(visit.next_at, "release", visit)
    return {**_summarize(outcomes), "daily": daily}


def simulate_experiment(cfg: dict[str, Any], engine_factory: Callable, provider_factory: Callable,
                        start_at: datetime, model_version: str = "unknown") -> dict[str, Any]:
    """주입 가능한 순수 실험 실행부. 외부 서비스 요청/저장 없이 결과를 반환한다."""
    seed, days, n = int(cfg["seed"]), int(cfg["days"]), int(cfg["n_users_per_group"])
    top_n, max_screens = int(cfg.get("top_n", 10)), int(cfg.get("max_screens_per_visit", 3))
    if days < 1 or top_n < 1 or max_screens < 1:
        raise ValueError("days, top_n and max_screens_per_visit must be positive")
    if set(cfg["groups"]) != {"control", "treatment"}:
        raise ValueError("Exactly control and treatment strategies are required")
    start_at = utc(start_at)
    engines = {name: engine_factory() for name in ("control", "treatment")}
    population = engines["control"].user_ids
    if population != engines["treatment"].user_ids:
        raise ValueError("A/B response engines must share the same population")
    groups = assign_groups(population, n, seed)
    fingerprint = hashlib.sha256(json.dumps({"config": cfg, "start_at": start_at.isoformat(),
                                             "model_version": model_version, "simulation_version": SIMULATION_VERSION},
                                            sort_keys=True).encode()).hexdigest()[:16]
    result: dict[str, Any] = {
        "schema_version": 2, "simulation_version": SIMULATION_VERSION,
        "source": "simulation", "experiment_id": fingerprint,
        "measured_at": datetime.now(timezone.utc).isoformat(), "config": copy.deepcopy(cfg),
        "model_version": model_version, "start_at": start_at.isoformat(),
        "end_at": (start_at + timedelta(days=days)).isoformat(),
        "randomization_unit": "user", "primary_metric": "assigned_user_cvr",
        "definitions": {
            "assigned_user_cvr": "users with >=1 simulated purchase / all randomly assigned users (ITT)",
            "user_cvr": "users with >=1 simulated purchase / users exposed to >=1 recommendation list",
            "user_ctr": "users with >=1 click / users exposed to >=1 recommendation list",
            "item_ctr": "clicked displayed items / displayed items; descriptive only, repeated observations per user",
            "ci": "95% Wald interval for treatment - control rate (or 1-alpha configured level)",
            "scope": "recommendation surfaces only; response next_screen search/browse ends this controlled visit",
            "observation_window": "[start_at, end_at); clicks and purchases counted at their action timestamps",
            "event_order": "event-time queue; exposure at display time, click feedback at view time; overlapping user visits wait",
        },
        "history_source": "frozen model history + simulated past views; Redis never used",
        "response_calibration": getattr(engines["control"], "calibration", "provided"),
        "groups": {}, "comparisons": {},
    }
    for name, strategy in cfg["groups"].items():
        engine, provider = engines[name], provider_factory(strategy)
        result["groups"][name] = {"strategy": strategy,
                                  **_simulate_arm(engine, provider, groups[name], cfg, start_at, fingerprint, name)}
    for metric, numerator, denominator in (("assigned_user_cvr", "purchased_users", "assigned_users"),
                                           ("user_cvr", "purchased_users", "exposed_users"),
                                           ("user_ctr", "clicked_users", "exposed_users")):
        a, b = result["groups"]["control"], result["groups"]["treatment"]
        if not a[denominator] or not b[denominator]:
            result["comparisons"][metric] = {"status": "insufficient_data", "reason": "No exposed users in an arm"}
            continue
        test = compare(GroupStats(a[numerator], a[denominator]), GroupStats(b[numerator], b[denominator]),
                       test=cfg.get("test", "z_test"), alpha=float(cfg.get("alpha", .05))).to_dict()
        test["status"] = "ok"
        test["small_counts"] = any(min(g[numerator], g[denominator] - g[numerator]) < 5 for g in (a, b))
        result["comparisons"][metric] = test
    result["limitations"] = ["Synthetic response model; not evidence of real customer conversion uplift.",
                              "Exposed-user comparisons are secondary; assigned-user CVR preserves randomization.",
                              "Wald CI and Z test are approximate; small event counts require more simulated users."]
    return result


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    from src.serving.model_loader import get_recommendation_models

    models = get_recommendation_models()
    seed = int(cfg["seed"])
    source = OnlineResponseEngine.from_config(seed=seed)
    start = utc(cfg["start_at"]) if cfg.get("start_at") else max(
        source.end_at, datetime.fromtimestamp(models.store.metadata["selection_cutoff"], tz=timezone.utc)
        + timedelta(seconds=1))
    if start.timestamp() < models.store.metadata["selection_cutoff"]:
        raise ValueError("A/B start_at must follow model selection cutoff")

    def engine_factory():
        engine = OnlineResponseEngine(source.world, source.population, source.cfg, source.personas, seed)
        engine.calibration = source.calibration
        return engine

    return simulate_experiment(cfg, engine_factory, StrategyProvider, start, models.cfg.get("version", "unknown"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--users-per-group", type=int)
    p.add_argument("--days", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--start-at")
    p.add_argument("--out", type=Path, help="New JSON artifact; existing files are never overwritten")
    args = p.parse_args()
    cfg = copy.deepcopy(load_config("abtest"))
    for key, value in (("n_users_per_group", args.users_per_group), ("days", args.days),
                       ("seed", args.seed), ("start_at", args.start_at)):
        if value is not None:
            cfg[key] = value
    out = args.out or artifact_dir() / "metrics" / f"abtest_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json"
    if out.exists():
        raise FileExistsError(out)
    result = run(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"[abtest] simulation saved: {out}")
    for name, g in result["groups"].items():
        print(f"  {name}: exposed {g['exposed_users']}/{g['assigned_users']}, "
              f"click users {g['clicked_users']}, purchase users {g['purchased_users']}")


if __name__ == "__main__":
    main()
