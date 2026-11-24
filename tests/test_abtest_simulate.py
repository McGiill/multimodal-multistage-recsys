from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from src.abtest.simulate import assign_groups, simulate_experiment
from src.common.schemas import Event
from src.simulator.online import OnlineResponseEngine, ReactionBatch

START = datetime(2026, 4, 1, tzinfo=timezone.utc)


class FakeEngine:
    user_ids: ClassVar[list[str]] = [f"U{i:02d}" for i in range(20)]
    calibration = "fixture"

    def signup_at(self, user_id):
        return START - timedelta(days=1)

    def expected_visits(self, user_id, days):
        return days * 10

    def respond(self, user_id, product_ids, at, session_id, *, impression_id=None):
        if "relevant" not in product_ids:
            return ReactionBatch([], [], "exit")
        events = [Event(event_id=f"{impression_id}-{kind}", user_id=user_id, session_id=session_id,
                        timestamp=at + timedelta(milliseconds=i + 1), event_type=kind,
                        product_id="relevant", impression_id=impression_id)
                  for i, kind in enumerate(("view", "cart", "purchase"))]
        return ReactionBatch(events, ["relevant"], "exit")


class FakeProvider:
    def __init__(self, name):
        self.name, self.calls, self.exposures, self.feedbacks = name, [], [], []

    def recommend(self, user_id, top_n, at, session_clicks, recent_clicks):
        self.calls.append((user_id, session_clicks, recent_clicks))
        return [SimpleNamespace(product_id="relevant" if self.name == "multistage" else "other")]

    def record_exposure(self, items, at):
        self.exposures.append(([x.product_id for x in items], at))

    def record_click(self, items, product_id, at):
        self.feedbacks.append(([x.product_id for x in items], product_id, at))


def configuration():
    return {"seed": 42, "days": 2, "n_users_per_group": 10,
            "groups": {"control": "popularity", "treatment": "multistage"}, "test": "z_test", "alpha": .05}


def test_assignment_is_stable_unique_and_balanced():
    a = assign_groups(FakeEngine.user_ids, 8, 9)
    assert a == assign_groups(list(reversed(FakeEngine.user_ids)), 8, 9)
    assert len(a["control"]) == len(a["treatment"]) == 8
    assert not set(a["control"]) & set(a["treatment"])
    assert a != assign_groups(FakeEngine.user_ids, 8, 10)
    with pytest.raises(ValueError):
        assign_groups(FakeEngine.user_ids, 11, 9)


def test_ab_uses_returned_products_and_user_level_inference():
    providers = {}

    def factory(name):
        providers[name] = FakeProvider(name)
        return providers[name]

    result = simulate_experiment(configuration(), FakeEngine, factory, START, "fixture-model")
    a, b = result["groups"]["control"], result["groups"]["treatment"]
    assert a["user_cvr"] == 0 and b["user_cvr"] == 1
    assert b["purchase_events"] > b["purchased_users"] == 10
    assert result["comparisons"]["assigned_user_cvr"]["diff"] == 1
    assert result["comparisons"]["assigned_user_cvr"]["small_counts"] is True
    assert "item_ctr" not in result["comparisons"]
    assert result["randomization_unit"] == "user" and result["source"] == "simulation"
    treatment = providers["multistage"]
    assert len(treatment.exposures) == b["impressions"]
    assert len(treatment.feedbacks) == b["impressions"]
    assert all(session == [] for _, session, _ in treatment.calls)  # each visit starts empty
    assert any(recent for _, _, recent in treatment.calls)  # past visits remain user history
    repeated = simulate_experiment(configuration(), FakeEngine, FakeProvider, START, "fixture-model")
    assert repeated["groups"] == result["groups"]


def test_no_visits_returns_insufficient_data_not_nan():
    class NoVisits(FakeEngine):
        def expected_visits(self, user_id, days):
            return 0

    result = simulate_experiment(configuration(), NoVisits, FakeProvider, START)
    assert result["groups"]["control"]["user_cvr"] is None
    assert result["comparisons"]["user_cvr"]["status"] == "insufficient_data"
    assert result["comparisons"]["assigned_user_cvr"]["status"] == "ok"


def test_ab_action_counts_and_rewards_use_half_open_observation_window(monkeypatch):
    from src.abtest import simulate

    end = START + timedelta(days=1)
    cfg = configuration() | {"days": 1, "n_users_per_group": 2}
    providers = {}

    class BoundaryEngine(FakeEngine):
        count = 0

        def respond(self, user_id, product_ids, at, session_id, *, impression_id=None):
            self.count += 1
            # First user acts just before end; second views exactly at end and buys afterwards.
            offset = 1 if self.count == 1 else 3
            events = [Event(event_id=f"{impression_id}-{kind}", user_id=user_id, session_id=session_id,
                            timestamp=at + timedelta(milliseconds=offset + i), event_type=kind,
                            product_id=product_ids[0], impression_id=impression_id)
                      for i, kind in enumerate(("view", "purchase"))]
            return ReactionBatch(events, product_ids, "exit")

    def factory(name):
        providers[name] = FakeProvider(name)
        return providers[name]

    monkeypatch.setattr(simulate, "_schedule", lambda engine, users, *_: [
        (end - timedelta(milliseconds=3), user, 0) for user in users])
    result = simulate_experiment(cfg, BoundaryEngine, factory, START)
    for name, values in result["groups"].items():
        assert values["impressions"] == 2
        assert values["clicked_items"] == values["clicked_users"] == 1
        assert values["purchase_events"] == values["purchased_users"] == 1
        assert list(values["daily"]) == [START.date().isoformat()]
        assert values["daily"][START.date().isoformat()]["purchases"] == 1
        provider = providers[cfg["groups"][name]]
        assert len(provider.exposures) == 2 and len(provider.feedbacks) == 1
        assert provider.feedbacks[0][2] < end


def test_ab_screens_and_click_feedback_are_globally_chronological(monkeypatch):
    from src.abtest import simulate

    cfg = configuration() | {"days": 1, "n_users_per_group": 3, "max_screens_per_visit": 2}
    providers, slow_users = {}, set()

    def schedule(engine, users, *_):
        slow_users.add(users[0])
        return [(START, users[0], 0), (START + timedelta(seconds=1), users[1], 0),
                (START + timedelta(seconds=2), users[0], 1), (START + timedelta(seconds=7), users[2], 0)]

    class DelayedEngine(FakeEngine):
        def respond(self, user_id, product_ids, at, session_id, *, impression_id=None):
            delayed = user_id in slow_users and session_id.endswith("-0") and impression_id.endswith("-0")
            event = Event(event_id=f"{impression_id}-view", user_id=user_id, session_id=session_id,
                          timestamp=at + timedelta(seconds=5 if delayed else .1), event_type="view",
                          product_id=product_ids[0], impression_id=impression_id)
            return ReactionBatch([event], product_ids, "home" if delayed else "exit")

    class AuditedProvider(FakeProvider):
        def __init__(self, name):
            super().__init__(name)
            self.timeline, self.snapshots = [], []

        def recommend(self, user_id, top_n, at, session_clicks, recent_clicks):
            self.timeline.append((at, "recommend"))
            self.snapshots.append((at, user_id, len(self.feedbacks), session_clicks.copy(), recent_clicks.copy()))
            return super().recommend(user_id, top_n, at, session_clicks, recent_clicks)

        def record_exposure(self, items, at):
            self.timeline.append((at, "exposure"))
            super().record_exposure(items, at)

        def record_click(self, items, product_id, at):
            self.timeline.append((at, "click"))
            super().record_click(items, product_id, at)

    def factory(name):
        providers[name] = AuditedProvider(name)
        return providers[name]

    monkeypatch.setattr(simulate, "_schedule", schedule)
    result = simulate_experiment(cfg, DelayedEngine, factory, START)
    for name, provider in providers.items():
        times = [at for at, _ in provider.timeline]
        assert times == sorted(times)
        first, other, continuation, deferred, final = provider.snapshots
        assert first[2] == other[2] == 0  # the first user's future click is unavailable at t=1
        assert other[0] == START + timedelta(seconds=1)
        assert continuation[0] == START + timedelta(seconds=6)
        assert continuation[2] == 2 and len(continuation[3]) == 1
        assert deferred[0] > START + timedelta(seconds=6.1)  # overlapping same-user visit waits
        assert deferred[1] == first[1] and deferred[3] == [] and len(deferred[4]) == 2
        assert final[0] == START + timedelta(seconds=7) and final[2] == 4
        group = "treatment" if name == "multistage" else "control"
        assert result["groups"][group]["impressions"] == result["groups"][group]["clicked_items"] == 5


def test_online_engine_preserves_response_state_through_cache_eviction():
    from tests.test_simulator import _setup

    cfg, _, world, personas, pop = _setup(n_users=8)
    a = OnlineResponseEngine(world, pop, cfg, personas, seed=51, cache_users=1)
    b = OnlineResponseEngine(world, pop, cfg, personas, seed=51, cache_users=20)
    users = a.user_ids[:2]
    at = datetime(2026, 2, 1, tzinfo=timezone.utc)
    ids = [str(x) for x in world.product_ids[:20]]
    for day in range(5):
        for user in users:
            session = f"{user}-{day}"
            x = a.respond(user, ids, at + timedelta(days=day), session)
            y = b.respond(user, ids, at + timedelta(days=day), session)
            assert x == y
            assert set(x.clicked_ids) <= set(ids)
            assert all(e.timestamp > at + timedelta(days=day) for e in x.events)
            np.testing.assert_array_equal(a._states[user].visit.viewed, b._states[user].visit.viewed)


def test_online_engine_rejects_invalid_exposure_and_time():
    from tests.test_simulator import _setup

    cfg, _, world, personas, pop = _setup(n_users=4)
    engine = OnlineResponseEngine(world, pop, cfg, personas)
    user = engine.user_ids[0]
    with pytest.raises(ValueError, match="before signup"):
        engine.respond(user, [], engine.signup_at(user) - timedelta(seconds=1), "s")
    with pytest.raises(ValueError, match="does not exist"):
        engine.respond(user, ["missing"], START, "s")
    with pytest.raises(ValueError, match="unique"):
        engine.respond(user, [str(world.product_ids[0])] * 2, START, "s")
