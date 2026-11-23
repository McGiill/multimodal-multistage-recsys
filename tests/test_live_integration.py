"""Opt-in integration: real Redis/model/persona engine and in-process FastAPI HTTP.

Run only against a fresh disposable Redis container with RUN_LIVE_INTEGRATION=1
and LIVE_REDIS_HOST set. Existing demo users and durable project logs are never
written: pytest supplies a temporary journal and all users have integration-* IDs.
"""

import copy
import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

pytestmark = pytest.mark.skipif(os.getenv("RUN_LIVE_INTEGRATION") != "1",
                                reason="requires an explicitly isolated Redis and existing model/data")


def test_stream_api_redis_durable_feedback_and_delivery_retries(monkeypatch):
    import redis
    from fastapi.testclient import TestClient

    from src.common.config import load_config
    from src.common.schemas import Event, Impression
    from src.recommendation.reranking.stage2 import get_rerank_service
    from src.serving.event_log import get_event_log
    from src.serving.feature_store import RedisFeatureStore, get_feature_store
    from src.serving.main import app
    from src.serving.model_loader import get_recommendation_models, load_models
    from src.simulator.online import OnlineResponseEngine
    from src.simulator.stream import run_stream

    host = os.getenv("LIVE_REDIS_HOST")
    assert host, "LIVE_REDIS_HOST must point to a fresh disposable Redis instance"
    monkeypatch.setenv("REDIS_HOST", host)
    monkeypatch.setenv("REDIS_PORT", os.getenv("LIVE_REDIS_PORT", "6379"))
    monkeypatch.setenv("PRELOAD_MODELS", "0")
    get_feature_store.cache_clear()
    features = get_feature_store()
    assert isinstance(features, RedisFeatureStore)
    assert features.client.dbsize() == 0, "Refusing to test against a nonempty Redis database"
    try:
        models = get_recommendation_models()
        engine = OnlineResponseEngine.from_config(seed=42)
    except FileNotFoundError as exc:
        pytest.skip(f"Existing model/simulator artifacts required: {exc}")
    get_rerank_service.cache_clear()
    run_id = uuid4().hex
    prefix = f"integration-{run_id}-"
    user = engine.user_ids[0]
    public_user = prefix + user
    log = get_event_log()
    cfg = copy.deepcopy(load_config("simulator"))
    cfg["stream"].update(user_prefix=prefix, top_n=10, events_per_sec=1000)
    at = max(datetime.now(timezone.utc), engine.end_at)
    calls = []

    with TestClient(app, raise_server_exceptions=False) as client:
        class InProcessApi:
            def request(self, method, path, **kwargs):
                response = client.request(method, path, **kwargs)
                assert response.status_code == 200, (path, response.status_code, response.text)
                result = response.json()
                calls.append((path, kwargs, result))
                return result

        result = run_stream(engine, InProcessApi(), cfg, max_impressions=30, users=[user],
                            sleep=lambda _: None, clock=lambda: at, run_id=run_id)
        assert result["impressions"] == 30 and result["events"] > 0
        events, impressions = log.export_snapshot()
        assert len(events) == result["events"] and len(impressions) == 30
        assert set(events.user_id) == set(impressions.user_id) == {public_user}
        assert all(len(ids) == 10 for ids in impressions.product_ids)
        views = events[events.event_type.eq("view")]
        assert features.get_recent_views(public_user) == views.product_id.tolist()[-features.recent_views_len:]
        for session, rows in views.groupby("session_id", sort=False):
            assert features.get_session_clicks(session) == rows.product_id.tolist()
            assert 0 < features.client.ttl(f"session_clicks:{session}") <= features.session_ttl_sec
            assert 0 < features.client.ttl(f"feature_events:{session}") <= features.session_ttl_sec
        assert features.get_recent_views("U00005") == []
        assert all(key.startswith((f"recent_views:{prefix}", "session_clicks:stream-", "feature_events:stream-"))
                   for key in features.client.scan_iter())
        event_payload = next(kwargs["json"] for path, kwargs, _ in calls if path == "/api/events")
        before = features.get_recent_views(public_user)
        assert client.post("/api/events", json=event_payload).json() == {"status": "duplicate"}
        assert features.get_recent_views(public_user) == before
        assert log.count_since() == result["events"]

        # Returned recommendations alone do not count; an acknowledged visible MAB item does.
        before_feedback = log.feedback_counts(models.folder.name)
        session = f"integration-{run_id}-feedback"
        response = client.get("/api/recommend", params={"user_id": public_user, "session_id": session,
                                                       "at": (at + timedelta(seconds=1)).isoformat()}).json()
        assert response["model_version"] == models.folder.name
        assert log.feedback_counts(models.folder.name) == before_feedback
        item = next(x for x in response["recommendations"] if x["reason"] == "mab_exploration")
        imp = Impression(impression_id=f"integration-{run_id}-shown", user_id=public_user, session_id=session,
                         timestamp=at + timedelta(seconds=1), source="home", context="isolated integration",
                         product_ids=[item["product_id"]], clicked_ids=[])
        payload = imp.model_dump(mode="json") | {"recommendation_id": response["recommendation_id"]}
        assert client.post("/api/impressions", json=payload).json()["status"] == "accepted"
        assert client.post("/api/impressions", json=payload).json()["status"] == "duplicate"
        arm = log.get_impression_arms(imp.impression_id)[item["product_id"]]
        previous = before_feedback.get(arm, {"impressions": 0, "clicks": 0})
        assert log.feedback_counts(models.folder.name)[arm]["impressions"] == previous["impressions"] + 1
        assert log.get_impression_model_version(imp.impression_id) == models.folder.name

        event = Event(event_id=f"integration-{run_id}-click", user_id=public_user, session_id=session,
                      timestamp=at + timedelta(seconds=2), event_type="view", product_id=item["product_id"],
                      impression_id=imp.impression_id, source="home")
        # Crash after durable insert and before Redis delivery: API retry completes its outbox.
        with monkeypatch.context() as patch:
            def fail_delivery(_):
                raise redis.ConnectionError("isolated simulated outage")
            patch.setattr(features, "push_event", fail_delivery)
            assert client.post("/api/events", json=event.model_dump(mode="json")).status_code == 500
        assert log.feature_pending(event.event_id)
        assert features.get_session_clicks(session) == []
        assert client.post("/api/events", json=event.model_dump(mode="json")).json()["status"] == "duplicate"
        assert not log.feature_pending(event.event_id)
        assert features.get_session_clicks(session) == [item["product_id"]]
        assert log.feedback_counts(models.folder.name)[arm]["clicks"] == previous["clicks"] + 1

        # Crash after Redis success: retrying the same ID must not append a second feature.
        event2 = event.model_copy(update={"event_id": event.event_id + "-retry",
                                          "timestamp": at + timedelta(seconds=3)})
        with monkeypatch.context() as patch:
            def fail_checkpoint(_):
                raise RuntimeError("isolated crash before feature-delivery acknowledgement")
            patch.setattr(log, "mark_feature_delivered", fail_checkpoint)
            assert client.post("/api/events", json=event2.model_dump(mode="json")).status_code == 500
        assert log.feature_pending(event2.event_id)
        assert features.get_session_click_count(session) == 2
        assert client.post("/api/events", json=event2.model_dump(mode="json")).json()["status"] == "duplicate"
        assert not log.feature_pending(event2.event_id)
        assert features.get_session_click_count(session) == 2
        assert log.feedback_counts(models.folder.name)[arm]["clicks"] == previous["clicks"] + 1
        assert log.feedback_counts("unrelated-model") == {}
    print({"stream": result, "model_version": models.folder.name, "transport": "FastAPI TestClient",
           "redis": "disposable real server", "durable_journal": "pytest temporary directory"})
    get_rerank_service.cache_clear()
    load_models.cache_clear()
