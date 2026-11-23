from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import requests

from src.common.schemas import Event
from src.simulator.stream import ApiClient, run_stream


class Engine:
    user_ids = ("U00005",)

    def signup_at(self, user):
        return datetime(2025, 1, 1, tzinfo=timezone.utc)

    def respond(self, user, ids, at, session, *, impression_id):
        return SimpleNamespace(events=[Event(event_id="test-view", user_id=user, session_id=session,
                                             event_type="view", product_id=ids[0], timestamp=at + timedelta(milliseconds=1),
                                             impression_id=impression_id)], next_screen="exit")


class Client:
    def __init__(self):
        self.calls = []

    def request(self, method, path, **args):
        self.calls.append((method, path, args))
        if path == "/api/recommend":
            return {"recommendation_id": "r1", "recommendations": [{"product_id": "p1"}, {"product_id": "p2"}]}
        return {"status": "accepted"}


def test_live_stream_acknowledges_actual_list_before_actions():
    client, sleeps = Client(), []
    result = run_stream(Engine(), client, {"seed": 42, "stream": {"events_per_sec": 20, "top_n": 1},
                                         "session_flow": {"max_screens": 3}}, max_impressions=1,
                        sleep=sleeps.append, clock=lambda: datetime(2026, 11, 1, tzinfo=timezone.utc), run_id="run")
    assert result == {"impressions": 1, "events": 1}
    assert [p for _, p, _ in client.calls] == ["/api/recommend", "/api/impressions", "/api/events"]
    imp = client.calls[1][2]["json"]
    ev = client.calls[2][2]["json"]
    assert imp["product_ids"] == ["p1"] and imp["clicked_ids"] == []
    assert imp["recommendation_id"] == "r1"
    assert ev["impression_id"] == imp["impression_id"]
    assert ev["user_id"] == "stream-U00005"  # manual demo member is not mutated
    assert sleeps == [0.05]


def test_transport_retry_preserves_event_payload():
    calls = []

    class Session:
        def request(self, *args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise requests.Timeout()
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"status": "accepted"})

    client = ApiClient("http://test", retries=1, session=Session(), sleep=lambda _: None)
    assert client.request("POST", "/api/events", json={"event_id": "stable"})["status"] == "accepted"
    assert calls[0] == calls[1]


def test_invalid_rate_fails_before_sending():
    with pytest.raises(ValueError):
        run_stream(Engine(), Client(), {"seed": 42, "stream": {"events_per_sec": 0}}, max_impressions=1)
