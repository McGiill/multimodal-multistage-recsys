from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

from src.common.schemas import Event, Impression
from src.serving.event_log import EventLog

AT = datetime(2026, 11, 1, tzinfo=timezone.utc)


def impression(**changes):
    values = {"impression_id": "i1", "user_id": "u", "session_id": "s", "timestamp": AT,
              "source": "home", "context": "test", "product_ids": ["p1"], "clicked_ids": []}
    return Impression(**(values | changes))


def event(**changes):
    values = {"event_id": "e1", "user_id": "u", "session_id": "s", "timestamp": AT + timedelta(seconds=1),
              "event_type": "view", "product_id": "p1", "impression_id": "i1", "source": "home"}
    return Event(**(values | changes))


def registered(tmp_path):
    log = EventLog(tmp_path / "events.sqlite3")
    log.register_recommendation("r1", "u", "s", [
        {"product_id": "p1", "is_exploration": True, "arm": "pants"},
        {"product_id": "p2", "is_exploration": True, "arm": "shirts"},
    ], "v1")
    return log


def test_only_acknowledged_items_count_and_feedback_survives_restart(tmp_path):
    log = registered(tmp_path)
    assert log.feedback_counts() == {}
    assert log.append_impression(impression(), "r1")
    assert not log.append_impression(impression(), "r1")
    assert log.feedback_counts("v1") == {"pants": {"impressions": 1, "clicks": 0}}
    assert log.feedback_counts("v2") == {}
    assert log.append_event(event())
    assert not log.append_event(event())
    assert log.claim_click(event()) == "pants"
    assert log.claim_click(event()) is None
    another = event(event_id="e2")
    assert log.append_event(another)
    assert log.claim_click(another) is None
    restarted = EventLog(log.path)
    assert restarted.feedback_counts("v1") == {"pants": {"impressions": 1, "clicks": 1}}
    assert restarted.get_impression_model_version("i1") == "v1"
    assert restarted.metrics()["ctr"] == 1


def test_reject_forged_unshown_or_cross_user_attribution(tmp_path):
    log = registered(tmp_path)
    with pytest.raises(ValueError, match="not returned"):
        log.append_impression(impression(product_ids=["other"]), "r1")
    with pytest.raises(ValueError, match="owner"):
        log.append_impression(impression(user_id="other"), "r1")
    with pytest.raises(ValueError, match="clicked_ids"):
        log.append_impression(impression(clicked_ids=["p1"]), "r1")
    log.append_impression(impression(), "r1")
    for changes in ({"product_id": "p2"}, {"user_id": "other"}, {"timestamp": AT - timedelta(seconds=1)}):
        with pytest.raises(ValueError):
            log.append_event(event(**changes))
    assert log.count_since() == 0


def test_idempotent_concurrent_events_and_feature_delivery_outbox(tmp_path):
    log = registered(tmp_path)
    log.append_impression(impression(), "r1")
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(lambda _: log.append_event(event()), range(12)))
    assert sum(outcomes) == 1
    assert log.feature_pending("e1")
    log.mark_feature_delivered("e1")
    assert not EventLog(log.path).feature_pending("e1")
    with pytest.raises(ValueError, match="Conflicting"):
        log.append_event(event(source="different"))


def test_all_event_types_persist_and_snapshot_uses_watermark(tmp_path):
    log = registered(tmp_path)
    log.append_impression(impression(), "r1")
    log.append_event(event(event_id="search", event_type="search", impression_id=None, product_id=None, query="jeans"))
    end = log.high_watermark()
    log.append_event(event())
    log.append_event(event(event_id="cart", event_type="cart"))
    log.append_event(event(event_id="buy", event_type="purchase"))
    ev, im = log.export_snapshot(end)
    assert ev.event_type.tolist() == ["search"]
    assert im.clicked_ids.tolist() == [[]]
    ev, im = log.export_snapshot()
    assert set(ev.event_type) == {"search", "view", "cart", "purchase"}
    assert im.clicked_ids.tolist() == [["p1"]]
    assert log.count_since(end) == 3
    assert log.metrics()["cvr"] == 1


def test_unlinked_click_never_gets_exploration_credit(tmp_path):
    log = registered(tmp_path)
    log.append_event(event(impression_id=None))
    assert log.claim_click(event(impression_id=None)) is None
    assert log.feedback_counts() == {}
