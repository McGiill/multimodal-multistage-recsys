"""시간 기반 8:1:1 분할: 구간 날짜, 방문 단위 배정, 노출 로그 분할."""

import pandas as pd
import pyarrow as pa
import pytest

from src.evaluation.split import boundaries, split_logs, summarize

CFG = {"period": {"start_date": "2026-01-01", "days": 90}}


def test_boundaries_are_72_9_9_days() -> None:
    b = boundaries(CFG)
    assert [(b[s][1] - b[s][0]).days for s in ("train", "valid", "test")] == [72, 9, 9]
    assert b["train"][0] == pd.Timestamp("2026-01-01")
    assert b["valid"][0] == pd.Timestamp("2026-03-14")
    assert b["test"][0] == pd.Timestamp("2026-03-23")
    assert b["test"][1] == pd.Timestamp("2026-04-01")  # 마지막 날 03-31 포함


def _logs():
    t = pd.Timestamp
    events = pd.DataFrame([
        ("u1", "S1", t("2026-01-05 10:00"), "view", "p1"),
        ("u1", "S2", t("2026-03-13 23:50"), "view", "p1"),   # 방문이 train 마지막 날 자정을 걸침
        ("u1", "S2", t("2026-03-14 00:10"), "cart", "p1"),
        ("u2", "S3", t("2026-03-20 12:00"), "search", None),
        ("u3", "S4", t("2026-03-30 09:00"), "view", "p9"),   # 로그 중간에 가입한 고객, 신상품
        ("u1", "S6", t("2026-04-01 00:00:05"), "view", "p1"),  # 마지막 날 자정 직전에 시작해 첫 기록이 다음 날
    ], columns=["user_id", "session_id", "timestamp", "event_type", "product_id"])
    impressions = pa.table({
        "session_id": ["S1", "S2", "S5"],                     # S5: 노출만 있고 이벤트 없는 방문
        "timestamp": pa.array([t("2026-01-05 09:59"), t("2026-03-13 23:49"), t("2026-03-25 08:00")]),
        "product_ids": [["p1", "p2"], ["p1"], ["p3"]],
    })
    return events, impressions


def test_sessions_stay_whole_and_impressions_follow() -> None:
    events, impressions = _logs()
    parts = split_logs(events, impressions, boundaries(CFG))
    train = parts["train"]["events"]
    assert set(train["session_id"]) == {"S1", "S2"}  # 자정을 걸친 S2도 시작 날짜 기준으로 train
    assert list(parts["valid"]["events"]["session_id"]) == ["S3"]
    assert list(parts["test"]["events"]["session_id"]) == ["S4", "S6"]
    assert parts["train"]["impressions"].column("session_id").to_pylist() == ["S1", "S2"]
    assert parts["test"]["impressions"].column("session_id").to_pylist() == ["S5"]
    assert parts["test"]["impressions"].column("product_ids").to_pylist() == [["p3"]]  # 목록 컬럼 유지
    assert sum(len(p["events"]) for p in parts.values()) == len(events)

    meta = summarize(parts, boundaries(CFG), pd.DataFrame({"user_id": ["u1", "u2", "u3"]}))
    assert meta["splits"]["test"]["users_without_train_history"] == 1
    assert meta["splits"]["test"]["items_unseen_in_train"] == 1


def test_session_before_period_is_rejected() -> None:
    events, impressions = _logs()
    events.loc[4, "timestamp"] = pd.Timestamp("2025-12-31")
    with pytest.raises(ValueError):
        split_logs(events, impressions, boundaries(CFG))
