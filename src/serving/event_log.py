"""Durable, idempotent event and displayed-impression journal.

Redis is a serving cache; this SQLite journal is the source for experiment metrics
and continuous training. A returned recommendation is not an impression. Only a
client acknowledgement of displayed products creates an impression.
"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.common.config import artifact_dir
from src.common.schemas import Event, Impression


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _payload(model: Any) -> dict[str, Any]:
    return model.model_dump(mode="json") if hasattr(model, "model_dump") else dict(model)


class EventLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS recommendations (
                    request_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                    session_id TEXT, model_version TEXT NOT NULL, items TEXT NOT NULL,
                    created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS impressions (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, impression_id TEXT UNIQUE NOT NULL,
                    request_id TEXT, user_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT UNIQUE NOT NULL,
                    user_id TEXT NOT NULL, session_id TEXT NOT NULL, event_type TEXT NOT NULL,
                    product_id TEXT, impression_id TEXT, timestamp TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS events_impression ON events(impression_id, event_type);
                CREATE TABLE IF NOT EXISTS exploration_clicks (
                    impression_id TEXT NOT NULL, product_id TEXT NOT NULL,
                    event_id TEXT NOT NULL, arm TEXT NOT NULL,
                    PRIMARY KEY(impression_id, product_id));
                CREATE TABLE IF NOT EXISTS feature_deliveries (
                    event_id TEXT PRIMARY KEY, delivered_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS bandit_stats (
                    model_version TEXT NOT NULL, arm TEXT NOT NULL,
                    shown INTEGER NOT NULL DEFAULT 0, clicks INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(model_version, arm));
                CREATE TABLE IF NOT EXISTS journal_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)
            con.execute("BEGIN IMMEDIATE")
            if con.execute("SELECT 1 FROM journal_metadata WHERE key='bandit_materialized'").fetchone() is None:
                rows = con.execute("SELECT i.impression_id,i.payload,r.items,r.model_version FROM impressions i "
                                   "JOIN recommendations r ON i.request_id=r.request_id").fetchall()
                clicks = {(row[0], row[1]) for row in con.execute(
                    "SELECT impression_id,product_id FROM exploration_clicks")}
                for row in rows:
                    shown = set(json.loads(row["payload"])["product_ids"])
                    for item in json.loads(row["items"]):
                        if item["product_id"] in shown and item.get("is_exploration") and item.get("arm") is not None:
                            self._bandit_add(con, row["model_version"], str(item["arm"]), 1,
                                             int((row["impression_id"], item["product_id"]) in clicks))
                con.execute("INSERT INTO journal_metadata VALUES ('bandit_materialized','1')")

    @staticmethod
    def _bandit_add(con: sqlite3.Connection, version: str, arm: str, shown: int, clicks: int) -> None:
        con.execute("INSERT INTO bandit_stats VALUES (?,?,?,?) ON CONFLICT(model_version,arm) DO UPDATE "
                    "SET shown=shown+excluded.shown,clicks=clicks+excluded.clicks", (version, arm, shown, clicks))

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA busy_timeout=30000")
            with con:
                yield con
        finally:
            con.close()

    def register_recommendation(self, request_id: str, user_id: str, session_id: str | None,
                                items: list[Any], model_version: str) -> bool:
        rows = [_payload(item) for item in items]
        ids = [item["product_id"] for item in rows]
        if not request_id or len(ids) != len(set(ids)):
            raise ValueError("Recommendation IDs must be unique")
        encoded = _json(rows)
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            old = con.execute("SELECT * FROM recommendations WHERE request_id=?", (request_id,)).fetchone()
            if old is not None:
                if (old["user_id"], old["session_id"], old["items"], old["model_version"]) != (
                    user_id, session_id, encoded, model_version
                ):
                    raise ValueError("Conflicting recommendation_id")
                return False
            con.execute("INSERT INTO recommendations VALUES (?,?,?,?,?,?)", (
                request_id, user_id, session_id, model_version, encoded, datetime.now(timezone.utc).isoformat()
            ))
        return True

    def feature_pending(self, event_id: str) -> bool:
        with self.connect() as con:
            row = con.execute("SELECT 1 FROM events e LEFT JOIN feature_deliveries d ON e.event_id=d.event_id "
                              "WHERE e.event_id=? AND d.event_id IS NULL", (event_id,)).fetchone()
        return row is not None

    def mark_feature_delivered(self, event_id: str) -> None:
        with self.connect() as con:
            con.execute("INSERT OR IGNORE INTO feature_deliveries SELECT event_id,? FROM events WHERE event_id=?",
                        (datetime.now(timezone.utc).isoformat(), event_id))

    def append_impression(self, impression: Impression, request_id: str | None = None) -> bool:
        payload = _payload(impression)
        payload.pop("recommendation_id", None)
        ids = list(payload["product_ids"])
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Displayed product_ids must be nonempty and unique")
        if payload["clicked_ids"]:
            raise ValueError("Initial clicked_ids must be empty; clicks arrive as events")
        encoded = _json(payload)
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            old = con.execute("SELECT payload,request_id FROM impressions WHERE impression_id=?",
                              (impression.impression_id,)).fetchone()
            if old is not None:
                if old["payload"] != encoded or old["request_id"] != request_id:
                    raise ValueError("Conflicting impression_id")
                return False
            rec = None
            if request_id:
                rec = con.execute("SELECT * FROM recommendations WHERE request_id=?", (request_id,)).fetchone()
                if rec is None:
                    raise ValueError("Unknown recommendation_id")
                if rec["user_id"] != impression.user_id or (
                    rec["session_id"] is not None and rec["session_id"] != impression.session_id
                ):
                    raise ValueError("Recommendation owner/session mismatch")
                if not set(ids) <= {x["product_id"] for x in json.loads(rec["items"])}:
                    raise ValueError("Displayed products were not returned by this recommendation")
            con.execute("INSERT INTO impressions(impression_id,request_id,user_id,session_id,timestamp,payload) "
                        "VALUES (?,?,?,?,?,?)", (impression.impression_id, request_id, impression.user_id,
                                               impression.session_id, payload["timestamp"], encoded))
            if rec is not None:
                for item in json.loads(rec["items"]):
                    if item["product_id"] in ids and item.get("is_exploration") and item.get("arm") is not None:
                        self._bandit_add(con, rec["model_version"], str(item["arm"]), 1, 0)
        return True

    @staticmethod
    def _validate_attribution(con: sqlite3.Connection, event: Event) -> sqlite3.Row | None:
        if not event.impression_id:
            return None
        row = con.execute("SELECT * FROM impressions WHERE impression_id=?", (event.impression_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown impression_id; acknowledge exposure before sending actions")
        imp = json.loads(row["payload"])
        if (event.user_id, event.session_id) != (row["user_id"], row["session_id"]):
            raise ValueError("Impression owner/session mismatch")
        if event.product_id not in imp["product_ids"]:
            raise ValueError("Event product was not displayed")
        shown_at = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
        at = event.timestamp
        if at.tzinfo is None:
            at = at.replace(tzinfo=timezone.utc)
        if shown_at.tzinfo is None:
            shown_at = shown_at.replace(tzinfo=timezone.utc)
        if at < shown_at:
            raise ValueError("Event precedes its impression")
        return row

    def append_event(self, event: Event) -> bool:
        payload = _payload(event)
        encoded = _json(payload)
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            old = con.execute("SELECT payload FROM events WHERE event_id=?", (event.event_id,)).fetchone()
            if old is not None:
                if old["payload"] != encoded:
                    raise ValueError("Conflicting event_id")
                return False
            self._validate_attribution(con, event)
            con.execute("INSERT INTO events(event_id,user_id,session_id,event_type,product_id,impression_id,timestamp,payload) "
                        "VALUES (?,?,?,?,?,?,?,?)", (event.event_id, event.user_id, event.session_id,
                                                    payload["event_type"], event.product_id, event.impression_id,
                                                    payload["timestamp"], encoded))
        return True

    def get_impression_arms(self, impression_id: str) -> dict[str, str]:
        with self.connect() as con:
            row = con.execute("SELECT i.payload,r.items FROM impressions i JOIN recommendations r "
                              "ON i.request_id=r.request_id WHERE i.impression_id=?", (impression_id,)).fetchone()
        if row is None:
            return {}
        shown = set(json.loads(row["payload"])["product_ids"])
        return {x["product_id"]: str(x["arm"]) for x in json.loads(row["items"])
                if x["product_id"] in shown and x.get("is_exploration") and x.get("arm") is not None}

    def get_impression_model_version(self, impression_id: str) -> str | None:
        with self.connect() as con:
            row = con.execute("SELECT r.model_version FROM impressions i JOIN recommendations r "
                              "ON i.request_id=r.request_id WHERE i.impression_id=?", (impression_id,)).fetchone()
        return row[0] if row else None

    def claim_click(self, event: Event) -> str | None:
        """Return an exploration arm once per displayed product, never for an unshown item."""
        if str(event.event_type.value) != "view" or not event.impression_id:
            return None
        arms = self.get_impression_arms(event.impression_id)
        arm = arms.get(event.product_id)
        if arm is None:
            return None
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            saved = con.execute("SELECT payload FROM events WHERE event_id=?", (event.event_id,)).fetchone()
            if saved is None or saved["payload"] != _json(_payload(event)):
                raise ValueError("Persist event before claiming feedback")
            self._validate_attribution(con, event)
            cur = con.execute("INSERT OR IGNORE INTO exploration_clicks VALUES (?,?,?,?)",
                              (event.impression_id, event.product_id, event.event_id, arm))
            if not cur.rowcount:
                return None
            version = con.execute("SELECT r.model_version FROM impressions i JOIN recommendations r "
                                  "ON i.request_id=r.request_id WHERE i.impression_id=?", (event.impression_id,)).fetchone()[0]
            self._bandit_add(con, version, arm, 0, 1)
            return arm

    def high_watermark(self) -> int:
        with self.connect() as con:
            return int(con.execute("SELECT COALESCE(MAX(seq),0) FROM events").fetchone()[0])

    def count_since(self, watermark: int = 0) -> int:
        with self.connect() as con:
            return int(con.execute("SELECT COUNT(*) FROM events WHERE seq>?", (watermark,)).fetchone()[0])

    def export_snapshot(self, high_watermark: int | None = None):
        """Return consistent event/impression frames; clicked_ids are derived from persisted view events."""
        import pandas as pd

        with self.connect() as con:
            con.execute("BEGIN")
            end = high_watermark if high_watermark is not None else con.execute(
                "SELECT COALESCE(MAX(seq),0) FROM events").fetchone()[0]
            events = [json.loads(x[0]) for x in con.execute("SELECT payload FROM events WHERE seq<=? ORDER BY seq", (end,))]
            impressions = [json.loads(x[0]) for x in con.execute("SELECT payload FROM impressions ORDER BY seq")]
        clicked: dict[str, list[str]] = {}
        for event in events:
            if event["event_type"] == "view" and event.get("impression_id"):
                ids = clicked.setdefault(event["impression_id"], [])
                if event["product_id"] not in ids:
                    ids.append(event["product_id"])
        for imp in impressions:
            imp["clicked_ids"] = clicked.get(imp["impression_id"], [])
        ev = pd.DataFrame(events, columns=list(Event.model_fields))
        im = pd.DataFrame(impressions, columns=list(Impression.model_fields))
        for frame in (ev, im):
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_localize(None)
        if high_watermark is not None:
            im = im[im.timestamp <= ev.timestamp.max()].reset_index(drop=True)
        return ev, im

    def metrics(self) -> dict[str, Any]:
        with self.connect() as con:
            imps = [json.loads(row[0]) for row in con.execute("SELECT payload FROM impressions")]
            clicks = con.execute("SELECT COUNT(*) FROM (SELECT DISTINCT impression_id, product_id FROM events "
                                 "WHERE event_type='view' AND impression_id IS NOT NULL)").fetchone()[0]
            buyers = con.execute("SELECT COUNT(DISTINCT user_id) FROM events WHERE event_type='purchase' "
                                 "AND impression_id IS NOT NULL").fetchone()[0]
            events = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        exposures = sum(len(imp["product_ids"]) for imp in imps)
        users = len({imp["user_id"] for imp in imps})
        return {"events": events, "impressions": len(imps), "item_exposures": exposures,
                "clicked_item_exposures": clicks, "exposed_users": users, "purchasing_users": buyers,
                "ctr": clicks / exposures if exposures else None, "cvr": buyers / users if users else None}

    def feedback_counts(self, model_version: str | None = None) -> dict[str, dict[str, int]]:
        with self.connect() as con:
            where, params = (" WHERE model_version=?", (model_version,)) if model_version is not None else ("", ())
            rows = con.execute("SELECT arm,SUM(shown),SUM(clicks) FROM bandit_stats" + where + " GROUP BY arm", params).fetchall()
        return {row[0]: {"impressions": row[1], "clicks": row[2]} for row in rows}


@lru_cache(maxsize=1)
def get_event_log() -> EventLog:
    return EventLog(os.getenv("EVENT_LOG_PATH") or artifact_dir() / "runtime" / "events.sqlite3")
