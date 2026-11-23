"""Live recommendation → displayed impression → virtual shopper actions → API.

Uses the same hidden-persona response model as A/B experiments. Default stream-*
user IDs keep generated traffic separate from hand-operated demo members.
``--max-impressions N`` provides a finite run; failed writes are retried with the
same IDs, and a persistent transport failure stops the process rather than losing
events silently. The API journal is the durable training source.
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import requests

from src.common.config import load_config
from src.common.schemas import Impression


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 15, retries: int = 3,
                 session: requests.Session | None = None, sleep: Callable = time.sleep) -> None:
        self.base_url, self.timeout, self.retries = base_url.rstrip("/"), timeout, retries
        self.session, self.sleep = session or requests.Session(), sleep

    def request(self, method: str, path: str, **kwargs):
        for attempt in range(self.retries + 1):
            try:
                res = self.session.request(method, self.base_url + path, timeout=self.timeout, **kwargs)
                res.raise_for_status()
                return res.json()
            except requests.HTTPError as exc:
                if exc.response is not None and 400 <= exc.response.status_code < 500:
                    raise  # schema/ownership errors need a fix, not retries
                if attempt == self.retries:
                    raise
            except (requests.Timeout, requests.ConnectionError):
                if attempt == self.retries:
                    raise
            self.sleep(min(2 ** attempt, 10))


def run_stream(engine: Any, client: Any, cfg: dict[str, Any], max_impressions: int | None = None,
               users: list[str] | None = None, sleep: Callable = time.sleep,
               clock: Callable = lambda: datetime.now(timezone.utc), run_id: str | None = None) -> dict[str, int]:
    options = cfg["stream"]
    rate = float(options["events_per_sec"])
    top_n = int(options.get("top_n", 10))
    if rate <= 0 or top_n < 1 or (max_impressions is not None and max_impressions < 0):
        raise ValueError("Positive rate/top_n and a nonnegative impression limit are required")
    members = users or engine.user_ids
    if not members or any(u not in engine.user_ids for u in members):
        raise ValueError("Select existing simulator users")
    rng = np.random.default_rng(cfg["seed"])
    run_id = run_id or uuid.uuid4().hex
    prefix = options.get("user_prefix", "stream-")
    sessions: dict[str, tuple[str, int]] = {}
    previous: dict[str, datetime] = {}
    result = {"impressions": 0, "events": 0}
    while max_impressions is None or result["impressions"] < max_impressions:
        user = str(rng.choice(members))
        public_user = prefix + user
        at = max(clock(), engine.signup_at(user), previous.get(user, datetime.min.replace(tzinfo=timezone.utc)))
        session, screens = sessions.get(user, (f"stream-{run_id}-{user}-{uuid.uuid4().hex[:8]}", 0))
        rec = client.request("GET", "/api/recommend", params={
            "user_id": public_user, "session_id": session, "top_n": top_n, "at": at.isoformat()
        })
        # Exactly this visible list is acknowledged. Returned-but-hidden products never count.
        ids = list(dict.fromkeys(item["product_id"] for item in rec["recommendations"]))[:top_n]
        if not ids:
            raise ValueError("Recommendation API returned no displayable products")
        imp_id = f"stream-{run_id}-imp-{result['impressions']}"
        imp = Impression(impression_id=imp_id, user_id=public_user, session_id=session, timestamp=at,
                         source="home", context="live recommendation response simulation", product_ids=ids, clicked_ids=[])
        payload = imp.model_dump(mode="json")
        payload["recommendation_id"] = rec.get("recommendation_id")
        if not payload["recommendation_id"]:
            raise ValueError("Recommendation API must return recommendation_id for exposure attribution")
        client.request("POST", "/api/impressions", json=payload)
        reaction = engine.respond(user, ids, at, session, impression_id=imp_id)
        for event in reaction.events:
            event = event.model_copy(update={"user_id": public_user, "event_id": f"{run_id}-{event.event_id}"})
            client.request("POST", "/api/events", json=event.model_dump(mode="json"))
            result["events"] += 1
            previous[user] = event.timestamp + timedelta(milliseconds=1)
            sleep(1 / rate)
        result["impressions"] += 1
        screens += 1
        if reaction.next_screen == "exit" or screens >= cfg["session_flow"]["max_screens"]:
            sessions.pop(user, None)
        else:
            sessions[user] = session, screens
        if not reaction.events:
            sleep(1 / rate)
        if result["impressions"] % 100 == 0:
            print(f"[stream] {result}", flush=True)
    return result


def main() -> None:
    from src.simulator.online import OnlineResponseEngine

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-impressions", type=int)
    p.add_argument("--users", help="Comma-separated existing simulator IDs (traffic uses stream.user_prefix)")
    args = p.parse_args()
    cfg = load_config("simulator")
    options = cfg["stream"]
    client = ApiClient(os.getenv("API_BASE_URL", "http://localhost:8000"),
                       timeout=options.get("timeout_sec", 15), retries=options.get("retries", 3))
    engine = OnlineResponseEngine.from_config()
    print(f"[stream] actual API recommendation traffic; target <= {options['events_per_sec']} events/sec", flush=True)
    print(run_stream(engine, client, cfg, args.max_impressions, args.users.split(",") if args.users else None))


if __name__ == "__main__":
    main()
