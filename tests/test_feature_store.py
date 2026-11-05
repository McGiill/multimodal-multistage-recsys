"""Feature Store (최근 조회 상품, 세션 클릭) 테스트.

메모리 구현은 항상, Redis 구현은 Redis가 떠 있을 때만 (REDIS_HOST, 없으면 localhost) 실행한다.
    docker compose up -d redis && pytest tests/test_feature_store.py -v
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime

import pytest
import redis

from src.common.config import env
from src.common.schemas import Event
from src.serving.feature_store import RECENT_KEY, SESSION_KEY, MemoryFeatureStore, RedisFeatureStore


def _event(session: str, kind: str, product: str | None, user: str = "U1") -> Event:
    return Event(event_id=f"{session}-{product}-{kind}", user_id=user, session_id=session,
                 event_type=kind, product_id=product, timestamp=datetime(2026, 3, 24))


def _redis() -> redis.Redis | None:
    client = redis.Redis(host=env("REDIS_HOST") or "localhost", port=int(env("REDIS_PORT", "6379")),
                         decode_responses=True, socket_connect_timeout=0.5)
    try:
        client.ping()
    except redis.exceptions.ConnectionError:
        return None
    return client


@pytest.fixture(params=["memory", "redis"])
def store(request):
    if request.param == "memory":
        yield MemoryFeatureStore(recent_views_len=2), lambda s: s
        return
    client = _redis()
    if client is None:
        pytest.skip("Redis 없음 (docker compose up -d redis)")
    tag = uuid.uuid4().hex[:8]  # 실제 키와 섞이지 않게 테스트마다 다른 ID
    yield RedisFeatureStore(client, recent_views_len=2, session_ttl_sec=60), lambda s: f"{s}-{tag}"
    keys = client.keys(f"*-{tag}")
    if keys:
        client.delete(*keys)


def test_keeps_views_per_session_and_user(store):
    fs, k = store
    for e in [_event(k("S1"), "view", "P000", k("U1")), _event(k("S1"), "search", None, k("U1")),
              _event(k("S1"), "cart", "P000", k("U1")), _event(k("S1"), "view", "P001", k("U1")),
              _event(k("S2"), "view", "P102", k("U1"))]:
        fs.push_event(e)
    assert fs.get_session_clicks(k("S1")) == ["P000", "P001"]  # view만
    assert fs.get_session_click_count(k("S1")) == 2
    assert fs.get_session_clicks(k("S2")) == ["P102"]
    assert fs.get_session_clicks(k("none")) == [] and fs.get_session_click_count(k("none")) == 0
    assert fs.get_recent_views(k("U1")) == ["P001", "P102"]   # 최근 2개
    assert fs.get_recent_views(k("U2")) == []


def test_redis_session_ttl_and_lookup_latency():
    client = _redis()
    if client is None:
        pytest.skip("Redis 없음 (docker compose up -d redis)")
    tag = uuid.uuid4().hex[:8]
    fs = RedisFeatureStore(client, recent_views_len=20, session_ttl_sec=60)
    try:
        for i in range(10):
            fs.push_event(_event(f"S-{tag}", "view", f"P{i:03d}", f"U-{tag}"))
        assert 0 < client.ttl(SESSION_KEY.format(f"S-{tag}")) <= 60
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            fs.get_session_clicks(f"S-{tag}")
            fs.get_recent_views(f"U-{tag}")
            times.append((time.perf_counter() - t0) * 1000)
        assert sorted(times)[94] < 10  # 명세: 피처 조회 10ms 이내 (p95)
    finally:
        client.delete(SESSION_KEY.format(f"S-{tag}"), RECENT_KEY.format(f"U-{tag}"))
