"""Redis Feature Store: 최근 조회 상품, 세션 내 클릭 수. 조회 < 10ms.
- push_event(event) / get_recent_views(user_id) / get_session_clicks(session_id) / get_session_click_count(session_id)

get_feature_store()는 환경변수 REDIS_HOST가 있으면 Redis(RedisFeatureStore), 없으면 메모리(MemoryFeatureStore)를 쓴다.
docker-compose는 .env의 REDIS_HOST=redis로 Redis에 붙고, 로컬에서 Redis 컨테이너만 띄웠으면 REDIS_HOST=localhost.
호출하는 쪽: routers/events.py, recommendation/pipeline.py
세션 클릭 = 세션 안의 view 이벤트 (세션 인코더 학습과 같은 기준).

Redis 키 (configs/serving.yaml 의 redis 항목)
  recent_views:{user_id}     list, 최근 recent_views_len개 (오래된 것 → 최근 순)
  session_clicks:{session_id} list, 마지막 이벤트 후 session_ttl_sec 지나면 만료

담당: D
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache

import redis

from src.common.config import env, load_config
from src.common.schemas import Event, EventType

RECENT_KEY = "recent_views:{}"
SESSION_KEY = "session_clicks:{}"


def is_click(event: Event) -> bool:
    return event.event_type == EventType.VIEW and bool(event.product_id)


class MemoryFeatureStore:
    """사용자별 최근 조회 상품(최대 recent_views_len개), 세션별 클릭 상품. 서버를 다시 켜면 비워지고 만료 없음."""

    def __init__(self, recent_views_len: int = 20) -> None:
        self.recent_views_len = recent_views_len
        self._recent: dict[str, deque[str]] = {}
        self._sessions: dict[str, list[str]] = {}

    def push_event(self, event: Event) -> None:
        if not is_click(event):
            return
        self._recent.setdefault(event.user_id, deque(maxlen=self.recent_views_len)).append(event.product_id)
        self._sessions.setdefault(event.session_id, []).append(event.product_id)

    def get_recent_views(self, user_id: str) -> list[str]:
        return list(self._recent.get(user_id, ()))

    def get_session_clicks(self, session_id: str) -> list[str]:
        return list(self._sessions.get(session_id, ()))

    def get_session_click_count(self, session_id: str) -> int:
        return len(self._sessions.get(session_id, ()))


class RedisFeatureStore:
    """MemoryFeatureStore와 같은 함수를 Redis list로. 이벤트 1건 저장 = 왕복 1번 (pipeline)."""

    def __init__(self, client: redis.Redis, recent_views_len: int = 20, session_ttl_sec: int = 1800) -> None:
        self.client = client
        self.recent_views_len = recent_views_len
        self.session_ttl_sec = session_ttl_sec

    def push_event(self, event: Event) -> None:
        if not is_click(event):
            return
        recent, session = RECENT_KEY.format(event.user_id), SESSION_KEY.format(event.session_id)
        pipe = self.client.pipeline(transaction=False)
        pipe.rpush(recent, event.product_id)
        pipe.ltrim(recent, -self.recent_views_len, -1)
        pipe.rpush(session, event.product_id)
        pipe.expire(session, self.session_ttl_sec)
        pipe.execute()

    def get_recent_views(self, user_id: str) -> list[str]:
        return self.client.lrange(RECENT_KEY.format(user_id), 0, -1)

    def get_session_clicks(self, session_id: str) -> list[str]:
        return self.client.lrange(SESSION_KEY.format(session_id), 0, -1)

    def get_session_click_count(self, session_id: str) -> int:
        return int(self.client.llen(SESSION_KEY.format(session_id)))


FeatureStoreLike = MemoryFeatureStore | RedisFeatureStore


@lru_cache(maxsize=1)
def get_feature_store() -> FeatureStoreLike:
    """서버 전체에서 하나. REDIS_HOST가 있으면 Redis (연결 실패 시 에러), 없으면 메모리."""
    cfg = load_config("serving")["redis"]
    host = env("REDIS_HOST")
    if not host:
        return MemoryFeatureStore(recent_views_len=cfg["recent_views_len"])
    client = redis.Redis(host=host, port=int(env("REDIS_PORT", "6379")), decode_responses=True)
    client.ping()
    return RedisFeatureStore(client, recent_views_len=cfg["recent_views_len"],
                             session_ttl_sec=cfg["session_ttl_sec"])
