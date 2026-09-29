"""docker 컨테이너용 실시간 로그 생성기: python -m src.simulator.stream

events_per_sec 속도로 이벤트를 만들어 API(POST /api/events)로 전송 → Redis 실시간 피처 갱신.
담당: A
"""

from __future__ import annotations

import time

from src.common.config import load_config


def main() -> None:
    cfg = load_config("simulator")
    rate = cfg["stream"]["events_per_sec"]
    print(f"[simulator] 행동 로그 생성 시작... ({rate} events/sec)", flush=True)
    while True:
        # TODO(A): behavior 모듈로 이벤트 생성 → requests.post(f"{API_BASE_URL}/api/events", json=...)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
