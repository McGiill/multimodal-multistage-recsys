"""응답시간 측정: warm-up 10회 후 100회 호출의 p95 (과제 측정 규격, 단건 요청).

    python -m src.evaluation.latency_bench --target search
    python -m src.evaluation.latency_bench --target search --model patrickjohncyh/fashion-clip
    python -m src.evaluation.latency_bench --target search --model patrickjohncyh/fashion-clip --out artifacts/metrics/latency_search_patrickjohncyh__fashion-clip_docker.json  # Docker 4코어 측정

- search: /api/search 를 앱 안에서 직접 호출(TestClient)해 text / image / hybrid 각각 측정
          → API 파싱·직렬화까지 포함한 서버 처리 시간. 기준 p95 < 200ms
- feature: Feature Store 조회 (세션 클릭 + 최근 조회 상품, 요청 1번에 필요한 만큼). 기준 p95 < 10ms
           REDIS_HOST가 있으면 Redis, 없으면 메모리 (docker compose up -d redis 후 REDIS_HOST=localhost)
- recommend: 추천 API 파싱·특징 조회·순위 계산·응답 등록·직렬화 포함 (TestClient, TCP 전송 제외)

결과: artifacts/metrics/latency_<target>_<timestamp>.json (기존 결과는 덮어쓰지 않는다)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import subprocess
import tempfile
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from src.common.config import ROOT, load_config
from src.search.catalog import model_slug
from src.serving.main import app


def measure(call: Callable[[int], Any], warmup: int = 10, n: int = 100) -> dict[str, float]:
    for i in range(warmup):
        call(i)
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        call(i)
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "mean_ms": round(float(arr.mean()), 2),
        "max_ms": round(float(arr.max()), 2),
    }


def cpu_limit() -> float | None:
    """컨테이너 CPU 제한 (cgroup v2 cpu.max, 예: docker --cpus 4 → 4.0). 제한이 없거나 컨테이너가 아니면 None."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
    except (OSError, ValueError):
        return None
    return None if quota == "max" else round(int(quota) / int(period), 2)


def machine_info() -> dict[str, Any]:
    import torch

    cpu = platform.processor()
    if platform.system() == "Darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False).stdout.strip()
    return {"platform": platform.platform(), "cpu": cpu, "cpu_count": os.cpu_count(), "cpu_limit": cpu_limit(),
            "torch_threads": torch.get_num_threads()}


def bench_search(model: str | None = None) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    from src.search.catalog import with_overrides
    from src.search.searcher import Searcher
    from src.serving.routers.search import get_searcher

    cfg = with_overrides(load_config("search"), model=model)
    eval_set = pd.read_parquet(ROOT / cfg["paths"]["eval_set"])
    texts = eval_set.loc[eval_set["query_type"] == "text", "query_text"].tolist()
    image_root = ROOT / cfg["paths"]["image_root"]
    images = [
        base64.b64encode((image_root / p).read_bytes()).decode()
        for p in eval_set.loc[eval_set["query_type"] == "image", "query_image_path"].head(20)
    ]

    searcher = Searcher(cfg)  # 로딩 시간은 측정에서 제외 (serve_device = CPU)
    app.dependency_overrides[get_searcher] = lambda: searcher
    client = TestClient(app)

    def post(payload: dict[str, Any]) -> None:
        r = client.post("/api/search", json=payload)
        r.raise_for_status()

    return {
        "device": searcher.encoder.device,
        "model_name": searcher.encoder.model_name,
        "index": cfg["index"]["type"],
        "n_products": len(searcher.product_ids),
        "text": measure(lambda i: post({"query_text": texts[i % len(texts)], "top_k": 10})),
        "image": measure(lambda i: post({"image_base64": images[i % len(images)], "top_k": 10})),
        "hybrid": measure(
            lambda i: post({"query_text": texts[i % len(texts)], "image_base64": images[i % len(images)], "top_k": 10})
        ),
    }


def bench_feature() -> dict[str, Any]:
    from src.common.schemas import Event
    from src.serving.feature_store import RedisFeatureStore, get_feature_store

    store = get_feature_store()
    user, session = f"bench-{uuid4().hex}", f"bench-{uuid4().hex}"
    for i in range(load_config("serving")["redis"]["recent_views_len"]):
        store.push_event(Event(event_id=f"{session}-{i}", user_id=user, session_id=session, event_type="view",
                               product_id=f"hm_{i:010d}", timestamp=datetime.now()))

    def lookup(_: int) -> None:
        store.get_session_clicks(session)
        store.get_recent_views(user)

    result = {"store": type(store).__name__, "lookup": measure(lookup)}
    if isinstance(store, RedisFeatureStore):
        store.client.delete(f"recent_views:{user}", f"session_clicks:{session}", f"feature_events:{session}")
    return result


def bench_recommend() -> dict[str, Any]:
    """실제 모델과 특징 저장소를 쓰되 데모 이력·영속 저널에 측정 데이터를 남기지 않는다."""
    from datetime import timezone

    from fastapi.testclient import TestClient

    from src.common.schemas import Event
    from src.recommendation.reranking.stage2 import get_rerank_service
    from src.serving.event_log import get_event_log
    from src.serving.feature_store import RedisFeatureStore, get_feature_store
    from src.serving.model_loader import get_recommendation_models

    models, features = get_recommendation_models(), get_feature_store()
    user, session = f"bench-{uuid4().hex}", f"bench-{uuid4().hex}"
    at = datetime.now(timezone.utc)
    old_log_path = os.environ.get("EVENT_LOG_PATH")
    try:
        with tempfile.TemporaryDirectory(prefix="recommend-bench-") as temporary:
            os.environ["EVENT_LOG_PATH"] = str(Path(temporary) / "events.sqlite3")
            get_event_log.cache_clear()
            client = TestClient(app)

            def request(_: int) -> None:
                response = client.get("/api/recommend", params={
                    "user_id": user, "session_id": session, "top_n": 50, "at": at.isoformat(),
                })
                response.raise_for_status()

            cold = measure(request)
            # 실제 상품 20개를 별도 측정 사용자에게만 넣어 세션 경로도 측정한다.
            for i, pid in enumerate(models.store.item_ids[:20]):
                features.push_event(Event(event_id=f"{session}-{i}", user_id=user, session_id=session,
                                          event_type="view", product_id=pid, timestamp=at))
            with_session = measure(request)
            return {"model_version": models.folder.name, "store": type(features).__name__, "top_n": 50,
                    "scope": "TestClient server processing including parsing, feature lookup, journal and serialization; excludes TCP and model loading",
                    "prediction_at": at.isoformat(), "user_profile": "unknown isolated benchmark user",
                    "session_inputs": "20 known products; isolated benchmark keys",
                    "warmup": 10, "requests_per_scenario": 100, "cold": cold, "session": with_session,
                    "passed": max(cold["p95_ms"], with_session["p95_ms"]) < 200}
    finally:
        if old_log_path is None:
            os.environ.pop("EVENT_LOG_PATH", None)
        else:
            os.environ["EVENT_LOG_PATH"] = old_log_path
        get_event_log.cache_clear()
        get_rerank_service.cache_clear()
        if isinstance(features, RedisFeatureStore):
            features.client.delete(f"recent_views:{user}", f"session_clicks:{session}", f"feature_events:{session}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=["search", "feature", "recommend"], default="search")
    parser.add_argument("--model", default=None, help="검색 모델 (기본: configs/search.yaml)")
    parser.add_argument("--out", default=None, help="새 결과 저장 경로 (기본: 시각을 붙인 JSON, 기존 파일 덮어쓰기 금지)")
    args = parser.parse_args()

    result = {"target": args.target, "measured_at": datetime.now().isoformat(timespec="seconds"), **machine_info()}
    if args.target == "search":
        result |= bench_search(args.model)
    elif args.target == "feature":
        result |= bench_feature()
    elif args.target == "recommend":
        result |= bench_recommend()

    suffix = f"_{model_slug(result['model_name'])}" if args.model and args.target == "search" else ""
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    out = ROOT / (args.out or Path(load_config("search")["paths"]["metrics_dir"]) / f"latency_{args.target}{suffix}_{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    print(f"[latency] {result['cpu']} / device={result.get('device')} / index={result.get('index')}")
    for mode in ("text", "image", "hybrid", "cold", "session"):
        if mode in result:
            r = result[mode]
            print(f"  {mode:6s} p50 {r['p50_ms']:7.1f}ms  p95 {r['p95_ms']:7.1f}ms  (기준 200ms)")
    if "lookup" in result:
        r = result["lookup"]
        print(f"  {result['store']} p50 {r['p50_ms']:.2f}ms  p95 {r['p95_ms']:.2f}ms  (기준 10ms)")
    print(f"[latency] 저장 → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
