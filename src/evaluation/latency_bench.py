"""응답시간 측정: warm-up 10회 후 100회 호출의 p95 (과제 측정 규격, 단건 요청).

    python -m src.evaluation.latency_bench --target search
    python -m src.evaluation.latency_bench --target search --model patrickjohncyh/fashion-clip

- search: /api/search 를 앱 안에서 직접 호출(TestClient)해 text / image / hybrid 각각 측정
          → API 파싱·직렬화까지 포함한 서버 처리 시간. 기준 p95 < 200ms
- 추천·후보·피처 조회 측정은 각 모듈 구현 후 추가

결과: artifacts/metrics/latency_<target>.json (실측 환경 포함 → docs 리포트에 기록)
"""

from __future__ import annotations

import argparse
import base64
import json
import platform
import subprocess
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

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


def machine_info() -> dict[str, Any]:
    import torch

    cpu = platform.processor()
    if platform.system() == "Darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False).stdout.strip()
    return {"platform": platform.platform(), "cpu": cpu, "torch_threads": torch.get_num_threads()}


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=["search"], default="search")
    parser.add_argument("--model", default=None, help="검색 모델 (기본: configs/search.yaml)")
    args = parser.parse_args()

    result = {"target": args.target, "measured_at": datetime.now().isoformat(timespec="seconds"), **machine_info()}
    if args.target == "search":
        result |= bench_search(args.model)

    suffix = f"_{model_slug(result['model_name'])}" if args.model else ""
    out = ROOT / load_config("search")["paths"]["metrics_dir"] / f"latency_{args.target}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[latency] {result['cpu']} / device={result.get('device')} / index={result.get('index')}")
    for mode in ("text", "image", "hybrid"):
        if mode in result:
            r = result[mode]
            print(f"  {mode:6s} p50 {r['p50_ms']:7.1f}ms  p95 {r['p95_ms']:7.1f}ms  (기준 200ms)")
    print(f"[latency] 저장 → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
