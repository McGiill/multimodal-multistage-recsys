"""합성 테스트 자료로 팀 스키마→학습→FAISS→CVR→API를 검증. 성능 입증용 아님."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pyarrow as pa

from src.common.config import load_config
from src.evaluation.eval_recommend import evaluate
from src.evaluation.split import boundaries, split_logs, summarize
from src.recommendation.training import build_index, train_candidate, train_ranking


def create_data(root):
    root = Path(root)
    out = root / "processed"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "events.parquet").exists():
        raise FileExistsError("Use a new smoke data directory")
    start = pd.Timestamp("2026-01-01")
    rng = np.random.default_rng(42)
    n = 600
    products = pd.DataFrame(
        {
            "product_id": [f"P{i:05}" for i in range(n)],
            "name": [f"Test garment {i}" for i in range(n)],
            "category_l1": [f"C{i % 4}" for i in range(n)],
            "category_l2": [f"T{i % 12}" for i in range(n)],
            "category_l3": [f"S{i % 24}" for i in range(n)],
            "brand": [f"B{i % 5}" for i in range(n)],
            "color": [f"color{i % 3}" for i in range(n)],
            "pattern": ["Solid"] * n,
            "price": [10.0 + i % 40 for i in range(n)],
            "created_at": [
                start + pd.Timedelta(days=28)
                if i >= 550
                else start - pd.Timedelta(days=100)
                for i in range(n)
            ],
        }
    )
    users = pd.DataFrame(
        {
            "user_id": [f"U{i:05}" for i in range(24)],
            "age_group": ["20s" if i % 2 else "40s" for i in range(24)],
            "gender": ["F" if i % 2 else "M" for i in range(24)],
            "signup_at": [
                start + pd.Timedelta(days=25)
                if i == 23
                else start - pd.Timedelta(days=300)
                for i in range(24)
            ],
        }
    )
    rows = []
    for day in range(30):
        for u in range(24):
            at = start + pd.Timedelta(days=day, hours=u % 12 + 1)
            if at < users.iloc[u].signup_at:
                continue
            session = f"S{day:02}_{u:02}"
            for turn in range(3):
                pool = np.arange(550 if day < 28 else 600)
                pool = pool[pool % 4 == u % 4]
                item = int(rng.choice(pool))
                t = at + pd.Timedelta(minutes=turn * 5)

                def add(kind, seconds, u=u, session=session, t=t, item=item):
                    rows.append(
                        {
                            "event_id": f"E{len(rows):08}",
                            "user_id": f"U{u:05}",
                            "session_id": session,
                            "timestamp": t + pd.Timedelta(seconds=seconds),
                            "event_type": kind,
                            "product_id": f"P{item:05}",
                        }
                    )

                add("view", 0)
                if rng.random() < (0.65 if item % 3 == 0 else 0.08):
                    add("cart", 20)
                    add("purchase", 40)
    products.to_parquet(out / "products.parquet", index=False)
    users.to_parquet(out / "users.parquet", index=False)
    pd.DataFrame(rows).to_parquet(out / "events.parquet", index=False)
    # Deliberately invalid hidden file: the model must never read it.
    (out / "users_hidden.parquet").write_bytes(b"NOT MODEL INPUT")
    events = pd.DataFrame(rows)
    bounds = boundaries({"period": {"start_date": str(start.date()), "days": 30}})
    parts = split_logs(events, pa.Table.from_pandas(events[["session_id", "timestamp"]]), bounds)
    splits = root / "splits"
    splits.mkdir()
    for name, part in parts.items():
        part["events"].to_parquet(splits / f"{name}_events.parquet", index=False)
    meta = summarize(parts, bounds, users)
    (splits / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def smoke(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    data = root / "data"
    bundle = root / "model"
    create_data(data)
    cfg = copy.deepcopy(load_config("recommend"))
    cfg["threads"] = 2
    cfg["candidate"].update(
        embedding_dim=16,
        history_len=5,
        epochs=2,
        batch_size=128,
        faiss_index_type="flat",
    )
    cfg["ranking"].update(
        embedding_dim=8, dnn_hidden=[32, 16], dropout=0.0, epochs=2, batch_size=128
    )
    train_candidate(data, bundle, cfg)
    build_index(bundle)
    train_ranking(bundle)
    report = evaluate(bundle)
    os.environ["REC_MODEL_DIR"] = str(bundle.resolve())
    os.environ["PRELOAD_MODELS"] = "0"
    from fastapi.testclient import TestClient

    from src.serving.main import app

    with TestClient(app) as client:
        response = client.get(
            "/api/recommend",
            params={"user_id": "U00001", "top_n": 10, "at": "2026-01-29T15:00:00Z"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert len(body["recommendations"]) == 10
        scores = [r["score"] for r in body["recommendations"]]
        assert scores == sorted(scores, reverse=True)
        unknown = client.get(
            "/api/recommend",
            params={"user_id": "unknown", "at": "2026-01-29T15:00:00Z"},
        )
        assert unknown.status_code == 200
        assert (
            client.get(
                "/api/recommend",
                params={"user_id": "U00001", "at": "2026-01-29T15:00:00"},
            ).status_code
            == 400
        )
    (root / "api_response.json").write_text(
        json.dumps(body, indent=2), encoding="utf-8"
    )
    report["purpose"] = (
        "Integration smoke test only; 600 synthetic products, not project performance validation"
    )
    report["api_status"] = response.status_code
    (root / "smoke_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    smoke(a.output)
