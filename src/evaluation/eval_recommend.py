"""저장된 Stage1/2 모델의 전체 test 평가. 기준 통과 여부와 측정 범위를 함께 기록."""

import json
from pathlib import Path

import numpy as np

from src.evaluation.metrics import auc
from src.recommendation.ranking.features import rank_features
from src.recommendation.training import (
    arguments,
    predict,
    retrieval_metric,
)
from src.serving.model_loader import RecommendationModels


def evaluate(folder):
    folder = Path(folder)
    models = RecommendationModels(folder)
    cfg = models.cfg
    store = models.store
    data = dict(np.load(folder / "examples.npz", allow_pickle=False))
    ix = np.flatnonzero(data["split"] == 2)
    if not len(ix):
        raise ValueError("Empty test")
    retrieval = retrieval_metric(
        models.tower, models.index, store, data, ix, cfg["candidate"]["n_candidates"]
    )
    ci = np.flatnonzero((data["split"] == 2) & data["mature"])
    if len(np.unique(data["label"][ci])) < 2:
        raise ValueError("CVR test requires both labels")
    s, d = rank_features(
        store,
        data["target"][ci],
        data["history"][ci],
        data["profile"][ci],
        data["time"][ci],
    )
    prob = predict(models.ranker, s, d)
    y = data["label"][ci]
    clipped = np.clip(prob, 1e-7, 1 - 1e-7)
    cvr = {
        "auc": auc(y, prob),
        "rows": len(ci),
        "positives": int(y.sum()),
        "logloss": float(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)).mean()),
    }
    import time

    rng = np.random.default_rng(123)
    chosen = rng.choice(ix, 220, replace=len(ix) < 220)
    lat = []
    total = []
    counts = []
    for j, i in enumerate(chosen):
        begin = time.perf_counter()
        ids, _, state = models.retriever.retrieve(
            data["user"][i], cfg["candidate"]["n_candidates"], data["time"][i]
        )
        middle = time.perf_counter()
        scores = models.score(ids, state, data["time"][i])
        np.argsort(-scores)
        end = time.perf_counter()
        if j >= 20:
            lat.append((middle - begin) * 1000)
            total.append((end - begin) * 1000)
            counts.append(len(ids))
    report = {
        "retrieval": retrieval,
        "cvr": cvr,
        "latency": {
            "candidate_p95_ms": float(np.percentile(lat, 95)),
            "candidate_p50_ms": float(np.percentile(lat, 50)),
            "candidate_max_ms": float(max(lat)),
            "total_p95_ms": float(np.percentile(total, 95)),
            "min_candidates": min(counts),
            "requests": 200,
            "warmups": 20,
            "scope": "includes history lookup; excludes model startup and HTTP",
        },
        "catalog_items": len(store.item_ids),
        "users": len(store.users),
        "events": store.metadata.get("source_events", len(store.events)),
        "history_events_saved": len(store.events),
        "data_audit": store.metadata.get("data_audit", {}),
        "history_policy": store.metadata.get("history_policy"),
        "label_definition": "Purchase within 24h attributed only to latest strictly earlier view of same user/item, within the same split; fully observed windows only",
        "split_examples": {
            name: {
                "retrieval_views": int((data["split"] == j).sum()),
                "cvr_rows": int(((data["split"] == j) & data["mature"]).sum()),
                "cvr_positives": int(data["label"][(data["split"] == j) & data["mature"]].sum()),
            } for j, name in enumerate(["train", "valid", "test"])
        },
        "boundary_view_exclusions": int((data["split"] == -1).sum()),
        "thresholds": {
            "recall": retrieval["recall_at_300"] >= 0.3,
            "cvr_auc": cvr["auc"] >= 0.7,
            "candidate_latency": bool(np.percentile(lat, 95) < 100),
            "candidate_count": min(counts) >= 300,
        },
        "limitations": [
            "Offline snapshot only; Redis streaming and Stage3 not integrated.",
            "Observed product price is consumed as supplied, not inferred currency.",
            "Results describe the supplied simulator dataset, not real shoppers or an online experiment.",
            "Frozen holdout history: valid uses train; test uses train+valid before their nominal boundaries.",
            "CVR AUC uses logged views, not unviewed FAISS candidates; exposure-selection bias remains.",
            "Supplied static catalog cannot validate historical price changes.",
            "Next-view retrieval target; CVR is conditional on a view.",
        ],
    }
    (folder / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    _, out, _ = arguments()
    print(json.dumps(evaluate(out), indent=2))
