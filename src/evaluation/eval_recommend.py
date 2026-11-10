"""저장된 Stage1/2 모델의 전체 test 평가. 기준 통과 여부와 측정 범위를 함께 기록."""

import json
from pathlib import Path

import numpy as np

from src.evaluation.metrics import auc
from src.recommendation.training import (
    arguments,
    predict,
    ranking_examples,
    ranking_inputs,
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
    # Stage 2 (ranking.objective): ctcvr = test 노출 목록에서 노출 후 구매(1) vs 나머지(0),
    #                              ctr = 클릭(1) vs 보여줬지만 안 누름(0), cvr = test 조회 후 24시간 구매
    objective = models.objective
    rdata = ranking_examples(folder, store, cfg)
    ci = np.flatnonzero((rdata["split"] == 2) & rdata["mature"])
    if len(np.unique(rdata["label"][ci])) < 2:
        raise ValueError(f"{objective} test requires both labels")
    s, d = ranking_inputs(cfg, store, models.tower, models.item_vectors, rdata, ci)
    prob = predict(models.ranker, s, d)
    y = rdata["label"][ci]
    clipped = np.clip(prob, 1e-7, 1 - 1e-7)
    ranking = {
        "objective": objective,
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
        "ranking": ranking,
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
        "label_definition": {
            "ctcvr": "Purchase within the window after being shown and clicked in that impression (1) vs shown-but-not-clicked "
                     "or clicked-but-not-purchased (0); negatives per click sampled from the same impression; "
                     "impressions without clicks skipped; fully observed windows only",
            "ctr": "Click (1) vs shown-but-not-clicked (0) in the same impression list; negatives per click sampled; "
                   "impressions without clicks skipped",
            "cvr": "Purchase within 24h attributed only to latest strictly earlier view of same user/item, within the same split; "
                   "fully observed windows only",
        }[objective],
        "ranking_features": cfg["ranking"].get("features", 1),
        "split_examples": {
            name: {
                "retrieval_views": int((data["split"] == j).sum()),
                "ranking_rows": int(((rdata["split"] == j) & rdata["mature"]).sum()),
                "ranking_positives": int(rdata["label"][(rdata["split"] == j) & rdata["mature"]].sum()),
            } for j, name in enumerate(["train", "valid", "test"])
        },
        "boundary_view_exclusions": int((data["split"] == -1).sum()),
        "thresholds": {
            "recall": retrieval["recall_at_300"] >= 0.3,
            "ranking_auc": ranking["auc"] >= 0.7,
            "candidate_latency": bool(np.percentile(lat, 95) < 100),
            "candidate_count": min(counts) >= 300,
        },
        "limitations": [
            "Offline snapshot only; Redis streaming and Stage3 not integrated.",
            "Observed product price is consumed as supplied, not inferred currency.",
            "Results describe the supplied simulator dataset, not real shoppers or an online experiment.",
            "Frozen holdout history: valid uses train; test uses train+valid before their nominal boundaries.",
            "Ranking AUC uses logged impressions/views, not unshown FAISS candidates; exposure-selection bias remains.",
            "Supplied static catalog cannot validate historical price changes.",
            "Next-view retrieval target; " + {"ctcvr": "CTCVR is purchase given being shown.", "ctr": "CTR is conditional on being shown.",
                                              "cvr": "CVR is conditional on a view."}[objective],
        ],
    }
    (folder / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    _, out, _ = arguments()
    print(json.dumps(evaluate(out), indent=2))
