"""Stage 1/2 학습 공통 실행. 데이터/모델 경로 및 설정은 명시적으로 저장."""

import argparse
import copy
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import yaml

from src.common.config import ROOT, data_dir, load_config
from src.evaluation.metrics import auc
from src.recommendation.candidate.retriever import search_available
from src.recommendation.candidate.two_tower import make_model as tower
from src.recommendation.ranking.deepfm import make_model as ranker
from src.recommendation.ranking.features import (
    FeatureStore,
    prepare_examples,
    prepare_impression_examples,
    rank_features,
    split_info,
)


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=data_dir())
    p.add_argument("--output", type=Path)
    p.add_argument("--config", type=Path)
    a = p.parse_args()
    cfg = (
        yaml.safe_load(a.config.read_text(encoding="utf-8-sig"))
        if a.config
        else copy.deepcopy(load_config("recommend"))
    )
    out = a.output or ROOT / cfg["paths"]["model_dir"] / cfg.get(
        "version", "stage12-cvr-v1"
    )
    return a.data_dir, Path(out), cfg


def seed(cfg):
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg.get("threads", 4))
    faiss.omp_set_num_threads(cfg.get("threads", 4))
    torch.use_deterministic_algorithms(True)
    return np.random.default_rng(cfg["seed"])


def tensors(data, ix):
    return (
        torch.as_tensor(data["profile"][ix]),
        torch.as_tensor(data["history"][ix]),
        torch.as_tensor(data["length"][ix]),
    )


@torch.inference_mode()
def make_index(model, cfg):
    model.eval()
    n = len(model.cats) - 1
    vectors = np.concatenate(
        [
            model.item(torch.arange(s, min(s + 4096, n + 1))).numpy()
            for s in range(1, n + 1, 4096)
        ]
    )
    c = cfg["candidate"]
    kind = c.get("faiss_index_type", "flat")
    if kind == "flat":
        index = faiss.IndexFlatIP(vectors.shape[1])
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(
            vectors.shape[1], c.get("hnsw_m", 32), faiss.METRIC_INNER_PRODUCT
        )
        index.hnsw.efConstruction = c.get("ef_construction", 200)
        index.hnsw.efSearch = max(c.get("ef_search", 600), c["n_candidates"])
    else:
        raise ValueError("candidate.faiss_index_type must be flat or hnsw")
    index.add(vectors)
    return index, vectors


@torch.inference_mode()
def retrieval_metric(model, index, store, data, ix, k):
    hits = []
    ranks = []
    model.eval()
    for start in range(0, len(ix), 128):
        batch = ix[start : start + 128]
        q = model.user(*tensors(data, batch)).numpy()
        for j, i in enumerate(batch):
            ids, _ = search_available(index, q[j], store.created, data["time"][i], k)
            pos = np.flatnonzero(ids == data["target"][i])
            hits.append(bool(len(pos)))
            ranks.append(1 / np.log2(pos[0] + 2) if len(pos) else 0.0)
    return {
        "queries": len(ix),
        "recall_at_300": float(np.mean(hits)) if hits else None,
        "ndcg_at_300": float(np.mean(ranks)) if ranks else None,
    }


def train_candidate(root, out, cfg):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new version directory")
    out.mkdir(parents=True, exist_ok=True)
    rng = seed(cfg)
    meta, cuts = split_info(root)
    print("Loading official split files and public features", flush=True)
    store = FeatureStore(root / "processed", cfg["candidate"]["history_len"]).fit(cuts[1])
    store.metadata.update(
        selection_cutoff=cuts[2], history_policy="frozen_split",
        history_cutoffs=cuts[1:3], source_meta=meta,
        timestamp_convention="naive source timestamps interpreted as UTC",
        source_events=store.audit["input_events"], data_audit=store.audit,
        objective=cfg["ranking"].get("objective", "cvr"),
    )
    print("Data audit", store.audit, flush=True)
    store.excluded_events.to_parquet(out / "excluded_pre_signup_events.parquet", index=False)
    store.save(out / "features")
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("Preparing strict-past training / frozen holdout histories", flush=True)
    data = prepare_examples(
        store, root, cfg["ranking"].get("cvr_window_seconds", 86400)
    )
    np.savez_compressed(out / "examples.npz", **data)
    train = np.flatnonzero(data["split"] == 0)
    valid = np.flatnonzero(data["split"] == 1)
    limit = cfg.get("validation_limit", 10000)
    if limit and len(valid) > limit:
        valid = np.sort(rng.choice(valid, limit, replace=False))
    if not len(train) or not len(valid):
        raise ValueError("Empty train or validation views")
    print(f"Examples ready: train={len(train)}, validation sample={len(valid)}", flush=True)
    c = cfg["candidate"]
    available_order = np.argsort(store.created[1:], kind="stable") + 1
    available_times = store.created[available_order]
    model = tower(store, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"])
    best = -1.0
    best_state = None
    logs = []
    for epoch in range(c["epochs"]):
        model.train()
        total = 0.0
        order = rng.permutation(train)
        for start in range(0, len(order), c["batch_size"]):
            ix = order[start : start + c["batch_size"]]
            targets = data["target"][ix]
            # Vectorized uniform draws from time-eligible catalog (exclude the target).
            counts = np.searchsorted(available_times, data["time"][ix], side="right")
            if (counts < 2).any():
                raise ValueError("At least 2 available products needed")
            negatives = available_order[rng.integers(0, counts[:, None], size=(len(ix), c["negative_ratio"]))]
            clash = negatives == targets[:, None]
            while clash.any():
                row, col = np.where(clash)
                negatives[row, col] = available_order[rng.integers(0, counts[row])]
                clash = negatives == targets[:, None]
            ids = np.column_stack([targets, negatives])
            q = model.user(*tensors(data, ix))
            logits = (q[:, None] * model.item(torch.as_tensor(ids))).sum(-1) / 0.1
            loss = torch.nn.functional.cross_entropy(
                logits, torch.zeros(len(ix), dtype=torch.long)
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            total += loss.item() * len(ix)
        print(f"TwoTower epoch {epoch + 1}: indexing and validating", flush=True)
        index, _ = make_index(model, cfg)
        metric = retrieval_metric(model, index, store, data, valid, c["n_candidates"])
        logs.append(
            {"epoch": epoch + 1, "train_loss": total / len(train), "valid": metric}
        )
        print("TwoTower", logs[-1], flush=True)
        if metric["recall_at_300"] > best:
            best = metric["recall_at_300"]
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), out / "two_tower.pt")
    (out / "candidate_training.json").write_text(
        json.dumps(logs, indent=2), encoding="utf-8"
    )
    return out


def load_bundle(out):
    out = Path(out)
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    seed(cfg)
    store = FeatureStore.load(out / "features")
    model = tower(store, cfg)
    model.load_state_dict(
        torch.load(out / "two_tower.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    return cfg, store, model


def build_index(out):
    out = Path(out)
    cfg, store, model = load_bundle(out)
    index, vectors = make_index(model, cfg)
    np.save(out / "item_vectors.npy", vectors)
    faiss.serialize_index(index).tofile(out / "items.faiss")
    (out / "index_ids.json").write_text(json.dumps(store.item_ids), encoding="utf-8")
    return index


@torch.inference_mode()
def predict(model, sparse, dense):
    model.eval()
    out = []
    for start in range(0, len(sparse), 4096):
        out.extend(
            model(
                torch.as_tensor(sparse[start : start + 4096]),
                torch.as_tensor(dense[start : start + 4096]),
            )
            .sigmoid()
            .numpy()
        )
    return np.asarray(out)


def ranking_examples(out, store, cfg, root=None):
    """Stage 2 학습·평가 예시 (ranking.objective). 노출 기반은 ranking_examples_<목표>.npz로 저장해 두고 재사용.
    - ctcvr: 노출 후 구매 = 1, 보여줬지만 안 누름 + 눌렀지만 안 삼 = 0 (명세 "노출 후 비클릭" 음성 + 구매 목표)
    - ctr  : 클릭 = 1, 보여줬지만 안 누름 = 0
    - cvr  : 조회 후 24시간 안 구매 = 1 (조회한 상품만, examples.npz)"""
    out = Path(out)
    r = cfg["ranking"]
    objective = r.get("objective", "cvr")
    if objective == "cvr":
        return dict(np.load(out / "examples.npz", allow_pickle=False))
    if objective not in ("ctr", "ctcvr"):
        raise ValueError(f"ranking.objective는 ctcvr / ctr / cvr 중 하나: {objective}")
    path = out / f"ranking_examples_{objective}.npz"
    if not path.exists():
        rng = np.random.default_rng(cfg.get("seed", 42))
        window = r.get("cvr_window_seconds", 86400) if objective == "ctcvr" else None
        data = prepare_impression_examples(
            store, root or data_dir(), r.get("impression_negatives", 4), rng, purchase_window=window
        )
        np.savez_compressed(path, **data)
    return dict(np.load(path, allow_pickle=False))


@torch.inference_mode()
def ranking_inputs(cfg, store, tower, item_vectors, data, ix, chunk=200_000):
    """예시 행 ix → Stage 2 입력 (sparse, dense). 특징 v2면 Two-Tower 취향 점수를 같이 계산한다
    (같은 사용자·시각은 사용자 벡터를 한 번만 계산). 메모리를 아끼려고 chunk 행씩 만든다."""
    version = cfg["ranking"].get("features", 1)
    parts = []
    for start in range(0, len(ix), chunk):
        rows = ix[start:start + chunk]
        affinity = None
        if version >= 2:
            codes, _ = pd.factorize(pd.MultiIndex.from_arrays([data["user"][rows], data["time"][rows]]))
            first = rows[np.unique(codes, return_index=True)[1]]
            q = np.concatenate([tower.user(*tensors(data, first[i:i + 4096])).numpy()
                                for i in range(0, len(first), 4096)])
            affinity = (q[codes] * item_vectors[data["target"][rows] - 1]).sum(1)
        parts.append(rank_features(store, data["target"][rows], data["history"][rows], data["profile"][rows],
                                   data["time"][rows], version, affinity))
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def train_ranking(out, root=None):
    out = Path(out)
    cfg, store, tower = load_bundle(out)
    rng = seed(cfg)
    r = cfg["ranking"]
    objective = r.get("objective", "cvr")
    data = ranking_examples(out, store, cfg, root)
    item_vectors = np.load(out / "item_vectors.npy") if r.get("features", 1) >= 2 else None
    sets = {}
    for j, name in enumerate(["train", "valid"]):
        ix = np.flatnonzero((data["split"] == j) & data["mature"])
        s, d = ranking_inputs(cfg, store, tower, item_vectors, data, ix)
        y = data["label"][ix]
        if len(np.unique(y)) < 2:
            raise ValueError(f"{name}: {objective} needs both positive and negative examples")
        sets[name] = (s, d, y)
    model = ranker(store, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=r["lr"])
    best = -1.0
    state = None
    logs = []
    s, d, y = sets["train"]
    for epoch in range(r["epochs"]):
        model.train()
        order = rng.permutation(len(y))
        loss_sum = 0.0
        for start in range(0, len(order), r["batch_size"]):
            ix = order[start : start + r["batch_size"]]
            logits = model(torch.as_tensor(s[ix]), torch.as_tensor(d[ix]))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, torch.as_tensor(y[ix])
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            loss_sum += loss.item() * len(ix)
        vs, vd, vy = sets["valid"]
        score = auc(vy, predict(model, vs, vd))
        logs.append({"epoch": epoch + 1, "loss": loss_sum / len(y), "valid_auc": score})
        print(objective.upper(), logs[-1], flush=True)
        if score > best:
            best = score
            state = copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    torch.save(model.state_dict(), out / f"ranking_{objective}.pt")
    (out / "ranking_training.json").write_text(
        json.dumps({"objective": objective, "features": r.get("features", 1), "epochs": logs}, indent=2),
        encoding="utf-8",
    )
