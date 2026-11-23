"""Retrain on accumulated durable logs, with isolated chronological snapshots.

``--once`` checks the threshold once; ``--prepare-only`` writes a reviewable job
without training. A process lock prevents concurrent jobs. A failed/deferred job
never advances the watermark. Successful training never changes the active model.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import ROOT, artifact_dir, data_dir, load_config
from src.ct.registry import ModelRegistry
from src.serving.event_log import EventLog, get_event_log


class DeferredTraining(ValueError):
    """Data exists, but chronological, fully observed holdouts are not ready."""


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


@contextmanager
def job_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def prepare_snapshot(source: Path, destination: Path, log: EventLog, watermark: int,
                     cfg: dict[str, Any]) -> dict[str, Any]:
    """Original data becomes history; fresh sessions are split chronologically 80/10/10.

    The latest data must actually reach training. Keeping the old March holdout
    dates would put all new November events in test and never teach the model.
    Whole sessions stay together, and exact cutoffs exclude spillover leakage.
    """
    if destination.exists():
        raise FileExistsError("Snapshot already exists; use a new version")
    live, live_imp = log.export_snapshot(watermark)
    if live.empty:
        raise DeferredTraining("No new events")
    products = pd.read_parquet(source / "processed/products.parquet")
    users = pd.read_parquet(source / "processed/users.parquet")
    valid_products = set(products.product_id)
    keep = live.event_type.eq("search") | live.product_id.isin(valid_products)
    unknown_products = int((~keep).sum())
    live = live[keep].copy()
    if live.empty:
        raise DeferredTraining("No events referring to the current catalog")
    for i, row in live_imp.iterrows():
        live_imp.at[i, "product_ids"] = [p for p in row.product_ids if p in valid_products]
        live_imp.at[i, "clicked_ids"] = [p for p in row.clicked_ids if p in valid_products]
    live_imp = live_imp[live_imp.product_ids.map(len) > 0].copy()
    first = pd.concat([live[["session_id", "timestamp"]], live_imp[["session_id", "timestamp"]]])
    starts = first.groupby("session_id").timestamp.min().sort_values(kind="stable")
    if len(starts) < 10:
        raise DeferredTraining("At least 10 new sessions are needed for 80/10/10 holdouts")
    train_end, valid_end = starts.iloc[int(len(starts) * .8)], starts.iloc[int(len(starts) * .9)]
    end = max(live.timestamp.max(), live_imp.timestamp.max() if not live_imp.empty else live.timestamp.max())
    end += pd.Timedelta(microseconds=1)
    window = cfg["ranking"].get("cvr_window_seconds", 86400) if cfg["ranking"].get("objective") in {"cvr", "ctcvr"} else 0
    if train_end >= valid_end or (valid_end - train_end).total_seconds() <= window or (end - valid_end).total_seconds() <= window:
        raise DeferredTraining("Fresh validation/test periods need distinct sessions and fully observed purchase windows")
    base = pd.read_parquet(source / "processed/events.parquet")
    base_imp = pd.read_parquet(source / "processed/impressions.parquet")
    for frame in (base, base_imp):
        frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True).dt.tz_localize(None)
    events = pd.concat([base, live], ignore_index=True).drop_duplicates("event_id", keep="last")
    impressions = pd.concat([base_imp, live_imp], ignore_index=True).drop_duplicates("impression_id", keep="last")
    # Online visitors may not exist in the synthetic cohort. Do not infer private demographics.
    records = pd.concat([events[["user_id", "timestamp"]], impressions[["user_id", "timestamp"]]])
    missing = set(records.user_id) - set(users.user_id)
    if missing:
        public = [{"user_id": u, "age_group": "unknown", "gender": "unknown",
                   "signup_at": records.loc[records.user_id.eq(u), "timestamp"].min()} for u in sorted(missing)]
        users = pd.concat([users, pd.DataFrame(public)], ignore_index=True)
    users["signup_at"] = pd.to_datetime(users.signup_at, utc=True).dt.tz_localize(None)
    all_first = pd.concat([events[["session_id", "timestamp"]], impressions[["session_id", "timestamp"]]])
    all_starts = all_first.groupby("session_id").timestamp.min()
    labels = all_starts.map(lambda t: "train" if t < train_end else "valid" if t < valid_end else "test")
    start = min(events.timestamp.min(), impressions.timestamp.min())
    cuts = [start, train_end, valid_end, end]
    meta: dict[str, Any] = {"ratios": {"train": .8, "valid": .1, "test": .1},
                           "unit": "new sessions chronologically; original data retained as history",
                           "cutoffs": [t.tz_localize("UTC").timestamp() for t in cuts],
                           "live_watermark": watermark, "live_events": len(live),
                           "unknown_product_events_excluded": unknown_products, "splits": {}}
    parts = {}
    for i, name in enumerate(("train", "valid", "test")):
        ev = events[events.session_id.map(labels).eq(name)].copy()
        im = impressions[impressions.session_id.map(labels).eq(name)].copy()
        live_rows = live[live.session_id.map(labels).eq(name)]
        if not ev.event_type.eq("view").any():
            raise DeferredTraining(f"{name} has no views")
        if cfg["ranking"].get("objective") == "ctcvr":
            mature = im.timestamp + pd.Timedelta(seconds=window) < cuts[i + 1]
            if not mature.any() or not ev.event_type.eq("purchase").any():
                raise DeferredTraining(f"{name} needs mature impressions and purchases")
        parts[name] = ev, im
        meta["splits"][name] = {"from": cuts[i].isoformat(), "to": cuts[i + 1].isoformat(),
                                "events": len(ev), "impressions": len(im), "live_events": len(live_rows)}
    if meta["splits"]["train"]["live_events"] == 0:
        raise DeferredTraining("No fresh observations reach training")
    (destination / "processed").mkdir(parents=True)
    (destination / "splits").mkdir()
    products.to_parquet(destination / "processed/products.parquet", index=False)
    users.to_parquet(destination / "processed/users.parquet", index=False)
    events.to_parquet(destination / "processed/events.parquet", index=False)
    impressions.to_parquet(destination / "processed/impressions.parquet", index=False)
    for name, (ev, im) in parts.items():
        ev.to_parquet(destination / f"splits/{name}_events.parquet", index=False)
        im.to_parquet(destination / f"splits/{name}_impressions.parquet", index=False)
    write_json(destination / "splits/meta.json", meta)
    return meta


def train_snapshot(snapshot: Path, model_dir: Path, cfg: dict[str, Any]) -> dict[str, float]:
    # Imports are deferred: monitoring/threshold checks do not load torch or FAISS.
    from src.evaluation.eval_recommend import evaluate
    from src.recommendation.session.encoder import SessionConfig
    from src.recommendation.session.train import fit, view_sessions
    from src.recommendation.training import build_index, train_candidate, train_ranking

    train_candidate(snapshot, model_dir, cfg)
    build_index(model_dir)
    train_ranking(model_dir, snapshot)
    session_cfg = SessionConfig.from_recommend_config(cfg)
    model, metrics = fit(
        view_sessions(pd.read_parquet(snapshot / "splits/train_events.parquet"), session_cfg.min_len),
        view_sessions(pd.read_parquet(snapshot / "splits/valid_events.parquet"), session_cfg.min_len),
        pd.read_parquet(snapshot / "processed/products.parquet").product_id.tolist(), session_cfg,
    )
    model.save(model_dir / "session_encoder.pt")
    write_json(model_dir / "session_metrics.json", metrics)
    metrics = evaluate(model_dir)
    return {"recall_at_300": metrics["retrieval"]["recall_at_300"], "ranking_auc": metrics["ranking"]["auc"]}


def run_once(log: EventLog | None = None, cfg: dict[str, Any] | None = None,
             recommendation_cfg: dict[str, Any] | None = None, source: Path | None = None,
             work_dir: Path | None = None, registry: ModelRegistry | None = None,
             trainer: Callable = train_snapshot, prepare_only: bool = False) -> dict[str, Any]:
    cfg = cfg or load_config("ct")
    rec = copy.deepcopy(recommendation_cfg or load_config("recommend"))
    log = log or get_event_log()
    work = work_dir or artifact_dir() / "ct"
    work.mkdir(parents=True, exist_ok=True)
    with job_lock(work / "trigger.lock") as acquired:
        if not acquired:
            return {"status": "busy"}
        checkpoint = work / "trigger_state.json"
        state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {"watermark": 0}
        count = log.count_since(int(state["watermark"]))
        if count < int(cfg["retrain"]["new_log_trigger"]):
            return {"status": "waiting", "new_events": count, "watermark": state["watermark"]}
        end = log.high_watermark()
        version = "ct-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        snapshot = work / "snapshots" / version
        model_dir = ROOT / cfg["registry"]["dir"] / version
        rec["version"] = version
        rec["session"]["model_file"] = str(model_dir / "session_encoder.pt")
        job: dict[str, Any] = {"version": version, "watermark": end, "previous_watermark": state["watermark"],
                               "new_events": count, "snapshot": str(snapshot), "model_dir": str(model_dir)}
        try:
            job["snapshot_meta"] = prepare_snapshot(source or data_dir(), snapshot, log, end, rec)
            write_json(snapshot / "recommend_config.json", rec)
            if prepare_only:
                job["status"] = "prepared"
            else:
                metrics = trainer(snapshot, model_dir, rec)
                registry = registry or ModelRegistry(ROOT / cfg["registry"]["dir"] / "registry")
                registry.register(version, model_dir, metrics=metrics, data_watermark=end, snapshot=snapshot)
                passed = registry.validate(version, metrics, cfg["registry"]["validation"])
                job.update(status="validated" if passed else "rejected", metrics=metrics, activated=False)
                write_json(checkpoint, {"watermark": end, "version": version, "updated_at": datetime.now(timezone.utc).isoformat()})
        except DeferredTraining as exc:
            job.update(status="deferred", reason=str(exc))
        except Exception as exc:  # noqa: BLE001 - preserve job failure/checkpoint on any training backend error
            job.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        write_json(work / "jobs" / f"{version}.json", job)
        write_json(work / "trigger_latest.json", job)
        return job


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--once", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    cfg = load_config("ct")
    while True:
        result = run_once(cfg=cfg, prepare_only=args.prepare_only)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if args.once or args.prepare_only:
            if result["status"] == "failed":
                raise SystemExit(1)
            return
        time.sleep(float(cfg["retrain"].get("poll_interval_sec", 60)))


if __name__ == "__main__":
    main()
