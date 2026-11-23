"""Monitor measured HitRate and durable displayed-impression CTR.

Alerts are written locally and printed, never sent to an external service.
Missing/stale measurements are reported rather than silently treated as zero.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.config import ROOT, artifact_dir, env, load_config
from src.recommendation.runtime_config import serving_version
from src.serving.event_log import EventLog, get_event_log


def inspect_metrics(metrics: dict[str, Any], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    alerts = []
    for key, threshold in thresholds.items():
        value = metrics.get(key)
        if value is None or not isinstance(value, (float, int)) or not math.isfinite(value):
            alerts.append({"metric": key, "status": "unavailable", "value": None, "threshold": threshold})
        elif value < threshold:
            alerts.append({"metric": key, "status": "below_threshold", "value": value, "threshold": threshold})
    return alerts


def _evaluation_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        at = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return at.replace(tzinfo=timezone.utc) if at.tzinfo is None else at.astimezone(timezone.utc)


def collect_metrics(log: EventLog, evaluation_path: Path, expected_version: str | None = None) -> dict[str, Any]:
    metrics = log.metrics()
    metrics.update(hitrate_at_50=None, model_version=None, evaluation_measured_at=None,
                   evaluation_path=None, evaluation_read_errors=[])
    results = []
    for path in map(Path, glob.glob(str(evaluation_path))):
        try:
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                raise TypeError("Evaluation must be a JSON object")
            result = payload.get("view", payload)
            kind = result.get("target_type", "view") if isinstance(result, dict) else None
            if kind not in ("user", "view"):  # user = 사용자 단위 주 지표 (2026-11-18), view = 조회 단위 보조 지표
                continue
            if result.get("k") != 50 or "multistage" not in result:
                continue
            at = _evaluation_time(result.get("measured_at"))
            results.append((kind == "user", at.timestamp() if at else path.stat().st_mtime, path, result))
        except (OSError, ValueError, TypeError) as exc:
            metrics["evaluation_read_errors"].append({"path": str(path), "reason": str(exc)})
    matching = [row for row in results if row[3].get("model_version") == expected_version]
    candidates = matching or results
    if candidates:
        _, _, path, result = max(candidates, key=lambda row: (row[0], row[1], str(row[2])))
        metrics.update(model_version=result.get("model_version"), evaluation_path=str(path),
                       evaluation_measured_at=result.get("measured_at"))
        if expected_version is None or result.get("model_version") == expected_version:
            metrics["hitrate_at_50"] = result.get("multistage", {}).get("hit_rate")
    return metrics


def monitor_once(log: EventLog | None = None, cfg: dict[str, Any] | None = None,
                 output: Path | None = None) -> dict[str, Any]:
    cfg = cfg or load_config("ct")
    m = cfg["monitor"]
    path = ROOT / m.get("evaluation_path", "artifacts/metrics/recommend_final*.json")
    expected = env("MODEL_VERSION") or serving_version(load_config("recommend"))
    override = env("REC_MODEL_DIR")
    if override:
        expected = Path(override).name
    elif expected in {"latest", "active"}:
        from src.ct.registry import resolve_model_version
        active = resolve_model_version(expected)
        expected = active.name if active else serving_version(load_config("recommend"))
    metrics = collect_metrics(log or get_event_log(), path, expected)
    alerts = inspect_metrics(metrics, m["thresholds"])
    if metrics["model_version"] and metrics["model_version"] != expected:
        alerts.append({"metric": "evaluation", "status": "model_version_mismatch",
                       "value": metrics["model_version"], "expected": expected})
    stamp = metrics["evaluation_measured_at"]
    at = _evaluation_time(stamp)
    if at and m.get("max_evaluation_age_hours"):
        if (datetime.now(timezone.utc) - at).total_seconds() > m["max_evaluation_age_hours"] * 3600:
            alerts.append({"metric": "evaluation", "status": "stale", "value": stamp})
    elif at is None and metrics["evaluation_path"]:
        alerts.append({"metric": "evaluation", "status": "timestamp_unavailable", "value": stamp})
    if metrics["evaluation_read_errors"]:
        alerts.append({"metric": "evaluation", "status": "invalid_artifact",
                       "value": metrics["evaluation_read_errors"]})
    result = {"measured_at": datetime.now(timezone.utc).isoformat(), "metrics": metrics, "alerts": alerts,
              "expected_model_version": expected,
              "definitions": {"ctr": "unique clicked (impression,item) / displayed item exposures",
                              "cvr": "exposed users purchasing an attributed displayed item / exposed users",
                              "online_scope": "all persisted displayed impressions, across model versions"}}
    folder = output or artifact_dir() / "ct"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "monitor_latest.json"
    temporary = target.with_suffix(f".{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(target)
    if alerts:
        with (folder / "alerts.jsonl").open("a") as f:
            f.write(json.dumps({"measured_at": result["measured_at"], "alerts": alerts}, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    cfg = load_config("ct")
    while True:
        try:
            monitor_once(cfg=cfg)
        except (OSError, ValueError) as exc:
            print(f"[monitor] measurement failed: {exc}", flush=True)
            if args.once:
                raise
        if args.once:
            break
        time.sleep(float(cfg["monitor"]["interval_sec"]))


if __name__ == "__main__":
    main()
