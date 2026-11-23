import copy
import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.common.schemas import Event, Impression
from src.ct.monitor import collect_metrics, inspect_metrics
from src.ct.registry import ModelRegistry
from src.ct.trigger import DeferredTraining, job_lock, prepare_snapshot, run_once
from src.serving.event_log import EventLog


def source_and_log(tmp_path, sessions=20):
    source = tmp_path / "source"
    processed = source / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame([{"product_id": "p1", "created_at": "2025-01-01", "price": 10},
                  {"product_id": "p2", "created_at": "2025-01-01", "price": 20}]).to_parquet(processed / "products.parquet")
    pd.DataFrame([{"user_id": "old", "signup_at": "2025-01-01", "age_group": "20s", "gender": "F"}]).to_parquet(processed / "users.parquet")
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ev = Event(event_id="base", user_id="old", session_id="base", event_type="view", product_id="p1", timestamp=base)
    imp = Impression(impression_id="base", user_id="old", session_id="base", timestamp=base,
                     source="home", context="base", product_ids=["p1", "p2"], clicked_ids=["p1"])
    pd.DataFrame([ev.model_dump()]).assign(event_type="view").to_parquet(processed / "events.parquet")
    pd.DataFrame([imp.model_dump()]).to_parquet(processed / "impressions.parquet")
    log = EventLog(tmp_path / "live.sqlite3")
    for i in range(sessions):
        at = datetime(2026, 4, 1, tzinfo=timezone.utc) + timedelta(days=i)
        session = f"live-{i}"
        log.append_impression(Impression(impression_id=session, user_id="new", session_id=session, timestamp=at,
                                        source="home", context="live", product_ids=["p1", "p2"], clicked_ids=[]))
        for j, kind in enumerate(["view", "purchase"], 1):
            log.append_event(Event(event_id=f"{session}-{kind}", user_id="new", session_id=session,
                                   timestamp=at + timedelta(seconds=j), event_type=kind, product_id="p1", impression_id=session))
    return source, log


def config(tmp_path):
    return {"retrain": {"new_log_trigger": 10}, "registry": {"dir": str(tmp_path / "models"),
                                                            "validation": {"recall_at_300": .3, "ranking_auc": .7}}}


REC = {"ranking": {"objective": "ctr"}, "session": {}, "candidate": {}}


def test_snapshot_places_live_data_in_training_and_preserves_sessions(tmp_path):
    source, log = source_and_log(tmp_path)
    dest = tmp_path / "snapshot"
    meta = prepare_snapshot(source, dest, log, log.high_watermark(), REC)
    assert meta["splits"]["train"]["live_events"] == 32
    assert meta["splits"]["valid"]["live_events"] == 4
    assert meta["splits"]["test"]["live_events"] == 4
    groups = [set(pd.read_parquet(dest / f"splits/{name}_events.parquet").session_id) for name in ("train", "valid", "test")]
    assert not groups[0] & groups[1] and not groups[1] & groups[2] and not groups[0] & groups[2]
    profiles = pd.read_parquet(dest / "processed/users.parquet").set_index("user_id")
    assert profiles.loc["new", "gender"] == "unknown"
    assert "users_hidden.parquet" not in {p.name for p in (dest / "processed").iterdir()}


def test_purchase_holdout_waits_for_mature_observations(tmp_path):
    source, log = source_and_log(tmp_path)
    cfg = copy.deepcopy(REC)
    cfg["ranking"] = {"objective": "ctcvr", "cvr_window_seconds": 7 * 86400}
    with pytest.raises(DeferredTraining, match="fully observed"):
        prepare_snapshot(source, tmp_path / "snapshot", log, log.high_watermark(), cfg)
    assert not (tmp_path / "snapshot").exists()


def test_trigger_checkpoints_success_not_failure_and_never_promotes(tmp_path):
    source, log = source_and_log(tmp_path)
    registry = ModelRegistry(tmp_path / "registry")
    calls = []

    def train(snapshot, output, cfg):
        calls.append(snapshot)
        assert pd.read_parquet(snapshot / "splits/train_events.parquet").user_id.eq("new").any()
        return {"recall_at_300": .4, "ranking_auc": .8}

    args = {"log": log, "cfg": config(tmp_path), "recommendation_cfg": REC, "source": source,
            "work_dir": tmp_path / "work", "registry": registry, "trainer": train}
    result = run_once(**args)
    assert result["status"] == "validated" and not result["activated"]
    assert registry.resolve_active() is None
    assert run_once(**args)["status"] == "waiting"
    assert len(calls) == 1
    (tmp_path / "work/trigger_state.json").unlink()

    def fail(*args):
        raise RuntimeError("training interrupted")

    result = run_once(**(args | {"trainer": fail}))
    assert result["status"] == "failed"
    assert not (tmp_path / "work/trigger_state.json").exists()


def test_job_lock_prevents_concurrent_workers(tmp_path):
    with job_lock(tmp_path / "lock") as first:
        assert first
        with job_lock(tmp_path / "lock") as second:
            assert not second


def test_registry_requires_validation_complete_bundle_and_retains_old_models(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    model = tmp_path / "model"
    model.mkdir()
    registry.register("v1", model)
    with pytest.raises(ValueError, match="validated"):
        registry.activate("v1")
    assert not registry.validate("v1", {"auc": .5}, {"auc": .7})
    assert registry.validate("v1", {"auc": .8}, {"auc": .7})
    (model / "config.json").write_text(json.dumps({"ranking": {"objective": "ctr"}}))
    with pytest.raises(ValueError, match="incomplete"):
        registry.activate("v1")
    for f in ["two_tower.pt", "items.faiss", "index_ids.json", "item_vectors.npy", "features/metadata.json", "ranking_ctr.pt"]:
        p = model / f
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"fixture")
    assert registry.activate("v1") == model
    assert registry.resolve_active() == model
    assert registry.retention_candidates(1) == []


def test_monitor_distinguishes_missing_metrics_from_below_threshold(tmp_path):
    log = EventLog(tmp_path / "events.sqlite3")
    values = collect_metrics(log, tmp_path / "missing.json")
    assert values["ctr"] is None and values["hitrate_at_50"] is None
    alerts = inspect_metrics({"ctr": .01, "hitrate_at_50": None}, {"ctr": .02, "hitrate_at_50": .2})
    assert [x["status"] for x in alerts] == ["below_threshold", "unavailable"]


def test_real_training_pipeline_consumes_snapshot_and_writes_separate_session_model(tmp_path):
    from src.common.config import load_config

    source, log = source_and_log(tmp_path)
    products = pd.read_parquet(source / "processed/products.parquet")
    products = pd.concat([products, pd.DataFrame([{"product_id": "p3", "created_at": "2025-01-01", "price": 30}])])
    products.to_parquet(source / "processed/products.parquet", index=False)
    # Add a second view in each session while retaining one unclicked impression item.
    for i in range(20):
        at = datetime(2026, 4, 1, tzinfo=timezone.utc) + timedelta(days=i)
        imp_id = f"second-{i}"
        log.append_impression(Impression(impression_id=imp_id, user_id="new", session_id=f"live-{i}",
                                        timestamp=at + timedelta(seconds=3), source="home", context="test",
                                        product_ids=["p2", "p3"], clicked_ids=[]))
        log.append_event(Event(event_id=imp_id, user_id="new", session_id=f"live-{i}", event_type="view",
                               product_id="p2", timestamp=at + timedelta(seconds=4), impression_id=imp_id))
    rec = copy.deepcopy(load_config("recommend"))
    rec["ranking"].update(objective="ctr", features=1, epochs=1, dnn_hidden=[8], embedding_dim=4, batch_size=32)
    rec["candidate"].update(epochs=1, embedding_dim=4, batch_size=16, history_len=3, n_candidates=3,
                            faiss_index_type="flat", image_features={"enabled": False})
    rec["session"].update(epochs=1, embedding_dim=4, hidden_dim=4, batch_size=16, max_len=3, eval_k=2)
    registry = ModelRegistry(tmp_path / "registry")
    ct = config(tmp_path)
    ct["registry"]["validation"] = {"recall_at_300": 0, "ranking_auc": 0}
    result = run_once(log=log, cfg=ct, recommendation_cfg=rec, source=source,
                      work_dir=tmp_path / "real-work", registry=registry)
    assert result["status"] == "validated", result
    assert not result["activated"] and registry.resolve_active() is None
    assert set(result["metrics"]) == {"recall_at_300", "ranking_auc"}
    output = registry.activate(result["version"])
    assert registry.resolve_active() == output
    assert (output / "session_encoder.pt").is_file()
    assert (output / "ranking_ctr.pt").is_file()
    assert (output / "metrics.json").is_file()
    saved = json.loads((output / "config.json").read_text())
    assert saved["session"]["model_file"] == str(output / "session_encoder.pt")


def test_monitor_reads_latest_matching_view_result_from_combined_evaluation(tmp_path, monkeypatch):
    from src.ct.monitor import monitor_once

    log = EventLog(tmp_path / "events.sqlite3")
    current = {"k": 50, "target_type": "view", "model_version": "current",
               "measured_at": "2020-01-01T00:00:00Z", "multistage": {"hit_rate": .21}}
    wrong = current | {"model_version": "other", "measured_at": "2026-11-01T00:00:00Z"}
    (tmp_path / "recommend_final_old.json").write_text(json.dumps(current | {"measured_at": "2019-01-01"}))
    (tmp_path / "recommend_final_both.json").write_text(json.dumps({"view": current, "purchase": wrong}))
    (tmp_path / "recommend_final_wrong.json").write_text(json.dumps(wrong))
    (tmp_path / "recommend_final_purchase.json").write_text(json.dumps(wrong | {"target_type": "purchase"}))
    pattern = tmp_path / "recommend_final*.json"
    values = collect_metrics(log, pattern, "current")
    assert values["hitrate_at_50"] == .21
    assert values["evaluation_path"].endswith("recommend_final_both.json")
    monkeypatch.setenv("MODEL_VERSION", "absent")
    monkeypatch.delenv("REC_MODEL_DIR", raising=False)
    result = monitor_once(log, {"monitor": {"evaluation_path": str(pattern), "thresholds": {"hitrate_at_50": .2},
                                          "max_evaluation_age_hours": 1}}, tmp_path / "monitor")
    assert {x["status"] for x in result["alerts"]} == {"unavailable", "model_version_mismatch", "stale"}
    assert result["metrics"]["hitrate_at_50"] is None
