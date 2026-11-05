"""최종 평가(eval_pipeline)의 세션 매칭: 같은 방문에서 그 조회보다 먼저 본 상품만."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import eval_pipeline


def test_session_prefixes_only_earlier_views_in_same_session(tmp_path, monkeypatch):
    (tmp_path / "splits").mkdir()
    t = pd.Timestamp("2026-03-24 10:00:00")
    sec = pd.Timedelta(seconds=1)
    pd.DataFrame({
        "user_id": ["U1", "U1", "U1", "U1", "U1"],
        "session_id": ["S1", "S1", "S1", "S1", "S2"],
        "timestamp": [t, t + sec, t + 2 * sec, t + 3 * sec, t + 4 * sec],
        "event_type": ["view", "search", "view", "view", "view"],
        "product_id": ["A", None, "B", "C", "D"],
    }).to_parquet(tmp_path / "splits" / "test_events.parquet")
    monkeypatch.setattr(eval_pipeline, "data_dir", lambda: tmp_path)

    base = t.timestamp()
    users = np.array(["U1", "U1", "U1", "U2"])
    times = np.array([base, base + 3, base + 4, base])
    assert eval_pipeline.session_prefixes(users, times) == [[], ["A", "B"], [], []]


def test_run_compares_multistage_with_both_baselines(monkeypatch):
    from types import SimpleNamespace

    q = SimpleNamespace(k=50, users=np.array(["U1"]), sessions=[[]])
    values = {"multistage": 0.3, "two_tower_only": 0.2, "popularity": 0.1}
    calls = []

    def fake_eval(strategy, q, use_session, name=""):
        calls.append((name, type(strategy).__name__, use_session))
        return dict.fromkeys(("hit_rate", "ndcg", "coverage"), values[name])

    monkeypatch.setattr(eval_pipeline, "load_queries", lambda limit: q)
    monkeypatch.setattr(eval_pipeline, "evaluate_strategy", fake_eval)

    result = eval_pipeline.run(["multistage", "two_tower_only", "popularity"])
    assert calls == [("multistage", "MultiStageStrategy", True), ("two_tower_only", "TwoTowerOnlyStrategy", False),
                     ("popularity", "PopularityStrategy", False)]
    assert result["multistage"]["lift_vs_two_tower_only"]["hit_rate"] == pytest.approx(0.5)
    assert result["multistage"]["lift_vs_popularity"]["hit_rate"] == pytest.approx(2.0)
    assert "lift_vs_popularity" not in result["two_tower_only"]  # 베이스라인끼리는 비교 안 함


def test_tradeoff_changes_only_exploration_slots_and_restores(monkeypatch):
    from types import SimpleNamespace

    from src.evaluation import tradeoff_exploration as tx
    from src.recommendation.reranking import RerankConfig

    base = RerankConfig.from_recommend_config()
    explorer = object()
    svc = SimpleNamespace(reranker=SimpleNamespace(config=base, explorer=explorer))
    seen = []

    def fake_eval(strategy, q, use_session, name=""):
        cfg = svc.reranker.config
        seen.append((cfg.exploration_slots, cfg.exploration_enabled, cfg.new_item_slots, use_session))
        assert svc.reranker.explorer is not explorer  # 개수마다 새 MAB 통계
        return {"hit_rate": 0.1, "ndcg": 0.05, "coverage": 0.2, "latency_p95_ms": 1.0, "passed": {}}

    monkeypatch.setattr(tx, "load_queries", lambda sample: SimpleNamespace(users=np.array(["U1"]), k=50))
    monkeypatch.setattr(tx, "get_recommendation_models", lambda: None)
    monkeypatch.setattr(tx, "get_rerank_service", lambda models: svc)
    monkeypatch.setattr(tx, "evaluate_strategy", fake_eval)

    result = tx.run([0, 1, 2, 3], sample=10)
    assert seen == [(n, n > 0, base.new_item_slots, True) for n in range(4)]
    assert [r["in_spec_range"] for r in result["rows"]] == [False, True, True, False]
    assert svc.reranker.config is base and svc.reranker.explorer is explorer
