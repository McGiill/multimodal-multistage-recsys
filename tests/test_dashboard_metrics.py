import importlib
import json
from pathlib import Path

import pytest

from dashboard.metrics import recommendation_rows, recommendation_runs


def test_new_and_existing_recommendation_artifacts():
    old = {"k": 50, "multistage": {"hit_rate": .1, "ndcg": .03, "coverage": .2, "latency_p95_ms": 40}}
    assert list(recommendation_runs(old)) == ["평가 결과"]
    assert recommendation_rows(old)[0]["전략"] == "multistage"
    new = {"view": old, "purchase": {"k": 50}}
    assert list(recommendation_runs(new)) == ["view", "purchase"]
    assert recommendation_rows(new["purchase"]) == []


@pytest.mark.parametrize("page,filename,payload", [
    ("1_검색품질.py", "search_fixture.json", {"model_name": "fixture", "k": 10, "results": {
        "all": {"n_queries": 10, "mrr": .6, "ndcg@10": .5},
        "clip_only_all": {"n_queries": 10, "mrr": .4, "ndcg@10": .3}}}),
    ("2_추천성능.py", "recommend_final_fixture.json", {"view": {"k": 50, "n_queries": 20,
        "multistage": {"hit_rate": .1, "ndcg": .03, "coverage": .2, "latency_p95_ms": 40}}, "purchase": {"k": 50}}),
    ("3_AB테스트.py", "abtest_fixture.json", {"source": "simulation", "groups": {
        "control": {"assigned_users": 10, "exposed_users": 8, "clicked_users": 4, "purchased_users": 1,
                    "assigned_user_cvr": .1, "user_ctr": .5, "user_cvr": .125, "item_ctr": .05},
        "treatment": {"assigned_users": 10, "exposed_users": 8, "clicked_users": 5, "purchased_users": 2,
                      "assigned_user_cvr": .2, "user_ctr": .625, "user_cvr": .25, "item_ctr": .06}},
        "comparisons": {"user_cvr": {"status": "ok", "diff": .125, "ci_low": -.1, "ci_high": .35,
                                      "control_rate": .125, "treatment_rate": .25, "p_value": .4,
                                      "lift": 1, "significant": False, "small_counts": True}}}),
])
def test_dashboard_renders_artifacts_and_empty_state(tmp_path, monkeypatch, page, filename, payload):
    from streamlit.testing.v1 import AppTest

    dashboard = Path(__file__).resolve().parents[1] / "dashboard"
    monkeypatch.syspath_prepend(str(dashboard))
    metrics = importlib.import_module("metrics")
    monkeypatch.setattr(metrics, "ARTIFACTS", tmp_path)
    folder = tmp_path / "metrics"
    folder.mkdir()
    empty = AppTest.from_file(str(dashboard / "pages" / page)).run(timeout=15)
    assert not empty.exception
    assert empty.info
    (folder / filename).write_text(json.dumps(payload), encoding="utf-8")
    full = AppTest.from_file(str(dashboard / "pages" / page)).run(timeout=15)
    assert not full.exception
    assert full.dataframe
