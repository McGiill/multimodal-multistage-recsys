"""통합 검증: 공개 입력, 시점 누수, 구매 귀속, 신규 상품, 재로딩, 실제 API."""

import copy
import shutil

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from scripts.stage12_smoke import create_data
from src.common.config import load_config
from src.recommendation.ranking.features import (
    purchase_labels,
    rank_features,
    stamp,
)
from src.recommendation.training import (
    build_index,
    make_index,
    predict,
    train_candidate,
    train_ranking,
)
from src.serving.model_loader import RecommendationModels, load_models


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("rec")
    data = root / "data"
    create_data(data)
    cfg = copy.deepcopy(load_config("recommend"))
    cfg["threads"] = 2
    cfg["candidate"].update(
        embedding_dim=16,
        history_len=5,
        epochs=1,
        batch_size=128,
        faiss_index_type="flat",
    )
    cfg["ranking"].update(
        embedding_dim=8, dnn_hidden=[32, 16], dropout=0.0, epochs=1, batch_size=128
    )
    bundle = root / "bundle"
    train_candidate(data, bundle, cfg)
    build_index(bundle)
    train_ranking(bundle)
    return data, bundle, RecommendationModels(bundle)


def test_public_input_whitelist_and_unknown_user(trained):
    _, _, models = trained
    assert models.store.users.columns.tolist() == [
        "user_id",
        "age_group",
        "gender",
        "signup_at",
    ]
    assert "persona" not in models.store.metadata["public_user_fields"]
    h, l, p = models.store.state("unknown", stamp("2026-01-29T00:00:00Z"))
    assert not h.any() and l == 1 and p[2] == 1
    # Malformed users_hidden file exists, yet training succeeded without reading it.


def test_current_and_future_events_not_in_history(trained):
    _, _, models = trained
    store = models.store
    ev = store.events[store.events.event_type.eq("view")].iloc[100]
    h, _length, p = store.state(ev.user_id, ev["_time"])
    prior = store.events[
        (store.events.user_id == ev.user_id)
        & (store.events.event_type == "view")
        & (store.events["_time"] < ev["_time"])
    ]["_pid"].to_numpy()[-store.history_len :]
    np.testing.assert_array_equal(h[: len(prior)], prior)
    assert not h[len(prior) :].any()
    assert p[3] == pytest.approx(
        min(
            np.log1p(
                len(
                    store.events[
                        (store.events.user_id == ev.user_id)
                        & (store.events.event_type == "cart")
                        & (store.events["_time"] < ev["_time"])
                    ]
                )
            )
            / 5,
            1,
        )
    )


def test_labels_exact_timestamp_last_view_and_window():
    v = pd.DataFrame(
        {"user_id": ["u"] * 3, "product_id": ["a", "a", "b"], "_time": [1.0, 4.0, 8.0]}
    )
    buy = pd.DataFrame(
        {
            "user_id": ["u"] * 4,
            "product_id": ["a", "b", "a", "z"],
            "_time": [5.0, 8.0, 30.0, 9.0],
        }
    )
    np.testing.assert_array_equal(purchase_labels(v, buy, 10), [0, 1, 0])


def test_maturity_and_feature_parity(trained):
    _, bundle, models = trained
    data = dict(np.load(bundle / "examples.npz", allow_pickle=False))
    i = int(np.flatnonzero(data["split"] == 2)[0])
    at = data["time"][i]
    h, length, p = models.store.state(data["user"][i], at)
    assert length == data["length"][i]
    np.testing.assert_array_equal(h, data["history"][i])
    np.testing.assert_allclose(p, data["profile"][i])
    ids, _, state = models.retriever.retrieve(data["user"][i], 300, at)
    s, d = rank_features(
        models.store,
        ids,
        np.tile(h, (len(ids), 1)),
        np.tile(p, (len(ids), 1)),
        np.full(len(ids), at),
    )
    np.testing.assert_allclose(
        predict(models.ranker, s, d), models.score(ids, state, at), atol=1e-7
    )
    assert not data["mature"][data["time"] >= stamp("2026-01-30T00:00:00Z")].any()


def test_future_products_filtered_and_hnsw_supported(trained):
    _, _, models = trained
    at = stamp("2026-01-28T12:00:00Z")
    ids, _, _ = models.retriever.retrieve("U00001", 1000, at)
    assert len(ids) == 550 and np.all(models.store.created[ids] <= at)
    cfg = copy.deepcopy(models.cfg)
    cfg["candidate"]["faiss_index_type"] = "hnsw"
    index, _ = make_index(models.tower, cfg)
    from src.recommendation.candidate.retriever import Retriever

    ids, _, _ = Retriever(models.tower, index, models.store).retrieve("U00001", 300, at)
    assert len(ids) == 300 and np.all(models.store.created[ids] <= at)


def test_model_loading_cached_and_id_mapping_checked(trained, tmp_path):
    _, bundle, _ = trained
    assert load_models(str(bundle)) is load_models(str(bundle))
    other = tmp_path / "copy"
    shutil.copytree(bundle, other)
    (other / "index_ids.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="ID mismatch"):
        RecommendationModels(other)


def test_future_signup_unknown_and_price_used(trained):
    _, _, models = trained
    at = stamp("2026-01-01T12:00:00Z")
    h, _length, p = models.store.state("U00023", at)
    assert not h.any() and np.count_nonzero(p) == 1
    # Changing price changes the item representation without mutating shared model.
    model = copy.deepcopy(models.tower)
    ids = torch.tensor([1])
    before = model.item(ids).detach()
    with torch.no_grad():
        model.numeric[1, 0] += 1
    assert not torch.allclose(before, model.item(ids))


def test_api_real_models_and_time_validation(trained, monkeypatch):
    _, bundle, _ = trained
    monkeypatch.setenv("REC_MODEL_DIR", str(bundle))
    monkeypatch.setenv("PRELOAD_MODELS", "0")
    from src.serving.main import app

    with TestClient(app) as client:
        args = {"user_id": "U00001", "top_n": 10, "at": "2026-01-28T12:00:00Z"}
        response = client.get("/api/recommend", params=args)
        assert response.status_code == 200, response.text
        body = response.json()
        assert len(body["recommendations"]) == 10
        assert all(int(r["product_id"][1:]) < 550 for r in body["recommendations"])
        scores = [r["score"] for r in body["recommendations"]]
        assert scores == sorted(scores, reverse=True) and all(
            0 <= p <= 1 for p in scores
        )
        assert all(
            r["reason"] == "cvr" and not r["is_exploration"]
            for r in body["recommendations"]
        )
        assert body["pipeline_latency"]["reranking_ms"] == 0
        assert (
            client.get(
                "/api/recommend", params={**args, "at": "2026-01-01T00:00:00Z"}
            ).status_code
            == 400
        )
        assert (
            client.get(
                "/api/recommend", params={**args, "at": "2026-01-28T12:00:00"}
            ).status_code
            == 400
        )
        assert (
            client.get("/api/recommend", params={**args, "top_n": 0}).status_code == 422
        )
        assert (
            client.get(
                "/api/recommend", params={**args, "user_id": "not_registered"}
            ).status_code
            == 200
        )


def test_holdout_history_frozen_and_test_events_not_saved(trained):
    source, _, models = trained
    store = models.store
    valid_end = stamp("2026-01-28T00:00:00Z")
    assert (store.events["_time"] < valid_end).all()
    assert (store.events["_split"] < 2).all()
    test = pd.read_parquet(source / "splits/test_events.parquet")
    assert not store.events.event_id.isin(test.event_id).any()
    user = "U00001"
    h1, _, _ = store.state(user, stamp("2026-01-28T12:00:00Z"))
    h2, _, _ = store.state(user, stamp("2026-01-30T12:00:00Z"))
    np.testing.assert_array_equal(h1, h2)
    history = store.events[(store.events.user_id == user) & store.events.event_type.eq("view")]
    expected = history["_pid"].to_numpy()[-store.history_len:]
    np.testing.assert_array_equal(h1[:len(expected)], expected)


def test_faiss_unicode_path(tmp_path):
    import faiss

    from src.common.faiss_utils import load_index, save_index

    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1.0, 0.0]], np.float32))
    path = tmp_path / "한글 인덱스.faiss"
    save_index(index, path)
    loaded = load_index(path)
    assert loaded.ntotal == 1


def test_pre_signup_events_excluded_with_audit(tmp_path):
    from src.recommendation.ranking.features import FeatureStore

    create_data(tmp_path)
    file = tmp_path / "splits/train_events.parquet"
    events = pd.read_parquet(file)
    bad = events.iloc[[0]].copy()
    bad["event_id"] = "bad_before_signup"
    bad["timestamp"] = pd.Timestamp("2024-01-01")
    pd.concat([events, bad]).to_parquet(file, index=False)
    store = FeatureStore(tmp_path / "processed")
    assert store.audit["pre_signup_events_excluded"] == 1
    assert store.audit["excluded_by_split"] == {"0": 1}
    assert store.excluded_events.event_id.tolist() == ["bad_before_signup"]
    assert not store.events.event_id.eq("bad_before_signup").any()


def test_valid_history_uses_train_only(trained):
    _, _, models = trained
    store = models.store
    user = "U00001"
    h1, _, _ = store.state(user, stamp("2026-01-25T12:00:00Z"))
    h2, _, _ = store.state(user, stamp("2026-01-27T12:00:00Z"))
    np.testing.assert_array_equal(h1, h2)
    prior = store.events[
        (store.events.user_id == user) & store.events.event_type.eq("view")
        & (store.events["_time"] < stamp("2026-01-25T00:00:00Z"))
    ]["_pid"].to_numpy()[-store.history_len:]
    np.testing.assert_array_equal(h2[:len(prior)], prior)
