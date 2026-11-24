"""추천 경계 검증: 실제 후보 검색/세션 결합/재정렬과 오프라인 정답의 시간 규칙."""

import copy
import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from types import SimpleNamespace

import faiss
import numpy as np
import pandas as pd
import pytest
import torch

from src.common.config import load_config
from src.evaluation import eval_pipeline
from src.recommendation import pipeline
from src.recommendation.candidate.retriever import Retriever
from src.recommendation.reranking.popularity import PopularityConfig, PopularityLists
from src.recommendation.reranking.reranker import RerankConfig
from src.recommendation.reranking.stage2 import RerankService
from src.recommendation.runtime_config import resolve_runtime_config
from src.recommendation.session.blending import blend_final_scores, percentile_scores
from src.recommendation.session.encoder import (
    ItemIndex,
    SessionConfig,
    SessionEncoder,
    SessionModel,
)

NOW = datetime(2026, 3, 31, 12, tzinfo=timezone.utc)
NOW_TS = NOW.timestamp()
BASE = pd.Timestamp("2026-03-01", tz="UTC")


@pytest.fixture
def pipeline_bundle(monkeypatch):
    """학습 가중치 경계만 고정하고 FAISS·세션·재정렬은 실제 구현을 함께 실행."""
    def build(*, cold=False, new_slots=0, n_candidates=9):
        items = [f"P{i}" for i in range(1, 10)]
        created = [NOW_TS - 30 * 86400] * 6 + [NOW_TS - 86400] * 2 + [NOW_TS + 86400]
        products = pd.DataFrame({
            "product_id": items,
            "category_l1": ["Ladieswear", "Divided", "Menswear", "Menswear", "Menswear",
                            "Sport", "Ladieswear", "Menswear", "Menswear"],
            "category_l2": ["Tops", "Tops", "Tops", "Tops", "Denim", "Sports", "Tops", "Denim", "Denim"],
            "category_l3": ["Detail"] * 9,
            "created_at": pd.to_datetime(created, unit="s", utc=True),
        })
        history = np.array([0, 0, 0, 0, 0] if cold else [3, 4, 3, 4, 3], dtype=np.int64)

        def append_recent(h, length, recent):
            values = np.r_[h[h > 0], recent][-len(h):]
            out = np.zeros_like(h)
            out[:len(values)] = values
            return out, max(len(values), 1)

        store = SimpleNamespace(
            item_ids=items, pmap={p: i + 1 for i, p in enumerate(items)}, products=products,
            created=np.r_[0, created], metadata={"selection_cutoff": NOW_TS - 86400},
            state=lambda user, at: (history.copy(), max(np.count_nonzero(history), 1), np.array([0], np.int64)),
            append_recent=append_recent,
        )
        index = faiss.IndexFlatIP(2)
        index.add(np.c_[np.linspace(1, 0.2, 9), np.zeros(9)].astype(np.float32))
        tower = SimpleNamespace(user=lambda *args: torch.tensor([[1.0, 0.0]]))
        raw_scores = np.array([0, .95, .9, .85, .8, .2, .7, .99, .1, 1.0], np.float32)
        models = SimpleNamespace(
            store=store, retriever=Retriever(tower, index, store), objective="cvr",
            cfg={"candidate": {"n_candidates": n_candidates}, "seed": 77, "final_eval_k": 3},
            score=lambda ids, state, at: raw_scores[np.asarray(ids)],
        )
        popular_items = np.repeat(np.arange(1, 7), [12, 9, 4, 3, 2, 1])
        log_times = np.full(len(popular_items), NOW_TS - 120, dtype=float)
        popularity = PopularityLists(log_times, popular_items, log_times, popular_items, 9,
                                     PopularityConfig(list_size=2, trending_min_views=1))
        rerank = RerankService(items, products, popularity=popularity, config=RerankConfig(
            min_history=5, diversity_enabled=False, dedupe_field=None,
            new_item_slots=new_slots, exploration_enabled=False,
        ))
        session_cfg = SessionConfig(embedding_dim=2, hidden_dim=2, n_candidates=9,
                                    blend_candidates=False, final_score_weight=.2)
        encoder = SessionEncoder(10, embedding_dim=2, hidden_dim=2)
        with torch.no_grad():
            for parameter in encoder.parameters():
                parameter.zero_()
            encoder.item_emb.weight[1:] = torch.tensor([
                [1, 0], [1, .1], [1, .2], [.8, .6], [0, 1], [.3, .8], [-1, 0], [-.1, 1], [0, 1],
            ])
        session = SessionModel(encoder, ItemIndex(items), session_cfg)
        models.cfg["session"] = {k: v for k, v in asdict(session_cfg).items() if k != "seed"}
        monkeypatch.setattr(pipeline, "get_recommendation_models", lambda: models)
        monkeypatch.setattr(eval_pipeline, "get_recommendation_models", lambda: models)
        monkeypatch.setattr(pipeline, "get_rerank_service", lambda models: rerank)
        monkeypatch.setattr(eval_pipeline, "get_rerank_service", lambda models: rerank)
        monkeypatch.setattr(pipeline, "get_session_model", lambda *args: session)
        return models, rerank

    return build


@pytest.mark.parametrize("use_session", [False, True])
def test_offline_evaluation_never_reads_live_redis(pipeline_bundle, monkeypatch, use_session):
    pipeline_bundle()

    def fail_if_live_store_is_read():
        raise AssertionError("Offline evaluation attempted to read contaminated demo Redis")

    monkeypatch.setattr(pipeline, "get_feature_store", fail_if_live_store_is_read)
    q = eval_pipeline.Queries(np.array(["U00005"]), np.array([NOW_TS]), [{"P1"}],
                              [["P3", "P5"]], 9, 4)
    strategy = pipeline.MultiStageStrategy()
    first = eval_pipeline.evaluate_strategy(strategy, q, use_session, "multistage")
    second = eval_pipeline.evaluate_strategy(strategy, q, use_session, "multistage")
    for metric in ("hit_rate", "ndcg", "coverage"):
        assert first[metric] == second[metric]


def test_no_session_eval_does_not_use_offline_prefix_either(pipeline_bundle, monkeypatch):
    pipeline_bundle()
    q = eval_pipeline.Queries(np.array(["U00005"]), np.array([NOW_TS]), [{"P7"}],
                              [["P3", "P5"]], 9, 1)
    monkeypatch.setattr(pipeline, "get_feature_store", lambda: pytest.fail("Redis access"))
    no_session = eval_pipeline.evaluate_strategy(pipeline.MultiStageStrategy(), q, False,
                                                 "multistage_no_session", session_weight=1.0)
    with_session = eval_pipeline.evaluate_strategy(pipeline.MultiStageStrategy(), q, True,
                                                   "multistage", session_weight=1.0)
    assert no_session["hit_rate"] == 1.0
    assert with_session["hit_rate"] == 0.0


@pytest.mark.parametrize("clicks", [[], ["P3", "P5"]])
def test_audience_filters_candidate_session_and_new_item_pools(pipeline_bundle, clicks):
    pipeline_bundle(new_slots=1, n_candidates=2)
    response = pipeline.MultiStageStrategy().recommend("U00005", 3, at=NOW, audience="men",
                                                        session_clicks=clicks)
    assert len(response.recommendations) == 3
    ids = {item.product_id for item in response.recommendations}
    assert ids <= {"P3", "P4", "P5", "P6", "P8"}
    assert "P8" in ids  # 신상품 풀의 높은 여성 점수(P7)를 거르고 남성 신상품을 확보
    assert any(item.reason == "new_item" for item in response.recommendations)


def test_audience_cold_start_fetches_beyond_wrong_audience_popular_items(pipeline_bundle):
    pipeline_bundle(cold=True)
    response = pipeline.MultiStageStrategy().recommend("new-user", 4, at=NOW, audience="men",
                                                        session_clicks=[])
    assert {item.product_id for item in response.recommendations} == {"P3", "P4", "P5", "P6"}
    assert {item.reason for item in response.recommendations} <= {"popular", "trending"}


def test_zero_session_weight_preserves_raw_scores_and_session_can_change_final_order(pipeline_bundle):
    models, _ = pipeline_bundle()
    strategy = pipeline.MultiStageStrategy()
    plain = strategy.recommend("U00005", 8, at=NOW, session_clicks=[], session_weight=0)
    zero = strategy.recommend("U00005", 8, at=NOW, session_clicks=["P3", "P5"], session_weight=0)
    full = strategy.recommend("U00005", 8, at=NOW, session_clicks=["P3", "P5"], session_weight=1)
    assert [x.product_id for x in zero.recommendations] == [x.product_id for x in plain.recommendations]
    for item in zero.recommendations:
        expected = models.score([models.store.pmap[item.product_id]], None, NOW_TS)[0]
        assert item.score == float(expected)
        assert item.reason == "cvr"
    assert full.recommendations[0].product_id == "P5"
    assert full.recommendations[0].reason == "cvr_session"
    assert all(0 <= item.score <= 1 for item in full.recommendations)


def test_percentile_blending_is_scale_invariant_and_ties_receive_average_rank():
    ranking = np.array([.001, .01, .01, .1])
    session = np.array([.9, .5, .5, -.1])
    np.testing.assert_allclose(percentile_scores(ranking), [0, .5, .5, 1])
    np.testing.assert_allclose(percentile_scores(np.ones(4)), [.5] * 4)
    actual = blend_final_scores(ranking, session, .25)
    np.testing.assert_allclose(actual, [.25, .5, .5, .75])
    np.testing.assert_allclose(actual, blend_final_scores(ranking * 1000 + 9, session * 2 + 1, .25))
    zero = blend_final_scores(ranking, session, 0)
    np.testing.assert_array_equal(zero, ranking)
    assert not np.shares_memory(zero, ranking)


@pytest.mark.parametrize("ranking,session,weight", [([1], [1], -1), ([1], [1], 1.1),
                                                   ([1, 2], [1], .2), ([np.nan], [1], .2)])
def test_invalid_session_blend_is_rejected(ranking, session, weight):
    with pytest.raises(ValueError):
        blend_final_scores(ranking, session, weight)


def stamp_at(seconds):
    return BASE + pd.Timedelta(seconds=seconds)


def purchase_data(*, shown_at=20, view_at=21, bought_at=40, end=200,
                  shown_user="U1", shown_session="S1", displayed=("P1",)):
    events = pd.DataFrame([
        {"user_id": "U1", "session_id": "S1", "timestamp": stamp_at(10), "event_type": "view",
         "product_id": "P2", "impression_id": None},
        {"user_id": "U2", "session_id": "S1", "timestamp": stamp_at(11), "event_type": "view",
         "product_id": "P3", "impression_id": None},
        {"user_id": "U1", "session_id": "S1", "timestamp": stamp_at(view_at), "event_type": "view",
         "product_id": "P1", "impression_id": "I1"},
        {"user_id": "U1", "session_id": "S1", "timestamp": stamp_at(bought_at), "event_type": "purchase",
         "product_id": "P1", "impression_id": None},
        {"user_id": "U1", "session_id": "S2", "timestamp": stamp_at(end), "event_type": "search",
         "product_id": None, "impression_id": None},
    ])
    impressions = pd.DataFrame([{"impression_id": "I1", "user_id": shown_user,
                                 "session_id": shown_session, "timestamp": stamp_at(shown_at),
                                 "product_ids": list(displayed)}])
    return events, impressions


def purchase_rows(events, impressions, window=60, cutoff=0):
    return eval_pipeline.purchase_impression_targets(events, impressions, window, ["P1", "P2", "P3"],
                                                     BASE.timestamp() + cutoff)


def test_purchase_query_predicts_before_target_view_and_isolates_users(tmp_path, monkeypatch):
    events, impressions = purchase_data()
    split = tmp_path / "splits"
    split.mkdir()
    events.to_parquet(split / "test_events.parquet")
    impressions.to_parquet(split / "test_impressions.parquet")
    monkeypatch.setattr(eval_pipeline, "data_dir", lambda: tmp_path)
    models = SimpleNamespace(
        cfg={"seed": 77, "final_eval_k": 3, "ranking": {"cvr_window_seconds": 60}},
        store=SimpleNamespace(item_ids=["P1", "P2", "P3"], metadata={"selection_cutoff": BASE.timestamp()}),
        folder=tmp_path / "isolated-v1",
    )
    query = eval_pipeline.load_purchase_queries(models)
    assert query.targets == [{"P1"}]
    assert query.times.tolist() == [stamp_at(20).timestamp()]
    assert query.sessions == [["P2"]]  # P1의 정답 조회와 다른 사용자 U2의 P3 모두 제외
    assert query.k == 3
    assert query.model_version == "isolated-v1"


@pytest.mark.parametrize("change", [
    {"shown_user": "U2"}, {"shown_session": "other-session"}, {"displayed": ("P2",)},
    {"view_at": 19}, {"bought_at": 21}, {"bought_at": 81}, {"end": 79},
])
def test_purchase_targets_reject_wrong_join_and_unobserved_or_invalid_time(change):
    assert purchase_rows(*purchase_data(**change)) == []


def test_purchase_window_and_model_cutoff_are_inclusive():
    events, impressions = purchase_data(bought_at=80, end=80)
    assert purchase_rows(events, impressions, cutoff=20) == [("U1", stamp_at(20).timestamp(), "S1", {"P1"})]
    assert purchase_rows(events, impressions, cutoff=21) == []


def test_purchase_attribution_uses_latest_strictly_preceding_view():
    events, impressions = purchase_data()
    second = impressions.iloc[0].copy()
    second["impression_id"], second["timestamp"] = "I2", stamp_at(25)
    impressions = pd.concat([impressions, second.to_frame().T], ignore_index=True)
    latest = events.iloc[2].copy()
    latest["impression_id"], latest["timestamp"] = "I2", stamp_at(26)
    events = pd.concat([events, latest.to_frame().T], ignore_index=True)
    assert purchase_rows(events, impressions) == [("U1", stamp_at(25).timestamp(), "S1", {"P1"})]


def test_view_evaluation_uses_bundle_seed_and_k(tmp_path, monkeypatch):
    count, limit, seed = 20, 5, 77
    np.savez(tmp_path / "examples.npz", split=np.full(count, 2), user=np.array([f"U{i}" for i in range(count)]),
             time=np.arange(count), target=np.ones(count, dtype=np.int64))
    (tmp_path / "splits").mkdir()
    purchase_data()[0].to_parquet(tmp_path / "splits/test_events.parquet")
    monkeypatch.setattr(eval_pipeline, "data_dir", lambda: tmp_path)
    models = SimpleNamespace(cfg={"seed": seed, "final_eval_k": 3, "paths": {"model_dir": "."},
                                  "version": "isolated"}, folder=tmp_path,
                             store=SimpleNamespace(item_ids=["P1"]))
    monkeypatch.setattr(eval_pipeline, "get_recommendation_models", lambda: models)
    monkeypatch.setattr(eval_pipeline, "session_prefixes", lambda users, times, **kwargs: [[] for _ in users])
    query = eval_pipeline.load_queries(limit)
    expected = np.sort(np.random.default_rng(seed).choice(count, limit, replace=False))
    assert query.users.tolist() == [f"U{i}" for i in expected]
    assert query.k == 3


def test_evaluation_result_reports_the_seed_that_sampled_its_queries(tmp_path, monkeypatch):
    count, limit, seed = 20, 5, 77
    np.savez(tmp_path / "examples.npz", split=np.full(count, 2), user=np.array([f"U{i}" for i in range(count)]),
             time=np.arange(count), target=np.ones(count, dtype=np.int64))
    (tmp_path / "splits").mkdir()
    purchase_data()[0].to_parquet(tmp_path / "splits/test_events.parquet")
    monkeypatch.setattr(eval_pipeline, "data_dir", lambda: tmp_path)
    models = SimpleNamespace(cfg={"seed": seed, "final_eval_k": 3, "paths": {"model_dir": "."},
                                  "version": "isolated"}, folder=tmp_path,
                             store=SimpleNamespace(item_ids=["P1"]))
    monkeypatch.setattr(eval_pipeline, "get_recommendation_models", lambda: models)
    monkeypatch.setattr(eval_pipeline, "session_prefixes", lambda users, times, **kwargs: [[] for _ in users])
    monkeypatch.setattr(eval_pipeline, "get_rerank_service", SimpleNamespace(cache_clear=lambda: None))
    monkeypatch.setattr(eval_pipeline, "evaluate_strategy", lambda *args, **kwargs:
                        {"hit_rate": 1, "ndcg": 1, "coverage": 1})
    result = eval_pipeline.run(["two_tower_only"], limit=limit)
    assert result["seed"] == seed
    assert result["k"] == 3


def test_legacy_bundle_fills_only_missing_runtime_settings_without_mutation():
    bundle = {"seed": 77, "session": {"live_history": False, "final_score_weight": 0},
              "reranking": {"mab": {"epsilon": .75}}}
    original = copy.deepcopy(bundle)
    resolved = resolve_runtime_config(bundle)
    assert resolved["seed"] == 77
    assert resolved["session"]["live_history"] is False
    assert resolved["session"]["final_score_weight"] == 0
    assert resolved["session"]["max_len"] == load_config("recommend")["session"]["max_len"]
    assert resolved["reranking"]["mab"]["epsilon"] == .75
    assert bundle == original
    legacy = resolve_runtime_config({"session": {"live_history": False}})
    assert legacy["session"]["final_score_weight"] == load_config("recommend")["session"]["final_score_weight"]


def test_bundle_session_config_controls_default_final_scores(pipeline_bundle):
    models, _ = pipeline_bundle()
    models.cfg["session"]["final_score_weight"] = 1.0
    response = pipeline.MultiStageStrategy().recommend("U00005", 1, at=NOW, session_clicks=["P3", "P5"])
    assert response.recommendations[0].product_id == "P5"
    models.cfg["session"]["final_score_weight"] = 0.0
    response = pipeline.MultiStageStrategy().recommend("U00005", 1, at=NOW, session_clicks=["P3", "P5"])
    assert response.recommendations[0].product_id == "P7"
    assert response.recommendations[0].reason == "cvr"


def test_bundle_reranking_popularity_and_mab_settings_are_used(pipeline_bundle, monkeypatch):
    from src.recommendation.reranking import stage2

    models, _ = pipeline_bundle()
    models.cfg["reranking"] = {"cold_start_min_events": 17, "new_item_min_slots": 0,
                                "popularity": {"list_size": 7},
                                "mab": {"epsilon": .75, "exploration_slots": 0}}
    models.store.events = pd.DataFrame({"event_type": ["view", "purchase"], "_time": [NOW_TS - 2, NOW_TS - 1],
                                        "_pid": [3, 3]})
    monkeypatch.setattr(stage2, "load_product_codes", lambda: None)
    service = stage2.get_rerank_service.__wrapped__(models)
    assert service.config.min_history == 17
    assert service.config.new_item_slots == 0
    assert service.config.mab.epsilon == .75
    assert service.config.mab.seed == 77
    assert service.popularity.config.list_size == 7


def test_image_experiment_freezes_session_checkpoint_before_either_arm_trains(tmp_path, monkeypatch):
    from src.evaluation import eval_recommend, experiment_recommend

    source = tmp_path / "original_session.pt"
    source.write_bytes(b"session-before-experiment")
    cfg = copy.deepcopy(load_config("recommend"))
    cfg["paths"]["model_dir"] = str(tmp_path)
    cfg["session"]["model_file"] = source.name
    cfg["reranking"]["mab"]["epsilon"] = .33
    trained = []

    def fake_train(root, folder, arm_config):
        folder.mkdir()
        trained.append(copy.deepcopy(arm_config))
        source.write_bytes(b"a-new-live-session-checkpoint")

    monkeypatch.setattr(experiment_recommend, "train_candidate", fake_train)
    monkeypatch.setattr(experiment_recommend, "build_index", lambda folder: None)
    monkeypatch.setattr(experiment_recommend, "train_ranking", lambda folder, root: None)
    monkeypatch.setattr(eval_recommend, "evaluate", lambda folder:
                        {"retrieval": {"recall_at_300": .1}, "ranking": {"auc": .5}})
    monkeypatch.setattr(eval_pipeline, "run", lambda *args, **kwargs: {})
    output = tmp_path / "experiment"
    result = experiment_recommend.run_experiment(tmp_path / "data", output, cfg, limit=2)
    expected = b"session-before-experiment"
    assert result["session_checkpoint"]["sha256"] == hashlib.sha256(expected).hexdigest()
    for name in ("baseline", "image"):
        assert (output / name / "session_encoder.pt").read_bytes() == expected
    assert source.read_bytes() != expected
    assert all(c["session"]["final_score_weight"] == 0 for c in trained)
    assert all(c["reranking"]["mab"]["epsilon"] == .33 for c in trained)


@pytest.mark.parametrize("relative_source", [False, True])
@pytest.mark.parametrize("target", ["view", "purchase"])
def test_ct_model_evaluates_its_snapshot_even_when_global_data_is_different(tmp_path, monkeypatch,
                                                                         relative_source, target):
    snapshot, legacy, folder = tmp_path / "snapshot", tmp_path / "legacy", tmp_path / "bundle"
    folder.mkdir()
    events, impressions = purchase_data()
    for root, old in ((snapshot, False), (legacy, True)):
        (root / "splits").mkdir(parents=True)
        ev, imp = events.copy(), impressions.copy()
        if old:
            ev["product_id"] = "P3"
            imp["product_ids"] = [["P3"]]
        ev.to_parquet(root / "splits/test_events.parquet")
        imp.to_parquet(root / "splits/test_impressions.parquet")
    np.savez(folder / "examples.npz", split=np.array([2]), user=np.array(["U1"]),
             time=np.array([stamp_at(21).timestamp()]), target=np.array([1]))
    cfg = copy.deepcopy(load_config("recommend"))
    cfg["ranking"]["cvr_window_seconds"] = 60
    source = snapshot.name if relative_source else str(snapshot)
    models = SimpleNamespace(folder=folder, cfg=cfg, store=SimpleNamespace(
        item_ids=["P1", "P2", "P3"], metadata={"source_data_dir": source, "selection_cutoff": BASE.timestamp()},
    ))
    monkeypatch.setattr(eval_pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(eval_pipeline, "data_dir", lambda: legacy)
    monkeypatch.setattr(eval_pipeline, "get_recommendation_models", lambda: models)
    q = eval_pipeline.load_queries(target=target)
    assert q.targets == [{"P1"}]
    assert q.sessions == [["P2"]]
    assert str(snapshot / "splits/test_events.parquet") in q.input_sha256
    assert not any(str(legacy) in path for path in q.input_sha256)
    if target == "purchase":
        assert str(snapshot / "splits/test_impressions.parquet") in q.input_sha256

    # 기록된 snapshot이 없어졌다고 기존 DATA_DIR로 되돌아가면 다른 실험이 된다.
    models.store.metadata["source_data_dir"] = "missing-snapshot"
    with pytest.raises(FileNotFoundError, match="Model evaluation source data is missing"):
        eval_pipeline.load_queries(target=target)
