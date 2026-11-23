"""2026-11-19 추가분: Stage 1 검색 보정, 최종 순위 모델, 다양성 칸, 사용자 단위 정답, 재학습 데이터 사본."""

import copy
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.stage12_smoke import create_data
from src.common.config import load_config
from src.evaluation.user_targets import load_user_targets
from src.recommendation import pipeline
from src.recommendation.candidate.prior import PriorIndex, RetrievalPrior
from src.recommendation.final_rank import FEATURES, FinalRankContext, FinalRanker
from src.recommendation.final_rank.context import HALF_LIFE, GroupStats
from src.recommendation.final_rank.train import train_final_ranker
from src.recommendation.refit import build_refit_root
from src.recommendation.reranking.aggregate import DiversityConfig, diversify
from src.recommendation.reranking.stage2 import get_rerank_service
from src.recommendation.runtime_config import serving_version
from src.recommendation.training import build_index, train_candidate, train_ranking
from src.serving.model_loader import RecommendationModels


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("final")
    data = root / "data"
    create_data(data)
    cfg = copy.deepcopy(load_config("recommend"))
    cfg["threads"] = 2
    cfg["candidate"].update(embedding_dim=16, history_len=5, epochs=1, batch_size=128, faiss_index_type="flat")
    cfg["ranking"].update(embedding_dim=8, dnn_hidden=[32, 16], dropout=0.0, epochs=1, batch_size=128)
    cfg["final_rank"]["training"].update(max_iter=20, negatives=40)
    out = root / "bundle"
    train_candidate(data, out, cfg)
    build_index(out)
    train_ranking(out, data)
    return data, out


def test_prior_index_equals_explicit_score():
    """보정 색인의 내적 = Two-Tower 내적 + β·인기 + γ·소분류 비율 + δ·중분류 비율."""
    rng = np.random.default_rng(0)
    n = 30
    cats = np.zeros((n + 1, 6), np.int64)
    cats[1:, 1] = rng.integers(1, 4, n)
    cats[1:, 2] = rng.integers(1, 7, n)
    pop = rng.random(n + 1).astype(np.float32)
    store = SimpleNamespace(item_ids=[f"P{i}" for i in range(n)], cats=cats, frozen_time=lambda t: np.asarray(t),
                            popularity=lambda ids, times, days=7: pop[ids])
    V = rng.normal(size=(n, 4)).astype(np.float32)
    prior = RetrievalPrior(popularity=0.3, category_l3=0.1, category_l2=0.2)
    index = PriorIndex(V, store, prior)
    q, history = rng.normal(size=(1, 4)).astype(np.float32), np.array([3, 3, 7, 0, 0])
    scores, ids = index.index(1000.0).search(index.query(q, history), n)
    s3 = np.bincount(cats[[3, 3, 7], 2], minlength=7) / 3
    s2 = np.bincount(cats[[3, 3, 7], 1], minlength=4) / 3
    explicit = V @ q[0] + 0.3 * pop[1:] + 0.1 * s3[cats[1:, 2]] + 0.2 * s2[cats[1:, 1]]
    np.testing.assert_allclose(scores[0], explicit[ids[0]], rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(ids[0], np.argsort(-explicit))
    assert not RetrievalPrior().active and RetrievalPrior.from_config({"candidate": {}}) == RetrievalPrior()

    # 최근 가중 소분류 비율: 이력 k개 전 조회의 무게 0.5 ** (k / 반감), 같은 소분류 열에 더해진다
    recent = RetrievalPrior(popularity=0.3, category_l3=0.1, category_l2=0.2, recent_l3=0.5, recent_half_life=2.0)
    index = PriorIndex(V, store, recent)
    scores, ids = index.index(1000.0).search(index.query(q, history), n)
    w = 0.5 ** (np.array([2, 1, 0]) / 2.0)
    r3 = np.bincount(cats[[3, 3, 7], 2], weights=w / w.sum(), minlength=7)
    np.testing.assert_allclose(scores[0], (explicit + 0.5 * r3[cats[1:, 2]])[ids[0]], rtol=1e-5, atol=1e-5)
    assert RetrievalPrior(recent_l3=0.1).active
    assert RetrievalPrior.from_config({"candidate": {"retrieval_prior": {"recent_l3": 0.2, "recent_half_life": 3}}}) \
        == RetrievalPrior(recent_l3=0.2, recent_half_life=3.0)


def test_diversify_keeps_head_and_prefers_unexposed_tail():
    ids = np.arange(1, 11)
    scores = np.linspace(1, 0.1, 10)
    codes = np.arange(11)
    counts = np.zeros(11, np.int64)
    counts[[4, 5]] = 100  # 순위 4·5위 상품은 이미 많이 노출됨
    order = diversify(ids, scores, codes, counts, head=3, tail=2, window=10, penalty=1.0)
    ranked = ids[np.argsort(-order, kind="stable")]
    assert list(ranked[:3]) == [1, 2, 3]
    assert 4 not in ranked[3:5] and 5 not in ranked[3:5]
    assert list(ids[np.argsort(-diversify(ids, scores, codes, counts, 3, 0, 10, 1.0), kind="stable")]) == list(ids)
    cfg = DiversityConfig.from_config({"tail_share": 0.06, "window": 1200, "penalty": 1.0})
    assert cfg.tail(50) == 3 and cfg.tail(10) == 1 and DiversityConfig().tail(50) == 0


def test_user_targets_first_event_and_all_interactions(tmp_path):
    (tmp_path / "splits").mkdir()
    t0 = pd.Timestamp("2026-03-23 10:00")
    pd.DataFrame({
        "user_id": ["A", "A", "A", "B", "B"],
        "timestamp": [t0 + pd.Timedelta(minutes=m) for m in (5, 0, 9, 1, 2)],
        "event_type": ["view", "search", "purchase", "cart", "view"],
        "product_id": ["P1", None, "P2", "P3", "UNKNOWN"],
    }).to_parquet(tmp_path / "splits/test_events.parquet")
    targets = load_user_targets(tmp_path, "test", {"P1": 1, "P2": 2, "P3": 3})
    assert targets.users == ["A", "B"]
    assert targets.at[0] == t0.tz_localize("UTC").timestamp()  # 검색도 방문 시작으로 본다
    assert list(targets.truth[0]) == [1, 2] and list(targets.purchases[0]) == [2]
    assert list(targets.truth[1]) == [3] and not len(targets.purchases[1])


def test_final_ranker_trains_loads_and_serves(bundle, monkeypatch):
    data, out = bundle
    meta = train_final_ranker(out, data, evaluate=True)
    assert (out / "final_ranker.pkl").exists() and meta["features"] == list(FEATURES)
    assert 0 <= meta["valid_cross_fit_top50_before_stage3"]["coverage"] <= 1
    models = RecommendationModels(out)
    assert models.final_ranker is not None and isinstance(models.final_context, FinalRankContext)
    monkeypatch.setattr(pipeline, "get_recommendation_models", lambda: models)
    get_rerank_service.cache_clear()
    user = models.store.users.user_id.iloc[0]
    at = datetime.fromtimestamp(models.store.metadata["selection_cutoff"] + 3600, tz=timezone.utc)
    res = pipeline.MultiStageStrategy().recommend(user, 20, at=at, session_clicks=[])
    ids = [r.product_id for r in res.recommendations]
    assert len(ids) == len(set(ids)) == 20
    created = models.store.products.set_index("product_id")["created_at"]
    assert (pd.to_datetime(created[ids], utc=True) <= pd.Timestamp(at)).all()
    service = get_rerank_service(models)
    assert service.exposure.sum() == 20  # 다양성 칸용 노출 횟수
    audience = pipeline.MultiStageStrategy().recommend(user, 10, at=at, session_clicks=[], audience="women")
    assert len(audience.recommendations) <= 10


def test_user_level_evaluation_reports_spec_metrics(bundle, monkeypatch):
    from src.evaluation import eval_pipeline
    from src.recommendation.strategies import baselines

    data, out = bundle
    if not (out / "final_ranker.pkl").exists():
        train_final_ranker(out, data, evaluate=False)
    models = RecommendationModels(out)
    for module in (eval_pipeline, pipeline, baselines):
        monkeypatch.setattr(module, "get_recommendation_models", lambda: models)
    monkeypatch.setattr(eval_pipeline, "evaluation_data_root", lambda _: data)
    res = eval_pipeline.run_users(["multistage", "two_tower_only", "popularity"], split="test")
    assert res["target_type"] == "user" and res["n_users"] > 0 and 0 <= res["stage1_recall@300"] <= 1
    for name in ("multistage", "two_tower_only", "popularity"):
        m = res[name]
        assert 0 <= m["hit_rate"] <= 1 and 0 <= m["ndcg"] <= 1 and 0 < m["coverage"] <= 1
        assert set(m["passed"]) == {"hit_rate", "ndcg", "coverage"}
    assert "lift_vs_popularity" in res["multistage"]


def test_final_rank_features_shape_and_sources(bundle):
    _, out = bundle
    models = RecommendationModels(out)
    ctx = FinalRankContext(models.store, models.item_vectors, load_config("recommend")["final_rank"])
    user = models.store.users.user_id.iloc[1]
    at = models.store.metadata["selection_cutoff"] + 60
    q, state = models.retriever.user_query(user, at)
    stage1, _ = models.retriever.search(q, state, at, 100)
    info = ctx.user_info(user, at, state)
    ids, src = ctx.pool(info, at, q, stage1)
    assert set(stage1.tolist()) <= set(ids.tolist()) and (models.store.created[ids] <= at).all()
    X = ctx.features(ids, src, info, at, q, models.score(ids, state, at))
    assert X.shape == (len(ids), len(FEATURES)) and np.isfinite(X).all()
    with pytest.raises(ValueError):
        FinalRanker([], features=[*FEATURES[:-1], "unknown_feature"])


def test_final_ranker_trained_on_older_features_selects_its_columns():
    class FirstColumn:  # 첫 열을 확률로 쓰는 가짜 모델
        def predict_proba(self, X):
            return np.c_[1 - X[:, 0], X[:, 0]]

    old = FinalRanker([FirstColumn()], features=("tt", "dfm"))  # 특징 추가 전 묶음처럼 일부 이름만
    X = np.random.default_rng(0).random((5, len(FEATURES))).astype(np.float32)
    np.testing.assert_allclose(old.predict(X), X[:, FEATURES.index("tt")], rtol=1e-6)
    assert FinalRanker([FirstColumn()]).columns is None


def test_group_stats_recency_visits_and_last_visit():
    day = 86400.0
    stats = GroupStats.of(np.array([5, 5, 7]), np.array([0.0, 2 * day, 2 * day + 100]), 3 * day)
    assert stats.codes.tolist() == [5, 7]
    np.testing.assert_allclose(stats.days_since, [1.0, (day - 100) / day], rtol=1e-6)
    assert stats.visit_days.tolist() == [2, 1]
    w = 0.5 ** (np.array([3 * day, day, day - 100]) / HALF_LIFE)
    np.testing.assert_allclose(stats.share_rw, [(w[0] + w[1]) / w.sum(), w[2] / w.sum()], rtol=1e-5)
    np.testing.assert_allclose(stats.last_visit, [0.5, 0.5])  # 마지막 3시간 안 조회 2개
    assert stats.rank.tolist() == [0, 1]
    assert stats.pick(stats.share_rw, np.array([7, 9]), -1.0)[1] == -1.0  # 본 적 없는 범주
    assert stats.log_days_since(np.array([9]))[0] == -1.0
    empty = GroupStats.of(np.array([], np.int64), np.array([]), day)
    assert empty.pick(empty.share_rw, np.array([5]), 0.0).tolist() == [0.0]


def test_refit_root_moves_early_valid_sessions(bundle, tmp_path):
    data, _ = bundle
    root = build_refit_root(data, tmp_path / "refit", 1)
    meta = json.loads((root / "splits/meta.json").read_text())
    original = pd.read_parquet(data / "splits/valid_events.parquet")
    moved = pd.read_parquet(root / "splits/train_events.parquet")
    kept = pd.read_parquet(root / "splits/valid_events.parquet")
    assert len(moved) + len(kept) == len(original) + len(pd.read_parquet(data / "splits/train_events.parquet"))
    assert not set(kept.session_id) & set(moved.session_id)  # 세션은 쪼개지지 않는다
    assert meta["splits"]["valid"]["from"] > load_json(data)["splits"]["valid"]["from"]
    assert pd.read_parquet(root / "splits/test_events.parquet").equals(pd.read_parquet(data / "splits/test_events.parquet"))


def load_json(data):
    return json.loads((data / "splits/meta.json").read_text())


def test_serving_version_prefers_refit_bundle(tmp_path):
    cfg = {"version": "v1", "refit": {"enabled": True, "suffix": "-refit"}, "paths": {"model_dir": str(tmp_path)}}
    assert serving_version(cfg) == "v1"
    (tmp_path / "v1-refit").mkdir()
    (tmp_path / "v1-refit/config.json").write_text("{}")
    assert serving_version(cfg) == "v1-refit"
    assert serving_version({**cfg, "refit": {"enabled": False}}) == "v1"
