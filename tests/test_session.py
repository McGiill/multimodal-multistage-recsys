"""세션 기반 추천 테스트 (가짜 세션으로 검증, 실제 로그 불필요).

실행: pytest tests/test_session.py -v
"""

from __future__ import annotations

import random
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.recommendation.session.encoder import (
    ItemIndex,
    SessionConfig,
    SessionEncoder,
    SessionModel,
    blend_scores,
    build_session_context,
    pad_sequences,
    session_interest,
)
from src.recommendation.session.train import (
    encode_sessions,
    fit,
    make_examples,
    view_sessions,
)

# 가짜 세계: 카테고리 4개 × 상품 25개. 세션은 한 카테고리 안에서 정해진 순서(i → i+1)로 이동
N_CAT, PER_CAT = 4, 25
PIDS = [f"P{c}{i:02d}" for c in range(N_CAT) for i in range(PER_CAT)]
CATEGORY = {p: f"cat{p[1]}" for p in PIDS}


def chain_sessions(n: int, seed: int = 42) -> list[list[str]]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        c, i = rng.randrange(N_CAT), rng.randrange(PER_CAT)
        out.append([f"P{c}{(i + j) % PER_CAT:02d}" for j in range(rng.randint(2, 6))])
    return out


def small_config(**kw) -> SessionConfig:
    params = dict(embedding_dim=16, hidden_dim=16, max_len=5, epochs=15, lr=0.01,
                  batch_size=64, eval_k=5, n_candidates=5)
    params.update(kw)
    return SessionConfig(**params)


@pytest.fixture(scope="module")
def trained() -> tuple[SessionModel, dict]:
    return fit(chain_sessions(600), chain_sessions(100, seed=7), PIDS, small_config(), log=None)


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

def test_config_loads_recommend_yaml():
    cfg = SessionConfig.from_recommend_config()
    assert cfg.encoder == "gru" and cfg.seed == 42
    assert 0 <= cfg.long_short_alpha <= 1 and cfg.interest_field == "category_l2"


@pytest.mark.parametrize("kw", [dict(encoder="lstm"), dict(long_short_alpha=1.5),
                                dict(min_len=1), dict(temperature=0), dict(interest_field="brand")])
def test_config_rejects_bad_values(kw):
    with pytest.raises(ValueError):
        SessionConfig(**kw)


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------

def test_item_index_is_sorted_one_based():
    idx = ItemIndex(["b", "a", "c", "a"])
    assert len(idx) == 4  # padding 포함
    assert idx.encode(["a", "zzz", "c"]) == [1, 3]
    assert list(idx.positions(["c", "zzz"])) == [3, 0]
    assert idx.decode([2, 1]) == ["b", "a"]


def test_view_sessions_orders_filters_and_collapses():
    t0 = datetime(2026, 3, 1)
    rows = [
        ("S1", t0 + timedelta(seconds=3), "view", "B"),
        ("S1", t0 + timedelta(seconds=1), "view", "A"),
        ("S1", t0 + timedelta(seconds=2), "view", "A"),   # 연속 중복 → 합침
        ("S1", t0 + timedelta(seconds=4), "cart", "B"),   # view만 사용
        ("S1", t0 + timedelta(seconds=5), "search", None),
        ("S2", t0, "view", "C"),                          # 1개 → 제외
        ("S3", t0, "view", "A"),
        ("S3", t0 + timedelta(seconds=1), "view", "B"),
        ("S3", t0 + timedelta(seconds=2), "view", "A"),   # 연속 아님 → 유지
    ]
    events = pd.DataFrame(rows, columns=["session_id", "timestamp", "event_type", "product_id"])
    assert view_sessions(events, min_len=2) == [["A", "B"], ["A", "B", "A"]]


def test_make_examples_prefixes_and_truncation():
    seq, length, target = make_examples([[1, 2, 3, 4]], max_len=2)
    assert target.tolist() == [2, 3, 4]
    assert length.tolist() == [1, 2, 2]
    assert seq.tolist() == [[1, 0], [1, 2], [2, 3]]


def test_encode_sessions_drops_unknown_then_short():
    idx = ItemIndex(["A", "B"])
    assert encode_sessions([["A", "X", "B"], ["A", "X"]], idx, min_len=2) == [[1, 2]]


def test_pad_sequences_keeps_most_recent():
    seq, length = pad_sequences([[1, 2, 3, 4, 5]], max_len=3)
    assert seq.tolist() == [[3, 4, 5]] and length.tolist() == [3]


# ---------------------------------------------------------------------------
# 모델 · 학습
# ---------------------------------------------------------------------------

def test_encoder_output_is_normalized():
    import torch

    enc = SessionEncoder(10, 8, 8)
    out = enc(torch.tensor([[1, 2, 0], [3, 0, 0]]), torch.tensor([2, 1]))
    assert out.shape == (2, 8)
    assert torch.allclose(out.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert float(enc.item_vectors()[0].detach().abs().sum()) == 0.0


def test_training_learns_session_pattern(trained):
    _, metrics = trained
    gru, pop = metrics["valid"]["session_gru"], metrics["valid"]["popularity"]
    assert metrics["loss"][-1] < metrics["loss"][0]
    assert gru["hit_rate@5"] > 0.8, gru
    assert gru["hit_rate@5"] > pop["hit_rate@5"] + 0.5, (gru, pop)


def test_training_is_reproducible():
    a = fit(chain_sessions(100), chain_sessions(20, 7), PIDS, small_config(epochs=2), log=None)[1]
    b = fit(chain_sessions(100), chain_sessions(20, 7), PIDS, small_config(epochs=2), log=None)[1]
    assert a["loss"] == b["loss"]


def test_fit_rejects_empty_data():
    with pytest.raises(ValueError):
        fit([["P000"]], [], PIDS, small_config(), log=None)


# ---------------------------------------------------------------------------
# 서빙 (SessionModel)
# ---------------------------------------------------------------------------

def test_top_k_predicts_next_and_excludes_clicked(trained):
    model, _ = trained
    ids, scores = model.top_k(["P103", "P104"], k=3)
    assert ids[0] == "P105"
    assert "P103" not in ids and "P104" not in ids
    assert list(scores) == sorted(scores, reverse=True)


def test_top_k_respects_allowed_mask(trained):
    model, _ = trained
    allowed = np.ones(len(model.index), dtype=bool)
    allowed[model.index.encode(["P105"])] = False  # 아직 등록 전 상품
    ids, _ = model.top_k(["P103", "P104"], k=3, allowed=allowed)
    assert "P105" not in ids


def test_empty_or_unknown_session(trained):
    model, _ = trained
    assert model.encode([]) is None and model.encode(["UNKNOWN"]) is None
    assert model.top_k([], 5) == ([], pytest.approx(np.zeros(0)))
    assert np.all(model.score([], ["P000", "P001"]) == 0)


def test_score_unknown_product_is_zero(trained):
    model, _ = trained
    s = model.score(["P103"], ["P104", "UNKNOWN"])
    assert s[1] == 0 and s[0] > 0


def test_merge_without_session_keeps_long_term(trained):
    model, _ = trained
    ids, scores = model.merge_candidates([], ["P000", "P001"], [0.9, 0.8])
    assert ids == ["P000", "P001"] and list(scores) == pytest.approx([0.9, 0.8])


def test_merge_adds_session_items_and_blends(trained):
    model, _ = trained
    long_ids, long_scores = ["P000", "P001", "P002"], [0.9, 0.8, 0.7]
    ids, scores = model.merge_candidates(["P203", "P204"], long_ids, long_scores)
    assert set(long_ids) <= set(ids)
    assert "P205" in ids  # 세션 관심(다음 상품)이 후보에 추가됨
    assert len(ids) == len(set(ids)) <= len(long_ids) + model.config.n_candidates
    assert list(scores) == sorted(scores, reverse=True)


def test_merge_alpha_extremes(trained, monkeypatch):
    model, _ = trained
    long_ids, long_scores = ["P000", "P001", "P002"], [0.9, 0.8, 0.7]
    monkeypatch.setattr(model.config, "long_short_alpha", 1.0)
    ids, _ = model.merge_candidates(["P203", "P204"], long_ids, long_scores)
    assert ids[:3] == long_ids  # 장기만
    monkeypatch.setattr(model.config, "long_short_alpha", 0.0)
    ids, _ = model.merge_candidates(["P203", "P204"], long_ids, long_scores)
    assert ids[0] == "P205"  # 단기만


@pytest.fixture(scope="module")
def partly_trained() -> SessionModel:
    """카테고리 3(P3xx)은 학습 세션에 한 번도 안 나옴 → 임베딩이 초기 무작위 값."""
    sessions = [s for s in chain_sessions(600) if not s[0].startswith("P3")]
    return fit(sessions, [], PIDS, small_config(epochs=3), log=None)[0]


def test_untrained_items_are_never_recommended(partly_trained):
    model = partly_trained
    assert not model.trained[model.index.encode(["P300", "P324"])].any()
    for clicks in (["P003", "P004"], ["P110"], ["P220", "P221", "P222"]):
        ids, _ = model.top_k(clicks, k=60)  # 학습된 상품 75개 중 클릭 제외 거의 전부
        assert not any(p.startswith("P3") for p in ids)


def test_untrained_items_score_zero_and_are_ignored_as_input(partly_trained):
    model = partly_trained
    assert model.encode(["P300", "P301"]) is None  # 학습 안 된 클릭만 → 세션 없음
    assert np.allclose(model.encode(["P300", "P003"]), model.encode(["P003"]))
    s = model.score(["P003"], ["P004", "P305"])
    assert s[1] == 0 and s[0] != 0


def test_trained_mask_survives_save_load(partly_trained, tmp_path):
    path = tmp_path / "session.pt"
    partly_trained.save(path)
    assert np.array_equal(SessionModel.load(path).trained, partly_trained.trained)


def test_save_and_load_roundtrip(trained, tmp_path):
    model, _ = trained
    path = tmp_path / "session.pt"
    model.save(path)
    loaded = SessionModel.load(path)
    assert loaded.config == model.config
    assert np.allclose(loaded.encode(["P001", "P002"]), model.encode(["P001", "P002"]))


def test_allowed_mask_maps_stage1_order():
    model = SessionModel(SessionEncoder(len(PIDS) + 1, 8, 8), ItemIndex(PIDS), SessionConfig())
    stage1 = ["P101", "unknown", "P000"]  # Stage 1 번호 순서, 모르는 상품 섞임
    mask = model.allowed_mask(stage1, np.array([True, True, False]))
    assert mask.shape == (len(model.index),)
    assert np.flatnonzero(mask).tolist() == model.index.encode(["P101"])


def test_blend_scores():
    assert blend_scores(np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.6).tolist() == pytest.approx([0.6, 0.4])


def test_session_latency_with_50k_items():
    pids = [f"hm_{i:010d}" for i in range(50_000)]
    cfg = SessionConfig()
    model = SessionModel(SessionEncoder(len(pids) + 1, cfg.embedding_dim, cfg.hidden_dim),
                         ItemIndex(pids), cfg)
    clicks = pids[:10]
    long_ids, long_scores = pids[100:400], np.linspace(1, 0, 300)
    for _ in range(3):
        model.merge_candidates(clicks, long_ids, long_scores)
    t0 = time.perf_counter()
    for _ in range(20):
        model.merge_candidates(clicks, long_ids, long_scores)
    assert (time.perf_counter() - t0) / 20 * 1000 < 30  # 추천 API 200ms 중 여유 있게


# ---------------------------------------------------------------------------
# session_context
# ---------------------------------------------------------------------------

def test_session_interest_majority_and_recency():
    cats = {"a1": "아우터", "a2": "아우터", "t1": "상의", "t2": "상의"}
    assert session_interest(["a1", "t1", "a2"], cats) == "아우터"
    assert session_interest(["a1", "t1"], cats) == "상의"  # 동률 → 최근
    assert session_interest([], cats) is None
    assert session_interest(["unknown"], cats) is None


def test_session_interest_accepts_series():
    cats = pd.Series({"a1": "아우터", "t1": "상의"})
    assert session_interest(["t1", "a1", "a1"], cats) == "아우터"


def test_build_session_context_fields():
    ctx = build_session_context([f"P0{i:02d}" for i in range(15)], CATEGORY, max_len=10)
    assert ctx.recent_clicks == [f"P0{i:02d}" for i in range(5, 15)]
    assert ctx.session_interest == "cat0"
    assert set(ctx.model_dump()) == {"recent_clicks", "session_interest"}

