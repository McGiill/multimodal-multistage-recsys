"""이미지 상품 정렬, 체크포인트 독립성, 조건부 후보 검색과 단일 변수 실험."""

import copy
import json
import os
import shutil
from types import SimpleNamespace

import faiss
import numpy as np
import pandas as pd
import pytest
import torch

from scripts.stage12_smoke import create_data
from src.common.config import load_config
from src.evaluation.experiment_recommend import (
    make_configs,
    model_environment,
    run_experiment,
)
from src.recommendation.candidate.image_features import load_image_features
from src.recommendation.candidate.retriever import search_available
from src.recommendation.candidate.two_tower import TwoTower, make_model
from src.recommendation.training import build_index, load_bundle, train_candidate


def write_features(folder, ids, vectors):
    folder.mkdir()
    pd.DataFrame({"product_id": ids}).to_parquet(folder / "product_ids.parquet", index=False)
    np.save(folder / "image_emb.npy", np.asarray(vectors, dtype=np.float32))
    return {"enabled": True, "embeddings_dir": str(folder), "projection_dim": 4}


def test_image_features_align_by_id_and_add_padding(tmp_path):
    cfg = write_features(tmp_path / "emb", ["C", "A", "B", "extra"], [[0, 3], [4, 0], [3, 4], [1, 1]])
    vectors, meta = load_image_features(["A", "B", "C"], cfg)
    np.testing.assert_allclose(vectors, [[0, 0], [1, 0], [.6, .8], [0, 1]])
    assert meta["aligned_items"] == 3 and meta["source_items"] == 4
    assert len(meta["image_emb_sha256"]) == 64
    assert load_image_features(["A"], {"enabled": False, "embeddings_dir": "/not/read"}) == (None, None)


@pytest.mark.parametrize("ids,vectors,match", [
    (["A", "A"], [[1, 0], [0, 1]], "unique"),
    (["A", "B"], [[1, 0]], "aligned"),
    (["B"], [[1, 0]], "missing"),
    (["A"], [[0, 0]], "nonzero"),
    (["A"], [[float("nan"), 1]], "finite"),
])
def test_invalid_image_inputs_fail_instead_of_silent_fallback(tmp_path, ids, vectors, match):
    cfg = write_features(tmp_path / "emb", ids, vectors)
    with pytest.raises(ValueError, match=match):
        load_image_features(["A"], cfg)


def test_image_projection_learns_and_padding_stays_zero():
    torch.manual_seed(42)
    model = TwoTower(np.zeros((4, 1), dtype=np.int64), np.zeros((4, 2)), [2], 3, dim=8,
                     image_vectors=[[0, 0], [1, 0], [0, 1], [.6, .8]], image_projection_dim=4)
    before = model.item(torch.tensor([1])).detach()
    with torch.no_grad():
        model.image_vectors[1] = torch.tensor([0., 1.])
    after = model.item(torch.tensor([1]))
    assert not torch.allclose(before, after)
    after.sum().backward()
    assert model.image_projection.weight.grad.abs().sum() > 0
    assert not model.item(torch.tensor([0])).any()
    assert "image_vectors" in model.state_dict()


def test_disabled_image_model_preserves_old_checkpoint_keys():
    store = SimpleNamespace(cats=np.zeros((3, 1), np.int64), numeric=np.zeros((3, 2)),
                            sizes=[2], metadata={"profile_dim": 3})
    old = TwoTower(store.cats, store.numeric, store.sizes, 3, dim=8)
    cfg = {"candidate": {"embedding_dim": 8, "image_features": {"enabled": False}}}
    new = make_model(store, cfg)
    new.load_state_dict(old.state_dict(), strict=True)
    assert not any("image" in key for key in new.state_dict())
    with pytest.raises(ValueError, match="disagree"):
        make_model(store, {"candidate": {"embedding_dim": 8, "image_features": {"enabled": True}}})


def test_image_checkpoint_reloads_without_source_embeddings(tmp_path, monkeypatch):
    from src.recommendation import training

    monkeypatch.setattr(training, "ROOT", tmp_path)
    root = tmp_path / "data"
    create_data(root)
    products = pd.read_parquet(root / "processed/products.parquet")
    ids = products.product_id.tolist()[::-1]
    image_cfg = write_features(tmp_path / "images", ids, np.random.default_rng(7).normal(size=(len(ids), 12)))
    cfg = copy.deepcopy(load_config("recommend"))
    cfg["threads"] = 2
    cfg["validation_limit"] = 15
    cfg["candidate"].update(embedding_dim=8, history_len=5, epochs=1, batch_size=128,
                            faiss_index_type="flat", image_features=image_cfg)
    bundle = tmp_path / "bundle"
    train_candidate(root, bundle, cfg)
    _, store, model = load_bundle(bundle)
    expected = model.item(torch.arange(1, 6)).detach().numpy()
    assert store.metadata["image_features"]["aligned_items"] == len(ids)
    assert store.metadata["source_data_dir"] == "data"  # 호스트/Docker의 ROOT가 달라도 연결 가능한 경로
    shutil.rmtree(tmp_path / "images")  # 체크포인트가 원본 이미지 폴더에 의존하지 않아야 한다.
    _, _, reloaded = load_bundle(bundle)
    np.testing.assert_allclose(reloaded.item(torch.arange(1, 6)).detach().numpy(), expected)
    index = build_index(bundle)
    assert index.ntotal == len(ids)
    assert json.loads((bundle / "index_ids.json").read_text()) == store.item_ids


def test_allowed_mask_fetches_beyond_wrong_audience_and_future_items():
    index = faiss.IndexFlatIP(1)
    index.add(np.arange(8, 0, -1, dtype=np.float32)[:, None])
    allowed = np.array([False, False, False, False, False, True, True, True, True])
    created = np.array([-np.inf, 0, 0, 0, 0, 0, 200, 0, 0])
    ids, scores = search_available(index, [1], created, 100, 3, allowed=allowed)
    assert ids.tolist() == [5, 7, 8] and scores.tolist() == [4, 2, 1]
    ids, _ = search_available(index, [1], created, 100, 10, allowed=np.zeros(9, dtype=bool))
    assert not len(ids)
    with pytest.raises(ValueError, match="boolean"):
        search_available(index, [1], created, 100, 3, allowed=np.ones(8, dtype=bool))


def test_experiment_changes_only_images_and_restores_environment(tmp_path, monkeypatch):
    cfg = copy.deepcopy(load_config("recommend"))
    configs = make_configs(cfg, tmp_path)
    left, right = copy.deepcopy(configs["baseline"]), copy.deepcopy(configs["image"])
    assert left["candidate"]["image_features"]["enabled"] is False
    assert right["candidate"]["image_features"]["enabled"] is True
    left.pop("version")
    right.pop("version")
    left["candidate"]["image_features"]["enabled"] = True
    assert left == right
    assert cfg["candidate"]["image_features"]["enabled"] is False
    monkeypatch.setenv("REC_MODEL_DIR", "/previous")
    monkeypatch.delenv("DATA_DIR", raising=False)
    with pytest.raises(RuntimeError), model_environment(tmp_path / "image", tmp_path / "data"):
        assert os.environ["REC_MODEL_DIR"] == str(tmp_path / "image")
        raise RuntimeError("failed experiment")
    assert os.environ["REC_MODEL_DIR"] == "/previous" and "DATA_DIR" not in os.environ


def test_two_arm_experiment_trains_and_evaluates_isolated_bundles(tmp_path):
    root = tmp_path / "data"
    create_data(root)
    # 레거시 smoke fixture에는 event.impression_id가 없어 구매 평가용 연결을 보충한다.
    for split in ("train", "valid", "test"):
        path = root / "splits" / f"{split}_events.parquet"
        events = pd.read_parquet(path)
        impressions = pd.read_parquet(root / "splits" / f"{split}_impressions.parquet")
        lookup = {(r.user_id, r.session_id, r.timestamp + pd.Timedelta(seconds=1)): r.impression_id
                  for r in impressions.itertuples()}
        events["impression_id"] = [lookup.get((r.user_id, r.session_id, r.timestamp))
                                    if r.event_type == "view" else None for r in events.itertuples()]
        events.to_parquet(path, index=False)
    ids = pd.read_parquet(root / "processed/products.parquet").product_id.tolist()
    image_cfg = write_features(tmp_path / "images", ids, np.random.default_rng(4).normal(size=(len(ids), 8)))
    cfg = copy.deepcopy(load_config("recommend"))
    cfg["threads"] = 2
    cfg["validation_limit"] = 10
    cfg["candidate"].update(embedding_dim=8, history_len=5, epochs=1, batch_size=128,
                            faiss_index_type="flat", image_features=image_cfg)
    cfg["ranking"].update(embedding_dim=4, dnn_hidden=[16], epochs=1, batch_size=128, dropout=0.0)
    output = tmp_path / "experiment"
    # 두 arm에서 실제 candidate/ranking을 학습하고 새 조회/구매 평가 경로를 모두 실행한다.
    result = run_experiment(root, output, cfg, limit=5, strategies=("two_tower_only", "two_tower_ranking"))
    assert result["completed"] and result["automatic_promotion"] is False
    assert result["session_final_score_weight"] == 0
    assert result["input_sha256"]["splits/meta.json"]
    for name in ("baseline", "image"):
        arm = result["arms"][name]
        assert set(arm["pipeline"]) == {"view", "purchase"}
        assert arm["pipeline"]["view"]["n_queries"] == 5
        assert arm["pipeline"]["purchase"]["n_queries"] == 5
        assert (output / name / "ranking_ctcvr.pt").exists()
    assert json.loads((output / "comparison.json").read_text())["completed"]
    with pytest.raises(FileExistsError):
        run_experiment(root, output, cfg, limit=5)
