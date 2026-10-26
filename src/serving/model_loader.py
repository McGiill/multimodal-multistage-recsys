"""버전별 Stage 1/2 모델을 프로세스당 한 번 로드. live Redis 업데이트는 별도 단계."""

import json
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
import torch

from src.common.config import ROOT, env, load_config
from src.recommendation.candidate.retriever import Retriever
from src.recommendation.ranking.deepfm import make_model
from src.recommendation.ranking.features import rank_features
from src.recommendation.training import load_bundle


class RecommendationModels:
    def __init__(self, folder):
        folder = Path(folder)
        self.cfg, self.store, self.tower = load_bundle(folder)
        self.index = faiss.deserialize_index(
            np.fromfile(folder / "items.faiss", dtype=np.uint8)
        )
        if self.index.ntotal != len(self.store.item_ids):
            raise ValueError("Index count mismatch")
        if (
            json.loads((folder / "index_ids.json").read_text(encoding="utf-8"))
            != self.store.item_ids
        ):
            raise ValueError("Index ID mismatch")
        self.ranker = make_model(self.store, self.cfg)
        self.ranker.load_state_dict(
            torch.load(folder / "ranking_cvr.pt", map_location="cpu", weights_only=True)
        )
        self.ranker.eval()
        self.retriever = Retriever(self.tower, self.index, self.store)

    @torch.inference_mode()
    def score(self, ids, state, at):
        h, _length, p = state
        n = len(ids)
        if n == 0:
            return np.array([], dtype=np.float32)
        s, d = rank_features(
            self.store, ids, np.tile(h, (n, 1)), np.tile(p, (n, 1)), np.full(n, at)
        )
        return self.ranker(torch.as_tensor(s), torch.as_tensor(d)).sigmoid().numpy()


@lru_cache(maxsize=4)
def load_models(folder):
    return RecommendationModels(folder)


def get_recommendation_models():
    override = env("REC_MODEL_DIR")
    if override:
        return load_models(str(Path(override).resolve()))
    cfg = load_config("recommend")
    version = env("MODEL_VERSION", cfg.get("version", "stage12-cvr-v1"))
    if version == "latest":
        version = cfg.get("version", "stage12-cvr-v1")
    return load_models(str((ROOT / cfg["paths"]["model_dir"] / version).resolve()))
