"""검색기 동작 확인: 작은 가짜 카탈로그 + 가짜 인코더로 FAISS 인덱스까지 실제로 만들어 검색."""

import json

import numpy as np
import pandas as pd
import pytest

from src.common.faiss_utils import build_index, l2_normalize, load_index, save_index
from src.common.schemas import SearchType
from src.search.bm25 import BM25Searcher
from src.search.searcher import Searcher

DIM = 8
INDEX_CFG = {"type": "flat", "nlist": 4, "m": 2, "nbits": 8, "nprobe": 2, "hnsw_m": 8, "ef_construction": 40, "ef_search": 16}


def _catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": [f"hm_{i:010d}" for i in range(6)],
            "product_code": ["1", "1", "2", "3", "4", "5"],
            "name": ["Strap top", "Strap top", "Denim jeans", "Wool sweater", "Ankle boots", "Sun hat"],
            "description": ["jersey top", "jersey top", "slim jeans", "warm knit", "leather boots", "straw hat"],
            "category_l1": ["Ladieswear"] * 6,
            "category_l2": ["Garment Upper body"] * 2 + ["Garment Lower body", "Garment Upper body", "Shoes", "Accessories"],
            "category_l3": ["Vest top", "Vest top", "Trousers", "Sweater", "Boots", "Hat/brim"],
            "color": ["Black", "White", "Blue", "Grey", "Black", "Beige"],
            "pattern": ["Solid"] * 6,
            "price": [25000, 25000, 40000, 50000, 70000, 20000],
            "image_path": [f"images/{i}.jpg" for i in range(6)],
            "n_purchases": [10] * 6,
        }
    )


class FakeEncoder:
    model_name = "fake/clip"
    device = "cpu"

    def __init__(self, text_vecs: dict[str, np.ndarray]) -> None:
        self.text_vecs = text_vecs

    def encode_texts(self, texts, batch_size=64):
        return l2_normalize(np.stack([self.text_vecs[t] for t in texts]))

    def encode_images(self, images, batch_size=32):
        return l2_normalize(np.stack([np.asarray(im, dtype="float32") for im in images]))


@pytest.fixture()
def searcher(tmp_path) -> Searcher:
    rng = np.random.default_rng(0)
    image_emb = l2_normalize(rng.normal(size=(6, DIM)))
    text_emb = l2_normalize(image_emb + 0.1 * rng.normal(size=(6, DIM)))
    catalog = _catalog()

    cfg = {
        "clip": {"model_name": "fake/clip", "serve_device": "cpu"},
        "index": INDEX_CFG,
        "retrieval": {"candidate_k": 4},
        "text_query": {"image_weight": 0.5},
        "hybrid": {"text_weight": 0.5},
        "paths": {"catalog": str(tmp_path / "products.parquet"), "index_dir": str(tmp_path)},
    }
    catalog.to_parquet(tmp_path / "products.parquet", index=False)
    d = tmp_path / "fake__clip"
    d.mkdir()
    np.save(d / "image_emb.npy", image_emb)
    np.save(d / "text_emb.npy", text_emb)
    catalog[["product_id"]].to_parquet(d / "product_ids.parquet", index=False)
    save_index(build_index(image_emb, INDEX_CFG), d / "image.index")
    save_index(build_index(text_emb, INDEX_CFG), d / "text.index")
    (d / "meta.json").write_text(json.dumps({"model_name": "fake/clip", "pad_to_square": True}))

    # 텍스트 쿼리 "q3"는 3번 상품 벡터와 같도록 → 3번이 1위여야 함
    encoder = FakeEncoder({"q3": image_emb[3]})
    s = Searcher(cfg, encoder=encoder)
    s._image_emb_for_test = image_emb
    return s


@pytest.mark.parametrize("index_type", ["flat", "hnsw", "ivfflat", "ivfpq"])
def test_build_and_reload_every_index_type(tmp_path, index_type) -> None:
    vecs = l2_normalize(np.random.default_rng(1).normal(size=(400, 16)))
    cfg = INDEX_CFG | {"type": index_type, "m": 4}
    save_index(build_index(vecs, cfg), tmp_path / "x.index")
    index = load_index(tmp_path / "x.index", cfg)
    _, idx = index.search(vecs[:5], 3)
    hit = np.mean([i in row for i, row in enumerate(idx)])
    assert hit >= (0.6 if index_type == "ivfpq" else 1.0)


def test_text_search_ranks_matching_product_first(searcher) -> None:
    result = searcher.search("q3", None, top_k=3)
    assert result.search_type == SearchType.TEXT
    assert result.product_ids[0] == "hm_0000000003"
    assert result.items[0] == {"product_id": "hm_0000000003", "name": "Wool sweater", "score": result.scores[0], "price": 50000}
    assert result.total_count >= len(result.product_ids)


def test_image_search_returns_cosine_scores(searcher) -> None:
    q = searcher._image_emb_for_test[4]
    result = searcher.search(None, q, top_k=2)
    assert result.search_type == SearchType.IMAGE
    assert result.product_ids[0] == "hm_0000000004"
    assert result.scores[0] == pytest.approx(1.0, abs=1e-4)


def test_hybrid_search(searcher) -> None:
    result = searcher.search("q3", searcher._image_emb_for_test[3], top_k=5)
    assert result.search_type == SearchType.HYBRID
    assert result.product_ids[0] == "hm_0000000003"
    assert result.scores == sorted(result.scores, reverse=True)


def test_bm25_baseline() -> None:
    ids, scores = BM25Searcher(_catalog()).search("black strap top", top_k=2)
    assert ids[0] in {"hm_0000000000", "hm_0000000001"}
    assert ids[0] == "hm_0000000000"  # 색상(Black)까지 일치하는 상품이 위
    assert scores[0] >= scores[1]


def test_model_slug_and_overrides() -> None:
    from src.search.catalog import model_slug, with_overrides

    assert model_slug("hf-hub:Marqo/marqo-fashionCLIP") == "hf-hub__Marqo__marqo-fashionCLIP"
    cfg = {"clip": {"model_name": "a"}, "text_query": {"image_weight": 0.2}, "hybrid": {"text_weight": 0.5}}
    new = with_overrides(cfg, model="b", text_image_weight=0.7)
    assert new["clip"]["model_name"] == "b" and new["text_query"]["image_weight"] == 0.7
    assert cfg["clip"]["model_name"] == "a"  # 원본 설정은 그대로
