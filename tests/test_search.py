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


def test_search_within_conditions(searcher) -> None:
    rows = searcher.select_rows(l3=["Vest top", "Boots"])
    assert rows.tolist() == [0, 1, 4]
    assert searcher.select_rows(l3=["Vest top"], colors=["White"]).tolist() == [1]
    assert searcher.select_rows(l1=["Menswear"]).tolist() == []

    result = searcher.rank_rows(searcher._image_emb_for_test[4], rows, top_k=2)
    assert result.product_ids[0] == "hm_0000000004" and result.total_count == 3
    assert result.items[0]["name"] == "Ankle boots" and result.scores == sorted(result.scores, reverse=True)


def test_bm25_index_matches_rank_bm25() -> None:
    from rank_bm25 import BM25Okapi

    from src.search.bm25 import BM25Index, tokenize
    from src.search.catalog import product_texts

    docs = [tokenize(t) for t in product_texts(_catalog())]
    fast, ref = BM25Index(docs), BM25Okapi(docs)
    for query in ("black strap top", "jersey jersey", "unknown words", "Ankle BOOTS leather"):
        assert np.allclose(fast.scores(query), ref.get_scores(tokenize(query)))


def test_text_search_fuses_bm25(searcher) -> None:
    # CLIP 벡터는 3번(Wool sweater)을 가리키지만 단어는 4번(Ankle boots)과 일치
    searcher.encoder.text_vecs["ankle boots"] = searcher._image_emb_for_test[3]
    assert searcher.search("ankle boots", None, top_k=1).product_ids == ["hm_0000000003"]  # BM25 비중 0
    searcher.cfg["text_query"]["bm25_weight"] = 0.8
    result = searcher.search("ankle boots", None, top_k=2)
    assert result.product_ids[0] == "hm_0000000004"
    assert searcher.search_vectors(searcher._image_emb_for_test[3], None, 1).product_ids == ["hm_0000000003"]  # 문장 없으면 CLIP만


def test_image_search_attribute_bonus(searcher, tmp_path) -> None:
    from src.search import attributes

    cfg = searcher.cfg | {"seed": 42, "attributes": {"epochs": 200, "lr": 0.05, "weight_decay": 0.0}}
    attributes.fit(cfg)
    probe = attributes.AttributeProbe(tmp_path / "fake__clip" / "attributes.npz",
                                      {"color": searcher.colors, "category": searcher.category_l3})
    q = searcher._image_emb_for_test[0]  # 0번 상품 사진 (Black, Vest top)
    bonus = probe.bonus(q, np.arange(6), {"color": 1.0, "category": 0.0})
    assert bonus.argmax() in (0, 4) and bonus[0] > bonus[1]  # 검은색 상품(0, 4)에 가점, 흰색(1)보다 크게
    assert probe.class_prob("color", q, ["Black"]) > probe.class_prob("color", q, ["White"])
    assert probe.class_prob("color", q, ["Black", "White"]) > probe.class_prob("color", q, ["Black"])

    searcher.attributes = probe
    plain = searcher.search(None, q, top_k=6)
    assert plain.scores[0] == pytest.approx(1.0, abs=1e-4)  # 가점 비중 0 → 코사인 그대로
    searcher.cfg["image_query"] = {"color_weight": 1.0, "category_weight": 1.0}
    boosted = searcher.search(None, q, top_k=6)
    assert boosted.product_ids[0] == "hm_0000000000" and boosted.scores[0] > 1.0


def test_rank_rows_combines_text(searcher) -> None:
    image = searcher._image_emb_for_test
    rows = np.arange(6)
    assert searcher.rank_rows(image[4], rows, 1).product_ids == ["hm_0000000004"]  # 문장 없으면 이미지 유사도
    searcher.cfg["hybrid"]["text_weight"] = 0.9
    result = searcher.rank_rows(image[4], rows, 1, q_text=image[3])
    assert result.product_ids == ["hm_0000000003"] and result.search_type == SearchType.HYBRID
    assert searcher.rank_rows(image[4], np.array([], dtype=int), 1, q_text=image[3]).product_ids == []


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


def test_relevance_tag_keeps_old_result_names() -> None:
    from src.evaluation.eval_search import relevance_tag

    assert relevance_tag({"eval": {}}) == ""  # 기준이 없으면 기존 기준(design)
    assert relevance_tag({"eval": {"relevance": "design"}}) == ""
    assert relevance_tag({"eval": {"relevance": "color_family"}}) == "_color_family"


def test_index_tradeoff_overlap() -> None:
    from src.evaluation.index_tradeoff import overlap

    found = np.array([[1, 2, 3], [4, 5, 6]])
    truth = np.array([[1, 2, 9], [4, 5, 6]])
    assert overlap(found, truth) == pytest.approx((2 / 3 + 1) / 2)
    assert overlap(found, truth[:, :1]) == 1.0  # 정답 1위는 둘 다 후보 안에 있음


def test_similar_products_skip_same_design(searcher) -> None:
    result = searcher.similar_products("hm_0000000000", top_k=3)
    assert "hm_0000000000" not in result.product_ids
    assert "hm_0000000001" not in result.product_ids  # 같은 디자인(product_code 1)의 다른 색
    codes = [searcher.product_codes[r] for r in result.rows]
    assert len(codes) == len(set(codes)) and len(result.items) == len(result.product_ids) <= 3
    with pytest.raises(KeyError):
        searcher.similar_products("hm_9999999999", top_k=3)


def test_search_restricted_to_audience(searcher) -> None:
    q = searcher._image_emb_for_test[4]
    assert searcher.search(None, q, top_k=3, l1=["Menswear"]).product_ids == []  # 가짜 카탈로그는 전부 Ladieswear
    assert searcher.search(None, q, top_k=3, l1=["Ladieswear"]).product_ids == searcher.search(None, q, top_k=3).product_ids
