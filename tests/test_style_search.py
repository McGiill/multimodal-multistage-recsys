"""스타일 사진 검색 동작 확인: 가짜 검출기 + 가짜 인코더 + 작은 카탈로그로 실제 검색 흐름을 끝까지 돌린다.

사진은 위쪽 절반이 빨강(상의), 아래쪽 절반이 파랑(바지). 가짜 인코더는 평균 색을 벡터로 바꿔서
빨강 → 상의 상품 방향(e0), 파랑 → 바지 상품 방향(e2), 사진 전체 → 둘의 중간이 된다.
"""

import copy
import json

import numpy as np
import pytest
from PIL import Image

from src.common.config import load_config
from src.common.faiss_utils import build_index, l2_normalize, save_index
from src.common.schemas import SearchType
from src.search.detector import (
    DETECTOR,
    GROUNDING,
    LOW_SCORE,
    USER,
    Detection,
    best_in_slot,
    select_detections,
)
from src.search.query_parser import Intent
from src.search.searcher import Searcher
from src.search.style_search import (
    PHOTO_ITEM,
    PHOTO_STYLE,
    WHOLE_PHOTO,
    WHOLE_PHOTO_TEXT,
    StyleSearcher,
)

CFG = load_config("style_search")
DIM = 8
E = np.eye(DIM, dtype="float32")
TOP_BOX, BOTTOM_BOX = (0, 0, 100, 100), (0, 100, 100, 200)
INDEX_CFG = {"type": "flat", "nlist": 4, "m": 2, "nbits": 8, "nprobe": 2, "hnsw_m": 8, "ef_construction": 40, "ef_search": 16}

# (소분류, 색, 고객층, 상품 사진 벡터)
PRODUCTS = [
    ("T-shirt", "Black", "Ladieswear", E[0] + 0.3 * E[6]),        # 0: 긴소매 쪽
    ("T-shirt", "Red", "Ladieswear", 0.9 * E[0] + 0.3 * E[7]),    # 1: 민소매 쪽
    ("Trousers", "Blue", "Ladieswear", E[2]),                    # 2
    ("Trousers", "Black", "Baby/Children", 0.95 * E[2] + 0.1 * E[3]),  # 3: 아동복
    ("Sneakers", "White", "Ladieswear", E[4]),                   # 4
    ("Shorts", "Blue", "Ladieswear", 0.8 * E[2] + 0.2 * E[5]),    # 5
]
TEXTS = {"a garment with long sleeves": E[6], "a sleeveless garment": E[7]}


class ColorEncoder:
    model_name = "fake/clip"
    device = "cpu"

    def encode_images(self, images, batch_size=32):
        means = [np.asarray(im, dtype="float32").reshape(-1, 3).mean(0) for im in images]
        return l2_normalize(np.stack([r * E[0] + g * E[4] + b * E[2] for r, g, b in means]))

    def encode_texts(self, texts, batch_size=64):
        return l2_normalize(np.stack([TEXTS.get(t, E[5]) for t in texts]))


class FakeDetector:
    def __init__(self, raw):
        self.raw = raw

    def detect_raw(self, image):
        assert self.raw is not None, "상품 사진에는 검출을 하지 않아야 함"
        return self.raw


class FakeGrounding:
    def __init__(self, box):
        self.box, self.calls = box, []

    def find(self, image, kinds):
        self.calls.append(kinds)
        return Detection(kinds[0], CFG["kinds"][kinds[0]]["slot"], kinds[0], 0.3, self.box, GROUNDING) if self.box else None


def _photo() -> Image.Image:
    im = Image.new("RGB", (100, 200), (0, 0, 255))
    im.paste((255, 0, 0), (0, 0, 100, 100))
    return im


@pytest.fixture()
def searcher(tmp_path) -> Searcher:
    import pandas as pd

    image_emb = l2_normalize(np.stack([p[3] for p in PRODUCTS]))
    catalog = pd.DataFrame({
        "product_id": [f"hm_{i:010d}" for i in range(len(PRODUCTS))],
        "name": [f"item {i}" for i in range(len(PRODUCTS))],
        "price": [10.0] * len(PRODUCTS),
        "category_l1": [p[2] for p in PRODUCTS],
        "category_l3": [p[0] for p in PRODUCTS],
        "color": [p[1] for p in PRODUCTS],
    })
    catalog.to_parquet(tmp_path / "products.parquet", index=False)
    d = tmp_path / "fake__clip"
    d.mkdir()
    np.save(d / "image_emb.npy", image_emb)
    np.save(d / "text_emb.npy", image_emb)
    catalog[["product_id"]].to_parquet(d / "product_ids.parquet", index=False)
    save_index(build_index(image_emb, INDEX_CFG), d / "image.index")
    save_index(build_index(image_emb, INDEX_CFG), d / "text.index")
    (d / "meta.json").write_text(json.dumps({"model_name": "fake/clip", "pad_to_square": True}))
    cfg = {
        "clip": {"model_name": "fake/clip", "serve_device": "cpu"}, "index": INDEX_CFG,
        "retrieval": {"candidate_k": 4}, "text_query": {"image_weight": 0.5}, "hybrid": {"text_weight": 0.5},
        "paths": {"catalog": str(tmp_path / "products.parquet"), "index_dir": str(tmp_path)},
    }
    return Searcher(cfg, encoder=ColorEncoder())


def _style(searcher, raw, grounding=None) -> StyleSearcher:
    cfg = copy.deepcopy(CFG)
    cfg["detector"]["crop_margin"] = 0.0
    cfg["audience"]["mode"] = "off"
    cfg["photo_type"]["margin"] = -1.0 if raw is not None else 1.0  # None: 상품 사진으로 판단하게
    return StyleSearcher(searcher, FakeDetector(raw), cfg, grounding)


def _raw(kind, box, score=0.9):
    """검출 모델 출력 (라벨, 점수, 박스)."""
    return CFG["kinds"][kind]["labels"][0], score, box


def _ids(group):
    return group.result.product_ids


def test_photo_only_searches_each_item_in_its_category(searcher) -> None:
    s = _style(searcher, [_raw("pants", BOTTOM_BOX), _raw("tee", TOP_BOX)])
    out = s.search(_photo())
    assert out.intent == Intent.STYLE
    assert [g.slot for g in out.groups] == ["top", "bottom"]  # 부위 순서 (설정의 slots 순)
    top, bottom = out.groups
    assert _ids(top) == ["hm_0000000000", "hm_0000000001"] and top.basis == PHOTO_ITEM and top.box == TOP_BOX
    assert _ids(bottom)[0] == "hm_0000000002" and "hm_0000000005" in _ids(bottom)  # 하의 전체에서 (반바지 포함)
    assert "hm_0000000004" not in _ids(bottom)                                       # 다른 부위는 제외
    assert top.result.items[0]["category"] == "T-shirt" and top.result.items[0]["color"] == "Black"


def test_match_for_missing_item_uses_whole_photo_style(searcher) -> None:
    s = _style(searcher, [_raw("tee", TOP_BOX), _raw("pants", BOTTOM_BOX)])
    out = s.search(_photo(), "이 옷이랑 어울리는 신발")
    assert out.intent == Intent.MATCH
    [shoes] = out.groups
    assert shoes.slot == "shoes" and shoes.basis == PHOTO_STYLE and shoes.box is None
    assert _ids(shoes) == ["hm_0000000004"]


def test_color_filter_on_requested_item(searcher) -> None:
    s = _style(searcher, [_raw("tee", TOP_BOX), _raw("pants", BOTTOM_BOX)])
    [pants] = s.search(_photo(), "이 바지 검은색으로").groups
    assert pants.slot == "bottom" and pants.filters == ["color:black"]
    assert _ids(pants) == ["hm_0000000003"]


def test_detail_direction_relative_to_photo_item(searcher) -> None:
    s = _style(searcher, [_raw("tee", TOP_BOX), _raw("pants", BOTTOM_BOX)])
    top, bottom = s.search(_photo(), "소매 더 짧게").groups
    assert top.filters == ["sleeve:down"] and _ids(top) == ["hm_0000000001"]  # 기준 옷보다 소매 짧은 쪽만
    assert bottom.filters == [] and len(_ids(bottom)) == 3                     # 하의에는 소매 조건 없음


def test_requested_kind_with_other_kind_in_same_slot(searcher) -> None:
    s = _style(searcher, [_raw("shorts", BOTTOM_BOX)])
    [g] = s.search(_photo(), "바지").groups
    assert g.basis == PHOTO_ITEM and g.kind == "shorts"     # 사진 속 반바지를 기준으로
    assert set(_ids(g)) == {"hm_0000000002", "hm_0000000003"}  # 요청한 바지 종류에서


def test_no_detection_falls_back_to_whole_photo(searcher) -> None:
    [g] = _style(searcher, []).search(_photo()).groups
    assert g.slot == "all" and g.basis == WHOLE_PHOTO and g.result.total_count == len(PRODUCTS)


def test_product_photo_skips_detection(searcher) -> None:
    s = _style(searcher, None)  # 상품 사진으로 판단 → 검출 안 함 (FakeDetector가 불리면 실패)
    out = s.search(_photo())
    assert not out.is_person_photo and out.detections == []
    [g] = out.groups
    assert g.basis == WHOLE_PHOTO and g.result.total_count == 4  # 일반 이미지 검색과 같음 (FAISS 후보 candidate_k개)
    assert g.result.search_type == SearchType.IMAGE and g.result.items[0]["category"]

    [g] = s.search(_photo(), "이 바지 검은색으로").groups  # 텍스트로 종류를 정하면 그 종류에서
    assert g.slot == "bottom" and g.basis == WHOLE_PHOTO and _ids(g) == ["hm_0000000003"]


class FakeProbe:
    """사진이 요청한 색일 확률을 정해 둔 색 분류기."""

    def __init__(self, prob):
        self.prob = prob

    def class_prob(self, name, q_image, values):
        assert name == "color" and values
        return self.prob

    def bonus(self, q_image, rows, weights):
        return np.zeros(len(rows))


def test_product_photo_description_uses_general_hybrid(searcher) -> None:
    searcher.attributes = FakeProbe(0.9)  # 사진이 요청한 색과 같음 → 색 변경이 아니라 설명
    [g] = _style(searcher, None).search(_photo(), "white sneakers").groups
    assert g.slot == "all" and g.basis == WHOLE_PHOTO_TEXT and g.result.search_type == SearchType.HYBRID
    assert g.filters == ["text:white sneakers"] and g.result.items[0]["color"]

    [g] = _style(searcher, None).search(_photo(), "floral").groups  # 사전에 없는 영어 단어만 있어도
    assert g.basis == WHOLE_PHOTO_TEXT and g.filters == ["text:floral"]


def test_product_photo_color_change_uses_color_filter(searcher) -> None:
    searcher.attributes = FakeProbe(0.01)  # 사진이 요청한 색이 아님 → 그 색 상품 안에서 가장 비슷한 것
    for text in ("white", "흰색으로"):
        [g] = _style(searcher, None).search(_photo(), text).groups
        assert g.basis == WHOLE_PHOTO and g.filters == ["color:white"]
        assert _ids(g) == ["hm_0000000004"] and g.result.search_type == SearchType.IMAGE

    searcher.attributes = FakeProbe(0.9)  # 같은 색이어도 한국어는 CLIP에 넣지 않으므로 색 조건
    [g] = _style(searcher, None).search(_photo(), "흰색으로").groups
    assert g.basis == WHOLE_PHOTO and g.filters == ["color:white"]


def test_product_photo_routes_that_keep_conditions(searcher) -> None:
    s = _style(searcher, None)
    [g] = s.search(_photo(), "바지").groups  # 한국어 종류
    assert g.slot == "bottom" and set(_ids(g)) == {"hm_0000000002", "hm_0000000003"}

    [g] = s.search(_photo(), "소매 더 짧게").groups  # 디테일
    assert g.filters == ["sleeve:down"]

    [g] = s.search(_photo(), "이 옷이랑 어울리는 신발").groups  # 어울림
    assert g.slot == "shoes" and _ids(g) == ["hm_0000000004"]

    [g] = s.search(_photo(), audience="women").groups  # 고객층을 직접 고르면 그 안에서
    assert g.result.total_count == 5 and "hm_0000000003" not in _ids(g)


def test_product_photo_unknown_korean_text_is_image_search(searcher) -> None:
    [g] = _style(searcher, None).search(_photo(), "예쁜 옷 찾아줘").groups
    assert g.basis == WHOLE_PHOTO and g.filters == [] and g.result.search_type == SearchType.IMAGE


def test_person_photo_free_text_scores_each_item(searcher) -> None:
    searcher.cfg["hybrid"]["text_weight"] = 0.8
    s = _style(searcher, [_raw("tee", TOP_BOX), _raw("pants", BOTTOM_BOX)])
    plain = s.search(_photo()).groups[1]
    assert _ids(plain)[0] == "hm_0000000002"

    top, bottom = s.search(_photo(), "floral").groups  # 가짜 인코더의 문장 벡터는 e5 → 5번 상품 쪽
    assert bottom.filters == ["text:floral"] and bottom.result.search_type == SearchType.HYBRID
    assert _ids(bottom)[0] == "hm_0000000005" and bottom.basis == PHOTO_ITEM
    assert set(_ids(top)) == {"hm_0000000000", "hm_0000000001"}  # 그 부위 안에서만


def test_audience_filter(searcher) -> None:
    s = _style(searcher, [_raw("pants", BOTTOM_BOX)])
    out = s.search(_photo(), audience="women")
    assert out.audience == "women" and _ids(out.groups[0])[0] == "hm_0000000002"
    assert "hm_0000000003" not in _ids(out.groups[0])  # 아동복 제외


def test_requested_item_uses_low_score_box(searcher) -> None:
    s = _style(searcher, [_raw("tee", TOP_BOX), _raw("pants", BOTTOM_BOX, score=0.3)])
    assert [g.slot for g in s.search(_photo()).groups] == ["top"]  # 텍스트 없으면 기준 점수 미만은 안 씀

    [g] = s.search(_photo(), "바지").groups  # 콕 집으면 낮은 점수 박스라도 사진 속 그 옷 기준
    assert g.basis == PHOTO_ITEM and g.box == BOTTOM_BOX and g.box_source == LOW_SCORE and g.detection_score == 0.3
    assert set(_ids(g)) == {"hm_0000000002", "hm_0000000003"}


def test_grounding_only_when_detector_has_no_candidate(searcher) -> None:
    grounding = FakeGrounding(BOTTOM_BOX)
    s = _style(searcher, [_raw("tee", TOP_BOX), _raw("pants", BOTTOM_BOX, score=0.3)], grounding)
    s.search(_photo(), "바지")
    s.search(_photo(), "상의")
    assert grounding.calls == []  # 낮은 점수 후보가 있거나 이미 찾은 부위면 부르지 않음

    s = _style(searcher, [_raw("tee", TOP_BOX)], grounding)
    [g] = s.search(_photo(), "바지").groups
    assert grounding.calls == [["pants"]]
    assert g.basis == PHOTO_ITEM and g.box == BOTTOM_BOX and g.box_source == GROUNDING

    [g] = _style(searcher, [_raw("tee", TOP_BOX)], FakeGrounding(None)).search(_photo(), "바지").groups
    assert g.basis == PHOTO_STYLE and g.box is None  # 그래도 못 찾으면 사진 전체 분위기로


def test_grounding_always_mode(searcher) -> None:
    grounding = FakeGrounding((0, 120, 100, 200))
    s = _style(searcher, [_raw("tee", TOP_BOX), _raw("pants", BOTTOM_BOX)], grounding)
    s.cfg["grounding"]["mode"] = "always"
    assert len(s.search(_photo()).groups) == 2 and grounding.calls == []  # 콕 집은 부위가 없으면 안 부름

    out = s.search(_photo(), "바지")  # YOLOS가 찾았어도 Grounding DINO 박스로
    [g] = out.groups
    assert grounding.calls == [["pants"]] and g.box == (0, 120, 100, 200) and g.box_source == GROUNDING
    assert [d.source for d in out.detections] == [DETECTOR, GROUNDING]  # 그 부위의 YOLOS 박스는 뺌

    s = _style(searcher, [_raw("tee", TOP_BOX), _raw("pants", BOTTOM_BOX, score=0.3)], FakeGrounding(None))
    s.cfg["grounding"]["mode"] = "always"
    [g] = s.search(_photo(), "바지").groups  # 못 찾으면 fallback과 같음 (낮은 점수 후보)
    assert g.box == BOTTOM_BOX and g.box_source == LOW_SCORE


def test_user_box_searches_only_that_item(searcher) -> None:
    s = _style(searcher, [_raw("tee", TOP_BOX), _raw("pants", BOTTOM_BOX)])
    out = s.search(_photo(), box=BOTTOM_BOX)
    [g] = out.groups
    assert g.slot == "bottom" and g.kind == "pants"  # 그 자리에서 검출 모델이 본 종류
    assert g.box_source == USER and g.detection_score is None and out.detections[-1].source == USER
    assert _ids(g)[0] == "hm_0000000002" and "hm_0000000005" in _ids(g)  # 하의 전체에서

    [g] = s.search(_photo(), "바지", box=BOTTOM_BOX).groups  # 텍스트로 종류를 말하면 그 종류에서
    assert set(_ids(g)) == {"hm_0000000002", "hm_0000000003"}

    [g] = _style(searcher, []).search(_photo(), box=BOTTOM_BOX).groups  # 무슨 옷인지 모르면 전체에서
    assert g.slot == "all" and g.kind is None and g.result.total_count == len(PRODUCTS)
    assert _ids(g)[0] == "hm_0000000002"  # 사진 전체가 아니라 지정한 영역(파랑) 기준


def test_user_box_on_product_photo_skips_detection(searcher) -> None:
    [g] = _style(searcher, None).search(_photo(), box=TOP_BOX).groups
    assert g.basis == PHOTO_ITEM and g.box == TOP_BOX and _ids(g)[0] == "hm_0000000000"


def test_match_with_user_box_uses_that_item(searcher) -> None:
    s = _style(searcher, [])
    [g] = s.search(_photo(), "이 옷이랑 어울리는 상의", box=TOP_BOX).groups
    assert g.slot == "top" and g.basis == PHOTO_STYLE
    assert g.result.scores[0] == pytest.approx(1 / np.sqrt(1.09), abs=1e-4)  # 사진 전체(빨강+파랑)가 아니라 영역(빨강) 기준

    out = s.search(_photo(), "이 바지랑 어울리는 상의", box=BOTTOM_BOX)
    assert out.detections[-1].kind == "pants"  # match에서는 "이 바지"가 지정한 영역의 종류


def test_best_in_slot() -> None:
    raw = [
        ("pants", 0.3, (0, 100, 100, 200)),
        ("shorts", 0.4, (0, 120, 100, 200)),
        ("skirt", 0.05, (0, 100, 100, 200)),                     # requested_threshold 미만
        ("shoe", 0.45, (0, 180, 20, 200)),                       # 다른 부위
        ("pants", 0.48, (0, 100, 3, 103)),                       # 너무 작음
    ]
    d = best_in_slot(raw, (100, 200), CFG, "bottom")
    assert (d.kind, d.score, d.source) == ("shorts", 0.4, LOW_SCORE)
    assert best_in_slot(raw, (100, 200), CFG, "top") is None


def test_select_detections() -> None:
    cfg = load_config("style_search")
    raw = [
        ("sleeve", 0.99, (0, 0, 50, 50)),                        # 부분 라벨 → 버림
        ("top, t-shirt, sweatshirt", 0.9, (0, 0, 100, 100)),
        ("sweater", 0.7, (2, 2, 100, 100)),                      # 같은 자리 다른 라벨 → 점수 낮아서 버림
        ("pants", 0.4, (0, 100, 100, 200)),                      # 점수 미달
        ("shoe", 0.8, (0, 180, 20, 200)),
        ("shoe", 0.75, (80, 180, 100, 200)),                     # 같은 종류는 하나만
        ("hat", 0.9, (0, 0, 3, 3)),                              # 너무 작음
    ]
    kept = select_detections(raw, (100, 200), cfg)
    assert [(d.kind, d.slot) for d in kept] == [("tee", "top"), ("shoes", "shoes")]
