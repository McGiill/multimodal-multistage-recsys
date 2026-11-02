"""스타일 사진 검색의 텍스트 해석: 요청 종류, 찾을 옷 / 기준 옷, 색, 디테일."""

import pytest

from src.common.config import load_config
from src.search.query_parser import Intent, parse_query

CFG = load_config("style_search")


@pytest.mark.parametrize(
    "text, intent, kinds, anchors",
    [
        (None, Intent.STYLE, [], []),
        ("이런 스타일 옷 추천해줘", Intent.STYLE, [], []),
        ("사진 속 신발 찾아줘", Intent.FIND, ["shoes"], []),
        ("반바지", Intent.FIND, ["shorts"], []),                 # "바지"로 잡히지 않음
        ("이 옷이랑 어울리는 바지 추천해줘", Intent.MATCH, ["pants"], []),
        ("바지에 어울리는 신발", Intent.MATCH, ["shoes"], ["pants"]),  # match 단어 앞 = 기준 옷
        ("후드티랑 어울리는 하의", Intent.MATCH, ["pants", "shorts", "skirt"], ["hoodie"]),  # 부위 → 종류 전부
        ("pants that go with this hoodie", Intent.MATCH, ["pants"], ["hoodie"]),  # 영어는 앞이 찾을 옷
        ("matching jeans", Intent.MATCH, ["pants"], []),
        ("어울리는 거 추천", Intent.MATCH, [], []),
    ],
)
def test_intent_and_targets(text, intent, kinds, anchors) -> None:
    p = parse_query(text, CFG)
    assert p.intent == intent
    assert p.kinds == kinds
    assert p.anchor_kinds == anchors


def test_colors_map_to_catalog_values() -> None:
    p = parse_query("이 바지 빨간색으로", CFG)
    assert p.kinds == ["pants"] and p.color_names == ["red"]
    assert {"Red", "Dark Red"} <= set(p.colors)
    assert parse_query("네이비 청바지", CFG).colors == ["Dark Blue"]
    assert parse_query("shredded jeans", CFG).colors == []  # 영어는 단어 경계로만 ("red" 아님)


@pytest.mark.parametrize(
    "text, details",
    [
        ("소매는 더 짧게 총장은 길게", {"sleeve": -1, "length": 1}),  # 가까운 방향 단어끼리 짝
        ("더 짧은 거", {"length": -1}),                                # 방향만 있으면 기장
        ("반팔 셔츠", {"sleeve": -1}),                                 # 단어 하나에 디테일 + 방향
        ("긴팔", {"sleeve": 1}),                                       # "긴"(방향)으로 잡히지 않음
        ("하이웨이스트 와이드 팬츠", {"rise": 1, "fit": 1}),
        ("short sleeve red tee", {"sleeve": -1}),
    ],
)
def test_details(text, details) -> None:
    assert parse_query(text, CFG).details == details
