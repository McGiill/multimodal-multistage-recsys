"""텍스트 검색 Query Understanding: 가격 규칙, 사전 규칙 대체, LLM 출력 검증·합치기 (실제 모델 없이)."""

import json

import pytest

from src.common.config import load_config
from src.search.query_filters import LLMExtractor, QueryFilters, QueryUnderstanding, parse_price, parse_rules

CFG = load_config("query_filters")
DICTIONARY = load_config(CFG["dictionary"])


@pytest.mark.parametrize(
    "text, lo, hi, rest",
    [
        ("black knit under $50", None, 50, "black knit"),
        ("jeans less than 40 dollars", None, 40, "jeans"),
        ("white shirt over $25", 25, None, "white shirt"),
        ("grey hoodie $20-$35", 20, 35, "grey hoodie"),
        ("dress between $20 and $40", 20, 40, "dress"),
        ("coat 30 to 60 dollars", 30, 60, "coat"),
        ("boots $80 or less", None, 80, "boots"),
        ("bag at least $15 and under $30", 15, 30, "bag"),
        ("$50 jeans", None, None, "$50 jeans"),           # 조건 없는 금액은 필터 아님
        ("size 8 to 10 shoes", None, None, "size 8 to 10 shoes"),  # $ / dollars 없는 범위는 가격 아님
        ("t-shirt under 9.99", None, 9.99, "t-shirt"),
    ],
)
def test_parse_price(text, lo, hi, rest) -> None:
    assert parse_price(text, CFG["price"]) == (lo, hi, rest)


def test_rules_fallback() -> None:
    f = parse_rules("men's striped navy shirt", DICTIONARY, CFG)
    assert f.kinds == ["shirt"] and f.colors == ["navy"] and f.patterns == ["stripe"] and f.audience == "men"
    assert parse_rules("dress for women", DICTIONARY, CFG).audience == "women"  # "women" 안의 "men"으로 잡히지 않음
    assert not parse_rules("something warm", DICTIONARY, CFG)


def _understanding(outputs: dict[str, str]) -> QueryUnderstanding:
    """generate가 정해진 문자열을 돌려주는 LLM (모델 로딩 없음)."""
    qu = QueryUnderstanding(CFG, DICTIONARY)
    qu.llm._load = lambda: True
    qu.llm.generate = lambda query: outputs[query]
    return qu


def _answer(**fields) -> str:
    empty = {"kind": [], "color": [], "pattern": [], "audience": None, "price_min": None, "price_max": None}
    return json.dumps(empty | fields)


def test_llm_result_is_used_and_price_comes_from_rules() -> None:
    qu = _understanding({
        "men's navy jeans less than $40": _answer(kind=["pants"], color=["navy"], audience="men", price_min=40, price_max=40),
    })
    f = qu.parse("men's navy jeans less than $40")
    assert f.source == "llm" and f.kinds == ["pants"] and f.colors == ["navy"] and f.audience == "men"
    assert (f.price_min, f.price_max) == (None, 40)  # LLM의 min=40(방향 오류) 대신 규칙 값
    assert f.text == "men's navy jeans"


def test_llm_price_only_when_number_in_query() -> None:
    qu = _understanding({
        "warm winter coat": _answer(kind=["coat"], price_max=100),       # 검색어에 없는 가격 → 버림
        "coat around 60 bucks": _answer(kind=["coat"], price_max=60),     # 규칙이 못 찾은 가격 → 숫자가 있으면 사용
    })
    assert qu.parse("warm winter coat").price_max is None
    assert qu.parse("coat around 60 bucks").price_max == 60


def test_llm_failure_falls_back_to_rules() -> None:
    qu = _understanding({"black boots": '{"kind": ["shoes"], "col'})  # 시간 초과로 덜 끝난 JSON
    f = qu.parse("black boots")
    assert f.source == "rules" and f.kinds == ["shoes"] and f.colors == ["black"]

    qu.llm._load = lambda: False  # 모델 로딩 실패
    assert qu.parse("red dress").source == "rules"


def test_llm_validate_drops_unknown_values() -> None:
    llm = LLMExtractor(CFG["llm"], kinds=["knit"], colors=["black"], patterns=["stripe"], audiences=["women"])
    out = llm.validate({"kind": ["knit", "spaceship", "knit"], "color": ["charcoal grey"], "pattern": [],
                        "audience": "aliens", "price_min": 0, "price_max": "50", "visual_description": " thick wool "})
    assert out == {"kind": ["knit"], "color": [], "pattern": [], "audience": None, "price_min": None, "price_max": None,
                   "visual_description": "thick wool"}


def test_llm_lists_capped_at_max_array_length() -> None:
    kinds = ["shirt", "tee", "knit", "hoodie", "cardigan"]
    llm = LLMExtractor(CFG["llm"] | {"max_array_length": 3}, kinds=kinds, colors=[], patterns=[], audiences=[])
    assert llm.schema()["properties"]["kind"]["maxItems"] == 3  # 생성 단계에서 제한
    assert llm.validate({"kind": kinds})["kind"] == ["shirt", "tee", "knit"]  # 그래도 넘치면 검증에서 자름


def test_llm_prompt_and_schema() -> None:
    few_shot = CFG["llm"] | {"few_shot": True, "describe": {"enabled": False}}  # 기본 모델용 형식
    llm = LLMExtractor(few_shot, kinds=["knit"], colors=["black"], patterns=["stripe"], audiences=["women"])
    schema = llm.schema()
    assert schema["properties"]["kind"]["items"]["enum"] == ["knit"]
    assert "visual_description" not in schema["properties"]
    messages = llm.messages("q")
    assert messages[-1] == {"role": "user", "content": "q"}
    assert len(messages) == 2 + 2 * len(CFG["llm"]["examples"])
    assert json.loads(messages[2]["content"])["price_max"] == 50  # 빈 칸은 [] / null로 채운 예시 답


def test_llm_lora_format() -> None:
    """LoRA 모델 형식: 예시 없음, 상황 풀이 지시 + 칸 (마지막 칸)."""
    lora = CFG["llm"] | {"few_shot": False, "describe": CFG["llm"]["describe"] | {"enabled": True}}
    llm = LLMExtractor(lora, kinds=["knit"], colors=["black"], patterns=["stripe"], audiences=["women"])
    messages = llm.messages("q")
    assert len(messages) == 2 and CFG["llm"]["describe"]["prompt"] in messages[0]["content"]
    assert list(llm.schema()["properties"])[-1] == "visual_description"


def test_use_llm_false_uses_rules_only() -> None:
    calls = []
    qu = _understanding({})
    qu.llm.generate = lambda query: calls.append(query) or _answer(kind=["coat"])
    f = qu.parse("black boots under $40", use_llm=False)
    assert calls == [] and f.source == "rules" and f.kinds == ["shoes"] and f.price_max == 40
    assert qu.parse("black boots under $40").source == "llm"


def test_cache_skips_generation() -> None:
    calls = []
    qu = _understanding({})
    qu.llm.generate = lambda query: calls.append(query) or _answer(kind=["coat"])
    qu.parse("long coat")
    qu.parse("long coat")
    assert calls == ["long coat"]


def test_catalog_values() -> None:
    qu = QueryUnderstanding(CFG | {"llm": {"enabled": False}}, DICTIONARY)
    values = qu.catalog_values(QueryFilters(kinds=["knit"], colors=["black"], patterns=["stripe"], audience="kids"))
    assert values == {"category_l3": DICTIONARY["kinds"]["knit"]["l3"], "color": ["Black"], "pattern": ["Stripe"],
                      "category_l1": ["Baby/Children"]}
    assert qu.catalog_values(QueryFilters()) == {"category_l3": None, "color": None, "pattern": None, "category_l1": None}
