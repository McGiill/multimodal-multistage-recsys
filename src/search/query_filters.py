"""Query Understanding (텍스트 검색): 영어 검색어 속 조건 → 필터. 설정 configs/query_filters.yaml.

    "black knit under $50"       → kind=knit, color=black, price_max=50
    "men's grey hoodie $20-$35"  → kind=hoodie, color=grey, audience=men, price 20~35
    "warm winter coat"           → kind=coat ("warm winter"는 필터 없이 CLIP·BM25 점수에만)

1. 가격: 규칙. 소형 LLM은 이하/이상 방향을 자주 틀려서 ("less than $40" → min=max=40) 규칙이 찾으면 규칙 값을 쓴다.
   규칙이 못 찾고 LLM만 찾은 가격은 그 숫자가 검색어에 있을 때만 쓴다. 가격 표현은 문장에서 빼고 인코딩한다.
2. 종류·색·무늬·고객층: 소형 LLM(Qwen3-0.6B, 4B 라벨로 LoRA 학습)이 정해진 보기 중에서 골라 JSON으로 답한다
   (lm-format-enforcer로 형식 강제). LLM이 꺼져 있거나, 요청이 LLM을 쓰지 않거나 (use_llm=False),
   실패하면 (로딩 실패 / 형식 깨짐 / 시간 초과) 사전 단어 매칭 규칙으로 대체.
   describe가 켜져 있으면 상황 표현을 보이는 특징으로 풀어 쓴 visual_description도 받는다 (LoRA 모델의 출력 형식).
3. 카탈로그 값으로 변환: kind → category_l3, color → color, pattern → pattern, audience → category_l1.
"""

from __future__ import annotations

import json
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from src.common.config import ROOT
from src.search.query_parser import parse_query

_NUMBER = r"(\d+(?:\.\d+)?)"
_CURRENCY = r"(?:dollars?|usd|bucks)"


@dataclass
class QueryFilters:
    kinds: list[str] = field(default_factory=list)     # style_search.yaml kinds 이름
    colors: list[str] = field(default_factory=list)    # style_search.yaml colors 이름
    patterns: list[str] = field(default_factory=list)  # query_filters.yaml patterns 이름
    audience: str | None = None                        # women | men | kids
    price_min: float | None = None                     # 달러 (카탈로그 price와 같은 단위)
    price_max: float | None = None
    text: str = ""                                     # 가격 표현을 뺀 문장 (CLIP·BM25 점수에 사용)
    source: str = "rules"                              # 종류·색·무늬·고객층을 뽑은 곳: llm | rules
    description: str = ""                              # LLM의 visual_description (아직 검색 점수에는 쓰지 않음)

    def __bool__(self) -> bool:
        return bool(self.kinds or self.colors or self.patterns or self.audience
                    or self.price_min is not None or self.price_max is not None)


def _words_regex(words: list[str]) -> str:
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


def _word_pattern(word: str) -> re.Pattern[str]:
    return re.compile(r"(?<![a-z0-9])" + re.escape(word.lower()) + r"(?![a-z0-9])")


# ---------------------------------------------------------------------------- 가격 (규칙)
def parse_price(text: str, cfg: dict[str, Any]) -> tuple[float | None, float | None, str]:
    """검색어 속 가격 조건 → (최소 달러, 최대 달러, 가격 표현을 뺀 문장). cfg = query_filters.yaml의 price."""
    low = text.lower()
    amount = rf"\$\s*{_NUMBER}|{_NUMBER}\s*\$?(?:\s*{_CURRENCY})?"
    rules = [  # (정규식, 종류) — 범위를 먼저 찾아서 "$20-$35"의 "$35"가 따로 잡히지 않게
        (rf"(?:between\s+|from\s+)?(\$\s*)?{_NUMBER}(\s*{_CURRENCY})?\s*(?:-|–|~|to|and)\s*(\$\s*)?{_NUMBER}(\s*{_CURRENCY})?",
         "range"),
        (rf"(?:{_words_regex(cfg['max_words'])})\s+(?:{amount})", "max"),
        (rf"(?:{_words_regex(cfg['min_words'])})\s+(?:{amount})", "min"),
        (rf"(?:{amount})\s+(?:{_words_regex(cfg['max_suffixes'])})", "max"),
        (rf"(?:{amount})\s+(?:{_words_regex(cfg['min_suffixes'])})", "min"),
    ]
    lo: float | None = None
    hi: float | None = None
    used = [False] * len(low)
    spans: list[tuple[int, int]] = []
    for pattern, kind in rules:
        for m in re.finditer(r"(?<![a-z0-9])(?:" + pattern + r")(?![a-z0-9])", low):
            if any(used[m.start():m.end()]):
                continue
            numbers = [float(g) for g in m.groups() if g and re.fullmatch(_NUMBER, g)]
            if kind == "range":
                if not any(g for i, g in enumerate(m.groups()) if i in (0, 2, 3, 5)):
                    continue  # 범위는 한쪽에라도 $ / dollars가 있어야 가격 ("size 8 to 10" 제외)
                a, b = sorted(numbers[:2])
                lo, hi = a, b
            elif kind == "max":
                hi = numbers[0] if hi is None else min(hi, numbers[0])
            else:
                lo = numbers[0] if lo is None else max(lo, numbers[0])
            used[m.start():m.end()] = [True] * (m.end() - m.start())
            spans.append((m.start(), m.end()))
    merged: list[list[int]] = []
    for s, e in sorted(spans):  # 가격 표현 사이 연결어도 같이 뺀다 ("at least $15 and under $30")
        if merged and re.fullmatch(r"\s*(?:and|,|&)?\s*", low[merged[-1][1]:s]):
            merged[-1][1] = e
        else:
            merged.append([s, e])
    rest = text
    for s, e in reversed(merged):
        rest = rest[:s] + " " + rest[e:]
    return lo, hi, " ".join(rest.split())


# ---------------------------------------------------------------------------- 종류·색·무늬·고객층 (규칙)
def parse_rules(text: str, dictionary: dict[str, Any], cfg: dict[str, Any]) -> QueryFilters:
    """사전 단어 매칭. 종류·색은 스타일 사진 검색과 같은 parse_query, 무늬·고객층은 query_filters.yaml 사전."""
    parsed = parse_query(text, dictionary)
    low = text.lower()
    patterns = [name for name, p in cfg["patterns"].items() if any(_word_pattern(w).search(low) for w in p["words"])]
    audience = next((name for name, a in cfg["audience"].items()
                     if any(_word_pattern(w).search(low) for w in a["words"])), None)
    return QueryFilters(kinds=parsed.kinds, colors=parsed.color_names, patterns=patterns, audience=audience,
                        text=text, source="rules")


# ---------------------------------------------------------------------------- 종류·색·무늬·고객층 (소형 LLM)
class LLMExtractor:
    """정해진 보기 중에서 고르는 JSON 추출. 처음 쓸 때 모델을 읽고, 읽기에 실패하면 이후 계속 None (→ 규칙)."""

    def __init__(self, cfg: dict[str, Any], kinds: list[str], colors: list[str], patterns: list[str],
                 audiences: list[str]) -> None:
        self.cfg = cfg
        self.options = {"kind": kinds, "color": colors, "pattern": patterns, "audience": audiences}
        self._model: Any = None
        self._failed = False
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, dict[str, Any] | None] = OrderedDict()

    def schema(self) -> dict[str, Any]:
        def options(name: str) -> dict[str, Any]:
            return {"type": "array", "items": {"type": "string", "enum": self.options[name]},
                    "maxItems": self.cfg["max_array_length"]}

        def nullable(spec: dict[str, Any]) -> dict[str, Any]:
            return {"anyOf": [spec, {"type": "null"}]}

        props = {
            "kind": options("kind"), "color": options("color"), "pattern": options("pattern"),
            "audience": nullable({"type": "string", "enum": self.options["audience"]}),
            "price_min": nullable({"type": "number"}), "price_max": nullable({"type": "number"}),
        }
        if self._describe:
            props["visual_description"] = {"type": "string", "maxLength": self.cfg["describe"]["max_length"]}
        return {"type": "object", "properties": props, "required": list(props)}

    @property
    def _describe(self) -> bool:
        return bool(self.cfg.get("describe", {}).get("enabled"))

    def messages(self, query: str) -> list[dict[str, str]]:
        """시스템 프롬프트 (+ 상황 풀이 지시) + 보기 목록 (+ few-shot 예시). LoRA 모델은 학습 때와 같이 예시 없이."""
        system = self.cfg["system_prompt"]
        if self._describe:
            system += " " + self.cfg["describe"]["prompt"]
        system += " " + " ".join(f"{name} options: {', '.join(values)}." for name, values in self.options.items())
        out = [{"role": "system", "content": system}]
        empty = {"kind": [], "color": [], "pattern": [], "audience": None, "price_min": None, "price_max": None}
        if self._describe:
            empty["visual_description"] = ""
        for ex in self.cfg["examples"] if self.cfg.get("few_shot", True) else []:
            answer = empty | {k: v for k, v in ex.items() if k != "query"}
            out += [{"role": "user", "content": ex["query"]}, {"role": "assistant", "content": json.dumps(answer)}]
        return out + [{"role": "user", "content": query}]

    def _load(self) -> bool:
        if self._model is not None or self._failed:
            return not self._failed
        try:
            import torch
            import transformers

            try:  # lm-format-enforcer 0.11.3이 transformers 5.x에서 없어진 경로를 import한다 → 같은 클래스로 연결
                from transformers.tokenization_utils import PreTrainedTokenizerBase  # noqa: F401
            except ImportError:
                import transformers.tokenization_utils as _tokenization_utils
                _tokenization_utils.PreTrainedTokenizerBase = transformers.PreTrainedTokenizerBase
            from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
            from lmformatenforcer.integrations.transformers import (
                build_token_enforcer_tokenizer_data, build_transformers_prefix_allowed_tokens_fn,
            )
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from src.search.encoder import resolve_device

            device = resolve_device(self.cfg["device"])
            dtype = torch.float16 if device == "cuda" else torch.float32
            local = ROOT / self.cfg["model_name"]  # 레포 기준 폴더 (LoRA 병합 모델)가 있으면 그것, 없으면 HF 모델 이름
            name = str(local) if local.is_dir() else self.cfg["model_name"]
            self.tokenizer = AutoTokenizer.from_pretrained(name)
            self._model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
            self.device = device
            self._parser = JsonSchemaParser(self.schema(), config=CharacterLevelParserConfig(
                max_consecutive_whitespaces=1, force_json_field_order=True,
            ))
            # 토큰 표(어휘 15만 개 → 문자열)는 한 번만 만든다. 쿼리마다 만들면 그것만 수 초 (0.6B GPU 5.0s → 1.3s)
            self._tokenizer_data = build_token_enforcer_tokenizer_data(self.tokenizer)
            self._build_fn = build_transformers_prefix_allowed_tokens_fn
            self._torch = torch
        except Exception as e:  # 모델 없음 / 네트워크 없음 / 라이브러리 없음 → 규칙으로
            print(f"[query_filters] LLM 로딩 실패 → 규칙으로 대체: {type(e).__name__}: {e}")
            self._failed = True
        return not self._failed

    def generate(self, query: str) -> str:
        prompt = self.tokenizer.apply_chat_template(
            self.messages(query), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self._lock, self._torch.no_grad():
            out = self._model.generate(
                **ids, max_new_tokens=self.cfg["max_new_tokens"], max_time=self.cfg["max_time_s"], do_sample=False,
                prefix_allowed_tokens_fn=self._build_fn(self._tokenizer_data, self._parser),
            )
        return self.tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def extract(self, query: str) -> dict[str, Any] | None:
        """JSON 필터. 실패하면 None."""
        if query in self._cache:
            self._cache.move_to_end(query)
            return self._cache[query]
        if not self._load():
            return None
        try:
            result = self.validate(json.loads(self.generate(query)))
        except Exception as e:  # 형식 깨짐 (시간 초과로 덜 끝난 JSON 포함)
            print(f"[query_filters] LLM 출력 실패 → 규칙으로 대체: {type(e).__name__}: {e}")
            result = None
        self._cache[query] = result
        if len(self._cache) > self.cfg["cache_size"]:
            self._cache.popitem(last=False)
        return result

    def validate(self, raw: dict[str, Any]) -> dict[str, Any]:
        """보기에 없는 값은 버리고 목록은 max_array_length개까지 (형식 강제가 있어도 방어적으로)."""
        out: dict[str, Any] = {}
        for name in ("kind", "color", "pattern"):
            values = raw.get(name) or []
            kept = [v for i, v in enumerate(values) if v in self.options[name] and v not in values[:i]]
            out[name] = kept[:self.cfg["max_array_length"]]
        out["audience"] = raw.get("audience") if raw.get("audience") in self.options["audience"] else None
        for name in ("price_min", "price_max"):
            value = raw.get(name)
            out[name] = float(value) if isinstance(value, (int, float)) and value > 0 else None
        if self._describe:
            out["visual_description"] = str(raw.get("visual_description") or "").strip()
        return out


# ---------------------------------------------------------------------------- 합치기 + 카탈로그 값
def _number_in_text(value: float, text: str) -> bool:
    return any(abs(float(n) - value) < 1e-9 for n in re.findall(_NUMBER, text))


class QueryUnderstanding:
    def __init__(self, cfg: dict[str, Any], dictionary: dict[str, Any]) -> None:
        self.cfg = cfg
        self.dictionary = dictionary
        llm = cfg.get("llm", {})
        self.llm = LLMExtractor(
            llm, kinds=list(dictionary["kinds"]), colors=list(dictionary["colors"]),
            patterns=list(cfg["patterns"]), audiences=list(cfg["audience"]),
        ) if llm.get("enabled") else None

    def parse(self, text: str | None, use_llm: bool = True) -> QueryFilters:
        """use_llm=False면 LLM이 켜져 있어도 규칙만 (CPU에서 LLM은 쿼리당 수 초라 화면의 일괄 검색 등에서 끔)."""
        if not text or not text.strip():
            return QueryFilters()
        lo, hi, rest = parse_price(text, self.cfg["price"])
        raw = self.llm.extract(text) if self.llm is not None and use_llm else None
        if raw is None:
            filters = parse_rules(text, self.dictionary, self.cfg)
        else:
            filters = QueryFilters(kinds=raw["kind"], colors=raw["color"], patterns=raw["pattern"],
                                   audience=raw["audience"], source="llm",
                                   description=raw.get("visual_description", ""))
            if lo is None and hi is None:  # 규칙이 못 찾은 가격: 숫자가 검색어에 있을 때만
                lo = raw["price_min"] if raw["price_min"] is not None and _number_in_text(raw["price_min"], text) else None
                hi = raw["price_max"] if raw["price_max"] is not None and _number_in_text(raw["price_max"], text) else None
                if lo is not None and hi is not None and lo > hi:
                    lo = hi = None
        filters.price_min, filters.price_max, filters.text = lo, hi, rest
        return filters

    def catalog_values(self, filters: QueryFilters) -> dict[str, list[str] | None]:
        """카탈로그 컬럼 → 허용 값 (None = 조건 없음)."""
        def union(names: list[str], table: dict[str, Any], key: str) -> list[str] | None:
            values: list[str] = []
            for name in names:
                values += [v for v in table[name][key] if v not in values]
            return values or None

        return {
            "category_l3": union(filters.kinds, self.dictionary["kinds"], "l3"),
            "color": union(filters.colors, self.dictionary["colors"], "values"),
            "pattern": union(filters.patterns, self.cfg["patterns"], "values"),
            "category_l1": union([filters.audience] if filters.audience else [], self.cfg["audience"], "category_l1"),
        }
