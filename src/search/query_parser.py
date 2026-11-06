"""Query Understanding: 스타일 사진 검색에 같이 들어온 텍스트를 해석한다 (LLM 없이 사전·규칙).

    "이 옷이랑 어울리는 바지 추천해줘" → intent=match, 찾을 옷=pants
    "이 바지 빨간색으로"               → intent=find,  찾을 옷=pants, 색=Red 계열
    "소매는 더 짧게 총장은 길게"       → 디테일 sleeve↓, length↑
    "" / None                          → intent=style (사진 속 옷 전부)

사전은 configs/style_search.yaml (targets, intents, colors, details, directions, detail_shortcuts).
- 긴 단어부터 찾고, 이미 잡힌 글자는 다시 쓰지 않는다 ("반바지"가 "바지"로, "긴팔"이 "긴"으로 잡히지 않게).
- 영어 단어는 단어 경계로 찾는다.
- 디테일 단어("소매")와 방향 단어("짧게")는 위치가 가까운 것끼리 짝짓는다. 방향만 있으면 default_detail.
- 사전에 없는 영어 단어는 free_text로 모은다 ("floral dress" → floral). 한국어는 FashionCLIP이 약해서 모으지 않는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

_HANGUL = re.compile(r"[가-힣]")
_ENGLISH_WORD = re.compile(r"[a-z][a-z0-9]*(?:[-/'][a-z0-9]+)*")


class Intent(str, Enum):
    STYLE = "style"   # 사진 속 옷 전부를 각각 비슷한 상품으로
    FIND = "find"     # 사진 속 특정 옷 (예: "사진 속 바지")
    MATCH = "match"   # 사진의 옷과 어울리는 다른 옷 (예: "어울리는 바지")


@dataclass
class ParsedQuery:
    intent: Intent = Intent.STYLE
    kinds: list[str] = field(default_factory=list)       # 찾을 옷 종류 (configs의 kinds 이름)
    anchor_kinds: list[str] = field(default_factory=list)  # match에서 기준이 되는 옷 ("바지에 어울리는 상의"의 바지)
    colors: list[str] = field(default_factory=list)      # 카탈로그 color 값 (예: Red, Dark Red)
    color_names: list[str] = field(default_factory=list)  # 색 이름 (예: red) — 응답 설명용
    details: dict[str, int] = field(default_factory=dict)  # 디테일 → 방향 (+1 = more 문장 쪽, -1 = less 쪽)
    free_text: str = ""                                  # 사전에 없는 영어 단어 (CLIP·BM25 점수에 사용)


def _pattern(word: str) -> re.Pattern[str]:
    if _HANGUL.search(word):
        return re.compile(re.escape(word))
    return re.compile(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])")


def _entries(cfg: dict[str, Any]) -> list[tuple[str, str, Any]]:
    """(단어, 종류, 값) 목록. 종류: target / intent / color / detail / direction / shortcut."""
    out: list[tuple[str, str, Any]] = []
    for t in cfg["targets"]:
        kinds = list(t.get("kinds", []))
        kinds += [k for k, v in cfg["kinds"].items() if v["slot"] in t.get("slots", [])]
        out += [(w, "target", kinds) for w in t["words"]]
    for intent, words in cfg["intents"].items():
        out += [(w, "intent", Intent(intent)) for w in words]
    out += [(w, "intent_target_before", Intent.MATCH) for w in cfg["match_target_before"]]
    for name, c in cfg["colors"].items():
        out += [(w, "color", (name, c["values"])) for w in c["words"]]
    for axis, d in cfg["details"].items():
        out += [(w, "detail", axis) for w in d["words"]]
    for sign, direction in (("up", +1), ("down", -1)):
        out += [(w, "direction", direction) for w in cfg["directions"][sign]]
    for word, (axis, direction) in cfg["detail_shortcuts"].items():
        out.append((word, "shortcut", (axis, +1 if direction == "up" else -1)))
    return sorted(out, key=lambda e: -len(e[0]))


def parse_query(text: str | None, cfg: dict[str, Any]) -> ParsedQuery:
    parsed = ParsedQuery()
    if not text or not text.strip():
        return parsed
    text = text.lower()
    used = [False] * len(text)
    hits: list[tuple[int, str, Any]] = []
    for word, kind, value in _entries(cfg):
        for m in _pattern(word.lower()).finditer(text):
            if any(used[m.start():m.end()]):
                continue
            used[m.start():m.end()] = [True] * (m.end() - m.start())
            hits.append((m.start(), kind, value))
    hits.sort(key=lambda h: h[0])

    rest = "".join(" " if u else c for c, u in zip(text, used))
    stopwords = set(cfg["free_text_stopwords"])
    parsed.free_text = " ".join(w for w in _ENGLISH_WORD.findall(rest) if w not in stopwords)

    intents = set()
    match_pos, target_before = None, False
    targets: list[tuple[int, list[str]]] = []
    axes: list[tuple[int, str]] = []
    directions: list[tuple[int, int]] = []
    for pos, kind, value in hits:
        if kind == "target":
            targets.append((pos, value))
        elif kind in ("intent", "intent_target_before"):
            intents.add(value)
            if value == Intent.MATCH and match_pos is None:
                match_pos, target_before = pos, kind == "intent_target_before"
        elif kind == "color":
            name, values = value
            if name not in parsed.color_names:
                parsed.color_names.append(name)
                parsed.colors += [v for v in values if v not in parsed.colors]
        elif kind == "shortcut":
            parsed.details[value[0]] = value[1]
        elif kind == "detail":
            axes.append((pos, value))
        elif kind == "direction":
            directions.append((pos, value))

    # 디테일 단어마다 가장 가까운 방향 단어 (뒤쪽 우선: "소매 더 짧게")
    free = list(directions)
    for pos, axis in axes:
        if not free:
            break
        after = [d for d in free if d[0] > pos]
        pick = after[0] if after else min(free, key=lambda d: pos - d[0])
        parsed.details[axis] = pick[1]
        free.remove(pick)
    if free and not axes:
        parsed.details.setdefault(cfg["default_detail"], free[0][1])

    # match: 한국어 "바지에 어울리는 상의" / "matching jeans"는 match 단어 뒤가 찾을 옷,
    #        영어 "a top that goes with these jeans"는 앞이 찾을 옷 (match_target_before)
    for pos, kinds in targets:
        is_anchor = match_pos is not None and ((pos < match_pos) != target_before)
        bucket = parsed.anchor_kinds if is_anchor else parsed.kinds
        bucket += [k for k in kinds if k not in bucket]

    if Intent.MATCH in intents:
        parsed.intent = Intent.MATCH
    elif parsed.kinds:
        parsed.intent = Intent.FIND
    return parsed


def description_text(text: str | None, parsed: ParsedQuery) -> str | None:
    """하이브리드 점수에 넣을 설명 문장. 영어 문장은 통째로, 한국어가 섞이면 사전 밖 영어 단어만. 없으면 None."""
    if not text or not text.strip():
        return None
    if _HANGUL.search(text):
        return parsed.free_text or None
    return text.strip() if parsed.free_text or parsed.kinds or parsed.colors else None
