"""스타일 사진 검색: 사람이 옷을 입은 사진 (+ 선택 텍스트) → 사진 속 옷마다 비슷한 상품.

흐름
1. 텍스트 해석 (query_parser): 요청 종류(style / find / match), 찾을 옷, 색, 디테일
2. 사진 종류: 사람이 입은 스타일 사진인지, 옷만 찍은 상품 사진인지 CLIP으로 판단
   - 스타일 사진: 옷 검출(detector)로 상의·바지·신발 등을 찾아 잘라낸다
     텍스트로 콕 집은 부위("상의")를 못 찾았으면 낮은 점수 후보 → Grounding DINO 순으로 다시 찾는다
     (grounding.mode: always면 콕 집은 부위는 처음부터 Grounding DINO로)
   - 상품 사진: 검출하지 않고 사진 전체를 그 상품으로 본다 (검출 모델이 착용 사진으로 학습돼서 상품 사진은 자주 틀림)
   - 사용자가 영역(box)을 지정하면 그 영역의 옷만 검색한다 (match면 그 옷과 어울리는 옷)
3. 고객층: 스타일 사진이면 사진 속 사람이 여성 / 남성 / 아이인지 판단해 category_l1을 제한
4. 옷별 검색: 잘라낸 사진 ↔ 상품 사진 유사도. 그 부위의 소분류 (+ 색, 디테일 조건) 안에서만 찾는다
5. 옷(부위)별로 묶어서 돌려준다

그룹마다 결과의 근거(basis)를 같이 준다.
- photo_item : 사진 속 그 옷과 비슷한 상품
- photo_style: 찾는 종류가 사진에 없을 때 (예: 상반신 사진 + "어울리는 바지") 사진 전체 분위기와 비슷한 상품
- whole_photo: 사진 전체가 검색 기준 (상품 사진, 또는 옷을 하나도 못 찾았을 때)

디테일 조절("소매 더 짧게")은 디테일 점수 = 상품 사진 · (more 문장 - less 문장)으로,
기준 옷(잘라낸 사진)보다 점수가 충분히 높은/낮은 상품만 남긴 뒤 유사도순으로 고른다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from PIL import Image

from src.common.config import load_config
from src.search.detector import (
    USER,
    Box,
    Detection,
    RawBox,
    _iou,
    best_in_slot,
    crop,
    kind_of_label,
    select_detections,
)
from src.search.query_parser import Intent, ParsedQuery, parse_query
from src.search.searcher import Searcher, SearchResult

PHOTO_ITEM, PHOTO_STYLE, WHOLE_PHOTO = "photo_item", "photo_style", "whole_photo"


class Detector(Protocol):
    def detect_raw(self, image: Image.Image) -> list[RawBox]: ...


class Grounder(Protocol):
    def find(self, image: Image.Image, kinds: list[str]) -> Detection | None: ...


@dataclass
class StyleGroup:
    slot: str                      # 부위 (top, bottom, ...). 부위 없이 사진 전체로 찾았으면 "all"
    kind: str | None               # 사진에서 찾은 옷 종류 (photo_item일 때)
    basis: str                     # photo_item / photo_style / whole_photo
    result: SearchResult
    box: Box | None = None
    detection_score: float | None = None
    filters: list[str] = field(default_factory=list)  # 적용한 조건 (예: "color:red", "sleeve:down")
    box_source: str | None = None  # 박스 출처: detector / low_score / grounding / user


@dataclass
class StyleSearchResult:
    intent: Intent
    is_person_photo: bool          # 스타일 사진으로 판단했는지 (False면 상품 사진)
    audience: str | None           # 제한한 고객층 (women / men / kids), 제한 안 했으면 None
    groups: list[StyleGroup]
    detections: list[Detection]    # 사진에서 찾은 옷 (콕 집은 부위를 다시 찾은 것, 사용자가 지정한 영역 포함)


@dataclass
class _Plan:
    slot: str
    basis: str
    l3: list[str] | None
    detection: int | None = None   # detections 안의 번호


class StyleSearcher:
    def __init__(self, searcher: Searcher, detector: Detector, cfg: dict[str, Any] | None = None,
                 grounding: Grounder | None = None) -> None:
        self.searcher = searcher
        self.detector = detector
        self.grounding = grounding  # None이면 Grounding DINO를 쓰지 않음
        self.cfg = cfg or load_config("style_search")
        self.slot_order = list(self.cfg["slots"])
        self._kind_of = kind_of_label(self.cfg)
        self._text_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ 진입점
    def search(self, image: Image.Image, query_text: str | None = None, top_k: int = 10,
               audience: str | None = None, box: Box | None = None) -> StyleSearchResult:
        parsed = parse_query(query_text, self.cfg)
        encoder = self.searcher.encoder
        photo_vec = encoder.encode_images([image])[0]
        is_person = self._is_person_photo(photo_vec)
        raw = self.detector.detect_raw(image) if is_person else []
        detections = select_detections(raw, image.size, self.cfg)
        user = None
        if box is not None:
            detections.append(self._user_detection(box, raw, parsed))
            user = len(detections) - 1
        elif is_person and parsed.intent == Intent.FIND:
            detections = self._find_requested(image, raw, parsed, detections)
        margin = self.cfg["detector"]["crop_margin"]
        item_vecs = encoder.encode_images([crop(image, d.box, margin) for d in detections]) if detections else []
        style_vec = item_vecs[user] if user is not None else photo_vec  # match: 지정한 영역의 옷과 어울리는
        audience_name, l1 = self._audience(photo_vec, audience, is_person)

        groups = []
        for plan in self._plans(parsed, detections, is_person, user):
            det = detections[plan.detection] if plan.detection is not None else None
            q = item_vecs[plan.detection] if det is not None else style_vec
            rows = self.searcher.select_rows(l3=plan.l3, l1=l1, colors=parsed.colors or None)
            rows, applied = self._apply_details(q, rows, parsed.details, plan.slot, top_k)
            result = self.searcher.rank_rows(q, rows, top_k)
            for item, r in zip(result.items, result.rows):
                item["category"] = str(self.searcher.category_l3[r])
                item["color"] = str(self.searcher.colors[r])
            groups.append(StyleGroup(
                slot=plan.slot, kind=det.kind if det else None, basis=plan.basis, result=result,
                box=det.box if det else None, detection_score=det.score if det else None,
                filters=[f"color:{c}" for c in parsed.color_names] + applied,
                box_source=det.source if det else None,
            ))
        groups.sort(key=lambda g: self.slot_order.index(g.slot) if g.slot in self.slot_order else len(self.slot_order))
        return StyleSearchResult(parsed.intent, is_person, audience_name, groups, detections)

    # ------------------------------------------------------------------ 무엇을 검색할지
    def _l3(self, kinds: list[str]) -> list[str]:
        return sorted({l3 for k in kinds for l3 in self.cfg["kinds"][k]["l3"]})

    def detected_l3(self, d: Detection) -> list[str] | None:
        """사진에서 찾은 옷: 부위 전체 소분류 (검출 모델이 셔츠 ↔ 티셔츠를 헷갈려도 정답이 빠지지 않게). 종류를 모르면 전체."""
        if d.kind is None:
            return None
        if d.slot in self.cfg["search_by_kind_slots"]:
            return self._l3([d.kind])
        return self._l3([k for k, v in self.cfg["kinds"].items() if v["slot"] == d.slot])

    def _user_detection(self, box: Box, raw: list[RawBox], parsed: ParsedQuery) -> Detection:
        """사용자가 지정한 영역이 무슨 옷인지: 텍스트로 말한 종류 > 그 자리에서 검출 모델이 본 종류 > 모름 (전체에서 검색)."""
        named = parsed.anchor_kinds if parsed.intent == Intent.MATCH else parsed.kinds  # match: "이 자켓이랑"의 자켓
        if named:
            kind = named[0]
        else:
            seen = [(score, label) for label, score, b in raw if label in self._kind_of and _iou(b, box) >= 0.5]
            kind = self._kind_of[max(seen)[1]] if seen else None
        return Detection(kind, self.cfg["kinds"][kind]["slot"] if kind else "all", "user", None, box, USER)

    def _find_requested(self, image: Image.Image, raw: list[RawBox], parsed: ParsedQuery,
                        detections: list[Detection]) -> list[Detection]:
        """텍스트로 콕 집은 부위를 다시 찾는다 (사진에 있다고 보고). grounding.mode에 따라
        fallback: 기준 점수로 못 찾았을 때만 YOLOS의 낮은 점수 후보 → 그래도 없으면 Grounding DINO
        always: 처음부터 Grounding DINO (박스가 더 정확하지만 느려서 GPU 서버용). 못 찾으면 fallback과 같음"""
        always = self.grounding is not None and self.cfg["grounding"]["mode"] == "always"
        found = list(detections)
        for slot in dict.fromkeys(self.cfg["kinds"][k]["slot"] for k in parsed.kinds):
            kinds = [k for k in parsed.kinds if self.cfg["kinds"][k]["slot"] == slot]
            det = self.grounding.find(image, kinds) if always else None
            if det is not None:  # always: 그 부위의 YOLOS 박스 대신
                found = [d for d in found if d.slot != slot]
            elif any(d.slot == slot for d in found):
                continue
            else:
                det = best_in_slot(raw, image.size, self.cfg, slot)
                if det is None and self.grounding is not None and not always:
                    det = self.grounding.find(image, kinds)
            if det is not None:
                found.append(det)
        return found

    def _plans(self, parsed: ParsedQuery, detections: list[Detection], is_person: bool,
               user: int | None = None) -> list[_Plan]:
        kinds = self.cfg["kinds"]
        if user is not None and parsed.intent != Intent.MATCH:  # 지정한 영역의 옷 하나만
            d = detections[user]
            return [_Plan(d.slot, PHOTO_ITEM, self._l3(parsed.kinds) if parsed.kinds else self.detected_l3(d), user)]
        targets = parsed.kinds
        if parsed.intent == Intent.MATCH and not targets:
            # "어울리는 거": 사진에 없는 기본 부위를 채운다 (다 있으면 사진 속 옷 전부)
            present = {d.slot for d in detections}
            missing = [s for s in self.cfg["match_default_slots"] if s not in present]
            targets = [k for k, v in kinds.items() if v["slot"] in missing]

        if not targets:
            plans = [_Plan(d.slot, PHOTO_ITEM, self.detected_l3(d), i) for i, d in enumerate(detections)]
            return plans or [_Plan("all", WHOLE_PHOTO, None)]

        plans = []
        for slot in dict.fromkeys(kinds[k]["slot"] for k in targets):
            wanted = self._l3([k for k in targets if kinds[k]["slot"] == slot])
            in_photo = sorted((i for i, d in enumerate(detections) if d.slot == slot),
                              key=lambda i: detections[i].kind not in targets)  # 같은 종류 먼저
            if in_photo:     # 사진 속 그 부위의 옷을 기준으로, 요청한 종류에서 (예: 반바지를 입었는데 "바지" → 바지 중에서)
                plans.append(_Plan(slot, PHOTO_ITEM, wanted, in_photo[0]))
            elif not is_person and parsed.intent != Intent.MATCH:
                plans.append(_Plan(slot, WHOLE_PHOTO, wanted))   # 상품 사진 + "이 바지 빨간색으로": 사진이 곧 그 옷
            else:            # 사진에 없음 → 사진 전체 분위기로 요청한 종류에서
                plans.append(_Plan(slot, PHOTO_STYLE, wanted))
        return plans

    # ------------------------------------------------------------------ 조건
    def _text(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in texts if t not in self._text_cache]
        if missing:
            for t, v in zip(missing, self.searcher.encoder.encode_texts(missing)):
                self._text_cache[t] = v
        return np.stack([self._text_cache[t] for t in texts])

    def _is_person_photo(self, photo_vec: np.ndarray) -> bool:
        p = self.cfg["photo_type"]
        person, product = self._text(p["prompts"]) @ photo_vec
        return bool(person - product > p["margin"])

    def _audience(self, photo_vec: np.ndarray, requested: str | None,
                  is_person: bool) -> tuple[str | None, list[str] | None]:
        a = self.cfg["audience"]
        groups = a["groups"]
        if requested in groups:
            return requested, groups[requested]["category_l1"]
        if requested == "all" or not is_person or (requested in (None, "auto") and a["mode"] == "off"):
            return None, None
        names = list(groups)
        logits = 100.0 * (self._text([groups[n]["prompt"] for n in names]) @ photo_vec)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        best = int(np.argmax(probs))
        if probs[best] < a["min_prob"]:
            return None, None
        return names[best], groups[names[best]]["category_l1"]

    def _apply_details(self, q: np.ndarray, rows: np.ndarray, details: dict[str, int], slot: str,
                       top_k: int) -> tuple[np.ndarray, list[str]]:
        applied = []
        for axis, sign in details.items():
            d = self.cfg["details"][axis]
            if (slot != "all" and slot not in d["slots"]) or len(rows) == 0:  # all: 상품 사진 전체
                continue
            more, less = d["anchors"].get(slot, d["anchors"]["default"])
            t = self._text([more, less])
            direction = t[0] - t[1]
            scores = self.searcher.image_emb[rows] @ direction
            gap = sign * (scores - float(q @ direction))  # 기준 옷보다 원하는 방향으로 얼마나 더 갔나
            keep = gap > self.cfg["detail_margin"] * scores.std()
            if keep.sum() < top_k:  # 너무 적으면 "기준 옷보다 조금이라도"로 완화
                keep = gap > 0
            rows = rows[keep]
            applied.append(f"{axis}:{'up' if sign > 0 else 'down'}")
        return rows, applied
