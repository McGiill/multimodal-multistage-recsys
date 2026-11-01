"""패션 아이템 검출: 스타일 사진(사람이 옷을 입은 사진)에서 상의·바지·신발 등의 위치를 찾는다.

기본 모델은 valentinafeve/yolos-fashionpedia (configs/style_search.yaml의 detector).
Fashionpedia 라벨 46개 중 옷·잡화만 쓰고 소매·주머니·지퍼 같은 부분 라벨은 버린다.
라벨 → 옷 종류(kind)와 부위(slot)는 configs의 kinds로 정한다.

상품 사진(사람 없이 옷만 찍은 사진)은 학습 데이터와 달라서 못 찾는 경우가 있다.
그때는 style_search가 사진 전체로 검색한다.

텍스트로 콕 집은 부위("상의")는 사진에 있다고 보고 다시 찾는다.
1) YOLOS 후보 중 기준 점수(threshold)는 못 넘었지만 requested_threshold 이상인 박스 (best_in_slot)
2) 그래도 없으면 Grounding DINO에 단어("pants.")를 주고 찾는다 (GroundingDetector, configs의 grounding)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)

from src.common.config import load_config
from src.search.encoder import resolve_device

Box = tuple[int, int, int, int]  # x1, y1, x2, y2 (픽셀)
RawBox = tuple[str, float, Box]  # 검출 모델 출력 (라벨, 점수, 박스)

# 박스 출처
DETECTOR = "detector"     # YOLOS, 기준 점수 이상
LOW_SCORE = "low_score"   # YOLOS, 기준 점수 미만이지만 텍스트로 콕 집은 부위라 사용
GROUNDING = "grounding"   # Grounding DINO (콕 집은 부위를 YOLOS가 아예 못 찾았을 때)
USER = "user"             # 사용자가 지정한 영역


@dataclass
class Detection:
    kind: str | None  # configs의 kinds 이름 (예: pants). 사용자가 지정한 영역은 모를 수 있음
    slot: str         # 부위 (예: bottom). 모르면 "all"
    label: str        # 검출 모델 라벨 (예: "pants")
    score: float | None
    box: Box
    source: str = DETECTOR


def box_area(box: Box) -> float:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _iou(a: Box, b: Box) -> float:
    inter = box_area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def kind_of_label(cfg: dict[str, Any]) -> dict[str, str]:
    return {label: kind for kind, k in cfg["kinds"].items() for label in k["labels"]}


def select_detections(raw: list[RawBox], image_size: tuple[int, int], cfg: dict[str, Any]) -> list[Detection]:
    """모델 출력 (라벨, 점수, 박스) → 쓸 만한 옷만. 점수순으로 겹친 박스를 지우고 종류마다 하나씩 남긴다."""
    d = cfg["detector"]
    kind_of = kind_of_label(cfg)
    image_area = image_size[0] * image_size[1]
    candidates = sorted(
        (r for r in raw if r[0] in kind_of and r[1] >= d["threshold"] and box_area(r[2]) >= d["min_area"] * image_area),
        key=lambda r: -r[1],
    )
    kept: list[Detection] = []
    for label, score, box in candidates:
        kind = kind_of[label]
        if any(k.kind == kind or _iou(k.box, box) > d["nms_iou"] for k in kept):
            continue
        kept.append(Detection(kind, cfg["kinds"][kind]["slot"], label, round(score, 4), box))
    return kept


def best_in_slot(raw: list[RawBox], image_size: tuple[int, int], cfg: dict[str, Any], slot: str) -> Detection | None:
    """텍스트로 콕 집은 부위: 기준 점수를 못 넘었어도 requested_threshold 이상이면 그 부위에서 점수가 가장 높은 박스.

    사용자가 그 부위를 말했으니 사진에 있다고 보고 놓치지 않는 쪽을 택한다.
    (Fashionpedia val 579장에서 그 부위 박스를 맞게 찾는 비율: 상의 47% → 74%, 아우터 40% → 74%, 하의 69% → 84%)
    """
    d = cfg["detector"]
    kind_of = kind_of_label(cfg)
    image_area = image_size[0] * image_size[1]
    candidates = [
        r for r in raw
        if r[0] in kind_of and cfg["kinds"][kind_of[r[0]]]["slot"] == slot
        and r[1] >= d["requested_threshold"] and box_area(r[2]) >= d["min_area"] * image_area
    ]
    if not candidates:
        return None
    label, score, box = max(candidates, key=lambda r: r[1])
    return Detection(kind_of[label], slot, label, round(score, 4), box, LOW_SCORE)


def crop(image: Image.Image, box: Box, margin: float) -> Image.Image:
    """박스 둘레에 여백을 조금 두고 잘라낸다 (옷 가장자리가 잘리지 않게)."""
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    w, h = image.size
    return image.crop((max(0, int(x1 - mx)), max(0, int(y1 - my)), min(w, int(x2 + mx)), min(h, int(y2 + my))))


class FashionDetector:
    def __init__(self, cfg: dict[str, Any] | None = None, device: str | None = None) -> None:
        self.cfg = cfg or load_config("style_search")
        d = self.cfg["detector"]
        self.device = resolve_device(device or d["device"])
        self.processor = AutoImageProcessor.from_pretrained(
            d["model_name"], size={"shortest_edge": d["shortest_edge"], "longest_edge": d["longest_edge"]}
        )
        self.model = AutoModelForObjectDetection.from_pretrained(d["model_name"]).to(self.device).eval()
        self.id2label = self.model.config.id2label

    @torch.inference_mode()
    def detect_raw(self, image: Image.Image) -> list[RawBox]:
        """모델 출력 그대로. 콕 집은 부위용으로 requested_threshold 이상까지 남긴다 (고르는 건 select_detections)."""
        d = self.cfg["detector"]
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        res = self.processor.post_process_object_detection(
            out, threshold=min(d["threshold"], d["requested_threshold"]), target_sizes=[image.size[::-1]]
        )[0]
        return [
            (self.id2label[int(label)], float(score), tuple(round(v) for v in box.tolist()))
            for score, label, box in zip(res["scores"], res["labels"], res["boxes"])
        ]

    def detect(self, image: Image.Image) -> list[Detection]:
        return select_detections(self.detect_raw(image), image.size, self.cfg)


class GroundingDetector:
    """Grounding DINO: 찾을 물체를 단어로 알려 주면 ("pants.") 그 위치를 찾는다 (configs의 grounding).

    정해진 라벨만 아는 YOLOS와 달리 어떤 단어든 되지만 느리다 (CPU 1장 약 3.2초, Mac GPU 약 0.6초).
    그래서 기본은 텍스트로 콕 집은 부위를 YOLOS가 아예 못 찾았을 때만 쓰고 (mode: fallback), 모델은 처음 쓸 때 로딩한다.
    """

    def __init__(self, cfg: dict[str, Any] | None = None, device: str | None = None) -> None:
        self.cfg = cfg or load_config("style_search")
        self.device = resolve_device(device or self.cfg["grounding"]["device"])
        self.processor = self.model = None

    @torch.inference_mode()
    def find(self, image: Image.Image, kinds: list[str]) -> Detection | None:
        """kinds(같은 부위의 옷 종류) 중 하나로 보이는 가장 점수 높은 박스."""
        g = self.cfg["grounding"]
        if self.model is None:
            self.processor = AutoProcessor.from_pretrained(g["model_name"])
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(g["model_name"]).to(self.device).eval()
        phrases = {k: g["phrases"][k] for k in kinds}
        inputs = self.processor(images=image, text=" ".join(dict.fromkeys(phrases.values())),
                                return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            out, inputs["input_ids"], threshold=g["threshold"], text_threshold=g["text_threshold"],
            target_sizes=[image.size[::-1]],
        )[0]
        if not len(res["scores"]):
            return None
        best = int(res["scores"].argmax())
        label = str(res["text_labels"][best])
        kind = next((k for k, p in phrases.items() if label and label in p), kinds[0])
        box = tuple(round(v) for v in res["boxes"][best].tolist())
        return Detection(kind, self.cfg["kinds"][kind]["slot"], label, round(float(res["scores"][best]), 4), box, GROUNDING)
