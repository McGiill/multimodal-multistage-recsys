"""검색 모듈 공통: 카탈로그 로딩, 상품 텍스트 구성, 모델별 인덱스 폴더."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import ROOT


def load_catalog(cfg: dict[str, Any], columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(ROOT / cfg["paths"]["catalog"], columns=columns)


def product_texts(catalog: pd.DataFrame) -> list[str]:
    """상품을 대표하는 텍스트. 핵심 속성을 앞에 둔다 (CLIP 텍스트는 77토큰에서 잘림)."""
    return (
        catalog["name"]
        + ". "
        + catalog["color"]
        + " "
        + catalog["pattern"]
        + " "
        + catalog["category_l3"]
        + ". "
        + catalog["description"].fillna("")
    ).tolist()


def model_slug(model_name: str) -> str:
    """폴더 이름용: 'hf-hub:Marqo/marqo-fashionCLIP' → 'hf-hub__Marqo__marqo-fashionCLIP'"""
    return model_name.replace(":", "__").replace("/", "__")


def with_overrides(
    cfg: dict[str, Any],
    model: str | None = None,
    text_image_weight: float | None = None,
    hybrid_text_weight: float | None = None,
) -> dict[str, Any]:
    """설정 파일은 그대로 두고 실험용으로 모델·가중치만 바꾼 복사본을 만든다 (CLI --model 등)."""
    cfg = copy.deepcopy(cfg)
    if model:
        cfg["clip"]["model_name"] = model
    if text_image_weight is not None:
        cfg["text_query"]["image_weight"] = text_image_weight
    if hybrid_text_weight is not None:
        cfg["hybrid"]["text_weight"] = hybrid_text_weight
    return cfg


def index_dir(cfg: dict[str, Any]) -> Path:
    """모델마다 인덱스를 따로 저장 → 모델을 바꿔 실험해도 이전 결과가 덮어써지지 않음."""
    return ROOT / cfg["paths"]["index_dir"] / model_slug(cfg["clip"]["model_name"])


def image_paths(cfg: dict[str, Any], catalog: pd.DataFrame) -> list[Path]:
    root = ROOT / cfg["paths"]["image_root"]
    return [root / p for p in catalog["image_path"]]
