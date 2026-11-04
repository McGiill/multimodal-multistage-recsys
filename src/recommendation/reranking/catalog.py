"""상품 카탈로그(products.parquet) → Stage 3 후보(Candidate) 변환.

Stage 2는 product_id와 점수만 주므로, 다양성 규칙·신규 상품 판단에 필요한
카테고리·등록일·product_code를 카탈로그에서 붙인다.
카탈로그는 DataFrame으로 받으므로 parquet을 직접 읽어도 되고,
Stage 2가 이미 메모리에 올린 상품 표를 넘겨도 된다.
담당: D
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.common.config import ROOT, load_config
from src.recommendation.reranking.items import Candidate

CATALOG_COLUMNS = [
    "product_id", "product_code", "category_l1", "category_l2", "category_l3", "created_at",
]


def catalog_path() -> Path:
    """configs/catalog.yaml 의 paths.output (기본 data/processed/products.parquet)."""
    return ROOT / load_config("catalog")["paths"]["output"]


def load_catalog(path: Path | None = None) -> pd.DataFrame:
    """products.parquet에서 Stage 3에 필요한 컬럼만 읽는다."""
    return prepare_catalog(pd.read_parquet(path or catalog_path(), columns=CATALOG_COLUMNS))


def prepare_catalog(products: pd.DataFrame) -> pd.DataFrame:
    """product_id 인덱스 + created_at UTC 통일.

    카탈로그의 created_at은 시간대 정보가 없고(자정), Stage 2는 UTC로 다룬다.
    시간대가 있는 시각과 없는 시각은 비교할 수 없으므로 여기서 UTC로 맞춘다.
    product_code가 없는 표(예: Stage 2 FeatureStore.products)도 받는다.
    """
    df = products.copy()
    if "product_code" not in df:
        df["product_code"] = None
    created = pd.to_datetime(df["created_at"])
    df["created_at"] = (created.dt.tz_localize("UTC") if created.dt.tz is None
                        else created.dt.tz_convert("UTC"))
    return df[CATALOG_COLUMNS].set_index("product_id")


def to_utc(t: datetime) -> datetime:
    """신규 상품 판단 기준 시각을 카탈로그와 같은 UTC로 맞춘다 (시간대 없으면 UTC로 간주)."""
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def to_candidates(
    product_ids: Sequence[str],
    scores: Sequence[float],
    catalog: pd.DataFrame,
    reason: str = "personalized",
) -> list[Candidate]:
    """(product_id, 점수) 목록에 카탈로그 정보를 붙여 Candidate로 만든다.

    catalog는 prepare_catalog()를 거친 표. 카탈로그에 없는 상품은 KeyError.
    """
    if len(product_ids) != len(scores):
        raise ValueError("product_ids와 scores 길이가 다름")
    missing = pd.Index(product_ids).difference(catalog.index)
    if len(missing):
        raise KeyError(f"카탈로그에 없는 상품 {len(missing)}개: {list(missing[:5])}")

    rows = catalog.loc[list(product_ids)]
    return [
        Candidate(
            product_id=pid,
            score=float(score),
            category_l1=row.category_l1,
            category_l2=row.category_l2,
            category_l3=row.category_l3,
            created_at=row.created_at.to_pydatetime(),
            reason=reason,
            product_code=None if pd.isna(row.product_code) else str(row.product_code),
        )
        for pid, score, row in zip(product_ids, scores, rows.itertuples(index=False))
    ]
