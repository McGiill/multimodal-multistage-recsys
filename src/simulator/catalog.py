"""시뮬레이션용 상품 세계(World): 카탈로그 + 상품 지도 + 검색 결과표를 배열로 묶는다.

상품 카탈로그 자체는 src/data/build_catalog.py가 만든다 (data/processed/products.parquet).
여기서는 시뮬레이션에 필요한 값만 숫자 배열로 바꿔 둔다.
- 카테고리·브랜드 → 정수 코드, 등록일 → 로그 시작일 기준 일수
- 상대 가격 = log(가격 / 같은 소분류 가격 중앙값) → 양말과 코트를 같은 기준으로 비교하지 않게
- 인기도 = H&M 구매 고객 수(전체 + 최근 가중)의 log 표준화
- 목록 재료: 카테고리 둘러보기(고객층·소분류별 인기순 / 신상품순 / 낮은 가격순), 홈(고객층별 인기순)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import ROOT


@dataclass
class World:
    product_ids: np.ndarray
    l1: np.ndarray                     # 정수 코드 (이름은 l1_names)
    l2: np.ndarray
    l3: np.ndarray
    brand: np.ndarray
    created_day: np.ndarray            # 로그 시작일 기준 등록일 (음수 = 로그 시작 전 등록)
    rel_price: np.ndarray
    popularity: np.ndarray             # 표준화된 인기도
    vectors: np.ndarray                # 상품 지도 벡터 (L2 정규화)
    l1_names: list[str]
    l2_names: list[str]
    l3_names: list[str]
    brand_names: list[str]
    browse: dict[str, dict[tuple[int, int], np.ndarray]] = field(repr=False)  # 정렬 → (고객층, 소분류) → 상품
    by_segment: dict[int, np.ndarray] = field(repr=False)           # 고객층 → 인기순 상품
    new_items: np.ndarray = field(repr=False)                       # 로그 기간 중 등록되는 상품
    search_vocab: pd.Index = field(repr=False, default_factory=lambda: pd.Index([], dtype=object))
    search_results: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 0), np.int32))
    item_query: dict[str, np.ndarray] = field(repr=False, default_factory=dict)  # 템플릿 → 상품별 검색어 번호

    @property
    def n(self) -> int:
        return len(self.product_ids)

    def registered(self, items: np.ndarray, day: int) -> np.ndarray:
        """그날 이미 등록된 상품만 (등록 전 상품은 화면에 나오지 않음)."""
        return items[self.created_day[items] <= day]


def _codes(values: pd.Series) -> tuple[np.ndarray, list[str]]:
    codes, names = pd.factorize(values, sort=True)
    return codes.astype(np.int32), [str(x) for x in names]


def build_world(catalog: pd.DataFrame, vectors: np.ndarray, stats: pd.DataFrame, start_date: str,
                search: dict[str, Any] | None = None) -> World:
    assert (stats["product_id"].to_numpy() == catalog["product_id"].to_numpy()).all(), "상품 지도와 카탈로그 순서가 다름"
    l1, l1_names = _codes(catalog["category_l1"])
    l2, l2_names = _codes(catalog["category_l2"])
    l3, l3_names = _codes(catalog["category_l3"])
    brand, brand_names = _codes(catalog["brand"])

    created = pd.to_datetime(catalog["created_at"]).to_numpy().astype("datetime64[D]")
    created_day = (created - np.datetime64(start_date, "D")).astype(np.int64)

    price = catalog["price"].to_numpy(dtype=float)
    l3_median = catalog.groupby("category_l3")["price"].transform("median").to_numpy(dtype=float)
    rel_price = np.log(price / l3_median)

    raw_pop = np.log1p(stats["pop_all"].to_numpy(dtype=float) + 3.0 * stats["pop_recent"].to_numpy(dtype=float))
    popularity = (raw_pop - raw_pop.mean()) / raw_pop.std()

    # 카테고리 둘러보기 정렬: 인기순 / 신상품순 / 낮은 가격순
    sort_keys = {"popular": -popularity, "newest": -created_day.astype(float), "price_low": rel_price}
    browse: dict[str, dict[tuple[int, int], list[int]]] = {}
    for sort, key in sort_keys.items():
        browse[sort] = {}
        for i in np.argsort(key, kind="stable"):
            browse[sort].setdefault((int(l1[i]), int(l3[i])), []).append(int(i))
    by_segment: dict[int, list[int]] = {}
    for i in np.argsort(-popularity, kind="stable"):
        by_segment.setdefault(int(l1[i]), []).append(int(i))

    world = World(
        product_ids=catalog["product_id"].to_numpy(),
        l1=l1, l2=l2, l3=l3, brand=brand,
        created_day=created_day,
        rel_price=rel_price,
        popularity=popularity,
        vectors=vectors.astype(np.float32),
        l1_names=l1_names, l2_names=l2_names, l3_names=l3_names, brand_names=brand_names,
        browse={s: {k: np.array(v, dtype=np.int32) for k, v in d.items()} for s, d in browse.items()},
        by_segment={k: np.array(v, dtype=np.int32) for k, v in by_segment.items()},
        new_items=np.flatnonzero(created_day >= 0).astype(np.int32),
    )
    if search is not None:
        world.search_vocab = search["vocab"]
        world.search_results = search["results"]
        world.item_query = {t: search["vocab"].get_indexer(q).astype(np.int32) for t, q in search["item_queries"].items()}
    return world


def load_catalog(cfg: dict[str, Any]) -> pd.DataFrame:
    return pd.read_parquet(ROOT / cfg["paths"]["catalog"])
