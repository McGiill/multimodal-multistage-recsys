"""H&M 원본 → 팀 공통 상품 카탈로그 (data/processed/products.parquet).

    python -m src.data.build_catalog            # configs/catalog.yaml 규모 (5만 개)
    python -m src.data.build_catalog --n 2000   # 개발용 소규모

검색 인덱스, 추천, 시뮬레이터가 모두 이 파일 하나를 상품 목록으로 쓴다.
- 이미지 파일이 실제로 있고, 거래가 min_purchases 이상인 상품만 대상
- 카테고리 3단계: index_group_name → product_group_name → product_type_name
- 가격: 상품별 최빈 거래가(= 정가) × 590 → 달러 (H&M 정규화 가격 = 실제 가격 ÷ 590)
- 브랜드: 매장 코너(section_name) 단위 가상 브랜드 (configs/catalog.yaml → brands)
- 등록일: 임의 생성, 브랜드마다 new_ratio 만큼 신규 상품 (configs/catalog.yaml → created_at)
- 판매량 같은 행동 지표는 넣지 않는다 (행동 로그에서 계산)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import ROOT, load_config

ARTICLE_COLUMNS = [
    "article_id",
    "product_code",
    "prod_name",
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "index_group_name",
    "section_name",
    "detail_desc",
]


def image_rel_path(article_id: str) -> str:
    """H&M 이미지 경로 규칙: images/앞3자리/상품ID.jpg"""
    return f"images/{article_id[:3]}/{article_id}.jpg"


def load_articles(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(
        raw_dir / "articles.csv",
        usecols=ARTICLE_COLUMNS,
        dtype={"article_id": str, "product_code": str},
    )
    df["article_id"] = df["article_id"].str.zfill(10)
    df["image_path"] = df["article_id"].map(image_rel_path)
    df["has_image"] = df["image_path"].map(lambda p: (raw_dir / p).is_file())
    return df


def purchase_stats(raw_dir: Path, chunksize: int, denominator: float) -> pd.DataFrame:
    """거래 csv(약 3,200만 행)를 나눠 읽어 상품별 거래 수와 최빈 거래가(달러)를 계산.

    할인 거래가 섞인 평균 대신 가장 많이 팔린 가격을 정가로 본다 (95% 이상이 9.99 같은 가격표 숫자).
    """
    parts = []
    reader = pd.read_csv(
        raw_dir / "transactions_train.csv",
        usecols=["article_id", "price"],
        dtype={"article_id": str, "price": "float64"},
        chunksize=chunksize,
    )
    for chunk in reader:
        cents = (chunk["price"] * denominator * 100).round().astype("int64")
        parts.append(chunk.groupby(["article_id", cents.rename("cents")]).size())
    counts = pd.concat(parts).groupby(level=[0, 1]).sum().reset_index(name="n")
    # 거래 수가 같은 가격이 여럿이면 높은 쪽 (할인 전 가격)
    mode = counts.sort_values(["article_id", "n", "cents"], ascending=[True, False, False]).drop_duplicates("article_id")
    stats = pd.DataFrame(
        {
            "n_purchases": counts.groupby("article_id")["n"].sum(),
            "list_price": mode.set_index("article_id")["cents"] / 100,
        }
    )
    stats.index = stats.index.str.zfill(10)
    return stats


def clean_name(name: str) -> str:
    """'Strap top (1)' 처럼 붙은 일련번호 제거."""
    return re.sub(r"\s*\(\d+\)\s*$", "", str(name)).strip()


def section_to_brand(brands: dict[str, Any]) -> dict[tuple[str, str], str]:
    """config의 brands → {(index_group_name, section_name): 브랜드}."""
    mapping: dict[tuple[str, str], str] = {}
    for brand, spec in brands.items():
        for group, sections in spec["sections"].items():
            for section in sections:
                if (group, section) in mapping:
                    raise ValueError(f"매장 코너가 두 브랜드에 들어 있음: {group} / {section}")
                mapping[(group, section)] = brand
    return mapping


def assign_brands(articles: pd.DataFrame, brands: dict[str, Any]) -> pd.Series:
    """매장 코너 → 브랜드. 같은 디자인(product_code)은 상품이 가장 많은 브랜드로 통일 (동률이면 브랜드명 순)."""
    mapping = section_to_brand(brands)
    keys = list(zip(articles["index_group_name"], articles["section_name"]))
    missing = sorted(set(keys) - mapping.keys())
    if missing:
        raise ValueError(f"브랜드가 지정되지 않은 매장 코너: {missing}")

    by_section = pd.DataFrame({"code": articles["product_code"], "brand": [mapping[k] for k in keys]})
    counts = by_section.value_counts().reset_index(name="n")
    top = counts.sort_values(["code", "n", "brand"], ascending=[True, False, True]).drop_duplicates("code")
    return articles["product_code"].map(top.set_index("code")["brand"])


def assign_created_at(catalog: pd.DataFrame, cfg: dict[str, Any], seed: int) -> pd.Series:
    """디자인(product_code) 단위로 등록일 생성: 브랜드마다 상품의 new_ratio 이상을 신규로 고른다."""
    rng = np.random.default_rng(seed)
    ref = pd.Timestamp(cfg["reference_date"])
    old_start, old_end = (pd.Timestamp(d) for d in cfg["old_range"])

    n_items = catalog.groupby("product_code").size()
    designs = catalog.drop_duplicates("product_code")[["product_code", "brand"]]
    new_codes: list[str] = []
    for _, codes in designs.groupby("brand")["product_code"]:
        codes = rng.permutation(codes.to_numpy())
        target = cfg["new_ratio"] * n_items[codes].sum()
        if target > 0:
            k = int(np.searchsorted(n_items[codes].cumsum().to_numpy(), target)) + 1
            new_codes.extend(codes[:k])

    codes = designs["product_code"].to_numpy()
    is_new = np.isin(codes, new_codes)
    new_dates = ref - pd.to_timedelta(rng.integers(0, cfg["new_days"], len(codes)), unit="D")
    old_dates = old_start + pd.to_timedelta(rng.integers(0, (old_end - old_start).days + 1, len(codes)), unit="D")
    dates = pd.Series(np.where(is_new, new_dates.to_numpy(), old_dates.to_numpy()), index=codes)
    return catalog["product_code"].map(dates)


def build_catalog(cfg: dict[str, Any], n_products: int) -> pd.DataFrame:
    raw_dir = ROOT / cfg["paths"]["raw_dir"]
    articles = load_articles(raw_dir)
    articles["brand"] = assign_brands(articles, cfg["brands"])
    stats = purchase_stats(raw_dir, cfg["transactions_chunksize"], cfg["price"]["denominator"])

    df = articles.join(stats, on="article_id")
    df["n_purchases"] = df["n_purchases"].fillna(0).astype("int64")

    # 거래가 없는 상품의 가격 = 같은 소분류(product_type) 중앙값 → 그래도 없으면 전체 중앙값
    # (min_purchases ≥ 1 이면 거래 없는 상품은 어차피 빠지므로 쓰이지 않음)
    type_median = df.groupby("product_type_name")["list_price"].transform("median")
    df["list_price"] = df["list_price"].fillna(type_median).fillna(df["list_price"].median())

    eligible = df[df["has_image"] & (df["n_purchases"] >= cfg["min_purchases"])]
    if len(eligible) < n_products:
        print(f"  [warn] 조건을 만족하는 상품이 {len(eligible):,}개뿐이라 전부 사용")
        n_products = len(eligible)
    sampled = eligible.sample(n=n_products, random_state=cfg["seed"]).sort_values("article_id")

    catalog = pd.DataFrame(
        {
            "product_id": "hm_" + sampled["article_id"],
            "product_code": sampled["product_code"],
            "name": sampled["prod_name"].map(clean_name),
            "description": sampled["detail_desc"].fillna(""),
            "brand": sampled["brand"],
            "category_l1": sampled["index_group_name"],
            "category_l2": sampled["product_group_name"],
            "category_l3": sampled["product_type_name"],
            "color": sampled["colour_group_name"],
            "pattern": sampled["graphical_appearance_name"],
            "price": sampled["list_price"].round(2),
            "image_path": sampled["image_path"],
        }
    ).reset_index(drop=True)
    catalog["created_at"] = assign_created_at(catalog, cfg["created_at"], cfg["seed"])
    return catalog


def summarize(catalog: pd.DataFrame, cfg: dict[str, Any]) -> None:
    print(f"  상품 수: {len(catalog):,}")
    print(f"  카테고리 수: L1 {catalog['category_l1'].nunique()}, "
          f"L2 {catalog[['category_l1', 'category_l2']].drop_duplicates().shape[0]}, "
          f"L3 {catalog[['category_l1', 'category_l2', 'category_l3']].drop_duplicates().shape[0]} (경로 기준)")
    q = catalog["price"].quantile([0.05, 0.5, 0.95]).tolist()
    print(f"  가격($) 5% / 50% / 95%: {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}")
    print(f"  색상 {catalog['color'].nunique()}종, 무늬 {catalog['pattern'].nunique()}종")
    sizes = catalog["brand"].value_counts()
    print(f"  브랜드 {len(sizes)}개 (상품 수 최소 {sizes.min():,} / 중앙값 {int(sizes.median()):,} / 최대 {sizes.max():,})")
    c = cfg["created_at"]
    is_new = catalog["created_at"] > pd.Timestamp(c["reference_date"]) - pd.Timedelta(days=c["new_days"])
    print(f"  신규 상품(기준일 {c['reference_date']}, {c['new_days']}일 이내): {is_new.sum():,}개 ({is_new.mean():.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=None, help="상품 수 (기본: configs/catalog.yaml)")
    args = parser.parse_args()

    cfg = load_config("catalog")
    np.random.seed(cfg["seed"])
    n_products = args.n or cfg["n_products"]

    print(f"[catalog] H&M 원본에서 상품 {n_products:,}개 카탈로그 생성")
    catalog = build_catalog(cfg, n_products)

    out = ROOT / cfg["paths"]["output"]
    out.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(out, index=False)
    summarize(catalog, cfg)
    print(f"[catalog] 저장 → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
