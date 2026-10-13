"""검색 평가셋(쿼리 → 관련 상품 + 관련도) 자동 생성: python -m src.evaluation.build_search_eval

출력: data/processed/search_eval.parquet
    query_id, query_type(text/image/hybrid), query_text, query_image_path, relevant_ids, relevance

쿼리 종류와 관련도 (2 = 정확히 일치, 1 = 부분 일치)
- text (상품명)  "black strap top"   2: 같은 상품명 + 같은 색      1: 같은 소분류 + 같은 색
- text (속성)    "white stripe t-shirt"
                                    2: 색 + 소분류 (+무늬) 일치    1: 색 + 소분류만 일치(무늬 다름)
- image         카탈로그에 없는 H&M 상품 사진 (같은 디자인의 다른 상품이 카탈로그에 있는 것만)
                                    2: 같은 디자인(product_code)   1: 같은 소분류 + 같은 색 ("비슷한 상품 찾기")
- hybrid        image 쿼리 사진 + 그 상품의 "색 소분류" 텍스트 → 관련도는 image와 동일

이미지 쿼리를 카탈로그 밖 상품으로 만드는 이유: 카탈로그 안 사진을 쓰면 자기 자신이 1위로 나와 의미가 없다.
쿼리는 valid / test 로 반씩 나눈다 → 가중치 등은 valid로 정하고, 보고하는 지표는 test로만 낸다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.common.config import ROOT, load_config
from src.data.build_catalog import image_rel_path

PLAIN_PATTERNS = {"Solid", "Unknown", "Other pattern"}


def _rows(query_type: str, text: str | None, image: str | None, grades: dict[str, int]) -> dict[str, Any]:
    ids = sorted(grades, key=lambda pid: -grades[pid])
    return {
        "query_type": query_type,
        "query_text": text,
        "query_image_path": image,
        "relevant_ids": ids,
        "relevance": [grades[pid] for pid in ids],
    }


def text_queries(catalog: pd.DataFrame, n: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    cat = catalog.assign(name_key=catalog["name"].str.lower(), color_key=catalog["color"].str.lower())
    by_type_color = cat.groupby(["category_l3", "color_key"])["product_id"].apply(set)
    by_type_color_pattern = cat.groupby(["category_l3", "color_key", "pattern"])["product_id"].apply(set)
    by_name_color = cat.groupby(["name_key", "color_key"])["product_id"].apply(set)

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx in rng.permutation(len(cat)):
        if len(out) >= n:
            break
        row = cat.iloc[idx]
        partial = by_type_color[(row["category_l3"], row["color_key"])]
        if len(out) % 2 == 0:  # 상품명 쿼리
            query = f"{row['color_key']} {row['name_key']}"
            exact = by_name_color[(row["name_key"], row["color_key"])]
        else:  # 속성 쿼리
            has_pattern = row["pattern"] not in PLAIN_PATTERNS
            pattern = f"{row['pattern'].lower()} " if has_pattern else ""
            query = f"{row['color_key']} {pattern}{row['category_l3'].lower()}"
            exact = by_type_color_pattern[(row["category_l3"], row["color_key"], row["pattern"])] if has_pattern else partial
        if query in seen:
            continue
        seen.add(query)
        grades = {pid: 1 for pid in partial} | {pid: 2 for pid in exact}
        out.append(_rows("text", query, None, grades))
    return out


def image_queries(
    cfg: dict[str, Any], catalog: pd.DataFrame, n_image: int, n_hybrid: int, rng: np.random.Generator
) -> list[dict[str, Any]]:
    articles = pd.read_csv(
        ROOT / cfg["paths"]["articles"],
        usecols=["article_id", "product_code", "colour_group_name", "product_type_name"],
        dtype={"article_id": str, "product_code": str},
    )
    articles["article_id"] = articles["article_id"].str.zfill(10)
    in_catalog = set(catalog["product_id"].str.removeprefix("hm_"))
    codes = set(catalog["product_code"])
    image_root = ROOT / cfg["paths"]["image_root"]

    held_out = articles[~articles["article_id"].isin(in_catalog) & articles["product_code"].isin(codes)]
    held_out = held_out.iloc[rng.permutation(len(held_out))]
    by_code = catalog.groupby("product_code")["product_id"].apply(set)
    by_type_color = catalog.groupby(["category_l3", "color"])["product_id"].apply(set)

    out: list[dict[str, Any]] = []
    for _, row in held_out.iterrows():
        if len(out) >= n_image + n_hybrid:
            break
        path = image_rel_path(row["article_id"])
        if not (image_root / path).is_file():
            continue
        similar = by_type_color.get((row["product_type_name"], row["colour_group_name"]), set())
        grades = {pid: 1 for pid in similar} | {pid: 2 for pid in by_code[row["product_code"]]}
        if len(out) < n_image:
            out.append(_rows("image", None, path, grades))
        else:
            text = f"{row['colour_group_name']} {row['product_type_name']}".lower()
            out.append(_rows("hybrid", text, path, grades))
    return out


def main() -> None:
    cfg = load_config("search")
    rng = np.random.default_rng(cfg["seed"])
    catalog = pd.read_parquet(ROOT / cfg["paths"]["catalog"])
    ev = cfg["eval"]

    rows = text_queries(catalog, ev["n_text_queries"], rng)
    rows += image_queries(cfg, catalog, ev["n_image_queries"], ev["n_hybrid_queries"], rng)
    df = pd.DataFrame(rows)
    df.insert(0, "query_id", [f"Q{i:05d}" for i in range(len(df))])
    df["split"] = np.where(rng.random(len(df)) < 0.5, "valid", "test")

    out = ROOT / cfg["paths"]["eval_set"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"[search-eval] 쿼리 {len(df):,}개 → {out.relative_to(ROOT)}")
    for qtype, g in df.groupby("query_type"):
        n_rel = g["relevant_ids"].map(len)
        n_exact = g["relevance"].map(lambda r: sum(x == 2 for x in r))
        n_test = int((g["split"] == "test").sum())
        print(f"  {qtype:6s} {len(g):5,}개 (test {n_test:,})  관련 상품 중앙값 {int(n_rel.median()):,}개 "
              f"(관련도 2: {int(n_exact.median()):,}개)")


if __name__ == "__main__":
    main()
