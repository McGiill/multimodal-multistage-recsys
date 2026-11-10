"""검색 평가셋(쿼리 → 관련 상품 + 관련도) 자동 생성: python -m src.evaluation.build_search_eval

    python -m src.evaluation.build_search_eval          # 설정 기준 (eval.relevance) → paths.eval_set
    python -m src.evaluation.build_search_eval --relevance design --out data/processed/search_eval.parquet  # 기존 기준

출력: configs/search.yaml의 paths.eval_set (data/processed/search_eval_v2.parquet, --out으로 변경)
    query_id, query_type(text/image/hybrid), query_text, query_image_path, relevant_ids, relevance

쿼리 종류와 관련도 (숫자가 클수록 더 정확한 정답, 0 = 오답)
- text (상품명)  "black strap top"   2: 같은 상품명 + 같은 색      1: 같은 소분류 + 같은 색
- text (속성)    "white stripe t-shirt"
                                    2: 색 + 소분류 (+무늬) 일치    1: 색 + 소분류만 일치(무늬 다름)
- image         카탈로그에 없는 H&M 상품 사진 (같은 디자인의 다른 상품이 카탈로그에 있는 것만)
    --relevance design (기존, 2026-10-10)
                                    2: 같은 디자인(product_code, 색 무관)   1: 같은 소분류 + 같은 색
    --relevance color_family (2026-11-09 확정: 이미지 검색의 정답은 비슷한 디자인 + 비슷한 색)
                                    3: 같은 디자인 + 같은 색 계열   2: 같은 소분류 + 같은 색 계열
                                    1: 같은 디자인 + 다른 색 계열 ("다른 색상 보기")
      같은 색 계열 = 색 라벨(colour_group_name)이 같거나 H&M 색 계열(perceived_colour_master_name)이 같음
      (색 라벨 50개 → 계열 20개, 예: Dark Blue / Blue / Light Blue → Blue. 계열은 상품마다 붙어 있어서
      같은 색 라벨이라도 계열이 다른 상품이 약 4% 있다 → 색 라벨이 같으면 계열과 상관없이 같은 색으로 본다).
      기존 기준은 쿼리가 카탈로그 밖 상품이라 "같은 디자인 + 같은 색"이 카탈로그에 없어서,
      남은 "같은 디자인의 다른 색"을 최고 정답으로 둔 것이었다 (색을 따르는 검색이 감점됨).
- hybrid        image 쿼리 사진 + 그 상품의 "색 소분류" 텍스트 → 관련도는 image와 동일

이미지 쿼리를 카탈로그 밖 상품으로 만드는 이유: 카탈로그 안 사진을 쓰면 자기 자신이 1위로 나와 의미가 없다.
쿼리는 valid / test 로 반씩 나눈다 → 가중치 등은 valid로 정하고, 보고하는 지표는 test로만 낸다.
두 기준은 쿼리와 분할이 같고 관련도만 다르다 (seed 고정, 관련도 계산은 난수를 쓰지 않음).
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import ROOT, load_config
from src.data.build_catalog import image_rel_path

PLAIN_PATTERNS = {"Solid", "Unknown", "Other pattern"}
RELEVANCE = ("design", "color_family")
UNKNOWN_FAMILIES = {"Unknown", "undefined"}  # 색 계열이 없으면 색 라벨(colour_group_name)을 그대로 계열로 쓴다


def color_family(master: pd.Series, group: pd.Series) -> pd.Series:
    """H&M 색 계열 (perceived_colour_master_name). 계열 값이 없는 상품은 색 라벨로 대신한다."""
    return master.where(~master.isin(UNKNOWN_FAMILIES), group)


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
    cfg: dict[str, Any], catalog: pd.DataFrame, n_image: int, n_hybrid: int, rng: np.random.Generator,
    relevance: str = "design",
) -> list[dict[str, Any]]:
    if relevance not in RELEVANCE:
        raise ValueError(f"relevance는 {RELEVANCE} 중 하나")
    articles = pd.read_csv(
        ROOT / cfg["paths"]["articles"],
        usecols=["article_id", "product_code", "colour_group_name", "product_type_name",
                 "perceived_colour_master_name"],
        dtype={"article_id": str, "product_code": str},
    )
    articles["article_id"] = articles["article_id"].str.zfill(10)
    articles["family"] = color_family(articles["perceived_colour_master_name"], articles["colour_group_name"])
    in_catalog = set(catalog["product_id"].str.removeprefix("hm_"))
    codes = set(catalog["product_code"])
    image_root = ROOT / cfg["paths"]["image_root"]

    held_out = articles[~articles["article_id"].isin(in_catalog) & articles["product_code"].isin(codes)]
    held_out = held_out.iloc[rng.permutation(len(held_out))]
    by_code = catalog.groupby("product_code")["product_id"].apply(set)
    by_type_color = catalog.groupby(["category_l3", "color"])["product_id"].apply(set)
    family_of = articles.set_index("article_id")["family"]
    catalog = catalog.assign(family=family_of.reindex(catalog["product_id"].str.removeprefix("hm_")).to_numpy())
    by_code_color = catalog.groupby(["product_code", "color"])["product_id"].apply(set)
    by_code_family = catalog.groupby(["product_code", "family"])["product_id"].apply(set)
    by_type_family = catalog.groupby(["category_l3", "family"])["product_id"].apply(set)

    out: list[dict[str, Any]] = []
    for _, row in held_out.iterrows():
        if len(out) >= n_image + n_hybrid:
            break
        path = image_rel_path(row["article_id"])
        if not (image_root / path).is_file():
            continue
        same_design = by_code[row["product_code"]]
        if relevance == "design":
            similar = by_type_color.get((row["product_type_name"], row["colour_group_name"]), set())
            grades = {pid: 1 for pid in similar} | {pid: 2 for pid in same_design}
        else:  # 같은 색 계열 = 색 라벨이 같거나 색 계열이 같음
            t, c, f, code = row["product_type_name"], row["colour_group_name"], row["family"], row["product_code"]
            similar = by_type_color.get((t, c), set()) | by_type_family.get((t, f), set())
            design_family = by_code_color.get((code, c), set()) | by_code_family.get((code, f), set())
            grades = {pid: 1 for pid in same_design} | {pid: 2 for pid in similar} | {pid: 3 for pid in design_family}
        if len(out) < n_image:
            out.append(_rows("image", None, path, grades))
        else:
            text = f"{row['colour_group_name']} {row['product_type_name']}".lower()
            out.append(_rows("hybrid", text, path, grades))
    return out


def main() -> None:
    cfg = load_config("search")
    official = cfg["eval"]["relevance"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--relevance", choices=RELEVANCE, default=official,
                        help=f"이미지·하이브리드 관련도 기준 (기본: eval.relevance = {official})")
    parser.add_argument("--out", default=None, help="저장 경로 (기본: paths.eval_set)")
    args = parser.parse_args()
    if args.relevance != official and args.out is None:
        parser.error(f"--relevance {args.relevance}는 설정 기준({official})과 달라서 --out으로 다른 파일에 저장해야 합니다")

    rng = np.random.default_rng(cfg["seed"])
    catalog = pd.read_parquet(ROOT / cfg["paths"]["catalog"])
    ev = cfg["eval"]

    rows = text_queries(catalog, ev["n_text_queries"], rng)
    rows += image_queries(cfg, catalog, ev["n_image_queries"], ev["n_hybrid_queries"], rng, args.relevance)
    df = pd.DataFrame(rows)
    df.insert(0, "query_id", [f"Q{i:05d}" for i in range(len(df))])
    df["split"] = np.where(rng.random(len(df)) < 0.5, "valid", "test")

    out = ROOT / (args.out or cfg["paths"]["eval_set"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"[search-eval] 쿼리 {len(df):,}개 ({args.relevance}) → {out.relative_to(ROOT)}")
    for qtype, g in df.groupby("query_type"):
        n_rel = g["relevant_ids"].map(len)
        top = g["relevance"].map(max)
        n_top = [sum(x == m for x in r) for r, m in zip(g["relevance"], top)]
        n_test = int((g["split"] == "test").sum())
        print(f"  {qtype:6s} {len(g):5,}개 (test {n_test:,})  관련 상품 중앙값 {int(n_rel.median()):,}개 "
              f"(가장 높은 관련도 {int(top.median())}: 중앙값 {int(np.median(n_top)):,}개)")


if __name__ == "__main__":
    main()
