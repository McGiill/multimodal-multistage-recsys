"""상품 지도(취향 공간): H&M 실제 구매 기록으로 "같이 산 상품끼리 가깝게" 상품 벡터를 학습한다.

    python -m src.simulator.item_space      # data/processed/simulator/ 에 저장 (run.py가 없으면 자동 실행)

방법: 고객 × 상품 구매 행렬(중복 제거, idf 가중, 고객별 정규화)을 truncated SVD로 분해한다.
- 상품 벡터 = V·√Σ, 고객 벡터 = U·√Σ (둘 다 L2 정규화 → 코사인 유사도로 취향 점수 계산)
- 실제 고객 벡터 일부를 "취향 시드"로 저장 → 가상 고객이 실제 고객 한 명의 취향 위치를 빌려 쓴다

CLIP 임베딩을 쓰지 않는 이유: 추천 모델이 같은 CLIP 피처를 쓰면 시뮬레이터 공식을 그대로 베끼게 된다.
이 벡터는 시뮬레이터 내부 전용이다 (추천 모델 입력으로 쓰면 정답 유출).

출력 (paths.space_dir)
- item_vectors.npy : 카탈로그 순서의 상품 벡터 (n_products × dim)
- item_stats.parquet : product_id, pop_recent(최근 N주 구매 고객 수), pop_all(전체 기간)
- seeds.parquet + seed_vectors.npy : 취향 시드 (실제 고객의 나이, 주 쇼핑 고객층, 자주 산 브랜드)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd

from src.common.config import ROOT, load_config
from src.data.build_catalog import assign_brands

FILES = ("item_vectors.npy", "item_stats.parquet", "seeds.parquet", "seed_vectors.npy", "meta.json")


def space_dir(cfg: dict[str, Any]) -> Path:
    return ROOT / cfg["paths"]["space_dir"]


def exists(cfg: dict[str, Any]) -> bool:
    return all((space_dir(cfg) / f).exists() for f in FILES)


def _read_transactions(raw_dir: Path, article_ids: pa.Array) -> tuple[np.ndarray, np.ndarray, np.ndarray, pa.Array]:
    """거래 csv → (고객 번호, 상품 번호, 날짜) 정수 배열. 고객 ID 문자열은 한 번만 사전화한다."""
    tbl = pacsv.read_csv(
        raw_dir / "transactions_train.csv",
        convert_options=pacsv.ConvertOptions(
            include_columns=["t_dat", "customer_id", "article_id"],
            column_types={"t_dat": pa.date32(), "customer_id": pa.string(), "article_id": pa.string()},
        ),
    )
    customers = pc.unique(tbl["customer_id"])
    cust = pc.index_in(tbl["customer_id"], value_set=customers).to_numpy().astype(np.int64)
    art = pc.fill_null(pc.index_in(tbl["article_id"], value_set=article_ids), -1).to_numpy().astype(np.int64)
    dates = tbl["t_dat"].to_numpy().astype("datetime64[D]")
    keep = art >= 0
    return cust[keep], art[keep], dates[keep], customers


def _mode_per_group(groups: np.ndarray, values: np.ndarray) -> pd.Series:
    """그룹(고객)별 최빈값 (동률이면 값이 작은 쪽)."""
    counts = pd.DataFrame({"g": groups, "v": values}).value_counts().reset_index(name="n")
    top = counts.sort_values(["g", "n", "v"], ascending=[True, False, True]).drop_duplicates("g")
    return top.set_index("g")["v"]


def build_item_space(sim_cfg: dict[str, Any], cat_cfg: dict[str, Any]) -> dict[str, Any]:
    sp = sim_cfg["item_space"]
    raw_dir = ROOT / sim_cfg["paths"]["raw_dir"]
    out = space_dir(sim_cfg)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(sim_cfg["seed"])
    t0 = time.perf_counter()

    articles = pd.read_csv(
        raw_dir / "articles.csv",
        usecols=["article_id", "product_code", "index_group_name", "section_name"],
        dtype={"article_id": str, "product_code": str},
    )
    articles["article_id"] = articles["article_id"].str.zfill(10)
    articles["brand"] = assign_brands(articles, cat_cfg["brands"])
    n_art = len(articles)

    cust, art, dates, customer_ids = _read_transactions(raw_dir, pa.array(articles["article_id"]))
    print(f"  거래 {len(cust):,}건 읽음 ({time.perf_counter() - t0:.0f}s)")

    # 같은 고객이 같은 상품을 여러 번 산 행은 하나로 (같은 날 여러 개 = 행 반복)
    key = np.unique(cust * n_art + art)
    pair_c, pair_a = key // n_art, key % n_art
    recent_from = dates.max() - np.timedelta64(sp["popularity_weeks"] * 7, "D")
    recent = np.unique((cust * n_art + art)[dates >= recent_from])
    pop_recent = np.bincount(recent % n_art, minlength=n_art)
    pop_all = np.bincount(pair_a, minlength=n_art)

    # 고객 × 상품 행렬: 2개 이상 산 고객만 (1개만 산 고객은 "같이 산" 정보가 없음)
    per_cust = np.bincount(pair_c)
    keep = per_cust[pair_c] >= sp["min_customer_items"]
    pair_c, pair_a = pair_c[keep], pair_a[keep]
    row_ids, rows = np.unique(pair_c, return_inverse=True)
    idf = np.log(len(row_ids) / np.maximum(np.bincount(pair_a, minlength=n_art), 1)).astype(np.float32)
    x = sparse.csr_matrix((idf[pair_a], (rows, pair_a)), shape=(len(row_ids), n_art), dtype=np.float32)
    x = normalize(x)
    print(f"  행렬 {x.shape[0]:,} 고객 × {x.shape[1]:,} 상품, 구매 쌍 {x.nnz:,}")

    u, s, vt = randomized_svd(x, n_components=sp["dim"], n_iter=sp["svd_iter"], random_state=sim_cfg["seed"])
    item_vec = normalize(vt.T * np.sqrt(s)).astype(np.float32)
    user_vec = normalize(u * np.sqrt(s)).astype(np.float32)
    print(f"  SVD {sp['dim']}차원 완료 ({time.perf_counter() - t0:.0f}s)")

    # 카탈로그 순서로 상품 벡터 / 인기도 저장
    catalog = pd.read_parquet(ROOT / sim_cfg["paths"]["catalog"], columns=["product_id"])
    art_index = pd.Series(np.arange(n_art), index="hm_" + articles["article_id"])
    cat_rows = art_index.loc[catalog["product_id"]].to_numpy()
    np.save(out / "item_vectors.npy", item_vec[cat_rows])
    pd.DataFrame(
        {"product_id": catalog["product_id"], "pop_recent": pop_recent[cat_rows], "pop_all": pop_all[cat_rows]}
    ).to_parquet(out / "item_stats.parquet", index=False)

    # 취향 시드: 충분히 산 실제 고객 중 일부
    n_items = np.bincount(rows)
    eligible = np.flatnonzero(n_items >= sp["seed_min_items"])
    chosen = np.sort(rng.choice(eligible, size=min(sp["max_seeds"], len(eligible)), replace=False))
    in_seed = np.isin(rows, chosen)
    seg = _mode_per_group(rows[in_seed], articles["index_group_name"].to_numpy()[pair_a[in_seed]])
    brand_counts = (
        pd.DataFrame({"r": rows[in_seed], "brand": articles["brand"].to_numpy()[pair_a[in_seed]]})
        .value_counts()
        .reset_index(name="n")
        .sort_values(["r", "n", "brand"], ascending=[True, False, True])
    )
    brand_counts["share"] = brand_counts["n"] / brand_counts.groupby("r")["n"].transform("sum")
    top_brands = brand_counts.groupby("r").head(3).groupby("r")["brand"].agg(list)
    top_share = brand_counts.groupby("r")["share"].first()

    ages = pacsv.read_csv(
        raw_dir / "customers.csv",
        convert_options=pacsv.ConvertOptions(
            include_columns=["customer_id", "age"], column_types={"customer_id": pa.string(), "age": pa.float64()}
        ),
    )
    seed_ids = pc.take(customer_ids, pa.array(row_ids[chosen]))
    age_pos = pc.index_in(seed_ids, value_set=ages["customer_id"])
    seed_age = pc.take(ages["age"], age_pos).to_numpy(zero_copy_only=False)

    seeds = pd.DataFrame(
        {
            "seed": np.arange(len(chosen)),
            "age": seed_age,
            "main_segment": seg.loc[chosen].to_numpy(),
            "top_brands": top_brands.loc[chosen].to_numpy(),
            "top_brand_share": top_share.loc[chosen].to_numpy(),
            "n_items": n_items[chosen],
        }
    )
    seeds.to_parquet(out / "seeds.parquet", index=False)
    np.save(out / "seed_vectors.npy", user_vec[chosen])

    meta = {
        "dim": sp["dim"],
        "n_customers": int(x.shape[0]),
        "n_pairs": int(x.nnz),
        "singular_values": [round(float(v), 3) for v in s],
        "date_range": [str(dates.min()), str(dates.max())],
        "popularity_from": str(recent_from),
        "n_seeds": len(seeds),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  시드 {len(seeds):,}명, 저장 → {out.relative_to(ROOT)} ({time.perf_counter() - t0:.0f}s)")
    return meta


def main() -> None:
    sim_cfg, cat_cfg = load_config("simulator"), load_config("catalog")
    print("[simulator] 상품 지도 학습 (H&M 구매 기록)")
    build_item_space(sim_cfg, cat_cfg)


if __name__ == "__main__":
    main()
