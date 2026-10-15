"""검색 결과표: 시뮬레이터가 쓸 수 있는 모든 검색어 → 검색 결과(상위 N개 상품)를 미리 계산해 둔다.

검색어는 "고객이 찾는 상품"의 속성으로 만든다 (configs/simulator.yaml → search.query_templates).
카탈로그에서 나올 수 있는 검색어를 전부 모아 한 번에 인코딩·검색하고 data/processed/simulator/ 에 저장한다.
→ 시뮬레이션 본체는 torch 없이 표만 조회한다 (빠르고, 나중에 병렬화도 쉬움).

엔진
- clip      : 실제 검색 엔진(src/search/searcher.py, configs/search.yaml 모델·인덱스). 인덱스가 있어야 함
- attribute : 속성 일치 검색 (색·종류·이름이 같은 상품을 인기순으로). 인덱스 없이 테스트할 때
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import load_config
from src.evaluation.build_search_eval import PLAIN_PATTERNS
from src.search.catalog import model_slug

TEMPLATES = ("color_name", "color_type", "color_pattern_type", "type")


def item_queries(catalog: pd.DataFrame) -> dict[str, np.ndarray]:
    """템플릿별로 상품마다 만들어지는 검색어 (상품 순서 그대로)."""
    color = catalog["color"].str.lower()
    l3 = catalog["category_l3"].str.lower()
    name = catalog["name"].str.lower()
    patterned = ~catalog["pattern"].isin(PLAIN_PATTERNS)
    color_type = color + " " + l3
    return {
        "color_name": (color + " " + name).to_numpy(),
        "color_type": color_type.to_numpy(),
        "color_pattern_type": np.where(patterned, color + " " + catalog["pattern"].str.lower() + " " + l3, color_type),
        "type": l3.to_numpy(),
    }


def _attribute_results(catalog: pd.DataFrame, queries: dict[str, np.ndarray], popularity: np.ndarray,
                       vocab: pd.Index, k: int) -> np.ndarray:
    """검색어를 만든 속성이 같은 상품을 인기순으로. 모자라면 같은 소분류 인기 상품으로 채운다."""
    order = np.argsort(-popularity, kind="stable")
    by_query: dict[str, list[int]] = {}
    for t in TEMPLATES:
        for i in order:
            by_query.setdefault(queries[t][i], []).append(int(i))
    by_type: dict[str, list[int]] = {}
    l3 = catalog["category_l3"].to_numpy()
    for i in order:
        by_type.setdefault(l3[i], []).append(int(i))
    first_item = {q: rows[0] for q, rows in by_query.items()}

    out = np.full((len(vocab), k), -1, dtype=np.int32)
    for qi, q in enumerate(vocab):
        rows = by_query[q][:k]
        if len(rows) < k:
            taken = set(rows)
            extra = [r for r in by_type[l3[first_item[q]]] if r not in taken]
            rows = rows + extra[: k - len(rows)]
        out[qi, : len(rows)] = rows
    return out


def _clip_results(catalog: pd.DataFrame, vocab: pd.Index, k: int) -> tuple[np.ndarray, str]:
    from src.search.searcher import Searcher

    cfg = load_config("search")
    searcher = Searcher(cfg, device=cfg["clip"]["index_device"])
    vectors = searcher.encoder.encode_texts(list(vocab), batch_size=256)
    pos = pd.Series(np.arange(len(catalog)), index=catalog["product_id"])
    out = np.full((len(vocab), k), -1, dtype=np.int32)
    for qi in range(len(vocab)):
        ids = searcher.search_vectors(vectors[qi], None, k).product_ids
        rows = pos.loc[ids].to_numpy()
        out[qi, : len(rows)] = rows
    return out, cfg["clip"]["model_name"]


def build_search_table(sim_cfg: dict[str, Any], catalog: pd.DataFrame, popularity: np.ndarray,
                       out_dir: Path | None = None) -> dict[str, Any]:
    """검색어 사전 + 결과표. out_dir가 있으면 같은 조건의 결과를 재사용하고, 없으면 만들어 저장한다."""
    engine, k = sim_cfg["search"]["engine"], sim_cfg["search"]["candidates"]
    queries = item_queries(catalog)
    vocab = pd.Index(sorted(set().union(*[set(q) for q in queries.values()])))
    model = load_config("search")["clip"]["model_name"] if engine == "clip" else "attribute"
    key = {"engine": engine, "model": model, "k": k, "n_queries": len(vocab), "n_products": len(catalog),
           "first_product": catalog["product_id"].iloc[0], "last_product": catalog["product_id"].iloc[-1]}

    path = out_dir / f"search_{engine}_{model_slug(model)}.npz" if out_dir else None
    if path is not None and path.exists():
        cached = np.load(path, allow_pickle=False)
        if json.loads(str(cached["key"])) == key:
            return {"vocab": vocab, "results": cached["results"], "item_queries": queries, "key": key}

    print(f"  검색어 {len(vocab):,}개 검색 결과표 생성 (엔진: {engine}, {model})")
    if engine == "clip":
        results, _ = _clip_results(catalog, vocab, k)
    elif engine == "attribute":
        results = _attribute_results(catalog, queries, popularity, vocab, k)
    else:
        raise ValueError(f"search.engine은 clip 또는 attribute: {engine}")

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, results=results, key=json.dumps(key))
    return {"vocab": vocab, "results": results, "item_queries": queries, "key": key}
