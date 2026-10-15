"""시뮬레이터 공통 단계: 상품 세계 불러오기, 고객 생성 + 행동 생성, 결과를 표로 변환.

run.py(본 생성)와 calibrate.py(전이 확률 보정용 소규모 생성)가 같이 쓴다.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from src.simulator import item_space
from src.simulator.behavior import EVENT_TYPES, SOURCES, Log, Simulator
from src.simulator.catalog import World, build_world, load_catalog
from src.simulator.personas import PersonaParams
from src.simulator.search_table import build_search_table
from src.simulator.users import Population, generate_users


@dataclass
class Inputs:
    """고객 생성 재료: 취향 시드(실제 H&M 고객)와 브랜드 유형."""

    seeds: pd.DataFrame
    seed_vectors: np.ndarray
    brand_types: dict[str, str]


def load_world(cfg: dict[str, Any]) -> World:
    space = item_space.space_dir(cfg)
    catalog = load_catalog(cfg)
    vectors = np.load(space / "item_vectors.npy")
    stats = pd.read_parquet(space / "item_stats.parquet")
    world = build_world(catalog, vectors, stats, cfg["period"]["start_date"])
    search = build_search_table(cfg, catalog, world.popularity, out_dir=space)
    return build_world(catalog, vectors, stats, cfg["period"]["start_date"], search=search)


def load_inputs(cfg: dict[str, Any], cat_cfg: dict[str, Any]) -> Inputs:
    space = item_space.space_dir(cfg)
    return Inputs(
        seeds=pd.read_parquet(space / "seeds.parquet"),
        seed_vectors=np.load(space / "seed_vectors.npy"),
        brand_types={b: spec["type"] for b, spec in cat_cfg["brands"].items()},
    )


def simulate(cfg: dict[str, Any], world: World, personas: dict[str, PersonaParams], inputs: Inputs,
             n_users: int | None = None, progress: bool = False) -> tuple[Population, pd.DataFrame, pa.Table]:
    if n_users is not None:
        cfg = copy.deepcopy(cfg)
        cfg["scale"]["n_users"] = n_users
    pop = generate_users(cfg, personas, world, inputs.seeds, inputs.seed_vectors, inputs.brand_types,
                         np.random.default_rng(cfg["seed"]))
    log = Simulator(world, cfg, personas).run(pop, progress=progress)
    events, impressions = to_tables(log, world, pop, cfg)
    return pop, events, impressions


def _list_column(lists: list, product_ids: np.ndarray) -> pa.Array:
    lengths = np.fromiter((len(x) for x in lists), dtype=np.int32, count=len(lists))
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    flat = np.concatenate([np.asarray(x, dtype=np.int64) for x in lists]) if len(lists) else np.zeros(0, np.int64)
    return pa.ListArray.from_arrays(pa.array(offsets), pa.array(product_ids[flat], type=pa.string()))


def to_tables(log: Log, world: World, pop: Population, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pa.Table]:
    start = np.datetime64(cfg["period"]["start_date"], "ms")
    user_ids = pop.public["user_id"].to_numpy()
    e = {k: np.asarray(v) for k, v in log.ev.items()}
    order = np.lexsort((np.arange(len(e["time"])), e["user"], e["time"]))
    item, query, imp = e["item"][order], e["query"][order], e["impression"][order]
    events = pd.DataFrame({
        "event_id": [f"E{i + 1:09d}" for i in range(len(order))],
        "user_id": user_ids[e["user"][order]],
        "session_id": [f"S{s + 1:08d}" for s in e["session"][order]],
        "timestamp": start + (e["time"][order] * 1000).astype("timedelta64[ms]"),
        "event_type": np.asarray(EVENT_TYPES, dtype=object)[e["type"][order]],
        "product_id": np.where(item >= 0, world.product_ids[np.clip(item, 0, None)], None),
        "query": [str(world.search_vocab[q]) if q >= 0 else None for q in query],
        "source": np.asarray(SOURCES, dtype=object)[e["source"][order]],
        "impression_id": [f"I{i + 1:09d}" if i >= 0 else None for i in imp],
    })

    m = log.imp
    n_imp = len(m["user"])
    impressions = pa.table({
        "impression_id": [f"I{i + 1:09d}" for i in range(n_imp)],
        "user_id": user_ids[np.asarray(m["user"], dtype=np.int64)] if n_imp else [],
        "session_id": [f"S{s + 1:08d}" for s in m["session"]],
        "timestamp": pa.array(start + (np.asarray(m["time"]) * 1000).astype("timedelta64[ms]")),
        "source": np.asarray(SOURCES, dtype=object)[np.asarray(m["source"], dtype=np.int64)] if n_imp else [],
        "context": m["context"],
        "product_ids": _list_column(m["items"], world.product_ids),
        "clicked_ids": _list_column(m["clicked"], world.product_ids),
    })
    return events, impressions
