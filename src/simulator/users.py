"""가상 고객 생성: 공개 정보(users.parquet)와 숨김 성향(users_hidden.parquet)을 따로 만든다.

- 페르소나: persona_ratio 비율대로 배정 → 가격 민감도, 브랜드 충성, 신상품 선호, 선호 카테고리 범위를 정함
- 취향 위치: H&M 실제 고객 한 명(취향 시드)의 위치를 빌리고 약간 흔든다 → 실제로 존재하는 취향 분포
- 주 쇼핑 고객층, 나이: 그 실제 고객에게서 가져옴. 성별은 고객층별 여성 비율로 배정
- 가입일: 대부분 로그 시작 전, new_user_ratio 만큼은 로그 기간 중 (이력이 적은 신규 사용자)

숨김 성향은 행동 생성과 페르소나별 분석에만 쓴다. 추천 모델 입력으로 쓰면 정답이 새어 나간다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.simulator.catalog import World
from src.simulator.personas import PersonaParams

AGE_BINS = [(0, 20, "10s"), (20, 30, "20s"), (30, 40, "30s"), (40, 50, "40s"), (50, 60, "50s"), (60, 200, "60s+")]


@dataclass
class Population:
    public: pd.DataFrame       # user_id, age_group, gender, signup_at
    hidden: pd.DataFrame       # user_id, persona, main_segment, preferred_l2, preferred_brands, ...
    taste: np.ndarray          # (n_users, dim) 취향 위치 (L2 정규화)


def age_group(age: np.ndarray) -> np.ndarray:
    out = np.empty(len(age), dtype=object)
    for lo, hi, name in AGE_BINS:
        out[(age >= lo) & (age < hi)] = name
    return out


def _uniform(rng: np.random.Generator, lo_hi: tuple[float, float], size: int) -> np.ndarray:
    return rng.uniform(lo_hi[0], lo_hi[1], size=size)


def _general_brand_pool(world: World, brand_types: dict[str, str]) -> dict[str, tuple[list[str], np.ndarray]]:
    """고객층별 종합 브랜드 후보와 뽑을 확률 (그 고객층에서 인기 있는 브랜드일수록 높게)."""
    pool = {}
    weight = np.exp(world.popularity)
    for code, segment in enumerate(world.l1_names):
        mask = world.l1 == code
        w = pd.Series(weight[mask]).groupby(np.asarray(world.brand_names, dtype=object)[world.brand[mask]]).sum()
        w = w[[b for b in w.index if brand_types.get(b) == "general"]]
        pool[segment] = (list(w.index), (w / w.sum()).to_numpy())
    return pool


def generate_users(cfg: dict[str, Any], personas: dict[str, PersonaParams], world: World,
                   seeds: pd.DataFrame, seed_vectors: np.ndarray, brand_types: dict[str, str],
                   rng: np.random.Generator) -> Population:
    n = cfg["scale"]["n_users"]
    ucfg = cfg["users"]
    start = np.datetime64(cfg["period"]["start_date"], "D")
    days = cfg["period"]["days"]

    names = list(cfg["persona_ratio"])
    persona = rng.choice(names, size=n, p=[cfg["persona_ratio"][p] for p in names])

    seed = rng.integers(0, len(seeds), size=n)
    jitter = rng.normal(size=(n, seed_vectors.shape[1])) / np.sqrt(seed_vectors.shape[1])
    taste = seed_vectors[seed] + ucfg["taste_jitter"] * jitter
    taste = (taste / np.linalg.norm(taste, axis=1, keepdims=True)).astype(np.float32)
    segment = seeds["main_segment"].to_numpy()[seed]

    female_p = np.array([ucfg["female_ratio_by_segment"][s] for s in segment])
    gender = np.where(rng.random(n) < female_p, "F", "M")
    ages = seeds["age"].to_numpy(dtype=float)[seed]
    known = seeds["age"].dropna().to_numpy(dtype=float)
    missing = np.isnan(ages)
    ages[missing] = rng.choice(known, size=missing.sum())

    is_new = rng.random(n) < ucfg["new_user_ratio"]
    before = int((start - np.datetime64(ucfg["signup_from"], "D")).astype(int))
    signup_day = np.where(is_new, rng.integers(0, days, size=n), -rng.integers(1, before + 1, size=n))
    signup_at = (start + signup_day.astype("timedelta64[D]")).astype("datetime64[s]") + rng.integers(
        0, 86400, size=n
    ).astype("timedelta64[s]")
    active_frac = (days - np.clip(signup_day, 0, None)) / days

    sigma = ucfg["activity_sigma"]
    activity = rng.lognormal(mean=-(sigma**2) / 2, sigma=sigma, size=n)

    price_sens = np.empty(n)
    new_aff = np.empty(n)
    loyalty = np.empty(n)
    expected = np.empty(n)
    pref_l2: list[list[str]] = [[] for _ in range(n)]
    pref_brands: list[list[str]] = [[] for _ in range(n)]
    brand_pool = _general_brand_pool(world, brand_types)
    top_brands = seeds["top_brands"].to_numpy()[seed]

    for name, p in personas.items():
        idx = np.flatnonzero(persona == name)
        price_sens[idx] = _uniform(rng, p.price_sensitivity, len(idx))
        new_aff[idx] = _uniform(rng, p.new_item_affinity, len(idx))
        loyalty[idx] = _uniform(rng, p.brand_loyalty, len(idx))
        expected[idx] = p.sessions * activity[idx] * active_frac[idx]
        for i in idx:
            pref_l2[i] = list(rng.choice(p.preferred_l2, size=p.n_preferred, replace=False))
            seed_brands = [str(b) for b in top_brands[i]]
            if p.brand_loyalty[1] >= 0.5:  # 브랜드충성: 실제 고객이 자주 산 종합 브랜드 1~2개
                general = [b for b in seed_brands if brand_types.get(b) == "general"]
                k = int(rng.integers(1, 3))
                if len(general) < k:
                    cands, probs = brand_pool[segment[i]]
                    extra = [b for b in rng.choice(cands, size=min(3, len(cands)), replace=False, p=probs) if b not in general]
                    general += extra
                pref_brands[i] = general[:k]
            else:  # 나머지: 자주 산 브랜드 1개를 약하게 선호
                pref_brands[i] = seed_brands[:1]

    user_id = np.array([f"U{i + 1:05d}" for i in range(n)])
    public = pd.DataFrame(
        {"user_id": user_id, "age_group": age_group(ages), "gender": gender, "signup_at": signup_at}
    )
    hidden = pd.DataFrame(
        {
            "user_id": user_id,
            "persona": persona,
            "main_segment": segment,
            "preferred_l2": pref_l2,
            "preferred_brands": pref_brands,
            "price_sensitivity": price_sens.round(3),
            "new_item_affinity": new_aff.round(3),
            "brand_loyalty": loyalty.round(3),
            "activity": activity.round(3),
            "expected_sessions": expected.round(2),
            "signup_day": signup_day,
            "seed": seed,
        }
    )
    return Population(public=public, hidden=hidden, taste=taste)
