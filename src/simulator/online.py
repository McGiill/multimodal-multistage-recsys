"""추천 목록에 대한 가상 고객 반응. A/B 실험과 실시간 스트림이 같은 규칙을 쓴다.

숨김 성향과 상품 지도는 이 모듈의 행동 생성에만 사용한다. 추천 모델에는 사용자 ID,
이미 발생한 조회만 전달한다. API 호출이나 파일 저장은 호출자가 담당한다.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import ROOT, load_config
from src.common.schemas import Event
from src.simulator.catalog import World, build_world, load_catalog
from src.simulator.personas import PersonaParams, load_personas, with_funnel
from src.simulator.response import ResponseModel, Shopper, Visit, base_scores
from src.simulator.users import Population


def stable_seed(seed: int, key: str) -> int:
    """Python hash/random 프로세스 상태에 영향받지 않는 사용자별 난수 시드."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "little")


def utc(value: datetime | str | pd.Timestamp) -> datetime:
    ts = pd.Timestamp(value)
    return (ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")).to_pydatetime()


@dataclass
class ReactionBatch:
    events: list[Event]
    clicked_ids: list[str]
    next_screen: str


@dataclass
class _UserState:
    rng: np.random.Generator
    cart: dict[int, int] = field(default_factory=dict)
    purchases: Counter = field(default_factory=Counter)
    session_id: str | None = None
    visit: Visit | None = None
    visit_number: int = 0
    event_number: int = 0
    last_at: datetime | None = None


class OnlineResponseEngine:
    def __init__(self, world: World, population: Population, cfg: dict[str, Any],
                 personas: dict[str, PersonaParams], seed: int = 42, cache_users: int = 128) -> None:
        if cache_users < 1:
            raise ValueError("cache_users must be positive")
        self.world, self.population, self.cfg = world, population, cfg
        self.personas, self.seed, self.cache_users = personas, seed, cache_users
        self.model = ResponseModel(world, cfg)
        self.user_ids = population.public["user_id"].astype(str).tolist()
        if len(set(self.user_ids)) != len(self.user_ids):
            raise ValueError("Duplicate simulator users")
        if population.hidden["user_id"].astype(str).tolist() != self.user_ids:
            raise ValueError("Public and hidden users must be aligned")
        if population.taste.shape != (len(self.user_ids), world.vectors.shape[1]):
            raise ValueError("Simulator taste vector shape mismatch")
        self._user_index = {u: i for i, u in enumerate(self.user_ids)}
        self._item_index = {str(p): i for i, p in enumerate(world.product_ids)}
        self._states: dict[str, _UserState] = {}
        self._shoppers: OrderedDict[str, Shopper] = OrderedDict()
        self.start_at = utc(cfg["period"]["start_date"])
        self.end_at = self.start_at + timedelta(days=cfg["period"]["days"])
        self.calibration = "provided"

    @classmethod
    def from_config(cls, seed: int | None = None, cfg: dict[str, Any] | None = None) -> OnlineResponseEngine:
        cfg = cfg or load_config("simulator")
        folder, space = ROOT / cfg["paths"]["output_dir"], ROOT / cfg["paths"]["space_dir"]
        catalog = load_catalog(cfg)
        world = build_world(catalog, np.load(space / "item_vectors.npy"),
                            pd.read_parquet(space / "item_stats.parquet"), cfg["period"]["start_date"])
        public = pd.read_parquet(folder / "users.parquet").sort_values("user_id").reset_index(drop=True)
        hidden = pd.read_parquet(folder / "users_hidden.parquet").set_index("user_id").loc[
            public["user_id"]].reset_index()
        # users.py는 persona → seed → jitter 순서로 난수를 소비한다. 저장된 seed와 일치 여부를
        # 검사하여 다른 생성 설정으로 만든 숨김 취향을 조용히 사용하는 것을 막는다.
        seed_vectors = np.load(space / "seed_vectors.npy")
        rng = np.random.default_rng(cfg["seed"])
        names = list(cfg["persona_ratio"])
        rng.choice(names, size=len(public), p=[cfg["persona_ratio"][p] for p in names])
        seed_indices = rng.integers(0, len(seed_vectors), size=len(public))
        if not np.array_equal(seed_indices, hidden["seed"].to_numpy()):
            raise ValueError("Saved users and simulator seed/config differ; use the generation configuration")
        jitter = rng.normal(size=(len(public), seed_vectors.shape[1])) / np.sqrt(seed_vectors.shape[1])
        taste = seed_vectors[seed_indices] + cfg["users"]["taste_jitter"] * jitter
        taste = (taste / np.maximum(np.linalg.norm(taste, axis=1, keepdims=True), 1e-9)).astype(np.float32)
        calibration_path = space / "persona_calibration.json"
        if not calibration_path.is_file():
            raise FileNotFoundError(f"Missing calibrated response parameters: {calibration_path}")
        saved = json.loads(calibration_path.read_text(encoding="utf-8"))
        personas = with_funnel(load_personas(cfg), saved["values"])
        engine = cls(world, Population(public, hidden, taste), cfg, personas,
                     cfg["seed"] if seed is None else seed)
        engine.calibration = saved.get("key", str(calibration_path))
        return engine

    def signup_at(self, user_id: str) -> datetime:
        return utc(self.population.public.iloc[self._user_index[user_id]]["signup_at"])

    def expected_visits(self, user_id: str, days: int) -> float:
        h = self.population.hidden.iloc[self._user_index[user_id]]
        return self.personas[h["persona"]].sessions * float(h["activity"]) * days / self.cfg["period"]["days"]

    def _shopper(self, user_id: str, state: _UserState) -> Shopper:
        if user_id in self._shoppers:
            self._shoppers.move_to_end(user_id)
            return self._shoppers[user_id]
        i = self._user_index[user_id]
        h, world = self.population.hidden.iloc[i], self.world
        l2 = np.array([world.l2_names.index(c) for c in h["preferred_l2"] if c in world.l2_names], np.int32)
        brands = np.array([world.brand_names.index(b) for b in h["preferred_brands"] if b in world.brand_names], np.int32)
        segment = world.l1_names.index(h["main_segment"]) if h["main_segment"] in world.l1_names else -1
        base = base_scores(world, self.cfg["utility"], self.population.taste[i], l2, segment, brands,
                           float(h["brand_loyalty"]), float(h["price_sensitivity"]),
                           np.random.default_rng(stable_seed(self.seed, f"base:{user_id}")))
        for item, count in state.purchases.items():
            base[item] -= 2.0 * count
        size = min(self.cfg["utility"]["target_pool"], world.n)
        pool = np.argsort(-base, kind="stable")[:size].astype(np.int32)
        shopper = Shopper(i, self.personas[h["persona"]], segment, float(h["new_item_affinity"]),
                          base, pool, state.cart)
        self._shoppers[user_id] = shopper
        if len(self._shoppers) > self.cache_users:
            self._shoppers.popitem(last=False)
        return shopper

    def respond(self, user_id: str, product_ids: list[str], at: datetime, session_id: str,
                *, impression_id: str | None = None) -> ReactionBatch:
        at = utc(at)
        if user_id not in self._user_index:
            raise ValueError(f"Unknown simulator user: {user_id}")
        if not session_id or len(set(product_ids)) != len(product_ids):
            raise ValueError("A session ID and unique displayed product IDs are required")
        if at < self.signup_at(user_id):
            raise ValueError("Cannot simulate a visit before signup")
        if any(p not in self._item_index for p in product_ids):
            raise ValueError("Displayed product does not exist in the simulator catalog")
        ids = np.array([self._item_index[p] for p in product_ids], dtype=np.int32)
        day = (at - self.start_at).days
        if np.any(self.world.created_day[ids] > day):
            raise ValueError("Cannot expose products before their registration")
        state = self._states.setdefault(user_id, _UserState(np.random.default_rng(stable_seed(self.seed, user_id))))
        if state.last_at is not None and at < state.last_at:
            raise ValueError("Simulator visits must be chronological for each user")
        shopper = self._shopper(user_id, state)
        pending: list[tuple[str, int, str]] = []
        if state.session_id != session_id:
            state.visit_number += 1
            state.visit = Visit(day, state.visit_number)
            state.session_id = session_id
            for item in self.model.checkout_cart(shopper, state.visit, state.rng):
                pending.append(("purchase", item, "cart"))
        state.visit.day = day
        reaction = self.model.react(shopper, state.visit, ids, state.rng)
        pending.extend((kind, item, "home") for kind, item in reaction.actions)
        events = []
        for j, (kind, item, source) in enumerate(pending, 1):
            state.event_number += 1
            if kind == "purchase":
                state.purchases[item] += 1
            events.append(Event(event_id=f"sim-{self.seed}-{user_id}-{state.event_number}", user_id=user_id,
                                session_id=session_id, event_type=kind, product_id=str(self.world.product_ids[item]),
                                timestamp=at + timedelta(milliseconds=j), source=source,
                                impression_id=impression_id if source == "home" else None))
        state.last_at = events[-1].timestamp if events else at
        return ReactionBatch(events, [str(self.world.product_ids[i]) for i in reaction.clicked], reaction.next_screen)
