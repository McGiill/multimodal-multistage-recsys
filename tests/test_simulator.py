"""시뮬레이터 동작 확인: 작은 가짜 카탈로그·상품 지도로 실제 생성 과정을 끝까지 돌려 본다 (H&M 원본, torch 불필요)."""

import copy

import numpy as np
import pandas as pd
import pytest

from src.common.config import load_config
from src.simulator.behavior import Simulator
from src.simulator.calibrate import _update
from src.simulator.catalog import build_world
from src.simulator.personas import load_personas
from src.simulator.pipeline import to_tables
from src.simulator.response import ResponseModel, Shopper, Visit
from src.simulator.rr_calibration import behavior_stats
from src.simulator.search_table import build_search_table
from src.simulator.users import generate_users

START = "2026-01-01"
DAYS = 14
SEGMENTS = ["Ladieswear", "Menswear"]
TYPES = [("Garment Upper body", "Sweater"), ("Garment Lower body", "Trousers"), ("Shoes", "Sneakers"),
         ("Accessories", "Bag")]


def _catalog(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    l2, l3 = zip(*[TYPES[i % len(TYPES)] for i in range(n)])
    created = pd.Timestamp("2025-06-01") + pd.to_timedelta(rng.integers(0, 200, n), unit="D")
    created = created.where(np.arange(n) % 12 != 0, pd.Timestamp(START) + pd.Timedelta(days=DAYS - 3))  # 신상품
    return pd.DataFrame({
        "product_id": [f"hm_{i:010d}" for i in range(n)],
        "product_code": [f"{i // 2:07d}" for i in range(n)],
        "name": [f"item {i // 2}" for i in range(n)],
        "description": "",
        "brand": [f"BR{(i % 5) + 1:02d}" for i in range(n)],
        "category_l1": [SEGMENTS[i % 2] for i in range(n)],
        "category_l2": list(l2),
        "category_l3": list(l3),
        "color": rng.choice(["Black", "White", "Blue"], n),
        "pattern": rng.choice(["Solid", "Stripe"], n),
        "price": rng.choice([9.99, 14.99, 19.99, 29.99], n),
        "image_path": "",
        "created_at": created,
    })


def _setup(n_users: int = 40, seed: int = 0):
    cfg = copy.deepcopy(load_config("simulator"))
    cfg["scale"]["n_users"] = n_users
    cfg["period"] = {"start_date": START, "days": DAYS}
    cfg["search"]["engine"] = "attribute"
    cfg["utility"]["target_pool"] = 100
    cfg["lists"]["home_pool"] = 60
    cfg["seed"] = seed

    rng = np.random.default_rng(seed)
    catalog = _catalog()
    vectors = rng.normal(size=(len(catalog), 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    stats = pd.DataFrame({"product_id": catalog["product_id"], "pop_all": rng.integers(0, 500, len(catalog)),
                          "pop_recent": rng.integers(0, 50, len(catalog))})
    world = build_world(catalog, vectors, stats, START)
    world = build_world(catalog, vectors, stats, START,
                        search=build_search_table(cfg, catalog, world.popularity, out_dir=None))

    seeds = pd.DataFrame({"seed": range(30), "age": rng.integers(18, 60, 30).astype(float),
                          "main_segment": [SEGMENTS[i % 2] for i in range(30)],
                          "top_brands": [[f"BR{(i % 5) + 1:02d}"] for i in range(30)],
                          "top_brand_share": 0.5, "n_items": 10})
    seed_vectors = rng.normal(size=(30, 8)).astype(np.float32)
    brand_types = {f"BR{i:02d}": "general" if i <= 3 else "specialty" for i in range(1, 6)}
    personas = load_personas(cfg)
    pop = generate_users(cfg, personas, world, seeds, seed_vectors, brand_types, np.random.default_rng(seed))
    return cfg, catalog, world, personas, pop


@pytest.fixture(scope="module")
def generated():
    cfg, catalog, world, personas, pop = _setup()
    log = Simulator(world, cfg, personas).run(pop, progress=False)
    events, impressions = to_tables(log, world, pop, cfg)
    return cfg, catalog, pop, events, impressions.to_pandas()


def test_output_columns_and_hidden_split(generated) -> None:
    _, _, pop, events, impressions = generated
    assert list(events.columns) == ["event_id", "user_id", "session_id", "timestamp", "event_type", "product_id",
                                    "query", "source", "impression_id"]
    assert list(pop.public.columns) == ["user_id", "age_group", "gender", "signup_at"]
    assert "persona" in pop.hidden.columns and "persona" not in events.columns
    assert set(events["event_type"]) <= {"search", "view", "cart", "purchase"}
    assert events["event_id"].is_unique and events["timestamp"].is_monotonic_increasing
    assert {"product_ids", "clicked_ids"} <= set(impressions.columns)


def test_events_follow_rules(generated) -> None:
    _, catalog, _, events, impressions = generated
    search = events[events["event_type"] == "search"]
    assert search["query"].notna().all() and search["product_id"].isna().all()
    others = events[events["event_type"] != "search"]
    assert others["product_id"].notna().all()

    # 조회는 모두 실제로 보여준 목록에서 클릭한 상품
    imp = impressions.set_index("impression_id")
    for _, e in events[events["event_type"] == "view"].iterrows():
        row = imp.loc[e["impression_id"]]
        assert e["product_id"] in row["product_ids"] and e["product_id"] in row["clicked_ids"]

    # 담기 전에 조회, 구매 전에 담기
    for kind, before in (("cart", "view"), ("purchase", "cart")):
        for _, e in events[events["event_type"] == kind].iterrows():
            prior = events[(events["user_id"] == e["user_id"]) & (events["product_id"] == e["product_id"])
                           & (events["event_type"] == before) & (events["timestamp"] <= e["timestamp"])]
            assert len(prior) > 0

    # 등록 전 상품은 나오지 않고, 모든 이벤트는 로그 기간 안
    created = events["product_id"].map(catalog.set_index("product_id")["created_at"])
    assert (events["timestamp"].dt.normalize() >= created.dt.normalize())[others.index].all()
    end = pd.Timestamp(START) + pd.Timedelta(days=DAYS)
    assert events["timestamp"].min() >= pd.Timestamp(START) and events["timestamp"].max() < end


def test_same_seed_same_result() -> None:
    a = _setup(n_users=10)
    b = _setup(n_users=10)
    ev_a, _ = to_tables(Simulator(a[2], a[0], a[3]).run(a[4], progress=False), a[2], a[4], a[0])
    ev_b, _ = to_tables(Simulator(b[2], b[0], b[3]).run(b[4], progress=False), b[2], b[4], b[0])
    pd.testing.assert_frame_equal(ev_a, ev_b)


def test_top_positions_get_more_clicks() -> None:
    cfg, _, world, personas, _ = _setup(n_users=1)
    model = ResponseModel(world, cfg)
    items = np.arange(20, dtype=np.int32)
    first = np.zeros(20)
    rng = np.random.default_rng(0)
    for _ in range(400):  # 모든 상품 점수가 같으면 차이는 순위 효과뿐
        shopper = Shopper(index=0, persona=personas["pragmatist"], segment=0, new_affinity=0.0,
                          base=np.full(world.n, 3.0, dtype=np.float32), pool=items)
        reaction = model.react(shopper, Visit(day=0), items, rng)
        if reaction.clicked:
            first[reaction.clicked[0]] += 1
    assert first[:5].sum() > 2 * first[-5:].sum()


def test_persona_config_is_validated() -> None:
    cfg = copy.deepcopy(load_config("simulator"))
    cfg["persona_ratio"]["trendsetter"] += 0.1
    with pytest.raises(ValueError):
        load_personas(cfg)
    cfg = copy.deepcopy(load_config("simulator"))
    cfg["personas"]["pragmatist"]["rr_group"] = "very/high"
    with pytest.raises(ValueError):
        load_personas(cfg)


def test_behavior_stats_matches_hand_count() -> None:
    """Retailrocket과 시뮬레이터에 같이 쓰는 지표: 조회 다음 이벤트, 담은 상품의 구매 시점."""
    t = pd.Timestamp("2026-01-01")
    rows = [  # (방문, 분, 종류, 상품)
        ("s1", 0, "view", "a"), ("s1", 1, "view", "b"), ("s1", 2, "cart", "b"), ("s1", 3, "purchase", "b"),
        ("s1", 4, "view", "c"), ("s1", 5, "cart", "c"),
        ("s2", 90, "search", None), ("s2", 91, "purchase", "c"), ("s2", 92, "view", "d"),
    ]
    ev = pd.DataFrame([{"user_id": "u", "session_id": s, "timestamp": t + pd.Timedelta(minutes=m),
                        "event_type": e, "product_id": p} for s, m, e, p in rows])
    st = behavior_stats(ev)
    # 조회 4번 다음: view, cart, cart, exit (search는 Retailrocket에 없어서 제외)
    assert st["view_to_view"] == 0.25 and st["view_to_cart"] == 0.5 and st["view_exit"] == 0.25
    assert st["cart_purchase_same_visit"] == 0.5 and st["cart_purchase_later"] == 0.5


def test_calibration_moves_toward_target() -> None:
    values = {"cart_rate": 0.02, "purchase_in_session": 0.2, "purchase_next_visit": 0.01, "exit_after_view": 0.3,
              "sessions": 30}
    goal = {"view_to_cart": 0.04, "view_exit": 0.5, "cart_purchase_same_visit": 0.3, "cart_purchase_later": 0.05}
    got = {"view_to_cart": 0.02, "view_exit": 0.6, "cart_purchase_same_visit": 0.2, "cart_purchase_later": 0.05}
    new = _update(values, got, goal)
    assert new["cart_rate"] == pytest.approx(0.04)          # 측정이 목표의 절반 → 2배
    assert new["exit_after_view"] < values["exit_after_view"]  # 측정 이탈이 목표보다 높음 → 낮춤
    assert new["purchase_in_session"] == pytest.approx(0.3)
    assert new["purchase_next_visit"] == pytest.approx(0.01)
