"""Stage 3 Re-ranking 테스트.

Stage 1/2 없이 가짜 후보로 명세 요구사항을 검증한다.
실행: pytest tests/test_reranking.py -v
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Sequence

import pandas as pd
import pytest

from src.recommendation.reranking import (
    ArmStats,
    Candidate,
    EpsilonGreedyExplorer,
    InMemoryStatsStore,
    MabConfig,
    RandomCategoryExplorer,
    UCBExplorer,
    build_explorer,
    RerankConfig,
    Reranker,
    load_catalog,
    max_run_length,
    prepare_catalog,
    slot_positions,
    to_candidates,
    to_utc,
)
from src.recommendation.reranking.catalog import catalog_path

NOW = datetime(2026, 10, 12, 12, 0, 0)
OLD = NOW - timedelta(days=60)
CATS = ["Garment Upper body", "Garment Lower body", "Shoes", "Accessories", "Underwear"]


def make(pid: str, score: float, l1: str = "Garment Upper body",
         l2: str = "T-shirt", created_at: datetime = OLD, reason: str = "personalized",
         product_code: str | None = None) -> Candidate:
    return Candidate(pid, score, l1, l2, "T-shirt - Solid", created_at, reason, product_code)


def random_candidates(n: int = 300, seed: int = 42, upper_ratio: float = 0.6) -> list[Candidate]:
    """H&M처럼 상의 비중이 높은 후보 집합."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        l1 = CATS[0] if rng.random() < upper_ratio else rng.choice(CATS[1:])
        out.append(make(f"P{i:05d}", rng.random(), l1, l2=f"{l1}-{rng.randint(0, 4)}"))
    return out


def cats_of(items, candidates: Sequence[Candidate]) -> list[str]:
    lookup = {c.product_id: c.category_l1 for c in candidates}
    return [lookup[i.product_id] for i in items]


def base_config(**kw) -> RerankConfig:
    # 가짜 후보는 category_l1에 상품군을 넣었으므로 다양성 기준도 l1
    params = dict(category_field="category_l1", new_item_slots=0, exploration_enabled=False)
    params.update(kw)
    return RerankConfig(**params)


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

def test_config_loads_recommend_yaml():
    cfg = RerankConfig.from_recommend_config()
    assert cfg.top_n == 10
    assert cfg.max_consecutive == 2
    assert cfg.min_history == 5
    assert cfg.new_item_max_age_days == 7
    assert cfg.new_item_slots >= 1
    assert 1 <= cfg.exploration_slots <= 2
    assert cfg.category_field == "category_l2"
    assert cfg.dedupe_field == "product_code"


def test_config_rejects_unknown_dedupe_field():
    with pytest.raises(ValueError):
        RerankConfig(dedupe_field="color")


def test_config_rejects_negative_slots():
    try:
        RerankConfig(exploration_slots=-1)
    except ValueError:
        return
    raise AssertionError("음수 슬롯 개수가 허용됨")


# ---------------------------------------------------------------------------
# 슬롯 위치 계산 (개수 → 위치)
# ---------------------------------------------------------------------------

def test_slot_positions_default():
    assert slot_positions(10, 1, 2) == ([5], [3, 7])


def test_slot_positions_scale_with_top_n():
    assert slot_positions(20, 1, 2) == ([10], [7, 13])


def test_slot_positions_collision_moves_to_next_free():
    new_pos, exp_pos = slot_positions(10, 1, 1)
    assert new_pos == [5] and exp_pos == [6]


def test_slot_positions_never_out_of_range_or_duplicated():
    for top_n in range(0, 15):
        for n_new in range(0, 3):
            for n_exp in range(0, 4):
                new_pos, exp_pos = slot_positions(top_n, n_new, n_exp)
                allp = new_pos + exp_pos
                assert len(allp) == len(set(allp))
                assert all(0 <= p < top_n for p in allp)
                assert len(allp) == min(top_n, n_new + n_exp)


# ---------------------------------------------------------------------------
# 기본 동작
# ---------------------------------------------------------------------------

def test_returns_top_n_unique_items():
    cands = random_candidates()
    res = Reranker(RerankConfig(category_field="category_l1")).rerank(cands, history_len=20, now=NOW)
    ids = [i.product_id for i in res.items]
    assert len(ids) == 10
    assert len(set(ids)) == 10


def test_order_follows_score_without_rules():
    cands = random_candidates()
    res = Reranker(base_config(diversity_enabled=False)).rerank(cands, 20, NOW)
    expected = [c.product_id for c in sorted(cands, key=lambda c: c.score, reverse=True)[:10]]
    assert [i.product_id for i in res.items] == expected


def test_stage2_reason_preserved():
    cands = [make(f"P{i}", 1 - i * 0.01, CATS[i % 5], reason="similar_users") for i in range(20)]
    res = Reranker(base_config()).rerank(cands, 20, NOW)
    assert all(i.reason == "similar_users" for i in res.items)


def test_empty_candidates_does_not_crash():
    res = Reranker(RerankConfig(category_field="category_l1")).rerank([], history_len=20, now=NOW)
    assert res.items == []


# ---------------------------------------------------------------------------
# 다양성: 동일 카테고리 연속 3개 이상 금지
# ---------------------------------------------------------------------------

def test_diversity_blocks_three_in_a_row():
    # 점수 상위 5개가 전부 같은 카테고리
    cands = [make(f"U{i}", 1.0 - i * 0.01, CATS[0]) for i in range(5)]
    cands += [make(f"S{i}", 0.5 - i * 0.01, CATS[2]) for i in range(5)]
    cands += [make(f"A{i}", 0.3 - i * 0.01, CATS[3]) for i in range(5)]
    res = Reranker(base_config()).rerank(cands, 20, NOW)
    assert max_run_length(cats_of(res.items, cands)) <= 2
    # 점수 1, 2위는 그대로 두고 3위 자리에 다른 카테고리가 들어와야 함
    assert [i.product_id for i in res.items[:3]] == ["U0", "U1", "S0"]


def test_diversity_holds_across_many_seeds_with_all_slots():
    """신규·탐색 슬롯 삽입 후에도 연속 규칙이 깨지지 않아야 함 (명세: 샘플 3건 이상)."""
    for seed in range(30):
        cands = random_candidates(seed=seed)
        new_pool = [make(f"N{seed}-{i}", 0.3, CATS[i % 5], created_at=NOW - timedelta(days=2))
                    for i in range(5)]
        res = Reranker(RerankConfig(category_field="category_l1"), RandomCategoryExplorer(seed=seed)).rerank(
            cands, 20, NOW, new_item_pool=new_pool)
        assert max_run_length(cats_of(res.items, cands + new_pool)) <= 2, f"seed={seed}"
        assert res.diversity_relaxed_positions == []


def test_diversity_disabled_allows_runs():
    cands = [make(f"U{i}", 1.0 - i * 0.01, CATS[0]) for i in range(10)]
    res = Reranker(base_config(diversity_enabled=False)).rerank(cands, 20, NOW)
    assert max_run_length(cats_of(res.items, cands)) == 10


def test_diversity_relaxed_when_only_one_category():
    """후보가 한 카테고리뿐이면 빈 결과 대신 규칙을 완화하고 기록한다."""
    cands = [make(f"U{i}", 1.0 - i * 0.01, CATS[0]) for i in range(10)]
    res = Reranker(base_config()).rerank(cands, 20, NOW)
    assert len(res.items) == 10
    assert res.diversity_relaxed_positions == list(range(2, 10))


# ---------------------------------------------------------------------------
# Cold-start: 이력 5개 미만 → 인기 + 트렌딩
# ---------------------------------------------------------------------------

def test_cold_start_uses_popular_and_trending():
    popular = [make(f"POP{i}", 0.9, CATS[i % 5]) for i in range(10)]
    trending = [make(f"TR{i}", 0.8, CATS[(i + 2) % 5]) for i in range(10)]
    res = Reranker(base_config()).rerank(
        random_candidates(), history_len=3, now=NOW, popular=popular, trending=trending)
    assert res.is_cold_start
    assert {i.reason for i in res.items} == {"popular", "trending"}
    assert [i.reason for i in res.items[:4]] == ["popular", "trending", "popular", "trending"]


def test_cold_start_zero_history():
    popular = [make(f"POP{i}", 0.9, CATS[i % 5]) for i in range(20)]
    res = Reranker(base_config()).rerank([], history_len=0, now=NOW, popular=popular)
    assert res.is_cold_start and len(res.items) == 10


def test_history_of_five_is_not_cold_start():
    res = Reranker(base_config()).rerank(random_candidates(), history_len=5, now=NOW)
    assert not res.is_cold_start


def test_cold_start_dedupes_popular_and_trending():
    shared = [make(f"P{i}", 0.9, CATS[i % 5]) for i in range(10)]
    res = Reranker(base_config()).rerank([], 0, NOW, popular=shared, trending=shared)
    ids = [i.product_id for i in res.items]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# 신규 상품 (등록 7일 이내) 노출 보장
# ---------------------------------------------------------------------------

def test_new_item_slot_filled_from_candidates():
    cands = random_candidates()
    fresh = make("NEW1", 0.01, CATS[3], created_at=NOW - timedelta(days=3))
    res = Reranker(base_config(new_item_slots=1)).rerank(cands + [fresh], 20, NOW)
    assert res.items[5].product_id == "NEW1"
    assert res.items[5].reason == "new_item"


def test_new_item_slot_uses_fallback_pool():
    fresh = make("NEW2", 0.1, CATS[3], created_at=NOW - timedelta(days=1))
    res = Reranker(base_config(new_item_slots=1)).rerank(
        random_candidates(), 20, NOW, new_item_pool=[fresh])
    assert "NEW2" in [i.product_id for i in res.items]


def test_item_older_than_7_days_is_not_new():
    stale = make("STALE", 0.01, CATS[3], created_at=NOW - timedelta(days=8))
    res = Reranker(base_config(new_item_slots=1)).rerank(
        random_candidates(), 20, NOW, new_item_pool=[stale])
    assert all(i.reason != "new_item" for i in res.items)
    assert "STALE" not in [i.product_id for i in res.items]


def test_exactly_7_days_is_not_new():
    # 팀 카탈로그 기준: now - 7일 "초과"만 신규
    edge = make("EDGE", 0.01, CATS[3], created_at=NOW - timedelta(days=7))
    res = Reranker(base_config(new_item_slots=1)).rerank(
        random_candidates(), 20, NOW, new_item_pool=[edge])
    assert "EDGE" not in [i.product_id for i in res.items]


def test_just_under_7_days_is_new():
    edge = make("EDGE", 0.01, CATS[3], created_at=NOW - timedelta(days=7) + timedelta(seconds=1))
    res = Reranker(base_config(new_item_slots=1)).rerank(
        random_candidates(), 20, NOW, new_item_pool=[edge])
    assert res.items[5].product_id == "EDGE"


# ---------------------------------------------------------------------------
# 탐색 슬롯 (MAB 자리)
# ---------------------------------------------------------------------------

def test_exploration_slots_count_and_flag():
    res = Reranker(RerankConfig(category_field="category_l1")).rerank(random_candidates(), 20, NOW)
    exp = [idx for idx, i in enumerate(res.items) if i.is_exploration]
    assert 1 <= len(exp) <= 2
    assert exp == [3, 7]
    assert all(res.items[idx].reason == "mab_exploration" for idx in exp)


def test_exploration_disabled_for_ab_control():
    res = Reranker(base_config()).rerank(random_candidates(), 20, NOW)
    assert not any(i.is_exploration for i in res.items)


def test_exploration_picks_outside_top_n():
    cands = random_candidates()
    top_ids = {c.product_id for c in sorted(cands, key=lambda c: c.score, reverse=True)[:10]}
    res = Reranker(RerankConfig(category_field="category_l1")).rerank(cands, 20, NOW)
    assert all(i.product_id not in top_ids for i in res.items if i.is_exploration)


def test_custom_explorer_is_injectable():
    class Fixed:
        def pick(self, candidates, n):
            return [c for c in candidates if c.product_id == "TARGET"]

    cands = random_candidates() + [make("TARGET", -1.0, CATS[4])]
    res = Reranker(base_config(exploration_enabled=True, exploration_slots=1), Fixed()) \
        .rerank(cands, 20, NOW)
    assert res.items[5].product_id == "TARGET" and res.items[5].is_exploration


# ---------------------------------------------------------------------------
# 출력 형식 · 성능
# ---------------------------------------------------------------------------

def test_request_top_n_overrides_config():
    res = Reranker(RerankConfig(category_field="category_l1")).rerank(random_candidates(), 20, NOW, top_n=20)
    assert len(res.items) == 20
    assert [idx for idx, i in enumerate(res.items) if i.is_exploration] == [7, 13]


def test_to_dict_has_spec_fields():
    res = Reranker(RerankConfig(category_field="category_l1")).rerank(random_candidates(), 20, NOW)
    for d in (i.to_dict() for i in res.items):
        assert set(d) == {"product_id", "score", "reason", "is_exploration"}


def test_reranking_is_fast_for_300_candidates():
    cands = random_candidates(300)
    reranker = Reranker(RerankConfig(category_field="category_l1"))
    for _ in range(10):  # warm-up (명세 측정 방식)
        reranker.rerank(cands, 20, NOW)
    times = sorted(reranker.rerank(cands, 20, NOW).reranking_ms for _ in range(100))
    assert times[94] < 10.0  # p95


# ---------------------------------------------------------------------------
# 같은 디자인 색상 변형 중복 제거 (product_code)
# ---------------------------------------------------------------------------

def _variants() -> list[Candidate]:
    # 같은 디자인(D1)의 색상 변형 3개가 점수 1~3위
    cands = [make(f"D1-{c}", 0.99 - i * 0.01, CATS[i % 2], product_code="D1")
             for i, c in enumerate(["black", "white", "navy"])]
    return cands + random_candidates(50)


def test_color_variants_shown_once():
    res = Reranker(base_config()).rerank(_variants(), 20, NOW)
    ids = [i.product_id for i in res.items]
    assert ids[0] == "D1-black"
    assert "D1-white" not in ids and "D1-navy" not in ids


def test_color_variants_allowed_when_dedupe_off():
    res = Reranker(base_config(dedupe_field=None)).rerank(_variants(), 20, NOW)
    assert {"D1-black", "D1-white", "D1-navy"} <= {i.product_id for i in res.items}


def test_missing_product_code_falls_back_to_product_id():
    # product_code가 없는 후보끼리는 서로 다른 상품으로 본다
    res = Reranker(base_config()).rerank(random_candidates(), 20, NOW)
    assert len(res.items) == 10


# ---------------------------------------------------------------------------
# 카탈로그 연동
# ---------------------------------------------------------------------------

def _fake_products() -> pd.DataFrame:
    # 실제 products.parquet처럼 created_at은 시간대 없는 자정
    return pd.DataFrame({
        "product_id": ["hm_0000000001", "hm_0000000002", "hm_0000000003"],
        "product_code": ["0000000", "0000000", "0000001"],
        "category_l1": ["Ladieswear", "Ladieswear", "Menswear"],
        "category_l2": ["Garment Upper body", "Garment Upper body", "Shoes"],
        "category_l3": ["T-shirt", "T-shirt", "Sneakers"],
        "created_at": pd.to_datetime(["2025-05-01", "2025-05-01", "2026-03-30"]),
        "price": [9.99, 9.99, 29.99],
    })


def test_prepare_catalog_makes_created_at_utc():
    cat = prepare_catalog(_fake_products())
    assert str(cat["created_at"].dt.tz) == "UTC"
    assert cat.index.name == "product_id"


def test_to_candidates_attaches_catalog_fields():
    cat = prepare_catalog(_fake_products())
    cands = to_candidates(["hm_0000000003", "hm_0000000001"], [0.5, 0.9], cat)
    assert [c.product_id for c in cands] == ["hm_0000000003", "hm_0000000001"]
    assert cands[0].category_l2 == "Shoes" and cands[0].product_code == "0000001"
    assert cands[1].score == 0.9 and cands[1].reason == "personalized"
    assert cands[0].created_at.tzinfo is not None


def test_to_candidates_rejects_unknown_product():
    cat = prepare_catalog(_fake_products())
    with pytest.raises(KeyError):
        to_candidates(["hm_9999999999"], [0.1], cat)


def test_catalog_without_product_code_is_accepted():
    # Stage 2 FeatureStore.products에는 product_code가 없다
    cat = prepare_catalog(_fake_products().drop(columns="product_code"))
    cands = to_candidates(["hm_0000000001"], [0.1], cat)
    assert cands[0].product_code is None


def test_rerank_with_catalog_candidates_and_utc_now():
    # 시간대 없는 요청 시각도 to_utc로 맞추면 신규 상품 판단이 동작한다
    cat = prepare_catalog(_fake_products())
    cands = to_candidates(list(cat.index), [0.9, 0.8, 0.1], cat)
    cfg = RerankConfig(top_n=3, new_item_slots=1, exploration_enabled=False)
    res = Reranker(cfg).rerank(cands, 20, to_utc(datetime(2026, 3, 31)))
    ids = [i.product_id for i in res.items]
    assert "hm_0000000003" in ids                                   # 3/30 등록 → 신규
    assert not {"hm_0000000001", "hm_0000000002"} <= set(ids)       # 같은 디자인은 1개만


def test_to_utc_keeps_aware_time():
    kst = timezone(timedelta(hours=9))
    assert to_utc(datetime(2026, 3, 31, 9, tzinfo=kst)) == datetime(2026, 3, 31, tzinfo=timezone.utc)


@pytest.mark.skipif(not catalog_path().exists(), reason="products.parquet 없음 (make catalog)")
def test_real_catalog_loads():
    cat = load_catalog()
    assert len(cat) > 0 and cat["product_code"].notna().all()
    sample = list(cat.index[:300])
    cands = to_candidates(sample, [1.0 - i / 300 for i in range(300)], cat)
    res = Reranker(RerankConfig.from_recommend_config()).rerank(
        cands, 20, to_utc(datetime(2026, 3, 31)))
    codes = [cat.loc[i.product_id, "product_code"] for i in res.items]
    assert len(codes) == len(set(codes))
    l2 = [cat.loc[i.product_id, "category_l2"] for i in res.items]
    assert max_run_length(l2) <= 2


# ---------------------------------------------------------------------------
# MAB (Epsilon-Greedy / UCB)
# ---------------------------------------------------------------------------

ARMS = ["Shoes", "Swimwear", "Accessories", "Nightwear"]


def arm_candidates(per_arm: int = 5) -> list[Candidate]:
    # arm = category_l2
    return [make(f"{a}-{i}", 0.5 - i * 0.01, l2=a) for a in ARMS for i in range(per_arm)]


def store_with(stats: dict[str, tuple[int, int]]) -> InMemoryStatsStore:
    store = InMemoryStatsStore()
    for arm, (shown, clicks) in stats.items():
        for k in range(shown):
            store.record(arm, k < clicks)
    return store


def test_mab_config_loads_recommend_yaml():
    cfg = MabConfig.from_recommend_config()
    assert cfg.algorithm in {"epsilon_greedy", "ucb"}
    assert cfg.arm_field == "category_l2" and cfg.seed == 42


def test_mab_config_rejects_thompson():
    with pytest.raises(ValueError):
        MabConfig(algorithm="thompson")


def test_build_explorer_follows_algorithm():
    assert isinstance(build_explorer(MabConfig(algorithm="epsilon_greedy")), EpsilonGreedyExplorer)
    assert isinstance(build_explorer(MabConfig(algorithm="ucb")), UCBExplorer)


def test_in_memory_store_counts_and_returns_copies():
    store = store_with({"Shoes": (4, 1)})
    got = store.get(["Shoes", "Swimwear"])
    assert got["Shoes"].shown == 4 and got["Shoes"].ctr == 0.25
    assert got["Swimwear"] == ArmStats()
    got["Shoes"].shown = 999
    assert store.get(["Shoes"])["Shoes"].shown == 4


@pytest.mark.parametrize("cls", [EpsilonGreedyExplorer, UCBExplorer])
def test_untried_arms_first(cls):
    # Shoes만 노출 이력이 있으면 나머지 arm을 먼저 시도
    explorer = cls(store=store_with({"Shoes": (100, 50)}))
    picked = explorer.pick(arm_candidates(), 3)
    assert "Shoes" not in {explorer.arm_of(c) for c in picked}


@pytest.mark.parametrize("cls", [EpsilonGreedyExplorer, UCBExplorer])
def test_pick_uses_distinct_arms_then_wraps(cls):
    explorer = cls()
    picked = explorer.pick(arm_candidates(per_arm=2), 6)
    arms = [explorer.arm_of(c) for c in picked]
    assert len(set(arms[:4])) == 4          # 한 바퀴는 arm 중복 없음
    assert set(arms[4:]) <= set(arms[:4])   # 그다음 두 번째 상품
    # 각 arm 안에서는 점수 최고 상품부터
    assert all(c.product_id.endswith("-0") for c in picked[:4])


@pytest.mark.parametrize("cls", [EpsilonGreedyExplorer, UCBExplorer])
def test_pick_handles_empty_and_small_pools(cls):
    explorer = cls()
    assert explorer.pick([], 3) == []
    assert len(explorer.pick(arm_candidates(per_arm=1), 10)) == len(ARMS)


def test_epsilon_zero_picks_best_ctr():
    stats = {"Shoes": (100, 30), "Swimwear": (100, 5), "Accessories": (100, 10), "Nightwear": (100, 1)}
    explorer = EpsilonGreedyExplorer(epsilon=0.0, store=store_with(stats))
    for _ in range(20):
        assert explorer.arm_of(explorer.pick(arm_candidates(), 1)[0]) == "Shoes"


def test_epsilon_greedy_prior_keeps_unlucky_arm_alive():
    # 좋은 arm이 초반 3번 노출에 클릭 0 → 보정 없으면 0%로 꼴찌, 보정하면 1/5=20%로 1위
    stats = {"Shoes": (3, 0), "Swimwear": (100, 3)}
    cands = [c for c in arm_candidates() if c.category_l2 in stats]
    plain = EpsilonGreedyExplorer(epsilon=0.0, ctr_prior=(0, 1), store=store_with(stats))
    smoothed = EpsilonGreedyExplorer(epsilon=0.0, store=store_with(stats))
    assert plain.arm_of(plain.pick(cands, 1)[0]) == "Swimwear"
    assert smoothed.arm_of(smoothed.pick(cands, 1)[0]) == "Shoes"


def test_mab_config_rejects_bad_prior():
    with pytest.raises(ValueError):
        MabConfig(ctr_prior=(2, 2))


def test_epsilon_one_spreads_choices():
    stats = {a: (100, 30 if a == "Shoes" else 1) for a in ARMS}
    explorer = EpsilonGreedyExplorer(epsilon=1.0, store=store_with(stats))
    firsts = {explorer.arm_of(explorer.pick(arm_candidates(), 1)[0]) for _ in range(200)}
    assert firsts == set(ARMS)


def test_ucb_bonus_favors_less_shown_arm():
    # 클릭률이 같으면 노출이 적은 arm의 UCB 점수가 더 높다
    explorer = UCBExplorer(store=store_with({"Shoes": (1000, 100), "Swimwear": (10, 1)}))
    picked = explorer.pick([c for c in arm_candidates() if c.category_l2 in {"Shoes", "Swimwear"}], 1)
    assert explorer.arm_of(picked[0]) == "Swimwear"


@pytest.mark.parametrize("cls", [EpsilonGreedyExplorer, UCBExplorer])
def test_bandit_converges_to_best_arm(cls):
    # 가짜 클릭률: Shoes 15%, 나머지 3% → 1,000번 반복 후 Shoes를 가장 많이 골라야 함
    true_ctr = {"Shoes": 0.15, "Swimwear": 0.03, "Accessories": 0.03, "Nightwear": 0.03}
    explorer = cls(seed=42)
    rng = random.Random(42)
    counts = dict.fromkeys(ARMS, 0)
    cands = arm_candidates()
    for _ in range(1000):
        arm = explorer.arm_of(explorer.pick(cands, 1)[0])
        counts[arm] += 1
        explorer.record_feedback(arm, rng.random() < true_ctr[arm])
    assert max(counts, key=counts.get) == "Shoes", counts
    assert counts["Shoes"] > 1000 / len(ARMS), counts


@pytest.mark.parametrize("algorithm", ["epsilon_greedy", "ucb"])
def test_reranker_with_bandit_keeps_rules(algorithm):
    explorer = build_explorer(MabConfig(algorithm=algorithm, arm_field="category_l1"))
    cfg = RerankConfig(category_field="category_l1", new_item_slots=0, exploration_slots=2)
    cands = random_candidates()
    res = Reranker(cfg, explorer).rerank(cands, 20, NOW)
    top_ids = {c.product_id for c in sorted(cands, key=lambda c: c.score, reverse=True)[:10]}
    explored = [i for i in res.items if i.is_exploration]
    assert len(explored) == 2
    assert all(i.reason == "mab_exploration" and i.product_id not in top_ids for i in explored)
    assert max_run_length(cats_of(res.items, cands)) <= 2
    assert len({i.product_id for i in res.items}) == 10
