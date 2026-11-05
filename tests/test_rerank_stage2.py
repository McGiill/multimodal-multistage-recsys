"""Stage 2 출력 → Stage 3 연결(RerankService) 테스트.

학습된 모델 없이 Stage 2 상품 표 모양의 가짜 데이터로 검증한다.
실행: pytest tests/test_rerank_stage2.py -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.recommendation.reranking import (
    InMemoryStatsStore,
    RerankConfig,
    RerankService,
    from_unix,
    get_rerank_service,
    history_count,
    max_run_length,
)
from src.recommendation.reranking import stage2
from src.recommendation.reranking.mab import EpsilonGreedyExplorer
from src.recommendation.reranking.popularity import PopularityConfig, PopularityLists

NOW = datetime(2026, 3, 31, 12, tzinfo=timezone.utc)
NOW_TS = NOW.timestamp()
L2 = ["Garment Upper body", "Garment Lower body", "Shoes", "Accessories"]


def make_products(n: int = 40, n_new: int = 3) -> pd.DataFrame:
    """store.products 모양 (product_id 정렬, created_at은 시간대 없는 자정, product_code 없음).

    앞쪽 상품일수록 상의 비중이 높고, 마지막 n_new개는 신규 상품(1일 전 등록).
    """
    rows = []
    for i in range(n):
        new = i >= n - n_new
        rows.append({
            "product_id": f"hm_{i:010d}",
            "category_l1": "Ladieswear",
            "category_l2": L2[0] if i < n // 2 else L2[i % len(L2)],
            "category_l3": "T-shirt",
            "brand": "b", "color": "c", "pattern": "p", "price": 10.0,
            "created_at": pd.Timestamp("2026-03-30") if new else pd.Timestamp("2025-06-01"),
        })
    return pd.DataFrame(rows)


def make_service(products: pd.DataFrame | None = None, **kw) -> RerankService:
    products = make_products() if products is None else products
    cfg = RerankConfig.from_recommend_config()
    return RerankService(products["product_id"].tolist(), products, cfg, **kw)


def warm_state(n_views: int = 10, history_len: int = 20) -> tuple:
    h = np.zeros(history_len, np.int64)
    h[:n_views] = np.arange(1, n_views + 1)
    return h, max(1, n_views), np.zeros(4, np.float32)


# ---------------------------------------------------------------------------
# 변환 함수
# ---------------------------------------------------------------------------

def test_from_unix_is_utc() -> None:
    t = from_unix(NOW_TS)
    assert t == NOW and t.tzinfo is not None


@pytest.mark.parametrize("n_views, expected", [(0, 0), (4, 4), (5, 5), (20, 20)])
def test_history_count_ignores_padding(n_views: int, expected: int) -> None:
    # state[1]은 이력 0개여도 1이므로 쓰면 안 된다
    assert history_count(warm_state(n_views)) == expected


def test_candidates_map_ids_to_catalog() -> None:
    products = make_products()
    svc = make_service(products)
    cands = svc.candidates(np.array([1, 40]), np.array([0.9, 0.1], np.float32))
    assert [c.product_id for c in cands] == ["hm_0000000000", "hm_0000000039"]
    assert cands[0].category_l2 == L2[0]
    assert cands[0].reason == "cvr"
    assert cands[1].created_at == datetime(2026, 3, 30, tzinfo=timezone.utc)
    assert cands[0].product_code is None  # store.products에는 product_code 없음


def test_product_codes_from_catalog() -> None:
    products = make_products()
    codes = {pid: "0001" for pid in products["product_id"][:2]}  # 1, 2번이 같은 디자인
    svc = make_service(products, product_codes=codes)
    c1, c2, c3 = svc.candidates([1, 2, 3], [0.3, 0.2, 0.1])
    assert c1.product_code == c2.product_code == "0001"
    assert c3.product_code is None  # 카탈로그에 없는 상품은 product_id 기준


def test_unknown_item_id_rejected() -> None:
    products = make_products()
    with pytest.raises(ValueError):
        RerankService(["hm_9999999999"], products, RerankConfig.from_recommend_config())


def test_padding_id_rejected() -> None:
    with pytest.raises(ValueError):
        make_service().candidates([0, 1], [0.5, 0.4])


def test_new_item_ids_by_request_time() -> None:
    svc = make_service()
    assert svc.new_item_ids(NOW_TS).tolist() == [38, 39, 40]
    # 등록 전 시각이면 신규 상품 없음, 8일 뒤면 7일이 지나 신규 아님
    assert svc.new_item_ids((NOW - timedelta(days=2)).timestamp()).size == 0
    assert svc.new_item_ids((NOW + timedelta(days=8)).timestamp()).size == 0


# ---------------------------------------------------------------------------
# rerank
# ---------------------------------------------------------------------------

def test_rerank_stage2_output() -> None:
    svc = make_service()
    ids = np.arange(1, 38)  # 신규 상품(38~40)은 후보에 없음
    scores = np.linspace(1, 0, len(ids)).astype(np.float32)
    new = svc.new_item_ids(NOW_TS)
    res = svc.rerank(ids, scores, warm_state(), NOW_TS, top_n=10,
                     new_ids=new, new_scores=np.full(len(new), 0.01, np.float32))

    assert not res.is_cold_start
    assert len(res.items) == 10
    lookup = svc.catalog["category_l2"]
    assert max_run_length([lookup[i.product_id] for i in res.items]) <= 2
    assert any(i.reason == "new_item" for i in res.items)
    # Stage 2 점수는 그대로
    score_of = {c.product_id: c.score for c in svc.candidates(ids, scores)}
    for item in res.items:
        if item.reason == "cvr":
            assert item.score == pytest.approx(score_of[item.product_id])


def test_rerank_cold_start_uses_fallback() -> None:
    svc = make_service()
    popular = svc.candidates(range(1, 11), np.linspace(1, 0.5, 10), reason="popular")
    res = svc.rerank(np.array([], np.int64), np.array([], np.float32), warm_state(0), NOW_TS,
                     top_n=5, popular=popular)
    assert res.is_cold_start
    assert res.items and all(i.reason in {"popular", "new_item", "mab_exploration"} for i in res.items)


def test_cold_start_without_lists_keeps_stage2() -> None:
    # 인기·트렌딩 목록이 아직 없으면 빈 진열 대신 Stage 2 순서로 채운다
    svc = make_service()
    ids = np.arange(1, 38)
    res = svc.rerank(ids, np.linspace(1, 0, len(ids)), warm_state(0), NOW_TS, top_n=10)
    assert not res.is_cold_start and len(res.items) == 10


# ---------------------------------------------------------------------------
# 서빙용 보관 (MAB 통계 유지)
# ---------------------------------------------------------------------------

def test_feedback_persists_across_requests() -> None:
    store = InMemoryStatsStore()
    svc = make_service(explorer=EpsilonGreedyExplorer(store=store))
    ids = np.arange(1, 38)
    scores = np.linspace(1, 0, len(ids)).astype(np.float32)

    first = svc.rerank(ids, scores, warm_state(), NOW_TS, top_n=10)
    assert first.exploration_arms
    clicked = set(list(first.exploration_arms)[:1])
    assert svc.record_feedback(first, clicked) == len(first.exploration_arms)

    arm = first.exploration_arms[next(iter(clicked))]
    stats = store.get([arm])[arm]
    assert stats.shown >= 1 and stats.clicks == 1


class FakeModels:
    """RecommendationModels처럼 객체 자체로 캐시 키가 된다 (SimpleNamespace는 hash 불가)."""

    def __init__(self, store: SimpleNamespace) -> None:
        self.store = store


def test_get_rerank_service_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    get_rerank_service.cache_clear()
    monkeypatch.setattr(stage2, "load_product_codes", lambda: None)
    products = make_products()
    events = pd.DataFrame({"event_type": pd.Series(dtype=str), "_time": pd.Series(dtype=float),
                           "_pid": pd.Series(dtype=int)})
    models = FakeModels(SimpleNamespace(item_ids=products["product_id"].tolist(), products=products,
                                        events=events, metadata={}))
    try:
        a, b = get_rerank_service(models), get_rerank_service(models)
        assert a is b  # 요청마다 새로 만들면 MAB 통계가 사라진다
    finally:
        get_rerank_service.cache_clear()


# ---------------------------------------------------------------------------
# cold-start 인기·트렌딩 목록
# ---------------------------------------------------------------------------

DAY = 86400.0


def make_popularity(n_items: int = 40) -> PopularityLists:
    """기준 시각 NOW_TS 기준: 상품 1은 구매 3번(7일 안), 상품 2는 구매 1번, 상품 3은 8일 전 구매(기간 밖).
    조회: 상품 4는 최근 1일에 5번(급상승), 상품 5는 매일 1번씩 꾸준히, 상품 6은 최근 1일 2번(최소 3번 미달)."""
    views = [(NOW_TS - 0.5 * DAY, 4)] * 5 + [(NOW_TS - d * DAY - 1, 5) for d in range(8)] + [(NOW_TS - 0.2 * DAY, 6)] * 2
    buys = [(NOW_TS - 1 * DAY, 1)] * 3 + [(NOW_TS - 2 * DAY, 2), (NOW_TS - 8 * DAY, 3)]
    views.sort(), buys.sort()
    cfg = PopularityConfig(window_days=7, trending_recent_days=1, trending_min_views=3, list_size=10)
    return PopularityLists(np.array([t for t, _ in views]), np.array([i for _, i in views]),
                           np.array([t for t, _ in buys]), np.array([i for _, i in buys]), n_items, cfg)


def test_popular_by_recent_purchases() -> None:
    ids, scores = make_popularity().popular(NOW_TS)
    assert ids[:2].tolist() == [1, 2]       # 구매 수 순
    assert 3 not in ids[:2]                 # 8일 전 구매는 기간 밖
    assert scores[0] == 1.0 and 0 < scores[1] < 1
    assert set(ids[2:]) <= {4, 5, 6}        # 구매 없는 상품은 조회 수 순으로 뒤에


def test_trending_by_view_growth_and_min_views() -> None:
    ids, scores = make_popularity().trending(NOW_TS)
    assert ids.tolist() == [4]              # 5는 꾸준해서 증가 없음(최근 1번 < 3), 6은 최근 조회 2번
    assert scores.tolist() == [1.0]


def test_lists_use_only_past_events() -> None:
    pop = make_popularity()
    ids, _ = pop.popular(NOW_TS - 1.5 * DAY)   # 상품 1의 구매(1일 전)는 아직 없음
    assert ids[0] == 2


def test_popularity_config_from_yaml() -> None:
    cfg = PopularityConfig.from_recommend_config()
    assert cfg.window_days >= 1 and cfg.list_size >= 50  # 평가 k=50을 채울 수 있어야 함


def test_cold_start_uses_popular_and_trending() -> None:
    svc = make_service(popularity=make_popularity())
    ids = np.arange(1, 38)
    res = svc.rerank(ids, np.linspace(1, 0, len(ids)), warm_state(0), NOW_TS, top_n=10)
    assert res.is_cold_start
    reasons = {i.reason for i in res.items}
    assert {"popular", "trending"} <= reasons
    assert res.items[0].product_id == "hm_0000000000"  # 번호 1 = 구매 최다


def test_cold_start_list_time_follows_frozen_history() -> None:
    # test 시각(valid 끝 이후) 요청이면 valid 끝 시각 기준 목록
    svc = make_service(popularity=make_popularity(), history_cutoffs=(NOW_TS - 3 * DAY, NOW_TS - 1.5 * DAY))
    assert svc.history_time(NOW_TS) == NOW_TS - 1.5 * DAY
    assert svc.history_time(NOW_TS - 2 * DAY) == NOW_TS - 3 * DAY   # valid 기간 → train 끝
    pop, _ = svc.cold_start_lists(NOW_TS, 10)
    assert pop[0].product_id == "hm_0000000001"  # 상품 1 구매(1일 전)는 기준 시각 이후라 빠짐


# ---------------------------------------------------------------------------
# MAB 피드백: 진열 즉시 노출, 클릭 이벤트로 클릭
# ---------------------------------------------------------------------------

def test_shown_then_click_updates_arm_stats() -> None:
    store = InMemoryStatsStore()
    svc = make_service(explorer=EpsilonGreedyExplorer(store=store))
    ids = np.arange(1, 38)
    res = svc.rerank(ids, np.linspace(1, 0, len(ids)).astype(np.float32), warm_state(), NOW_TS, top_n=10)
    assert svc.record_shown("U1", res) == len(res.exploration_arms) > 0
    pid, arm = next(iter(res.exploration_arms.items()))
    shown = store.get([arm])[arm].shown
    assert shown >= 1 and store.get([arm])[arm].clicks == 0

    assert not svc.record_click("U2", pid)          # 다른 사용자
    assert not svc.record_click("U1", "hm_9999999999")  # 탐색 칸이 아닌 상품
    assert svc.record_click("U1", pid)
    assert not svc.record_click("U1", pid)          # 같은 상품 두 번째 클릭은 안 셈
    assert store.get([arm])[arm].clicks == 1 and store.get([arm])[arm].shown == shown


def test_new_recommendation_replaces_pending_clicks() -> None:
    svc = make_service(explorer=EpsilonGreedyExplorer(store=InMemoryStatsStore()))
    ids = np.arange(1, 38)
    scores = np.linspace(1, 0, len(ids)).astype(np.float32)
    first = svc.rerank(ids, scores, warm_state(), NOW_TS, top_n=10)
    svc.record_shown("U1", first)
    cold = svc.rerank(ids, scores, warm_state(0), NOW_TS, top_n=10)  # 목록 없음 → Stage 2 순서, 탐색 칸 있음
    svc.record_shown("U1", cold)
    stale = set(first.exploration_arms) - set(cold.exploration_arms)
    assert all(not svc.record_click("U1", p) for p in stale)
