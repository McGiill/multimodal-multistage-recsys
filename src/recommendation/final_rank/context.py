"""최종 순위 후보 풀과 특징 (노출 구조: 카테고리 목록 안 인기·신상품·가격 순위, 고객 선호, 장바구니 등).

후보 풀 = Stage 1 상위 pool_size개 + 고객 이력의 상위 카테고리 기준 목록
  catpop   관심 소분류 5개의 최근 7일 인기 상위        catnew   관심 소분류 신상품 (등록 new_days일 이내)
  segpop   주 고객층(category_l1) 인기 상위            taste_tail 관심 소분류 3개 안 전체 인기 순위 밖 상품, 취향(Two-Tower) 순
  pricelow 관심 소분류 3개 저가순                      l2pop    관심 중분류 3개 인기 상위
  g13pop / g13cheap / g13new  (고객층, 소분류) 목록 — 쇼핑몰 카테고리 화면 단위 — 누적 조회순 / 저가순 / 신상품
모든 목록은 요청 시각에 등록된 상품만, 고객층 범위(allowed)가 있으면 그 안에서만 고른다.
기준 시각(인기 등)은 FeatureStore.frozen_time 규칙을 따라 평가·학습에서 미래 기록을 보지 않는다.

카테고리 의도 특징 (2026-11-22): 고객이 이번에 어느 카테고리를 볼지 — 시각이 있는 조회 기록 전체로 계산
  g13_/l3_days_since  그 (고객층, 소분류) / 소분류를 마지막으로 본 뒤 경과일      g13_visit_days  그 그룹을 본 날 수
  g13_/l3_share_rw    시간 감쇠(반감 7일) 비율        g13_last_visit  마지막 방문(마지막 조회 앞 3시간) 안 비율
  g13_rw_rank         감쇠 비율 기준 그룹 순위         l3_carted / l3_bought  그 소분류 장바구니·구매 수
  days_since_last     마지막 조회 뒤 경과일 (고객 단위)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

WEEK = 7 * 86400
SOURCES = ("tt300", "catpop", "catnew", "segpop", "taste_tail", "pricelow", "l2pop", "g13pop", "g13cheap", "g13new")
FEATURES = (
    "dfm", "tt", "tt_rank", "pop7", "pop1", "pop_in_l3", "pop_rank_l3", "new_rank_l3", "price_rank_l3",
    "age_days", "is_new", "in_hist", "new_aff", "view_cnt", "in_cart", "recent_l3",
    "pop_all", "pop_rank_g13", "price_rank_g13", "pop_rank_l1", "new_rank_g13",
    "s_l1", "s_l2", "s_l3", "s_brand", "s_color", "price_gap", "hist_len",
    *(f"src_{s}" for s in SOURCES),
    "g13_days_since", "l3_days_since", "g13_visit_days", "g13_share_rw", "l3_share_rw", "g13_last_visit",
    "g13_rw_rank", "l3_carted", "l3_bought", "days_since_last",
)
HALF_LIFE = 7 * 86400   # 시간 감쇠 비율의 반감 (7일)
LAST_VISIT = 3 * 3600   # 마지막 방문 = 마지막 조회 앞 3시간 안의 조회
DEFAULT_SOURCES = {"top_l3": 5, "top_l2": 3, "top_g13": 5, "catpop": 40, "segpop": 100, "taste_tail": 60,
                   "taste_tail_min_rank": 2000, "pricelow": 15, "l2pop": 30, "g13pop": 40, "g13cheap": 20,
                   "new_days": 7, "g13_new_days": 14}
_GROUP = 10 ** 10  # 그룹 번호 × _GROUP + 등록 시각 → 그룹 안 등록 순위를 한 번의 searchsorted로


def group_rank(values: np.ndarray, groups: np.ndarray, descending: bool) -> tuple[np.ndarray, np.ndarray]:
    """같은 그룹 안 순위 (0 = 맨 위)와 그룹 크기."""
    order = np.lexsort(((-values if descending else values), groups))
    g = groups[order]
    start = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
    size = np.diff(np.r_[start, len(g)])
    pos = np.empty(len(values), np.int64)
    pos[order] = np.arange(len(g)) - np.repeat(start, size)
    sz = np.empty(len(values), np.int64)
    sz[order] = np.repeat(size, size)
    return pos, sz


def _sorted_lists(groups: np.ndarray, key: np.ndarray) -> dict[int, np.ndarray]:
    """그룹별 상품 번호 (key 오름차순). 0번(padding) 제외."""
    ids = np.arange(1, len(groups))
    order = np.lexsort((key[ids], groups[ids]))
    ids, g = ids[order], groups[ids][order]
    cut = np.flatnonzero(np.diff(g)) + 1
    return {int(gg[0]): part for gg, part in zip(np.split(g, cut), np.split(ids, cut))}


class CutoffLists:
    """기록 기준 시각 하나에 대한 상품 순위·카테고리 목록 (요청마다 다시 만들지 않는다)."""

    def __init__(self, store: Any, at: float) -> None:
        n = len(store.item_ids)
        ids = np.arange(n + 1).clip(1)
        cats = store.cats
        self.l1, self.l2, self.l3 = cats[:, 0].astype(np.int64), cats[:, 1].astype(np.int64), cats[:, 2].astype(np.int64)
        self.g13 = self.l1 * 1000 + self.l3
        self.created = store.created
        self.price = store.numeric[:, 0]
        self.pop7 = store.popularity(ids, np.full(n + 1, at)); self.pop7[0] = 0
        self.pop1 = store.popularity(ids, np.full(n + 1, at), days=1); self.pop1[0] = 0
        self.pop_all = np.log1p(store.view_counts(at)).astype(np.float32); self.pop_all[0] = 0
        self.pop_rank_l3, size = group_rank(self.pop7, self.l3, True)
        self.pop_in_l3 = (self.pop_rank_l3 / np.maximum(size - 1, 1)).astype(np.float32)
        self.price_rank_l3, _ = group_rank(self.price, self.l3, False)
        self.pop_rank_g13, _ = group_rank(self.pop_all, self.g13, True)
        self.price_rank_g13, _ = group_rank(self.price, self.g13, False)
        self.pop_rank_l1, _ = group_rank(self.pop_all, self.l1, True)
        self.grank = np.empty(n + 1, np.int64)
        self.grank[np.argsort(-self.pop7, kind="stable")] = np.arange(n + 1)
        self.l3_pop, self.l3_cheap = _sorted_lists(self.l3, -self.pop7), _sorted_lists(self.l3, self.price)
        self.l2_pop, self.l1_pop = _sorted_lists(self.l2, -self.pop7), _sorted_lists(self.l1, -self.pop7)
        self.g13_pop, self.g13_cheap = _sorted_lists(self.g13, -self.pop_all), _sorted_lists(self.g13, self.price)
        self.created_key_l3 = np.sort(self.l3[1:] * _GROUP + self.created[1:])
        self.created_key_g13 = np.sort(self.g13[1:] * _GROUP + self.created[1:])

    def new_rank(self, keys: np.ndarray, groups: np.ndarray, ids: np.ndarray, at: float) -> np.ndarray:
        """그룹 안 신상품순 위치 = 요청 시각까지 등록된 같은 그룹 상품 중 이 상품보다 나중에 등록된 수."""
        base = groups * _GROUP
        return (np.searchsorted(keys, base + at, side="right")
                - np.searchsorted(keys, base + self.created[ids], side="right")).astype(np.float32)


@dataclass
class GroupStats:
    """고객 조회의 범주(그룹)별 요약. codes 오름차순, 나머지는 같은 순서."""
    codes: np.ndarray
    days_since: np.ndarray   # 마지막 조회 뒤 경과일 (요청 시각 기준)
    visit_days: np.ndarray   # 조회한 날 수
    share_rw: np.ndarray     # 시간 감쇠(반감 HALF_LIFE) 비율
    last_visit: np.ndarray   # 마지막 방문 안 비율
    rank: np.ndarray         # share_rw 순위 (0 = 가장 큼)

    @classmethod
    def of(cls, codes: np.ndarray, times: np.ndarray, at: float) -> GroupStats:
        if not len(codes):
            empty = np.array([], np.float32)
            return cls(np.array([], np.int64), empty, empty, empty, empty, np.array([], np.int64))
        u, inv = np.unique(codes, return_inverse=True)
        last = np.full(len(u), -np.inf)
        np.maximum.at(last, inv, times)
        day = np.floor(times / 86400).astype(np.int64)
        pairs = np.unique(inv * 1_000_000 + (day - day.min()))  # (그룹, 날) 쌍
        visit = np.bincount(pairs // 1_000_000, minlength=len(u))
        w = 0.5 ** (np.maximum(at - times, 0) / HALF_LIFE)
        share = np.bincount(inv, weights=w, minlength=len(u)) / w.sum()
        recent = times >= times.max() - LAST_VISIT
        last_visit = np.bincount(inv[recent], minlength=len(u)) / recent.sum()
        rank = np.empty(len(u), np.int64)
        rank[np.argsort(-share, kind="stable")] = np.arange(len(u))
        return cls(u, ((at - last) / 86400).astype(np.float32), visit.astype(np.float32),
                   share.astype(np.float32), last_visit.astype(np.float32), rank)

    def pick(self, values: np.ndarray, codes: np.ndarray, default: float) -> np.ndarray:
        """후보 범주 코드별 값. 고객이 본 적 없는 범주는 default."""
        out = np.full(len(codes), default, np.float32)
        if len(self.codes):
            pos = np.minimum(np.searchsorted(self.codes, codes), len(self.codes) - 1)
            hit = self.codes[pos] == codes
            out[hit] = values[pos[hit]]
        return out

    def log_days_since(self, codes: np.ndarray) -> np.ndarray:
        """log1p(마지막 조회 뒤 경과일). 본 적 없는 범주는 -1."""
        d = self.pick(self.days_since, codes, -1.0)
        return np.where(d >= 0, np.log1p(np.maximum(d, 0)), -1.0).astype(np.float32)


@dataclass
class UserInfo:
    history: np.ndarray            # 상품 번호, 오래된 것 → 최근 (state의 이력, 실시간 조회 포함)
    top_l3: list[int]
    top_l2: list[int]
    top_g13: list[int]
    main_l1: int | None
    shares: dict[int, np.ndarray]  # store.cats 열 번호 → 범주 코드별 이력 비율
    recent_l3: np.ndarray          # 최근 5개 이력의 소분류 비율
    view_ids: np.ndarray           # 기준 시각까지 조회한 상품 (정렬)
    view_cnt: np.ndarray           # 그 조회 횟수
    cart_open: np.ndarray          # 담았지만 아직 안 산 상품
    new_aff: float                 # 이력 조회 중 등록 7일 이내 상품 비율 (-1 = 이력 없음)
    price_mean: float | None
    g13_stats: GroupStats | None = None   # 카테고리 의도 특징 (조회 기록 전체, 시각 포함)
    l3_stats: GroupStats | None = None
    cart_l3: np.ndarray | None = None     # 소분류별 장바구니 수
    buy_l3: np.ndarray | None = None      # 소분류별 구매 수
    days_since_last: float = -1.0         # 마지막 조회 뒤 경과일 (-1 = 조회 없음)


def _top(codes: np.ndarray, k: int) -> list[int]:
    if not len(codes):
        return []
    g, c = np.unique(codes, return_counts=True)
    return [int(x) for x in g[np.argsort(-c, kind="stable")][:k]]


class FinalRankContext:
    def __init__(self, store: Any, item_vectors: np.ndarray, config: Mapping[str, Any] | None = None) -> None:
        self.store = store
        self.vectors = np.asarray(item_vectors, np.float32)
        cfg = dict(config or {})
        self.sources = {**DEFAULT_SOURCES, **(cfg.get("sources") or {})}
        self.pool_size = int(cfg.get("pool_size", 1000))
        self.n_codes = [int(store.cats[:, c].max()) + 1 for c in range(store.cats.shape[1])]
        self._lists: dict[int, CutoffLists] = {}

    # ---- 기준 시각별 목록 --------------------------------------------------
    def lists(self, at: float) -> CutoffLists:
        day = int(np.floor(self.store.frozen_time(np.array([float(at)]))[0] / 86400))
        if day not in self._lists:
            if len(self._lists) >= 2:
                self._lists.pop(next(iter(self._lists)))
            self._lists[day] = CutoffLists(self.store, at)
        return self._lists[day]

    # ---- 고객 ---------------------------------------------------------------
    def user_info(self, user_id: str, at: float, state: Sequence[Any]) -> UserInfo:
        store, cats = self.store, self.store.cats
        h = np.asarray(state[0])
        h = h[h > 0]
        cut = float(store.frozen_time(np.array([float(at)]))[0])
        empty = np.array([], dtype=np.int64)
        tv, iv = store.history.get((user_id, "view"), (np.array([]), empty))
        e = np.searchsorted(tv, cut, side="left")
        views_t, views = tv[:e], iv[:e]
        recent_t, recent = views_t[-50:], views[-50:]
        new_aff = float(np.mean(recent_t - store.created[recent] < WEEK)) if len(recent) else -1.0
        view_ids, view_cnt = np.unique(views, return_counts=True)
        tc, ic = store.history.get((user_id, "cart"), (np.array([]), empty))
        tp, ip = store.history.get((user_id, "purchase"), (np.array([]), empty))
        carted = ic[:np.searchsorted(tc, cut, side="left")]
        bought = ip[:np.searchsorted(tp, cut, side="left")]
        shares = {c: (np.bincount(cats[h, c], minlength=self.n_codes[c]) / len(h)).astype(np.float32) if len(h)
                  else np.zeros(self.n_codes[c], np.float32) for c in range(5)}
        last = h[-5:]
        recent_l3 = (np.bincount(cats[last, 2], minlength=self.n_codes[2]) / max(len(last), 1)).astype(np.float32)
        g13 = cats[h, 0].astype(np.int64) * 1000 + cats[h, 2]
        view_g13 = cats[views, 0].astype(np.int64) * 1000 + cats[views, 2]
        return UserInfo(history=h, top_l3=_top(cats[h, 2], self.sources["top_l3"]),
                        top_l2=_top(cats[h, 1], self.sources["top_l2"]), top_g13=_top(g13, self.sources["top_g13"]),
                        main_l1=int(np.bincount(cats[h, 0]).argmax()) if len(h) else None, shares=shares,
                        recent_l3=recent_l3, view_ids=view_ids, view_cnt=view_cnt,
                        cart_open=np.setdiff1d(np.unique(carted), np.unique(bought)), new_aff=new_aff,
                        price_mean=float(self.store.numeric[h, 0].mean()) if len(h) else None,
                        g13_stats=GroupStats.of(view_g13, views_t, float(at)),
                        l3_stats=GroupStats.of(cats[views, 2].astype(np.int64), views_t, float(at)),
                        cart_l3=np.bincount(cats[carted, 2], minlength=self.n_codes[2]).astype(np.float32),
                        buy_l3=np.bincount(cats[bought, 2], minlength=self.n_codes[2]).astype(np.float32),
                        days_since_last=float((float(at) - views_t[-1]) / 86400) if len(views_t) else -1.0)

    # ---- 후보 풀 -------------------------------------------------------------
    def pool(self, info: UserInfo, at: float, q: np.ndarray, stage1: np.ndarray, allowed: np.ndarray | None = None,
             extra: Sequence[int] = ()) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        L, s = self.lists(at), self.sources
        created = L.created

        def avail(ids: np.ndarray) -> np.ndarray:
            ok = created[ids] <= at
            if allowed is not None:
                ok &= allowed[ids]
            return ids[ok]

        def fresh(ids: np.ndarray, days: int) -> np.ndarray:
            age = at - created[ids]
            return avail(ids[(age >= 0) & (age < days * 86400)])

        def take(lists: dict[int, np.ndarray], groups: Sequence[int], k: int) -> np.ndarray:
            parts = [avail(lists[g])[:k] for g in groups if g in lists]
            return np.concatenate(parts) if parts else np.array([], np.int64)

        stage1 = np.asarray(stage1, np.int64)
        src = {"tt300": stage1[:300]}
        src["catpop"] = take(L.l3_pop, info.top_l3, s["catpop"])
        src["catnew"] = np.concatenate([fresh(L.l3_pop[g], s["new_days"]) for g in info.top_l3 if g in L.l3_pop] or [[]]).astype(np.int64)
        src["segpop"] = avail(L.l1_pop[info.main_l1])[:s["segpop"]] if info.main_l1 in L.l1_pop else np.array([], np.int64)
        tail = np.concatenate([avail(L.l3_pop[g]) for g in info.top_l3[:3] if g in L.l3_pop] or [[]]).astype(np.int64)
        tail = tail[L.grank[tail] >= s["taste_tail_min_rank"]]
        src["taste_tail"] = tail[np.argsort(-(self.vectors[tail - 1] @ np.ravel(q)), kind="stable")[:s["taste_tail"]]] if len(tail) else tail
        src["pricelow"] = take(L.l3_cheap, info.top_l3[:3], s["pricelow"])
        src["l2pop"] = take(L.l2_pop, info.top_l2[:3], s["l2pop"])
        src["g13pop"] = take(L.g13_pop, info.top_g13, s["g13pop"])
        src["g13cheap"] = take(L.g13_cheap, info.top_g13[:3], s["g13cheap"])
        src["g13new"] = np.concatenate([fresh(L.g13_pop[g], s["g13_new_days"]) for g in info.top_g13 if g in L.g13_pop] or [[]]).astype(np.int64)
        ids = np.unique(np.concatenate([stage1[:self.pool_size], *src.values(), np.asarray(extra, np.int64)]).astype(np.int64))
        return ids[ids > 0], src

    # ---- 특징 ---------------------------------------------------------------
    def features(self, ids: np.ndarray, src: Mapping[str, np.ndarray], info: UserInfo, at: float,
                 q: np.ndarray, dfm: np.ndarray) -> np.ndarray:
        L, cats = self.lists(at), self.store.cats
        n = len(ids)
        tt = self.vectors[ids - 1] @ np.ravel(q)
        age = np.maximum(at - L.created[ids], 0)
        view_cnt = np.zeros(n, np.float32)
        if len(info.view_ids):
            pos = np.minimum(np.searchsorted(info.view_ids, ids), len(info.view_ids) - 1)
            hit = info.view_ids[pos] == ids
            view_cnt[hit] = info.view_cnt[pos[hit]]
        F = {
            "dfm": dfm, "tt": tt, "tt_rank": np.argsort(np.argsort(-tt, kind="stable"), kind="stable") / max(n - 1, 1),
            "pop7": L.pop7[ids], "pop1": L.pop1[ids], "pop_in_l3": L.pop_in_l3[ids],
            "pop_rank_l3": np.minimum(L.pop_rank_l3[ids], 500),
            "new_rank_l3": np.minimum(L.new_rank(L.created_key_l3, L.l3[ids], ids, at), 500),
            "price_rank_l3": np.minimum(L.price_rank_l3[ids], 500),
            "age_days": np.log1p(age / 86400), "is_new": (age < WEEK).astype(np.float32),
            "in_hist": np.isin(ids, info.history).astype(np.float32), "new_aff": np.full(n, info.new_aff, np.float32),
            "view_cnt": view_cnt, "in_cart": np.isin(ids, info.cart_open).astype(np.float32),
            "recent_l3": info.recent_l3[cats[ids, 2]],
            "pop_all": L.pop_all[ids], "pop_rank_g13": np.minimum(L.pop_rank_g13[ids], 500),
            "price_rank_g13": np.minimum(L.price_rank_g13[ids], 500), "pop_rank_l1": np.minimum(L.pop_rank_l1[ids], 5000),
            "new_rank_g13": np.minimum(L.new_rank(L.created_key_g13, L.g13[ids], ids, at), 500),
        }
        for col, name in ((0, "s_l1"), (1, "s_l2"), (2, "s_l3"), (3, "s_brand"), (4, "s_color")):
            F[name] = info.shares[col][cats[ids, col]]
        F["price_gap"] = np.abs(L.price[ids] - info.price_mean) if info.price_mean is not None else np.full(n, -1.0)
        F["hist_len"] = np.full(n, len(info.history), np.float32)
        for s in SOURCES:
            F[f"src_{s}"] = np.isin(ids, src[s]).astype(np.float32)
        g, l3 = info.g13_stats, info.l3_stats
        if g is None or l3 is None:  # 이전 방식으로 만든 UserInfo
            g = l3 = GroupStats.of(np.array([], np.int64), np.array([]), at)
        g13_ids, l3_ids = L.g13[ids], cats[ids, 2].astype(np.int64)
        F["g13_days_since"] = g.log_days_since(g13_ids)
        F["l3_days_since"] = l3.log_days_since(l3_ids)
        F["g13_visit_days"] = np.log1p(g.pick(g.visit_days, g13_ids, 0.0))
        F["g13_share_rw"] = g.pick(g.share_rw, g13_ids, 0.0)
        F["l3_share_rw"] = l3.pick(l3.share_rw, l3_ids, 0.0)
        F["g13_last_visit"] = g.pick(g.last_visit, g13_ids, 0.0)
        F["g13_rw_rank"] = np.minimum(g.pick(g.rank.astype(np.float32), g13_ids, -1.0), 30)
        zeros = np.zeros(self.n_codes[2], np.float32)
        F["l3_carted"] = np.log1p((info.cart_l3 if info.cart_l3 is not None else zeros)[l3_ids])
        F["l3_bought"] = np.log1p((info.buy_l3 if info.buy_l3 is not None else zeros)[l3_ids])
        F["days_since_last"] = np.full(n, np.log1p(info.days_since_last) if info.days_since_last >= 0 else -1.0, np.float32)
        return np.column_stack([np.asarray(F[k], dtype=np.float32) for k in FEATURES])
