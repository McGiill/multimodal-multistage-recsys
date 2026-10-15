"""방문(세션) 단위 행동 생성: 화면에 목록을 보여주고 → 반응 함수(response.py)로 반응을 정한다.

한 방문의 흐름 (configs/simulator.yaml → session_flow)
    방문 시작 → 지난번 장바구니 처리(일부 구매)
    → 첫 화면 선택(페르소나별 search / browse / home 비율)
    → 목록 표시(노출 기록) → 반응: 클릭(view) → 담기(cart) → 구매(purchase)
    → 다음 화면(같은 목록 / 검색 / 둘러보기 / 홈) 또는 종료

화면별 목록
- search : 찾는 상품의 속성으로 검색어를 만들고 검색 결과표(실제 검색 엔진 결과)에서 상위 N개
- browse : 찾는 상품의 (고객층, 소분류) 카테고리 목록, 1~3페이지. 정렬은 인기순 / 신상품순 / 낮은 가격순 중
           페르소나별 비율로 고른다 (가성비추구는 낮은 가격순, 트렌드세터는 신상품순을 많이)
- home   : 기본은 고객층 인기 상품 + 신상품 일부. 추천 API가 준비되면 home_provider로 교체한다
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.simulator.catalog import World
from src.simulator.personas import PersonaParams
from src.simulator.response import ResponseModel, Shopper, Visit, base_scores
from src.simulator.users import Population

EVENT_TYPES = ("search", "view", "cart", "purchase")
SOURCES = ("search", "browse", "home", "cart")
DAY = 86400

HomeProvider = Callable[[Shopper, Visit, np.random.Generator], np.ndarray]


@dataclass
class Log:
    """생성 결과 (정수 코드). run.py가 문자열·ID로 바꿔 parquet으로 저장한다."""

    ev: dict[str, list] = field(default_factory=lambda: {
        k: [] for k in ("user", "session", "time", "type", "item", "query", "source", "impression")
    })
    imp: dict[str, list] = field(default_factory=lambda: {
        k: [] for k in ("user", "session", "time", "source", "context", "items", "clicked")
    })
    n_sessions: int = 0

    def event(self, user: int, session: int, t: float, etype: str, item: int = -1, query: int = -1,
              source: str = "search", impression: int = -1) -> None:
        e = self.ev
        e["user"].append(user)
        e["session"].append(session)
        e["time"].append(t)
        e["type"].append(EVENT_TYPES.index(etype))
        e["item"].append(item)
        e["query"].append(query)
        e["source"].append(SOURCES.index(source))
        e["impression"].append(impression)

    def impression(self, user: int, session: int, t: float, source: str, context: str, items: np.ndarray) -> int:
        i = self.imp
        i["user"].append(user)
        i["session"].append(session)
        i["time"].append(t)
        i["source"].append(SOURCES.index(source))
        i["context"].append(context)
        i["items"].append(np.asarray(items, dtype=np.int32))
        i["clicked"].append([])
        return len(i["user"]) - 1


class Simulator:
    def __init__(self, world: World, cfg: dict[str, Any], personas: dict[str, PersonaParams],
                 home_provider: HomeProvider | None = None) -> None:
        self.world, self.cfg, self.personas = world, cfg, personas
        self.model = ResponseModel(world, cfg)
        self.home_provider = home_provider or self.popular_home
        lists = cfg["lists"]
        self.length = lists["length"]
        self.pages = np.array(lists["browse_pages"], dtype=float)
        self.templates = list(cfg["search"]["query_templates"])
        self.template_p = np.array([cfg["search"]["query_templates"][t] for t in self.templates], dtype=float)
        self.days = cfg["period"]["days"]
        t = cfg["time"]
        self.hour_p = np.array(t["hour_weights"], dtype=float) / sum(t["hour_weights"])
        start_weekday = int((np.datetime64(cfg["period"]["start_date"], "D").astype(int) + 3) % 7)  # 월=0
        self.day_w = np.array(t["weekday_weights"], dtype=float)[(start_weekday + np.arange(self.days)) % 7]
        self.mean_sec = t["mean_seconds"]
        self.gap_sec = t["session_gap_min"] * 60

    # ------------------------------------------------------------------ 고객 준비
    def make_shopper(self, pop: Population, i: int, rng: np.random.Generator) -> Shopper:
        world, u, h = self.world, self.cfg["utility"], pop.hidden.iloc[i]
        l2 = np.array([world.l2_names.index(c) for c in h["preferred_l2"] if c in world.l2_names], dtype=np.int32)
        brands = np.array(
            [world.brand_names.index(b) for b in h["preferred_brands"] if b in world.brand_names], dtype=np.int32
        )
        segment = world.l1_names.index(h["main_segment"]) if h["main_segment"] in world.l1_names else -1
        base = base_scores(world, u, pop.taste[i], l2, segment, brands, float(h["brand_loyalty"]),
                           float(h["price_sensitivity"]), rng)
        pool = np.argpartition(-base, u["target_pool"])[: u["target_pool"]].astype(np.int32)
        return Shopper(index=i, persona=self.personas[h["persona"]], segment=segment,
                       new_affinity=float(h["new_item_affinity"]), base=base, pool=np.sort(pool))

    # ------------------------------------------------------------------ 화면별 목록
    def search_list(self, target: int, day: int, rng: np.random.Generator) -> tuple[np.ndarray, int]:
        template = self.templates[int(rng.choice(len(self.templates), p=self.template_p))]
        q = int(self.world.item_query[template][target])
        rows = self.world.search_results[q]
        return self.world.registered(rows[rows >= 0], day)[: self.length], q

    def browse_list(self, shopper: Shopper, target: int, day: int, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        w, sorts = self.world, shopper.persona.browse_sort
        sort = str(rng.choice(list(sorts), p=list(sorts.values())))
        key = (int(w.l1[target]), int(w.l3[target]))
        items = w.registered(w.browse[sort][key], day)
        page = int(rng.choice(len(self.pages), p=self.pages / self.pages.sum()))
        if page * self.length >= len(items):
            page = 0
        context = f"{w.l1_names[key[0]]}/{w.l3_names[key[1]]} ({sort})"
        return items[page * self.length: (page + 1) * self.length], context

    def popular_home(self, shopper: Shopper, visit: Visit, rng: np.random.Generator) -> np.ndarray:
        """추천 API 연결 전 기본 홈: 고객층 인기 상품 + 최근 등록 신상품 일부."""
        w, lists = self.world, self.cfg["lists"]
        seg_items = w.by_segment.get(shopper.segment, w.by_segment[next(iter(w.by_segment))])
        pool = w.registered(seg_items[: lists["home_pool"]], visit.day)
        n_new = round(self.length * lists["home_new_share"])
        created = w.created_day[seg_items]
        fresh = seg_items[(created > visit.day - lists["new_days"]) & (created <= visit.day)][:n_new]
        k = min(self.length - len(fresh), len(pool))
        weight = np.exp(lists["home_pop_power"] * w.popularity[pool])
        picked = rng.choice(pool, size=k, replace=False, p=weight / weight.sum())
        items = list(picked[np.argsort(-w.popularity[picked], kind="stable")])
        for j, item in enumerate(fresh):  # 신상품은 고르게 끼워 넣음
            items.insert(min(len(items), 2 + j * 5), item)
        return np.array(items[: self.length], dtype=np.int32)

    # ------------------------------------------------------------------ 시간
    def _gap(self, kind: str, rng: np.random.Generator) -> float:
        m = self.mean_sec[kind]
        return float(rng.lognormal(np.log(m) - 0.32, 0.8))

    def _session_starts(self, signup_day: int, expected: float, rng: np.random.Generator) -> np.ndarray:
        active = np.arange(max(0, signup_day), self.days)
        n = rng.poisson(expected)
        if n == 0 or len(active) == 0:
            return np.zeros(0)
        p = self.day_w[active] / self.day_w[active].sum()
        days = rng.choice(active, size=n, p=p)
        hours = rng.choice(24, size=n, p=self.hour_p)
        return np.sort(days * DAY + hours * 3600 + rng.uniform(0, 3600, size=n))

    # ------------------------------------------------------------------ 고객 한 명
    def simulate_user(self, pop: Population, i: int, log: Log, rng: np.random.Generator) -> None:
        h = pop.hidden.iloc[i]
        shopper = self.make_shopper(pop, i, rng)
        persona, flow_max = shopper.persona, self.cfg["session_flow"]["max_screens"]
        screens = list(persona.screens)
        screen_p = np.array([persona.screens[s] for s in screens])
        prev_end = -np.inf

        for number, planned in enumerate(self._session_starts(int(h["signup_day"]), float(h["expected_sessions"]), rng)):
            t = max(planned, prev_end + self.gap_sec)
            day = int(t // DAY)
            if day >= self.days:
                break
            session = log.n_sessions
            log.n_sessions += 1
            visit = Visit(day=day, number=number)

            for item in self.model.checkout_cart(shopper, visit, rng):
                t += self._gap("purchase", rng)
                log.event(i, session, t, "purchase", item=item, source="cart")

            screen = screens[int(rng.choice(len(screens), p=screen_p))]
            for _ in range(flow_max):
                if screen == "exit":
                    break
                query = -1
                if screen == "search":
                    items, query = self.search_list(self.model.pick_target(shopper, visit, rng), day, rng)
                    context = str(self.world.search_vocab[query])
                elif screen == "browse":
                    items, context = self.browse_list(shopper, self.model.pick_target(shopper, visit, rng), day, rng)
                else:
                    items, context = self.home_provider(shopper, visit, rng), self.world.l1_names[shopper.segment]
                t += self._gap("list", rng)
                if screen == "search":
                    log.event(i, session, t, "search", query=query, source="search")
                if len(items) == 0:
                    screen = screens[int(rng.choice(len(screens), p=screen_p))]
                    continue
                imp = log.impression(i, session, t, screen, context, items)
                reaction = self.model.react(shopper, visit, items, rng)
                for etype, item in reaction.actions:
                    t += self._gap(etype, rng)
                    log.event(i, session, t, etype, item=item, source=screen, impression=imp)
                log.imp["clicked"][imp] = reaction.clicked
                screen = reaction.next_screen
            prev_end = t

    def run(self, pop: Population, progress: bool = True) -> Log:
        log = Log()
        n = len(pop.hidden)
        for i in range(n):
            self.simulate_user(pop, i, log, np.random.default_rng([self.cfg["seed"], i]))
            if progress and (i + 1) % 1000 == 0:
                print(f"  고객 {i + 1:,}/{n:,}, 이벤트 {len(log.ev['user']):,}", flush=True)
        return log
