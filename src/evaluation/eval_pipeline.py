"""최종 추천 평가: 전체 파이프라인(Stage 1 → 세션 결합 → Stage 2 → Stage 3)의 test HitRate@50 / NDCG@50 / Coverage.

    python -m src.evaluation.eval_pipeline                                  # 단계별 전부 + 베이스라인
    python -m src.evaluation.eval_pipeline --strategies multistage,two_tower_only,popularity
    python -m src.evaluation.eval_pipeline --limit 2000                     # 빠른 확인용 (seed 고정 표본)

단계를 하나씩 더하며 같은 조건으로 비교 (어느 단계에서 성능이 바뀌는지 확인):
    two_tower_only        Stage 1 후보를 User Tower 유사도순 그대로 (명세 베이스라인)
    two_tower_ranking     + Stage 2: 같은 후보를 DeepFM 점수순으로 (ranking.objective: ctr 클릭 / cvr 구매 확률)
    multistage_no_session + Stage 3: 재정렬 (다양성·cold-start·신규 상품·MAB 탐색 칸)
    multistage            + 세션: 같은 방문에서 앞서 본 상품 (최근 조회 이력 + 세션 후보 결합)
    popularity            인기순 (명세 베이스라인)
전략마다 Stage 3(MAB 통계)를 새로 만들어서 앞 전략의 탐색 기록이 섞이지 않게 한다.

정답: test 조회 1건마다 "그 시점에 다음으로 본 상품" 1개 (팀 Stage 1 Recall@300과 같은 76,242건,
      학습 묶음의 examples.npz split == 2). 예측 시각 = 그 조회 시각, 사용자 이력은 모델 규칙(frozen_split)대로.
세션: 같은 방문에서 그 조회보다 먼저 본 상품들 (test_events.parquet의 session_id). 세션이 없으면 Stage 1 후보 그대로.
지표 (src/evaluation/metrics.py): HitRate@k, NDCG@k (정답 1개 → 1/log2(순위+1)), Coverage = 추천된 고유 상품 / 전체 상품.
k = configs/recommend.yaml 의 final_eval_k (명세 50). 목표: HitRate@50 ≥ 0.20, NDCG@50 ≥ 0.08, Coverage ≥ 0.20.
MAB 탐색 칸은 서빙과 같이 진열 즉시 노출만 센다 (평가 중 클릭 정보는 넣지 않음).

결과: artifacts/metrics/recommend_final.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import ROOT, artifact_dir, data_dir, load_config
from src.common.schemas import RecommendationItem, RecommendResponse
from src.evaluation.metrics import coverage, hit_rate_at_k, ndcg_at_k
from src.recommendation.pipeline import MultiStageStrategy
from src.recommendation.ranking.features import seconds, stamp
from src.recommendation.reranking.stage2 import get_rerank_service
from src.recommendation.strategies.baselines import (
    PopularityStrategy,
    TwoTowerOnlyStrategy,
    _response,
)
from src.serving.model_loader import get_recommendation_models

TARGETS = {"hit_rate": 0.20, "ndcg": 0.08, "coverage": 0.20}
BASELINES = ("two_tower_only", "popularity")  # 명세 베이스라인 → 결과에 lift_vs_<이름>
STAGE_CHAIN = ("two_tower_only", "two_tower_ranking", "multistage_no_session", "multistage")


class TwoTowerRankingStrategy:
    """평가 전용: Stage 1 후보 → DeepFM 점수순 (세션·재정렬 없음). 랭킹 단계만의 효과를 보려고 둔다."""

    def recommend(self, user_id: str, top_n: int, at: datetime | None = None) -> RecommendResponse:
        begin = time.perf_counter()
        models = get_recommendation_models()
        now = stamp(at or datetime.now(timezone.utc))
        ids, _, state = models.retriever.retrieve(user_id, models.cfg["candidate"]["n_candidates"], now)
        scores = models.score(ids, state, now)
        order = np.argsort(-scores, kind="stable")[:top_n]
        items = [RecommendationItem(product_id=models.store.item_ids[ids[i] - 1], score=float(scores[i]),
                                    reason=models.objective) for i in order]
        return _response(user_id, items, begin)


def session_prefixes(users: np.ndarray, times: np.ndarray) -> list[list[str]]:
    """(사용자, 조회 시각)마다 같은 방문에서 그보다 먼저 본 상품 (오래된 것 → 최근 순)."""
    ev = pd.read_parquet(data_dir() / "splits" / "test_events.parquet",
                         columns=["user_id", "session_id", "timestamp", "event_type", "product_id"])
    ev = ev[ev["event_type"] == "view"].copy()
    ev["t"] = seconds(ev["timestamp"])  # examples.npz의 time과 같은 변환 (UTC unix 초)
    ev = ev.sort_values(["session_id", "t"], kind="stable")
    session_of = dict(zip(zip(ev["user_id"], ev["t"]), ev["session_id"]))
    views = {s: (g["t"].to_numpy(), g["product_id"].tolist()) for s, g in ev.groupby("session_id", sort=False)}
    out = []
    for u, t in zip(users, times):
        s = session_of.get((u, float(t)))
        if s is None:
            out.append([])
            continue
        ts, items = views[s]
        out.append(items[: int(np.searchsorted(ts, t, side="left"))])
    return out


@dataclass
class Queries:
    """평가할 test 조회: 사용자, 예측 시각(unix 초), 정답 상품, 세션 클릭."""

    users: np.ndarray
    times: np.ndarray
    targets: list[set[str]]
    sessions: list[list[str]]
    n_items: int
    k: int


def load_queries(limit: int | None = None) -> Queries:
    """examples.npz의 test 조회 (limit이 있으면 seed 고정 표본)."""
    cfg = load_config("recommend")
    models = get_recommendation_models()
    folder = ROOT / cfg["paths"]["model_dir"] / cfg["version"]
    data = np.load(folder / "examples.npz", allow_pickle=False)
    ix = np.flatnonzero(data["split"] == 2)
    if limit:
        ix = np.sort(np.random.default_rng(cfg["seed"]).choice(ix, min(limit, len(ix)), replace=False))
    users, times = data["user"][ix], data["time"][ix]
    return Queries(users=users, times=times,
                   targets=[{models.store.item_ids[t - 1]} for t in data["target"][ix]],
                   sessions=session_prefixes(users, times),
                   n_items=len(models.store.item_ids), k=cfg["final_eval_k"])


def evaluate_strategy(strategy: Any, q: Queries, use_session: bool, name: str = "") -> dict[str, Any]:
    """조회마다 strategy.recommend(top_n=k) → HitRate@k / NDCG@k / Coverage / p95."""
    ranked, latency, t0 = [], [], time.perf_counter()
    for j, (u, t) in enumerate(zip(q.users, q.times)):
        at = datetime.fromtimestamp(float(t), tz=timezone.utc)
        kw = {"session_clicks": q.sessions[j]} if use_session else {}
        s = time.perf_counter()
        resp = strategy.recommend(str(u), q.k, at=at, **kw)
        latency.append((time.perf_counter() - s) * 1000)
        ranked.append([r.product_id for r in resp.recommendations])
        if (j + 1) % 10000 == 0:
            print(f"  [{name}] {j + 1:,}/{len(q.users):,} ({time.perf_counter() - t0:.0f}s)", flush=True)
    m = {
        "hit_rate": hit_rate_at_k(ranked, q.targets, q.k),
        "ndcg": ndcg_at_k(ranked, q.targets, q.k),
        "coverage": coverage(ranked, q.n_items),
        "latency_p95_ms": float(np.percentile(latency, 95)),
    }
    m["passed"] = {key: m[key] >= v for key, v in TARGETS.items()}
    print(f"[eval] {name:22s} HitRate@{q.k} {m['hit_rate']:.4f}  NDCG@{q.k} {m['ndcg']:.4f}  "
          f"Coverage {m['coverage']:.4f}  p95 {m['latency_p95_ms']:.1f}ms", flush=True)
    return m


def run(strategies: list[str], limit: int | None = None) -> dict[str, Any]:
    q = load_queries(limit)
    result: dict[str, Any] = {
        "k": q.k, "n_queries": len(q.users), "with_session": sum(bool(s) for s in q.sessions),
        "target": "next viewed product (test split, examples.npz)", "targets": TARGETS,
        "model_version": load_config("recommend")["version"],
        "session": {k: load_config("recommend")["session"].get(k) for k in ("live_history", "blend_candidates")},
        "stage_chain": [name for name in STAGE_CHAIN if name in strategies],
        "measured_at": datetime.now().isoformat(timespec="seconds"),
    }
    for name in strategies:
        get_rerank_service.cache_clear()  # 전략마다 MAB 통계를 새로 시작
        if name == "popularity":
            strategy, use_session = PopularityStrategy(), False
        elif name == "two_tower_only":
            strategy, use_session = TwoTowerOnlyStrategy(), False
        elif name == "two_tower_ranking":
            strategy, use_session = TwoTowerRankingStrategy(), False
        elif name in ("multistage", "multistage_no_session"):
            strategy, use_session = MultiStageStrategy(), name == "multistage"
        else:
            raise ValueError(f"지원하지 않는 전략: {name}")
        result[name] = evaluate_strategy(strategy, q, use_session, name)

    for baseline in BASELINES:
        if baseline in result:
            base = result[baseline]
            for name in strategies:
                if name not in BASELINES:
                    result[name][f"lift_vs_{baseline}"] = {
                        key: (result[name][key] / base[key] - 1) if base[key] else None
                        for key in ("hit_rate", "ndcg", "coverage")
                    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategies", default="multistage,multistage_no_session,two_tower_ranking,two_tower_only,popularity")
    parser.add_argument("--limit", type=int, default=None, help="평가할 조회 수 (기본: 전부)")
    args = parser.parse_args()
    result = run(args.strategies.split(","), args.limit)
    out = artifact_dir() / "metrics" / ("recommend_final.json" if not args.limit else "recommend_final_sample.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] 저장 → {out}")


if __name__ == "__main__":
    main()
