"""최종 추천 평가: 전체 파이프라인(Stage 1 → 세션 결합 → Stage 2 → Stage 3)의 test HitRate@50 / NDCG@50 / Coverage.

    python -m src.evaluation.eval_pipeline                                  # 사용자 단위 (주 지표, 명세 판정)
    python -m src.evaluation.eval_pipeline --limit 2000                     # 빠른 확인용 (seed 고정 고객 표본)
    python -m src.evaluation.eval_pipeline --target both                    # 보조: 조회 1건 단위 조회·구매 평가
    python -m src.evaluation.eval_pipeline --target view --strategies multistage,two_tower_only,popularity

사용자 단위 (run_users, 2026-11-18 정의): 고객마다 test 첫 이벤트 시각에 목록 1개, 정답 = test 기간에
조회·장바구니·구매한 상품 전부. 아래 '조회 정답'은 세션 효과 확인용 보조 지표다.

단계를 하나씩 더하며 같은 조건으로 비교 (어느 단계에서 성능이 바뀌는지 확인):
    two_tower_only        Stage 1 후보를 User Tower 유사도순 그대로 (명세 베이스라인)
    two_tower_ranking     + Stage 2: 같은 후보를 DeepFM 점수순으로 (ranking.objective: ctr 클릭 / cvr 구매 확률)
    multistage_no_session + Stage 3: 재정렬 (다양성·cold-start·신규 상품·MAB 탐색 칸)
    multistage            + 세션: 같은 방문에서 앞서 본 상품 (최근 조회 이력 + 세션 후보 결합)
    popularity            인기순 (명세 베이스라인)
전략마다 Stage 3(MAB 통계)를 새로 만들어서 앞 전략의 탐색 기록이 섞이지 않게 한다.

조회 정답: test 조회 1건마다 "그 시점에 다음으로 본 상품" 1개 (팀 Stage 1 Recall@300과 같은 쿼리,
      학습 묶음의 examples.npz split == 2). 예측 시각 = 그 조회 시각, 사용자 이력은 모델 규칙(frozen_split)대로.
구매 정답: 구매로 이어진 노출 시점에서 추천하고, 완전히 관측한 구매 창 안의 구매 상품을 맞힌다.
  구매 양성 노출의 순위 평가이며 전체 노출 구매율/AUC와는 다른 지표다.
세션: 같은 방문에서 그 조회보다 먼저 본 상품들 (test_events.parquet의 session_id). 세션이 없으면 Stage 1 후보 그대로.
지표 (src/evaluation/metrics.py): HitRate@k, NDCG@k (정답 1개 → 1/log2(순위+1)), Coverage = 추천된 고유 상품 / 전체 상품.
k = configs/recommend.yaml 의 final_eval_k (명세 50). 목표: HitRate@50 ≥ 0.20, NDCG@50 ≥ 0.08, Coverage ≥ 0.20.
MAB 탐색 칸은 서빙과 같이 진열 즉시 노출만 센다 (평가 중 클릭 정보는 넣지 않음).

결과: artifacts/metrics/recommend_final_<target>_<시각>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import ROOT, artifact_dir, data_dir, load_config
from src.common.schemas import RecommendationItem, RecommendResponse
from src.evaluation.metrics import coverage, hit_rate_at_k, ndcg_at_k, recall_at_k
from src.evaluation.user_targets import load_user_targets
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


def session_prefixes(users: np.ndarray, times: np.ndarray, session_ids=None,
                     *, root: Path | None = None) -> list[list[str]]:
    """(사용자, 조회 시각)마다 같은 방문에서 그보다 먼저 본 상품 (오래된 것 → 최근 순)."""
    root = data_dir() if root is None else Path(root)
    ev = pd.read_parquet(root / "splits" / "test_events.parquet",
                         columns=["user_id", "session_id", "timestamp", "event_type", "product_id"])
    ev = ev[ev["event_type"] == "view"].copy()
    ev["t"] = seconds(ev["timestamp"])  # examples.npz의 time과 같은 변환 (UTC unix 초)
    ev = ev.sort_values(["session_id", "t"], kind="stable")
    session_of = dict(zip(zip(ev["user_id"], ev["t"]), ev["session_id"]))
    views = {s: (g["t"].to_numpy(), g["product_id"].tolist())
             for s, g in ev.groupby(["user_id", "session_id"], sort=False)}
    out = []
    for j, (u, t) in enumerate(zip(users, times)):
        s = session_of.get((u, float(t))) if session_ids is None else session_ids[j]
        if s is None or (u, s) not in views:
            out.append([])
            continue
        ts, items = views[(u, s)]
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
    target: str = "view"
    model_version: str = ""
    definition: str = "next viewed product (test split, examples.npz)"
    seed: int = 42
    input_sha256: dict[str, str] = field(default_factory=dict)
    session_config: dict[str, Any] = field(default_factory=dict)


def input_fingerprints(paths) -> dict[str, str]:
    out = {}
    for path in paths:
        digest = hashlib.sha256()
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        out[str(path)] = digest.hexdigest()
    return out


def evaluation_data_root(models) -> Path:
    """모델이 학습한 데이터 스냅샷. 경로를 안 남긴 레거시 번들만 DATA_DIR을 사용한다."""
    source = getattr(models.store, "metadata", {}).get("source_data_dir")
    if source is None:
        return data_dir()
    root = Path(source)
    if not root.is_absolute():
        root = ROOT / root
    root = root.resolve()
    if not (root / "splits").is_dir():
        raise FileNotFoundError(f"Model evaluation source data is missing: {root}")
    return root


def load_queries(limit: int | None = None, target: str = "view") -> Queries:
    """examples.npz의 test 조회 (limit이 있으면 seed 고정 표본)."""
    models = get_recommendation_models()
    from src.recommendation.runtime_config import resolve_runtime_config

    cfg = models.cfg
    folder = getattr(models, "folder", ROOT / cfg["paths"]["model_dir"] / cfg["version"])
    if target == "purchase":
        return load_purchase_queries(models, limit)
    if target != "view":
        raise ValueError("target must be view or purchase")
    root = evaluation_data_root(models)
    data = np.load(folder / "examples.npz", allow_pickle=False)
    ix = np.flatnonzero(data["split"] == 2)
    if limit:
        ix = np.sort(np.random.default_rng(cfg["seed"]).choice(ix, min(limit, len(ix)), replace=False))
    users, times = data["user"][ix], data["time"][ix]
    return Queries(users=users, times=times,
                   targets=[{models.store.item_ids[t - 1]} for t in data["target"][ix]],
                   sessions=session_prefixes(users, times, root=root),
                   n_items=len(models.store.item_ids), k=cfg["final_eval_k"], model_version=folder.name,
                   seed=cfg["seed"], input_sha256=input_fingerprints([
                       folder / "examples.npz", root / "splits" / "test_events.parquet"]),
                   session_config=resolve_runtime_config(cfg)["session"])


def purchase_impression_targets(events, impressions, window_seconds, item_ids, cutoff):
    """구매를 24h 내 가장 최근 선행 조회에 귀속; 그 조회의 노출 시점에서 예측.

    구매한 상품을 이미 본 직후에 예측하는 누출을 피한다. 노출 후 window 전체를
    관측할 수 있는 양성 노출만 평가하며, 전체 노출 구매율/AUC를 뜻하지 않는다.
    """
    ev, imp = events.copy(), impressions.copy()
    ev["_t"], imp["_t"] = seconds(ev["timestamp"]), seconds(imp["timestamp"])
    if ev.empty:
        return []
    end = float(ev["_t"].max())
    imp = imp[(imp["_t"] >= cutoff) & (imp["_t"] + window_seconds <= end)]
    by_impression = {row["impression_id"]: row for row in imp.to_dict("records")}
    valid_items = set(item_ids)
    views = {key: group.sort_values("_t", kind="stable") for key, group in
             ev[ev["event_type"] == "view"].groupby(["user_id", "product_id"], sort=False)}
    targets = {}
    for buy in ev[ev["event_type"] == "purchase"].to_dict("records"):
        group = views.get((buy["user_id"], buy["product_id"]))
        if group is None or buy["product_id"] not in valid_items:
            continue
        pos = int(np.searchsorted(group["_t"].to_numpy(), buy["_t"], side="left")) - 1
        if pos < 0:
            continue
        view = group.iloc[pos]
        shown = by_impression.get(view.get("impression_id"))
        if shown is None or shown["user_id"] != buy["user_id"] or shown["session_id"] != view["session_id"]:
            continue
        shown_time = float(shown["_t"])
        if (shown_time <= view["_t"] < buy["_t"] <= shown_time + window_seconds
                and buy["product_id"] in shown["product_ids"]):
            targets.setdefault(shown["impression_id"], set()).add(buy["product_id"])
    return [(row["user_id"], row["_t"], row["session_id"], targets[row["impression_id"]])
            for row in imp.sort_values(["_t", "impression_id"], kind="stable").to_dict("records")
            if row["impression_id"] in targets]


def load_purchase_queries(models, limit=None) -> Queries:
    from src.recommendation.runtime_config import resolve_runtime_config

    cfg = models.cfg
    root = evaluation_data_root(models)
    split = root / "splits"
    ev = pd.read_parquet(split / "test_events.parquet")
    imp = pd.read_parquet(split / "test_impressions.parquet")
    window = models.cfg["ranking"].get("cvr_window_seconds", 86400)
    rows = purchase_impression_targets(ev, imp, window, models.store.item_ids,
                                      models.store.metadata["selection_cutoff"])
    if limit is not None:
        ix = np.sort(np.random.default_rng(cfg["seed"]).choice(len(rows), min(limit, len(rows)), replace=False))
        rows = [rows[i] for i in ix]
    users = np.array([r[0] for r in rows])
    times = np.array([r[1] for r in rows], dtype=float)
    sessions = session_prefixes(users, times, [r[2] for r in rows], root=root)
    return Queries(users, times, [r[3] for r in rows], sessions, len(models.store.item_ids),
                   cfg["final_eval_k"], "purchase", models.folder.name,
                   f"Purchased products attributed to an impression within {window}s; prediction at impression "
                   "before click; complete observation window; purchase-positive impressions only",
                   cfg["seed"], input_fingerprints([split / "test_events.parquet", split / "test_impressions.parquet"]),
                   session_config=resolve_runtime_config(cfg)["session"])


def evaluate_strategy(strategy: Any, q: Queries, use_session: bool, name: str = "",
                      session_weight=None) -> dict[str, Any]:
    """조회마다 strategy.recommend(top_n=k) → HitRate@k / NDCG@k / Coverage / p95."""
    if len(q.users) == 0:
        raise ValueError("No eligible evaluation queries (check target, split and full purchase observation window)")
    ranked, latency, t0 = [], [], time.perf_counter()
    for j, (u, t) in enumerate(zip(q.users, q.times)):
        at = datetime.fromtimestamp(float(t), tz=timezone.utc)
        is_multistage = isinstance(strategy, MultiStageStrategy) or name in {"multistage", "multistage_no_session"}
        kw = {"session_clicks": q.sessions[j] if use_session else []} if is_multistage or use_session else {}
        if is_multistage and session_weight is not None:
            kw["session_weight"] = session_weight
        s = time.perf_counter()
        resp = strategy.recommend(str(u), q.k, at=at, **kw)
        latency.append((time.perf_counter() - s) * 1000)
        ranked.append([r.product_id for r in resp.recommendations])
        if isinstance(strategy, MultiStageStrategy):
            # 이 평가에서 보여줬다고 정의한 K개만 센다. demo/Redis 클릭 보상은 사용하지 않는다.
            get_rerank_service(get_recommendation_models()).record_recommendation_feedback(resp.recommendations, [])
        if (j + 1) % 10000 == 0:
            print(f"  [{name}] {j + 1:,}/{len(q.users):,} ({time.perf_counter() - t0:.0f}s)", flush=True)
    m = {
        "hit_rate": hit_rate_at_k(ranked, q.targets, q.k),
        "ndcg": ndcg_at_k(ranked, q.targets, q.k),
        "coverage": coverage(ranked, q.n_items),
        "latency_p95_ms": float(np.percentile(latency, 95)),
    }
    m["passed"] = {key: m[key] >= v for key, v in TARGETS.items()} if q.target == "view" else {}
    print(f"[eval] {name:22s} HitRate@{q.k} {m['hit_rate']:.4f}  NDCG@{q.k} {m['ndcg']:.4f}  "
          f"Coverage {m['coverage']:.4f}  p95 {m['latency_p95_ms']:.1f}ms", flush=True)
    return m


def run(strategies: list[str], limit: int | None = None, target: str = "view", session_weight=None) -> dict[str, Any]:
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    q = load_queries(limit) if target == "view" else load_queries(limit, target)
    session_cfg = getattr(q, "session_config", {}) or load_config("recommend")["session"]
    query_digest = hashlib.sha256()
    for user, at, targets, clicks in zip(q.users, q.times, q.targets, q.sessions):
        query_digest.update(json.dumps([str(user), float(at), sorted(targets), clicks],
                                       ensure_ascii=False, separators=(",", ":")).encode())
        query_digest.update(b"\n")
    result: dict[str, Any] = {
        "k": q.k, "n_queries": len(q.users), "with_session": sum(bool(s) for s in q.sessions),
        "target": getattr(q, "definition", "next viewed product (test split, examples.npz)"),
        "target_type": target, "targets": TARGETS if target == "view" else {},
        "seed": getattr(q, "seed", load_config("recommend")["seed"]),
        "input_sha256": getattr(q, "input_sha256", {}),
        "evaluated_query_sha256": query_digest.hexdigest(),
        "model_version": getattr(q, "model_version", "") or load_config("recommend")["version"],
        "history_source": "explicit offline prefixes; no Redis reads",
        "latency_scope": "in-process strategy only; excludes HTTP and Redis",
        "session_weight": session_weight if session_weight is not None else session_cfg.get("final_score_weight", 0),
        "session": session_cfg,
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
        options = {"session_weight": session_weight} if session_weight is not None else {}
        result[name] = evaluate_strategy(strategy, q, use_session, name, **options)

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


USER_STRATEGIES = "multistage,two_tower_ranking,two_tower_only,popularity"


def run_users(strategies: list[str], limit: int | None = None, split: str = "test") -> dict[str, Any]:
    """사용자 단위 평가 (2026-11-18 정의, src/evaluation/user_targets.py) — 명세 기준 판정에 쓰는 주 지표.

    고객마다 그 기간 첫 이벤트 시각에 Top-k 목록을 한 번(세션 클릭 없이) 만들고, 그 기간에 조회·장바구니·구매한 상품
    전부를 정답으로 본다. HitRate / NDCG(정답 수 기준 IDCG) / Coverage(전체 목록의 고유 상품 ÷ 전체 상품)와
    구매만 정답인 보조 지표, Stage 1 사용자 단위 Recall@300을 같이 잰다. 다양성 칸의 노출 횟수가 고객 순서에 따라
    달라지므로 seed 고정 무작위 순서로 돈다. limit이면 그 순서의 앞 limit명만 (빠른 확인용).
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    models = get_recommendation_models()
    cfg, store = models.cfg, models.store
    targets = load_user_targets(evaluation_data_root(models), split, store.pmap)
    order = np.random.default_rng(cfg["seed"]).permutation(len(targets))[:limit]
    name_of = store.item_ids
    truth = {j: {name_of[i - 1] for i in targets.truth[j]} for j in order}
    buys = {j: {name_of[i - 1] for i in targets.purchases[j]} for j in order if len(targets.purchases[j])}
    k, n_cand = cfg["final_eval_k"], cfg["candidate"]["n_candidates"]
    stage1 = [[name_of[i - 1] for i in models.retriever.retrieve(targets.users[j], n_cand, targets.at[j])[0]] for j in order]
    result: dict[str, Any] = {
        "definition": "user-level: one Top-k list per user at the user's first event in the split (no session clicks, "
                      "history frozen at split start); truth = all items the user viewed/carted/purchased in the split",
        "target_type": "user", "split": split, "k": k, "n_users": len(order), "n_buyers": len(buys),
        "targets": TARGETS, "seed": cfg["seed"], "model_version": models.folder.name,
        f"stage1_recall@{n_cand}": recall_at_k(stage1, [truth[j] for j in order], n_cand),
        "stage1_target": 0.30, "history_source": "frozen bundle history; no Redis reads",
        "latency_scope": "in-process strategy only; excludes HTTP and Redis",
        "measured_at": datetime.now().isoformat(timespec="seconds"),
    }
    print(f"[eval:user] {len(order):,} users, Stage 1 Recall@{n_cand} {result[f'stage1_recall@{n_cand}']:.4f}", flush=True)
    for name in strategies:
        get_rerank_service.cache_clear()  # 전략마다 MAB 통계·다양성 노출 횟수를 새로 시작
        strategy = {"popularity": PopularityStrategy, "two_tower_only": TwoTowerOnlyStrategy,
                    "two_tower_ranking": TwoTowerRankingStrategy, "multistage": MultiStageStrategy}.get(name)
        if strategy is None:
            raise ValueError(f"지원하지 않는 전략 (사용자 단위): {name}")
        strategy = strategy()
        lists, latency, t0 = {}, [], time.perf_counter()
        for n, j in enumerate(order):
            at = datetime.fromtimestamp(float(targets.at[j]), tz=timezone.utc)
            kw = {"session_clicks": []} if isinstance(strategy, MultiStageStrategy) else {}
            s = time.perf_counter()
            resp = strategy.recommend(targets.users[j], k, at=at, **kw)
            latency.append((time.perf_counter() - s) * 1000)
            lists[j] = [r.product_id for r in resp.recommendations]
            if isinstance(strategy, MultiStageStrategy):
                get_rerank_service(get_recommendation_models()).record_recommendation_feedback(resp.recommendations, [])
            if (n + 1) % 2000 == 0:
                print(f"  [{name}] {n + 1:,}/{len(order):,} ({time.perf_counter() - t0:.0f}s)", flush=True)
        ranked = [lists[j] for j in order]
        m = {"hit_rate": hit_rate_at_k(ranked, [truth[j] for j in order], k),
             "ndcg": ndcg_at_k(ranked, [truth[j] for j in order], k),
             "coverage": coverage(ranked, len(name_of)),
             "purchase_hit_rate": hit_rate_at_k([lists[j] for j in buys], list(buys.values()), k) if buys else None,
             "purchase_ndcg": ndcg_at_k([lists[j] for j in buys], list(buys.values()), k) if buys else None,
             "latency_p95_ms": float(np.percentile(latency, 95))}
        m["passed"] = {key: m[key] >= v for key, v in TARGETS.items()}
        result[name] = m
        print(f"[eval:user] {name:20s} HitRate@{k} {m['hit_rate']:.4f}  NDCG@{k} {m['ndcg']:.4f}  "
              f"Coverage {m['coverage']:.4f}  구매 HitRate {m['purchase_hit_rate'] or 0:.4f}  p95 {m['latency_p95_ms']:.1f}ms",
              flush=True)
    for baseline in BASELINES:
        if baseline in result:
            for name in strategies:
                if name not in BASELINES:
                    result[name][f"lift_vs_{baseline}"] = {
                        key: (result[name][key] / result[baseline][key] - 1) if result[baseline][key] else None
                        for key in ("hit_rate", "ndcg", "coverage")}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategies", default=None,
                        help="기본: user → " + USER_STRATEGIES + " / view·purchase → 단계별 전부 + 베이스라인")
    parser.add_argument("--limit", type=int, default=None, help="평가할 고객(user) 또는 조회(view) 수 (기본: 전부)")
    parser.add_argument("--target", choices=["user", "view", "purchase", "both"], default="user",
                        help="user: 사용자 단위 (주 지표) / view·purchase·both: 조회 1건 단위 보조 지표")
    parser.add_argument("--split", choices=["valid", "test"], default="test", help="user 평가 기간")
    parser.add_argument("--session-weight", type=float, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    if args.target == "user":
        result = run_users((args.strategies or USER_STRATEGIES).split(","), args.limit, args.split)
    else:
        strategies = (args.strategies or "multistage,multistage_no_session,two_tower_ranking,two_tower_only,popularity").split(",")
        targets = ["view", "purchase"] if args.target == "both" else [args.target]
        results = {target: run(strategies, args.limit, target, args.session_weight) for target in targets}
        result = results if args.target == "both" else results[args.target]
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    out = Path(args.out) if args.out else artifact_dir() / "metrics" / f"recommend_final_{args.target}_{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[eval] 저장 → {out}")


if __name__ == "__main__":
    main()
