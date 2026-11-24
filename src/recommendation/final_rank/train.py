"""최종 순위 모델 학습: python -m src.recommendation.final_rank.train [--output 묶음] [--data-dir data]

선택용 묶음(train 기간만 학습한 Stage 1·2, configs/recommend.yaml version)으로 valid 기간 고객마다 첫 방문 시점의
후보 풀과 특징을 만들고, 그 고객이 valid 기간에 조회·장바구니·구매한 상품을 정답으로 GBDT를 학습한다.
- 2겹 교차: 고객을 두 묶음으로 나눠 서로 다른 묶음으로 학습 → 교차 점수로 valid 지표를 남긴다 (Stage 3 규칙 전 근사).
- 음성: 고객마다 풀에서 정답이 아닌 상품 negatives개 무작위 (seed 고정). 특징은 풀 전체 기준으로 계산한다.
- 서빙은 두 모델의 평균 확률 (final_ranker.pkl, 묶음 폴더에 저장).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from src.common.config import ROOT, data_dir, load_config
from src.evaluation.metrics import coverage, hit_rate_at_k, ndcg_at_k
from src.evaluation.user_targets import load_user_targets
from src.recommendation.final_rank import (
    FEATURES,
    FinalRankContext,
    FinalRanker,
    score_pool,
)
from src.recommendation.runtime_config import resolve_runtime_config
from src.serving.model_loader import RecommendationModels

K = 50


def user_pool(models, ctx, user_id, at, ranker=None):
    q, state = models.retriever.user_query(user_id, at)
    stage1, _ = models.retriever.search(q, state, at, ctx.pool_size)
    ids, scores, X, _ = score_pool(models, ctx, ranker, user_id, at, q, state, stage1)
    return ids, scores, X


def train_final_ranker(folder: str | Path, root: str | Path | None = None, evaluate: bool = True) -> dict:
    t0 = time.perf_counter()
    folder = Path(folder)
    models = RecommendationModels(folder)
    cfg = resolve_runtime_config(models.cfg)
    fr = cfg.get("final_rank") or {}
    tr = fr.get("training") or {}
    ctx = FinalRankContext(models.store, models.item_vectors, fr)
    targets = load_user_targets(Path(root) if root else data_dir(), "valid", models.store.pmap)
    folds = int(tr.get("folds", 2))
    fold = np.random.default_rng(cfg["seed"]).permutation(len(targets)) % folds
    negatives = int(tr.get("negatives", 400))
    rows = {f: ([], []) for f in range(folds)}
    rngs = {f: np.random.default_rng(cfg["seed"] + 1 + f) for f in range(folds)}
    for j, (user, at) in enumerate(zip(targets.users, targets.at)):
        ids, _, X = user_pool(models, ctx, user, at)
        y = np.isin(ids, targets.truth[j])
        neg = np.flatnonzero(~y)
        keep = np.r_[np.flatnonzero(y), rngs[fold[j]].choice(neg, min(negatives, len(neg)), replace=False)]
        rows[fold[j]][0].append(X[keep])
        rows[fold[j]][1].append(y[keep])
        if (j + 1) % 2000 == 0:
            print(f"[final_rank] rows {j + 1:,}/{len(targets):,} ({time.perf_counter() - t0:.0f}s)", flush=True)
    params = {k: tr[k] for k in ("max_iter", "learning_rate", "max_leaf_nodes", "l2_regularization") if k in tr}
    estimators, fits = [], []
    for f in range(folds):
        X, y = np.concatenate(rows[f][0]), np.concatenate(rows[f][1])
        est = HistGradientBoostingClassifier(early_stopping=True, validation_fraction=0.1, random_state=cfg["seed"],
                                             **params).fit(X, y)
        estimators.append(est)
        fits.append({"fold": f, "rows": int(len(y)), "positives": int(y.sum()), "iterations": est.n_iter_})
        print(f"[final_rank] fold {f} fitted {fits[-1]} ({time.perf_counter() - t0:.0f}s)", flush=True)
    del rows
    meta = {"trained_on": "valid split, user-level labels (view/cart/purchase), first event per user",
            "bundle": folder.name, "users": len(targets), "negatives_per_user": negatives, "folds": fits,
            "features": list(FEATURES)}
    if evaluate:  # 교차 점수: 고객이 속하지 않은 묶음의 모델로 점수 → valid 사용자 단위 지표 (Stage 3 규칙 전 근사)
        from src.recommendation.reranking.stage2 import get_rerank_service

        codes = get_rerank_service(models).design_codes
        lists = []
        for j, (user, at) in enumerate(zip(targets.users, targets.at)):
            ids, _, X = user_pool(models, ctx, user, at)
            others = [e for f, e in enumerate(estimators) if f != fold[j]] or estimators
            s = np.mean([e.predict_proba(X)[:, 1] for e in others], axis=0)
            order = np.argsort(-s, kind="stable")
            _, first = np.unique(codes[ids[order]], return_index=True)
            lists.append([models.store.item_ids[i - 1] for i in ids[order[np.sort(first)]][:K]])
        truth = [{models.store.item_ids[i - 1] for i in t} for t in targets.truth]
        meta["valid_cross_fit_top50_before_stage3"] = {
            "hit_rate@50": hit_rate_at_k(lists, truth, K), "ndcg@50": ndcg_at_k(lists, truth, K),
            "coverage": coverage(lists, len(models.store.item_ids))}
        print(f"[final_rank] valid (cross-fit) {meta['valid_cross_fit_top50_before_stage3']}", flush=True)
    FinalRanker(estimators, FEATURES, meta).save(folder)
    (folder / "final_ranker.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[final_rank] saved {folder / 'final_ranker.pkl'} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, help="선택용 묶음 폴더 (기본: model_dir/version)")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--no-eval", action="store_true", help="교차 점수 valid 지표 생략")
    a = p.parse_args()
    cfg = load_config("recommend")
    folder = a.output or ROOT / cfg["paths"]["model_dir"] / cfg["version"]
    train_final_ranker(folder, a.data_dir, evaluate=not a.no_eval)


if __name__ == "__main__":
    main()
