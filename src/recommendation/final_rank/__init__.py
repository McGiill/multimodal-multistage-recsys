"""최종 순위 (Stage 2 다음 단계): 후보 출처 확장 + 노출 구조 특징 + GBDT.

Stage 1(Two-Tower) 상위 pool_size개에 고객의 관심 카테고리 목록(인기·저가·신상품), 고객층 인기, 취향 롱테일을 더하고,
DeepFM(Stage 2, 명세 Ranking) 점수와 카테고리 목록 안 순위 같은 노출 구조 특징으로 학습한 GBDT가 최종 순서를 정한다.
정답 = 그 고객이 앞으로 조회·장바구니·구매할 상품 (valid 기간으로 학습, 2겹 교차). 설정: configs/recommend.yaml final_rank.
학습: python -m src.recommendation.final_rank.train
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.recommendation.final_rank.context import FEATURES, FinalRankContext

ARTIFACT = "final_ranker.pkl"


class FinalRanker:
    """교차 학습한 GBDT 여러 개의 평균 확률."""

    def __init__(self, estimators: Sequence[Any], features: Sequence[str] = FEATURES, meta: dict | None = None) -> None:
        unknown = [f for f in features if f not in FEATURES]
        if unknown:
            raise ValueError(f"Final ranker uses unknown features {unknown}; retrain the final ranker")
        self.estimators, self.features, self.meta = list(estimators), tuple(features), dict(meta or {})
        # 특징을 추가하기 전에 학습한 묶음은 자기 특징 이름의 열만 골라 쓴다 (같은 계산 → 같은 점수)
        self.columns = None if self.features == FEATURES else np.array([FEATURES.index(f) for f in self.features])

    def select(self, X: np.ndarray) -> np.ndarray:
        """현재 FEATURES 순서의 특징 행렬 → 이 모델이 학습한 열."""
        return X if self.columns is None else X[:, self.columns]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not len(X):
            return np.array([], np.float32)
        X = self.select(X)
        return np.mean([m.predict_proba(X)[:, 1] for m in self.estimators], axis=0).astype(np.float32)

    def save(self, folder: str | Path) -> Path:
        path = Path(folder) / ARTIFACT
        with path.open("wb") as f:
            pickle.dump({"estimators": self.estimators, "features": list(self.features), "meta": self.meta}, f)
        return path

    @classmethod
    def load(cls, folder: str | Path) -> FinalRanker | None:
        """학습 결과가 없는 묶음(이전 모델·테스트용 작은 묶음)은 None → 기존 순서(DeepFM)를 쓴다."""
        path = Path(folder) / ARTIFACT
        if not path.exists():
            return None
        with path.open("rb") as f:  # 같은 학습 파이프라인이 만든 파일만 읽는다 (외부 파일 금지)
            blob = pickle.load(f)
        return cls(blob["estimators"], blob["features"], blob.get("meta"))


def score_pool(models: Any, ctx: FinalRankContext, ranker: FinalRanker | None, user_id: str, at: float,
               q: np.ndarray, state: Sequence[Any], stage1: np.ndarray, allowed: np.ndarray | None = None,
               extra: Sequence[int] = ()) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """후보 풀 → (상품 번호, 최종 점수, 특징 행렬, 출처). ranker가 없으면 점수는 DeepFM (학습 데이터 만들 때)."""
    info = ctx.user_info(user_id, at, state)
    ids, src = ctx.pool(info, at, q, stage1, allowed, extra)
    dfm = models.score(ids, state, at)
    X = ctx.features(ids, src, info, at, q, dfm)
    scores = ranker.predict(X) if ranker is not None else np.asarray(dfm, np.float32)
    return ids, scores, X, src


__all__ = ["ARTIFACT", "FEATURES", "FinalRankContext", "FinalRanker", "score_pool"]
