"""사용자 단위 평가 대상 (2026-11-18 결정, 노션 '지표 정의' 페이지).

평가 기간에 상품을 조회·장바구니·구매한 고객마다 그 기간 첫 이벤트 시각(첫 방문 시작)에 목록을 한 번 만들고,
그 고객이 평가 기간에 조회·장바구니·구매한 상품 전부를 정답으로 본다 (같은 상품은 1개, 관련도 1).
모델 입력 이력은 평가 기간 시작 전까지로 고정 (FeatureStore frozen_split). 순위 모델 학습(valid)과 최종 평가(test)가 같이 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.recommendation.ranking.features import seconds

INTERACTIONS = ("view", "cart", "purchase")


@dataclass
class UserTargets:
    users: list[str]
    at: np.ndarray                 # unix 초, 그 기간 첫 이벤트
    truth: list[np.ndarray]        # 상품 번호 (store.pmap) — 조회·장바구니·구매
    purchases: list[np.ndarray]    # 상품 번호 — 구매만 (보조 지표)

    def __len__(self) -> int:
        return len(self.users)


def load_user_targets(root: str | Path, split: str, pmap: dict[str, int]) -> UserTargets:
    ev = pd.read_parquet(Path(root) / "splits" / f"{split}_events.parquet",
                         columns=["user_id", "timestamp", "event_type", "product_id"])
    ev["t"] = seconds(ev["timestamp"])
    first = ev.groupby("user_id")["t"].min()
    inter = ev[ev["event_type"].isin(INTERACTIONS) & ev["product_id"].isin(set(pmap))]
    def to_ids(products: pd.Series) -> np.ndarray:
        return np.unique(np.fromiter((pmap[p] for p in products), dtype=np.int64))

    truth = inter.groupby("user_id")["product_id"].agg(to_ids)
    buys = inter[inter["event_type"] == "purchase"].groupby("user_id")["product_id"].agg(to_ids)
    users = sorted(truth.index)
    empty = np.array([], dtype=np.int64)
    return UserTargets(users=users, at=np.array([float(first[u]) for u in users]),
                       truth=[truth[u] for u in users], purchases=[buys.get(u, empty) for u in users])
