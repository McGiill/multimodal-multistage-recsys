"""시점별 등록 상품만 FAISS 후보로 반환. 부족하면 있는 수만 반환한다."""

import numpy as np
import torch

from src.recommendation.candidate.prior import PriorIndex


def search_available(index, q, created, at, n, allowed=None):
    """상품 등록 시각과 요청 범위를 모두 만족할 때까지 FAISS fetch를 확장한다."""
    if allowed is not None:
        allowed = np.asarray(allowed)
        if allowed.dtype != np.bool_ or allowed.shape != (index.ntotal + 1,) or allowed[0]:
            raise ValueError("allowed must be a boolean item mask including false padding row")
    n = min(n, index.ntotal)
    if n <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    fetch = min(index.ntotal, max(n, 2 * n))
    while True:
        scores, ids = index.search(
            np.asarray(q, dtype=np.float32).reshape(1, -1), fetch
        )
        good = ids[0] >= 0
        pid = ids[0, good] + 1
        score = scores[0, good]
        valid = created[pid] <= at
        if allowed is not None:
            valid &= allowed[pid]
        pid, score = pid[valid], score[valid]
        if len(pid) >= n or fetch == index.ntotal:
            return pid[:n], score[:n]
        fetch = min(index.ntotal, fetch * 2)


class Retriever:
    def __init__(self, model, index, store, vectors=None, prior=None):
        """prior(RetrievalPrior)가 켜져 있으면 Two-Tower 내적에 인기·카테고리 선호 항을 더한 색인으로 찾는다 (prior.py).
        vectors: 상품 벡터 (item_vectors.npy). 끄거나 vectors가 없으면 기존 Two-Tower 색인만 쓴다."""
        self.model = model
        self.index = index
        self.store = store
        self.prior_index = (PriorIndex(vectors, store, prior)
                            if prior is not None and prior.active and vectors is not None else None)

    @torch.inference_mode()
    def user_query(self, user_id, at, recent=None):
        """User Tower 벡터 (1, d)와 state. recent: 저장된 이력 뒤에 붙일 최근 조회 상품 번호 (append_recent)."""
        if at is None:
            raise ValueError("Explicit prediction time required")
        h, l, p = self.store.state(user_id, at)
        if recent is not None and len(recent):
            h, l = self.store.append_recent(h, l, recent)
        q = self.model.user(
            torch.from_numpy(p[None]), torch.from_numpy(h[None]), torch.tensor([l])
        ).numpy()
        return q, (h, l, p)

    def search(self, q, state, at, n, allowed=None):
        if self.prior_index is not None:
            return search_available(self.prior_index.index(at), self.prior_index.query(q, state[0]),
                                    self.store.created, at, n, allowed=allowed)
        return search_available(self.index, q, self.store.created, at, n, allowed=allowed)

    def retrieve(self, user_id, n=300, at=None, recent=None, allowed=None):
        """recent: 저장된 이력 뒤에 붙일 최근 조회 상품 번호 (FeatureStore.append_recent). 반환 state에도 반영된다."""
        q, state = self.user_query(user_id, at, recent)
        ids, scores = self.search(q, state, at, n, allowed=allowed)
        return ids, scores, state
