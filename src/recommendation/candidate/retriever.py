"""시점별 등록 상품만 FAISS 후보로 반환. 부족하면 있는 수만 반환한다."""

import numpy as np
import torch


def search_available(index, q, created, at, n):
    n = min(n, index.ntotal)
    if n == 0:
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
        pid, score = pid[valid], score[valid]
        if len(pid) >= n or fetch == index.ntotal:
            return pid[:n], score[:n]
        fetch = min(index.ntotal, fetch * 2)


class Retriever:
    def __init__(self, model, index, store):
        self.model = model
        self.index = index
        self.store = store

    @torch.inference_mode()
    def retrieve(self, user_id, n=300, at=None):
        if at is None:
            raise ValueError("Explicit prediction time required")
        h, l, p = self.store.state(user_id, at)
        q = self.model.user(
            torch.from_numpy(p[None]), torch.from_numpy(h[None]), torch.tensor([l])
        ).numpy()
        ids, scores = search_available(self.index, q, self.store.created, at, n)
        return ids, scores, (h, l, p)
