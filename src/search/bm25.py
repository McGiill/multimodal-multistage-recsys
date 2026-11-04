"""BM25 키워드 매칭: 베이스라인(BM25Searcher)과 검색기의 텍스트 점수 융합용(BM25Index).

상품 텍스트(이름, 색상, 무늬, 소분류, 설명)에 대한 키워드 매칭. 이미지 쿼리는 처리하지 못한다.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from src.search.catalog import product_texts

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    """rank_bm25.BM25Okapi와 같은 점수를 역색인으로 계산 (API 응답시간용: 쿼리 단어가 나온 상품만 계산).

    BM25Okapi는 쿼리 단어마다 전체 상품을 파이썬으로 훑어서 느리고, 상품마다 단어 사전을 들고 있어 메모리도 크다.
    """

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25) -> None:
        self.n_docs = len(docs)
        self.k1 = k1
        lengths = np.array([len(d) for d in docs], dtype=np.float64)
        self.norm = k1 * (1 - b + b * lengths / lengths.mean())
        postings: dict[str, tuple[list[int], list[int]]] = {}
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                rows, tfs = postings.setdefault(term, ([], []))
                rows.append(i)
                tfs.append(tf)
        self.postings = {t: (np.array(r, dtype=np.int32), np.array(f, dtype=np.float64)) for t, (r, f) in postings.items()}
        idf = {t: math.log(self.n_docs - len(r) + 0.5) - math.log(len(r) + 0.5) for t, (r, _) in self.postings.items()}
        floor = epsilon * sum(idf.values()) / len(idf)  # BM25Okapi와 같이 음수 idf(흔한 단어)는 평균 idf의 epsilon배
        self.idf = {t: v if v >= 0 else floor for t, v in idf.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n_docs)
        for term in tokenize(query):
            if term in self.postings:
                rows, tf = self.postings[term]
                out[rows] += self.idf[term] * tf * (self.k1 + 1) / (tf + self.norm[rows])
        return out


class BM25Searcher:
    def __init__(self, catalog: pd.DataFrame) -> None:
        self.product_ids = catalog["product_id"].to_numpy()
        self.bm25 = BM25Okapi([tokenize(t) for t in product_texts(catalog)])

    def search(self, query: str, top_k: int = 10) -> tuple[list[str], list[float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        top = top[np.argsort(-scores[top])]
        return self.product_ids[top].tolist(), [round(float(s), 4) for s in scores[top]]
