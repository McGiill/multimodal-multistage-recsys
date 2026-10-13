"""베이스라인: BM25 텍스트 단독 검색 (rank-bm25).

상품 텍스트(이름, 색상, 무늬, 소분류, 설명)에 대한 키워드 매칭. 이미지 쿼리는 처리하지 못한다.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from src.search.catalog import product_texts

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Searcher:
    def __init__(self, catalog: pd.DataFrame) -> None:
        self.product_ids = catalog["product_id"].to_numpy()
        self.bm25 = BM25Okapi([tokenize(t) for t in product_texts(catalog)])

    def search(self, query: str, top_k: int = 10) -> tuple[list[str], list[float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        top = top[np.argsort(-scores[top])]
        return self.product_ids[top].tolist(), [round(float(s), 4) for s in scores[top]]
