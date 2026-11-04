"""Stage 1·2 출력 → Stage 3 입력 변환, 서빙용 Reranker 보관.

Stage 2 (pipeline.MultiStageStrategy)가 다루는 값
  ids    : 1부터 시작하는 내부 상품 번호 (store.item_ids[id - 1] = product_id)
  scores : CVR 점수
  state  : (최근 조회 상품 번호 h, 길이, 프로필) — FeatureStore.state
  now    : unix 초 (UTC)
Reranker는 카테고리·등록일이 붙은 Candidate와 datetime을 받으므로 여기서 바꾼다.

MAB 통계는 Reranker 안(InMemoryStatsStore)에 쌓이므로 요청마다 새로 만들면 매번 초기화된다.
get_rerank_service(models)는 모델 묶음당 한 번만 만든다 (model_loader.load_models와 같은 캐시 방식).
담당: D
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Collection, Mapping, Sequence

import numpy as np
import pandas as pd

from src.recommendation.reranking.catalog import load_catalog, prepare_catalog, to_candidates
from src.recommendation.reranking.items import Candidate, RerankResult
from src.recommendation.reranking.mab import ExplorationPicker
from src.recommendation.reranking.reranker import Reranker, RerankConfig

STAGE2_REASON = "cvr"


def from_unix(ts: float) -> datetime:
    """Stage 2의 unix 초 → UTC datetime (카탈로그 created_at과 같은 형식)."""
    return datetime.fromtimestamp(float(ts), tz=timezone.utc)


def history_count(state: Sequence[Any]) -> int:
    """사용자 조회 이력 수 (cold-start 판단용).

    state[1]은 이력이 없어도 1이라 쓰지 않고, state[0]의 빈칸(0)이 아닌 개수를 센다.
    history_len(기본 20)개에서 잘리지만 cold-start 기준(5개 미만) 판단에는 충분하다.
    """
    return int(np.count_nonzero(np.asarray(state[0])))


class RerankService:
    """Stage 2 번호·점수를 받아 Reranker를 돌린다. 서버에서 하나만 만들어 계속 쓴다.

    products: Stage 2 상품 표 (store.products, product_id + category_l1~l3 + created_at)
    product_codes: product_id → product_code (store.products에는 없어서 카탈로그에서 따로 받음).
        None이면 색상 변형 중복 제거 없이 product_id 기준.
    """

    def __init__(
        self,
        item_ids: Sequence[str],
        products: pd.DataFrame,
        config: RerankConfig | None = None,
        product_codes: Mapping[str, str] | pd.Series | None = None,
        explorer: ExplorationPicker | None = None,
    ) -> None:
        self.item_ids = list(item_ids)
        catalog = prepare_catalog(products)
        if product_codes is not None:
            catalog["product_code"] = pd.Series(product_codes, dtype=object).reindex(catalog.index)
        missing = pd.Index(self.item_ids).difference(catalog.index)
        if len(missing):
            raise ValueError(f"상품 표에 없는 상품 {len(missing)}개: {list(missing[:5])}")
        self.catalog = catalog.loc[self.item_ids]
        # 번호 i의 등록 시각(unix 초) = self._created[i - 1]
        self._created = (self.catalog["created_at"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy()
        # 요청마다 표를 조회하지 않도록 번호 순서로 미리 꺼내 둔다 (300개 변환 1ms 미만)
        self._rows = to_candidates(self.item_ids, np.zeros(len(self.item_ids)), self.catalog)
        self.reranker = Reranker(config or RerankConfig.from_recommend_config(), explorer)

    @property
    def config(self) -> RerankConfig:
        return self.reranker.config

    def candidates(self, ids: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray,
                   reason: str = STAGE2_REASON) -> list[Candidate]:
        """내부 번호(1부터) + 점수 → Candidate."""
        if len(ids) != len(scores):
            raise ValueError("ids와 scores 길이가 다름")
        if len(ids) and np.min(ids) < 1:  # 0(padding)은 _rows[-1]로 조용히 잘못 매핑됨
            raise ValueError("상품 번호는 1부터")
        return [replace(self._rows[int(i) - 1], score=float(s), reason=reason)
                for i, s in zip(ids, scores)]

    def new_item_ids(self, now: float) -> np.ndarray:
        """요청 시각 기준 신규 상품 번호(1부터). 등록 시각이 (now - new_item_days일, now] 안인 상품.

        Stage 2로 점수를 매겨 rerank(new_ids=..., new_scores=...)로 넘기면, 후보 300개에
        신규 상품이 없을 때 이 목록에서 신규 상품 칸을 채운다.
        """
        cutoff = now - self.config.new_item_max_age_days * 86400
        return np.flatnonzero((self._created > cutoff) & (self._created <= now)) + 1

    def rerank(
        self,
        ids: Sequence[int] | np.ndarray,
        scores: Sequence[float] | np.ndarray,
        state: Sequence[Any],
        now: float,
        top_n: int | None = None,
        popular: Sequence[Candidate] = (),
        trending: Sequence[Candidate] = (),
        new_ids: Sequence[int] | np.ndarray = (),
        new_scores: Sequence[float] | np.ndarray = (),
    ) -> RerankResult:
        """Stage 2 출력(후보 전체, top_n으로 자르기 전) → 최종 Top-N.

        신규 상품 목록은 점수 상위 top_n개만 넘긴다 (칸을 채우는 데 그 이상은 쓰이지 않음).
        """
        n = self.config.top_n if top_n is None else top_n
        history = history_count(state)
        if not popular and not trending:
            # 인기·트렌딩 목록이 아직 없으면 cold-start 사용자에게 빈 진열이 나가므로
            # Stage 2 순서로 채운다 (목록이 생기면 이 분기는 타지 않음)
            history = max(history, self.config.min_history)
        new_pool: list[Candidate] = []
        if len(new_ids):
            new_scores = np.asarray(new_scores, dtype=np.float32)
            keep = np.argsort(-new_scores, kind="stable")[:n]
            new_pool = self.candidates(np.asarray(new_ids)[keep], new_scores[keep])
        return self.reranker.rerank(
            self.candidates(ids, scores),
            history,
            from_unix(now),
            popular=popular,
            trending=trending,
            new_item_pool=new_pool,
            top_n=n,
        )

    def record_feedback(self, result: RerankResult, clicked_ids: Collection[str]) -> int:
        """진열한 MAB 탐색 칸의 클릭 여부를 기록한다 (Reranker.record_feedback)."""
        return self.reranker.record_feedback(result, clicked_ids)


def load_product_codes() -> pd.Series | None:
    """카탈로그(products.parquet)의 product_id → product_code. 파일이 없으면 None."""
    try:
        return load_catalog()["product_code"]
    except FileNotFoundError:
        return None


@lru_cache(maxsize=4)
def get_rerank_service(models: Any) -> RerankService:
    """모델 묶음(model_loader.RecommendationModels)당 RerankService 하나.

    load_models가 같은 버전에 같은 객체를 돌려주므로 MAB 통계가 요청 사이에 유지된다.
    """
    return RerankService(models.store.item_ids, models.store.products,
                         product_codes=load_product_codes())
