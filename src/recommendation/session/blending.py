"""확률과 코사인의 스케일 차이를 없앤 최종 순위 점수 결합.

결과는 구매 확률이 아니라 ranking score다. weight=0은 기존 DeepFM 점수를 그대로 반환한다.
후보와 신상품 풀을 함께 정규화해야 같은 값의 의미가 풀마다 달라지지 않는다.
"""

import numpy as np


def percentile_scores(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Ranking scores must be finite")
    if len(values) < 2:
        return np.full(len(values), 0.5, dtype=np.float32)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ranks = np.cumsum(counts) - (counts + 1) / 2
    return (ranks[inverse] / (len(values) - 1)).astype(np.float32)


def blend_final_scores(ranking, session, weight: float) -> np.ndarray:
    if not 0 <= weight <= 1:
        raise ValueError("session final_score_weight must be between 0 and 1")
    ranking, session = np.asarray(ranking), np.asarray(session)
    if ranking.shape != session.shape or ranking.ndim != 1:
        raise ValueError("Ranking and session scores must be aligned vectors")
    if weight == 0:
        return ranking.copy()
    return (1 - weight) * percentile_scores(ranking) + weight * percentile_scores(session)
