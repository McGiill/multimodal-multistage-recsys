"""Stage 2 Ranking 학습: python -m src.recommendation.ranking.train  (configs/recommend.yaml ranking.objective)

- ctr (기본, 명세 평가 프로토콜 "Ranking: 노출 후 비클릭 implicit negative"):
      노출 목록에서 클릭한 상품 = 1, 보여줬지만 클릭하지 않은 상품 = 0 (클릭 1개당 impression_negatives개)
- cvr (이전 방식): 조회 후 24시간 안 구매 = 1
"""

from src.recommendation.training import arguments, train_ranking

if __name__ == "__main__":
    root, out, _ = arguments()
    train_ranking(out, root)
