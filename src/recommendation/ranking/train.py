"""조회 후 24시간 구매 라벨로 CVR만 학습. 노출 미클릭은 사용하지 않는다."""

from src.recommendation.training import arguments, train_ranking

if __name__ == "__main__":
    _, out, _ = arguments()
    train_ranking(out)
