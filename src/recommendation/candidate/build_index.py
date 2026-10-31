"""상품 임베딩 오프라인 계산 및 FAISS 저장."""

from src.recommendation.training import arguments, build_index

if __name__ == "__main__":
    _, out, _ = arguments()
    build_index(out)
