"""python -m src.recommendation.candidate.train [--data-dir ... --output ... --config ...]"""

from src.recommendation.training import arguments, train_candidate

if __name__ == "__main__":
    root, out, cfg = arguments()
    train_candidate(root, out, cfg)
