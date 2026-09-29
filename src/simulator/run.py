"""배치 생성 엔트리포인트: python -m src.simulator.run

출력: data/processed/products.parquet, users.parquet, events.parquet
담당: A
"""

from __future__ import annotations

from src.common.config import load_config
from src.common.seed import set_seed


def main() -> None:
    cfg = load_config("simulator")
    set_seed(cfg["seed"])
    # TODO(A): catalog → users → behavior 순으로 생성 후 parquet 저장
    raise NotImplementedError


if __name__ == "__main__":
    main()
