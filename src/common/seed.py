"""재현성: 시뮬레이터 / 데이터 분할 / 학습 모두 seed=42."""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
