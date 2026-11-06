"""상품 속성 분류기: 상품 이미지 임베딩 → 색 / 소분류 (선형 분류기, linear probe).

이미지 쿼리 사진의 색·종류를 추정해서, 같은 색·같은 종류 상품에 가점을 준다 (Searcher의 image 경로).
CLIP 이미지 유사도만 쓰면 모양이 비슷한 다른 색 상품이 위로 많이 올라온다
(valid 이미지 쿼리 top-10 중 평균 3.9개가 같은 소분류·다른 색).

    python -m src.search.attributes                 # 저장된 상품 임베딩으로 분류기만 학습 (indexer도 마지막에 실행)
    python -m src.search.attributes --model patrickjohncyh/fashion-clip

학습 데이터는 카탈로그 상품(이미지 임베딩, 색·소분류 라벨)뿐이라 평가 쿼리와 겹치지 않는다.
저장: artifacts/indexes/search/<모델>/attributes.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.common.config import load_config
from src.search.catalog import index_dir, load_catalog, with_overrides

COLUMNS = {"color": "color", "category": "category_l3"}  # 속성 이름 → 카탈로그 컬럼
INPUT_SCALE = 10.0  # L2 정규화된 임베딩은 값이 작아서 키워서 학습 (가중치에 곱해 저장하므로 추론은 그대로)


def train_probe(x: np.ndarray, labels: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """softmax 회귀 (전체 배치 Adam). 반환: (클래스 이름, W, b) → 확률 = softmax(W @ q + b)"""
    classes, y = np.unique(labels, return_inverse=True)
    torch.manual_seed(cfg["seed"])
    xt, yt = torch.tensor(x, dtype=torch.float32) * INPUT_SCALE, torch.tensor(y)
    linear = torch.nn.Linear(x.shape[1], len(classes))
    p = cfg["attributes"]
    opt = torch.optim.Adam(linear.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    for _ in range(p["epochs"]):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(linear(xt), yt).backward()
        opt.step()
    return classes, linear.weight.detach().numpy() * INPUT_SCALE, linear.bias.detach().numpy()


def fit(cfg: dict[str, Any]) -> Path:
    """저장된 상품 이미지 임베딩으로 속성 분류기를 학습해 인덱스 폴더에 저장."""
    d = index_dir(cfg)
    ids = pd.read_parquet(d / "product_ids.parquet")["product_id"]
    catalog = load_catalog(cfg, columns=["product_id", *COLUMNS.values()]).set_index("product_id").loc[ids]
    image_emb = np.load(d / "image_emb.npy")
    arrays = {}
    for name, column in COLUMNS.items():
        classes, w, b = train_probe(image_emb, catalog[column].to_numpy(), cfg)
        arrays |= {f"{name}_classes": classes.astype(str), f"{name}_w": w, f"{name}_b": b}  # str: pickle 없이 저장
    out = d / "attributes.npz"
    np.savez(out, **arrays)
    return out


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


class AttributeProbe:
    """쿼리 사진의 속성 확률 → 상품마다 '그 상품의 속성일 확률' 가점."""

    def __init__(self, path: Path, item_values: dict[str, np.ndarray]) -> None:
        """item_values: 속성 이름(color / category) → 상품별 값 (검색기의 상품 순서)."""
        data = np.load(path, allow_pickle=False)
        self.classes, self.w, self.b, self.item_class = {}, {}, {}, {}
        for name in COLUMNS:
            classes = data[f"{name}_classes"]
            self.classes[name] = classes
            self.w[name], self.b[name] = data[f"{name}_w"], data[f"{name}_b"]
            lookup = {c: i for i, c in enumerate(classes)}
            # 분류기 학습 뒤 카탈로그에 새로 생긴 속성 값은 -1 → 가점 0
            self.item_class[name] = np.array([lookup.get(v, -1) for v in item_values[name]])

    def bonus(self, q_image: np.ndarray, rows: np.ndarray, weights: dict[str, float]) -> np.ndarray:
        total = np.zeros(len(rows))
        for name, weight in weights.items():
            if weight:
                prob = np.append(_softmax(self.w[name] @ q_image + self.b[name]), 0.0)  # 마지막 칸 = 모르는 값(-1)
                total += weight * prob[self.item_class[name][rows]]
        return total

    def class_prob(self, name: str, q_image: np.ndarray, values: list[str]) -> float:
        """쿼리 사진의 속성이 values 중 하나일 확률 (스타일 검색의 색 변경 판단)."""
        prob = _softmax(self.w[name] @ q_image + self.b[name])
        return float(prob[np.isin(self.classes[name], values)].sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="모델 이름 (기본: configs/search.yaml)")
    args = parser.parse_args()
    cfg = with_overrides(load_config("search"), model=args.model)
    out = fit(cfg)
    print(f"[attributes] 색 / 소분류 분류기 저장 → {out}")


if __name__ == "__main__":
    main()
