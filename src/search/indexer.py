"""상품 CLIP 임베딩 → FAISS 인덱스 구축.

    python -m src.search.indexer                     # 임베딩 계산 + 인덱스 구축
    python -m src.search.indexer --reuse-embeddings  # 임베딩은 재사용하고 인덱스만 다시 (index.type 실험용)
    python -m src.search.indexer --model patrickjohncyh/fashion-clip   # 설정 파일을 바꾸지 않고 모델 실험

저장 위치: artifacts/indexes/search/<모델 이름>/
    image_emb.npy / text_emb.npy   상품 이미지·텍스트 임베딩 (L2 정규화, 카탈로그 순서)
    image.index   / text.index     FAISS 인덱스
    product_ids.parquet            임베딩 행 순서 = 상품 ID
    meta.json                      모델, 인덱스 설정, 생성 시각
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.common.config import load_config
from src.common.faiss_utils import build_index, save_index
from src.common.seed import set_seed
from src.search.catalog import (
    image_paths,
    index_dir,
    load_catalog,
    product_texts,
    with_overrides,
)
from src.search.encoder import build_encoder


def compute_embeddings(cfg: dict[str, Any], catalog: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    clip = cfg["clip"]
    encoder = build_encoder(clip["model_name"], device=clip["index_device"], pad_square=clip["pad_to_square"])
    print(f"  모델 {clip['model_name']} ({encoder.device})")

    paths = image_paths(cfg, catalog)
    batches = encoder.encode_image_files(paths, batch_size=clip["batch_size"])
    n_batches = (len(paths) + clip["batch_size"] - 1) // clip["batch_size"]
    image_emb = np.concatenate(list(tqdm(batches, total=n_batches, desc="  이미지 인코딩")))
    text_emb = encoder.encode_texts(product_texts(catalog), batch_size=clip["batch_size"] * 4)
    return image_emb, text_emb


def load_embeddings(out_dir: Path, catalog: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    saved_ids = pd.read_parquet(out_dir / "product_ids.parquet")["product_id"]
    if not saved_ids.equals(catalog["product_id"]):
        raise ValueError("저장된 임베딩과 현재 카탈로그의 상품 목록이 다릅니다. --reuse-embeddings 없이 다시 실행하세요.")
    return np.load(out_dir / "image_emb.npy"), np.load(out_dir / "text_emb.npy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reuse-embeddings", action="store_true")
    parser.add_argument("--model", default=None, help="모델 이름 (기본: configs/search.yaml)")
    args = parser.parse_args()

    cfg = with_overrides(load_config("search"), model=args.model)
    set_seed(cfg["seed"])
    catalog = load_catalog(cfg)
    out_dir = index_dir(cfg)
    print(f"[indexer] 상품 {len(catalog):,}개 → {out_dir}")

    t0 = time.perf_counter()
    pad_to_square = cfg["clip"]["pad_to_square"]
    if args.reuse_embeddings:
        image_emb, text_emb = load_embeddings(out_dir, catalog)
        # 임베딩을 만들 때의 전처리 설정을 유지 (서빙 시 쿼리 이미지도 같은 방식으로 인코딩해야 함)
        pad_to_square = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))["pad_to_square"]
    else:
        image_emb, text_emb = compute_embeddings(cfg, catalog)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "image_emb.npy", image_emb)
        np.save(out_dir / "text_emb.npy", text_emb)
        catalog[["product_id"]].to_parquet(out_dir / "product_ids.parquet", index=False)
    t_embed = time.perf_counter() - t0

    t0 = time.perf_counter()
    save_index(build_index(image_emb, cfg["index"]), out_dir / "image.index")
    save_index(build_index(text_emb, cfg["index"]), out_dir / "text.index")
    t_index = time.perf_counter() - t0

    meta = {
        "model_name": cfg["clip"]["model_name"],
        "pad_to_square": pad_to_square,
        "index": cfg["index"],
        "n_products": len(catalog),
        "dim": int(image_emb.shape[1]),
        "embed_sec": round(t_embed, 1),
        "index_sec": round(t_index, 1),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[indexer] 완료: 임베딩 {t_embed:.1f}s, 인덱스 {t_index:.1f}s ({cfg['index']['type']})")


if __name__ == "__main__":
    main()
