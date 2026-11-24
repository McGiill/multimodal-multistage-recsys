"""최종 서빙 묶음: train + valid 앞 N일로 Stage 1(Two-Tower)·Stage 2(DeepFM)를 다시 학습한다.

    python -m src.recommendation.refit      → artifacts/models/<version><suffix>/  (데이터 사본: <version><suffix>-data/)

- 하이퍼파라미터·계수는 선택용 묶음(<version>, train만 학습)에서 valid로 이미 골랐다. 재학습은 그 설정 그대로
  최근 기록(valid 앞 N일)까지 배우게 해서 test 시점과의 간격을 줄인다. epoch는 valid 남은 기간으로 고른다. test는 쓰지 않는다.
- 공식 분할 파일은 바꾸지 않는다. valid 세션(시작 시각 기준)을 앞 N일 / 나머지로 나눈 사본을 묶음 옆에 만든다.
- 최종 순위 모델(final_ranker.pkl)은 선택용 묶음에서 학습한 것을 복사한다 (valid 전체 정답으로 학습).
설정: configs/recommend.yaml refit. 서빙은 model_loader.serving_version이 재학습 묶음을 고른다.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pandas as pd

from src.common.config import ROOT, data_dir, load_config
from src.recommendation.training import build_index, train_candidate, train_ranking


def build_refit_root(source: Path, destination: Path, valid_days_in_train: int) -> Path:
    meta = json.loads((source / "splits/meta.json").read_text(encoding="utf-8-sig"))
    valid_from = pd.Timestamp(meta["splits"]["valid"]["from"])
    cut = valid_from + pd.Timedelta(days=valid_days_in_train)
    if cut > pd.Timestamp(meta["splits"]["valid"]["to"]):
        raise ValueError("refit.valid_days_in_train must leave at least one valid day for epoch selection")
    (destination / "processed").mkdir(parents=True, exist_ok=True)
    (destination / "splits").mkdir(exist_ok=True)
    for name in ("users.parquet", "products.parquet"):
        shutil.copy2(source / "processed" / name, destination / "processed" / name)
    counts = {}
    for kind in ("events", "impressions"):
        train = pd.read_parquet(source / f"splits/train_{kind}.parquet")
        valid = pd.read_parquet(source / f"splits/valid_{kind}.parquet")
        start = valid.groupby("session_id")["timestamp"].transform("min")  # 세션을 쪼개지 않는다
        early = (start < cut).to_numpy()
        pd.concat([train, valid[early]], ignore_index=True).to_parquet(destination / f"splits/train_{kind}.parquet", index=False)
        valid[~early].reset_index(drop=True).to_parquet(destination / f"splits/valid_{kind}.parquet", index=False)
        shutil.copy2(source / f"splits/test_{kind}.parquet", destination / f"splits/test_{kind}.parquet")
        counts[kind] = {"moved_to_train": int(early.sum()), "kept_valid": int((~early).sum())}
    meta["splits"]["train"]["to"] = (cut - pd.Timedelta(days=1)).date().isoformat()
    meta["splits"]["valid"]["from"] = cut.date().isoformat()
    meta["refit"] = {"valid_days_in_train": valid_days_in_train, "source": str(source), "counts": counts,
                     "note": "valid sessions starting before the cut moved into train; test unchanged"}
    (destination / "splits/meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def main() -> None:
    cfg = load_config("recommend")
    refit = cfg.get("refit") or {}
    version, suffix = cfg["version"], refit.get("suffix", "-refit")
    models = ROOT / cfg["paths"]["model_dir"]
    selection, out = models / version, models / (version + suffix)
    root = models / (version + suffix + "-data")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} already exists; use a new version")
    print(f"[refit] data copy → {root}", flush=True)
    build_refit_root(data_dir(), root, int(refit.get("valid_days_in_train", 7)))
    cfg_r = copy.deepcopy(cfg)
    cfg_r["version"] = version + suffix
    cfg_r["refit"] = {**refit, "source_version": version}
    train_candidate(root, out, cfg_r)
    build_index(out)
    train_ranking(out, root)
    for name in ("final_ranker.pkl", "final_ranker.json"):
        if (selection / name).exists():
            shutil.copy2(selection / name, out / name)
    print(f"[refit] serving bundle ready: {out}", flush=True)


if __name__ == "__main__":
    main()
