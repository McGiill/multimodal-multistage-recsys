"""같은 데이터·seed에서 Item Tower의 이미지 특징 유무만 바꾸는 독립 실험.

    python -m src.evaluation.experiment_recommend --output artifacts/experiments/image-v1
    python -m src.evaluation.experiment_recommend --output artifacts/experiments/image-smoke --limit 200

두 새 모델을 학습한다. 기본 서빙 버전과 기존 metrics 파일은 바꾸지 않으며,
세션 최종 점수 가중치는 0으로 고정하여 이미지 효과와 세션 개선을 분리한다.
--limit는 평가 표본 수만 줄인다(학습량은 --config의 epochs 등으로 조절).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.common.config import data_dir, load_config
from src.recommendation.candidate.image_features import file_sha256
from src.recommendation.runtime_config import resolve_runtime_config
from src.recommendation.session.encoder import model_path
from src.recommendation.training import build_index, train_candidate, train_ranking


def make_configs(cfg: dict, embeddings_dir: Path | None = None) -> dict[str, dict]:
    """두 arm의 차이는 버전 이름과 image_features.enabled 하나뿐이다."""
    common = resolve_runtime_config(cfg)
    common["session"]["final_score_weight"] = 0.0
    features = common["candidate"].setdefault("image_features", {})
    if embeddings_dir is not None:
        features["embeddings_dir"] = str(embeddings_dir.resolve())
    if not features.get("embeddings_dir"):
        raise ValueError("An image embedding directory is required")
    arms = {}
    for name, enabled in (("baseline", False), ("image", True)):
        arm = copy.deepcopy(common)
        arm["version"] = f"{common.get('version', 'candidate')}-{name}"
        arm["candidate"]["image_features"]["enabled"] = enabled
        arms[name] = arm
    return arms


def dataset_manifest(root: Path) -> dict[str, str]:
    """실험 결과와 함께 입력 파일 checksum을 남긴다(숨김 성향은 학습에 사용하지 않음)."""
    paths = [root / "processed" / f"{name}.parquet" for name in ("products", "users")]
    paths += [root / "splits" / "meta.json"]
    paths += [root / "splits" / f"{split}_{kind}.parquet"
              for split in ("train", "valid", "test") for kind in ("events", "impressions")]
    return {str(p.relative_to(root)): file_sha256(p) for p in paths if p.exists()}


def _clear_serving_caches() -> None:
    from src.recommendation.reranking.stage2 import get_rerank_service
    from src.recommendation.session.encoder import get_session_model
    from src.serving.model_loader import load_models

    get_rerank_service.cache_clear()
    get_session_model.cache_clear()
    load_models.cache_clear()


@contextmanager
def model_environment(folder: Path, root: Path):
    """같은 프로세스의 arm 교체가 이전 모델/Redis 설정을 섞지 않도록 정리한다."""
    updates = {"REC_MODEL_DIR": str(folder.resolve()), "DATA_DIR": str(root.resolve())}
    previous = {key: os.environ.get(key) for key in updates}
    try:
        os.environ.update(updates)
        _clear_serving_caches()
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _clear_serving_caches()


def run_experiment(root: Path, output: Path, cfg: dict, *, embeddings_dir: Path | None = None,
                   limit: int | None = None, target: str = "both",
                   strategies: tuple[str, ...] = ("two_tower_only", "two_tower_ranking", "multistage")) -> dict:
    from src.evaluation.eval_pipeline import run
    from src.evaluation.eval_recommend import evaluate

    root, output = Path(root).resolve(), Path(output).resolve()
    if target not in ("view", "purchase", "both"):
        raise ValueError("target must be view, purchase, or both")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    configs = make_configs(cfg, embeddings_dir)
    source_session = model_path(configs["baseline"])
    if not source_session.exists() and any(s in {"multistage", "multistage_no_session"} for s in strategies):
        raise FileNotFoundError(f"Freeze a trained session checkpoint before this experiment: {source_session}")
    manifest = dataset_manifest(root)
    # 이미 사용한 실험 이름을 재사용하면 실패시켜 결과/모델 덮어쓰기를 막는다.
    output.mkdir(parents=True, exist_ok=False)
    session_snapshot = None
    if source_session.exists():
        # 두 arm의 긴 학습 사이에 공용 체크포인트가 바뀌어도 같은 세션 모델을 사용한다.
        frozen_session = output / "session_encoder.pt"
        shutil.copyfile(source_session, frozen_session)
        session_snapshot = {"source": str(source_session), "sha256": file_sha256(frozen_session)}
        for arm_config in configs.values():
            arm_config["session"]["model_file"] = str(frozen_session)
    result = {
        "experiment": "candidate_image_features", "seed": cfg["seed"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(root), "input_sha256": manifest, "limit": limit,
        "session_final_score_weight": 0.0,
        "session_checkpoint": session_snapshot,
        "changed_variable": "candidate.image_features.enabled",
        "automatic_promotion": False, "arms": {},
    }
    targets = ("view", "purchase") if target == "both" else (target,)
    for name, arm_config in configs.items():
        folder = output / name
        train_candidate(root, folder, arm_config)
        if session_snapshot is not None:
            shutil.copyfile(frozen_session, folder / "session_encoder.pt")
        build_index(folder)
        train_ranking(folder, root)
        with model_environment(folder, root):
            stage12 = evaluate(folder)
            pipeline = {truth: run(list(strategies), limit=limit, target=truth, session_weight=0.0)
                        for truth in targets}
        result["arms"][name] = {"model_dir": str(folder), "stage12": stage12, "pipeline": pipeline}
        (output / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if dataset_manifest(root) != manifest:
        raise RuntimeError("Input data changed during the image experiment; results are not comparable")
    result["delta_image_minus_baseline"] = {
        "recall_at_300": result["arms"]["image"]["stage12"]["retrieval"]["recall_at_300"]
                         - result["arms"]["baseline"]["stage12"]["retrieval"]["recall_at_300"],
        "ranking_auc": result["arms"]["image"]["stage12"]["ranking"]["auc"]
                       - result["arms"]["baseline"]["stage12"]["ranking"]["auc"],
    }
    result["completed"] = True
    (output / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=data_dir())
    parser.add_argument("--output", type=Path, required=True, help="새 실험 폴더; 기존 폴더는 거부")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--embeddings-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--target", choices=("view", "purchase", "both"), default="both")
    parser.add_argument("--strategies", default="two_tower_only,two_tower_ranking,multistage")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) if args.config else copy.deepcopy(load_config("recommend"))
    result = run_experiment(args.data_dir, args.output, cfg, embeddings_dir=args.embeddings_dir, limit=args.limit,
                            target=args.target, strategies=tuple(args.strategies.split(",")))
    print(json.dumps(result["delta_image_minus_baseline"], indent=2))
    print(f"[experiment] {args.output / 'comparison.json'}")


if __name__ == "__main__":
    main()
