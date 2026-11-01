"""Official team data -> Stage 1/2 CVR -> evaluation and real FastAPI request.

python scripts/stage12_team.py --action all
python scripts/stage12_team.py --action demo
"""

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import ROOT, load_config
from src.recommendation.training import build_index, train_candidate, train_ranking


def verify(root):
    manifest = root / "TEAM_SHA256SUMS.txt"
    if not manifest.exists():
        raise FileNotFoundError("Copy official SHA256SUMS.txt to data/TEAM_SHA256SUMS.txt")
    records = []
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        digest, name = line.split(maxsplit=1)
        rel = Path(name.removeprefix("data/"))
        path = (root / rel).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Checksum path outside data directory")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"Checksum mismatch: {rel}")
        records.append({"file": name, "sha256": actual, "bytes": path.stat().st_size})
    if len(records) != 9:
        raise ValueError("Expected 9 official data files")
    return records


def demo(out):
    os.environ["REC_MODEL_DIR"] = str(out.resolve())
    os.environ["PRELOAD_MODELS"] = "0"
    from datetime import UTC, datetime

    from fastapi.testclient import TestClient

    from src.serving.main import app
    from src.serving.model_loader import get_recommendation_models

    models = get_recommendation_models()
    at = datetime.fromtimestamp(models.store.metadata["selection_cutoff"] + 86400, UTC).isoformat()
    user = models.store.users.iloc[0].user_id
    with TestClient(app) as client:
        response = client.get("/api/recommend", params={"user_id": user, "top_n": 10, "at": at})
        assert response.status_code == 200, response.text
        body = response.json()
        items = body["recommendations"]
        assert len(items) == 10
        assert len({item["product_id"] for item in items}) == 10
        assert all(item["product_id"] in models.store.pmap for item in items)
        scores = [item["score"] for item in items]
        assert scores == sorted(scores, reverse=True)
        assert all(0 <= score <= 1 for score in scores)
        cold = client.get("/api/recommend", params={"user_id": "unknown", "at": at})
        assert cold.status_code == 200, cold.text
        invalid = client.get("/api/recommend", params={"user_id": user, "at": "2026-01-01T00:00:00Z"})
        assert invalid.status_code == 400
    result = {
        "status": response.status_code, "prediction_at": at,
        "unknown_user_status": cold.status_code, "before_model_cutoff_status": invalid.status_code,
        "response": body,
    }
    (out / "api_demo.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["verify", "train", "evaluate", "demo", "all"], default="demo")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = copy.deepcopy(load_config("recommend"))
    out = args.output or ROOT / cfg["paths"]["model_dir"] / cfg["version"]
    if args.action in ["verify", "train", "all"]:
        records = verify(args.data_dir)
        print(f"Verified {len(records)} official files", flush=True)
    if args.action in ["train", "all"]:
        train_candidate(args.data_dir, out, cfg)
        (out / "source_checksums.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        build_index(out)
        train_ranking(out)
    if args.action in ["evaluate", "all"]:
        from src.evaluation.eval_recommend import evaluate

        print(json.dumps(evaluate(out), indent=2), flush=True)
    if args.action in ["demo", "all"]:
        demo(out)


if __name__ == "__main__":
    main()
