"""Trade-off 실험: MAB 탐색 칸 개수(0~3개)별 HitRate@50 / NDCG@50 / Coverage.

    python -m src.evaluation.tradeoff_exploration              # configs/recommend.yaml 의 tradeoff 설정
    python -m src.evaluation.tradeoff_exploration --sample 0   # test 76,242건 전부 (탐색 칸 4종 × 약 25분)

평가 방식은 eval_pipeline과 같다 (전체 파이프라인, 세션 포함, 정답 = 다음 조회 상품, k = final_eval_k).
바꾸는 것은 탐색 칸 개수 하나뿐이고, 위치는 Top-N 균등 분할로 자동 계산 (reranker.slot_positions).
0개 = MAB 끔 (A/B Control 설정과 같음), 3개 = 명세 범위(1~2개) 밖 비교용.
개수마다 MAB 통계를 새로 시작한다. 오프라인 로그라 탐색 칸 클릭 정보가 없어 평가 중 MAB는 노출만 쌓인다
(통계가 학습되는 효과는 A/B 시뮬레이션에서 본다).

결과: artifacts/metrics/tradeoff_exploration.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from typing import Any

from src.common.config import artifact_dir, load_config
from src.evaluation.eval_pipeline import evaluate_strategy, load_queries
from src.recommendation.pipeline import MultiStageStrategy
from src.recommendation.reranking.mab import build_explorer
from src.recommendation.reranking.stage2 import get_rerank_service
from src.serving.model_loader import get_recommendation_models


def run(slots: list[int], sample: int | None) -> dict[str, Any]:
    q = load_queries(sample)
    svc = get_rerank_service(get_recommendation_models())
    base_cfg, base_explorer = svc.reranker.config, svc.reranker.explorer
    rows = []
    try:
        for n in slots:
            svc.reranker.config = replace(base_cfg, exploration_slots=n, exploration_enabled=n > 0)
            svc.reranker.explorer = build_explorer(base_cfg.mab)  # 개수마다 MAB 통계 초기화
            m = evaluate_strategy(MultiStageStrategy(), q, use_session=True, name=f"exploration_slots={n}")
            rows.append({"exploration_slots": n, "in_spec_range": 1 <= n <= 2, **m})
    finally:
        svc.reranker.config, svc.reranker.explorer = base_cfg, base_explorer
    return {
        "k": q.k, "n_queries": len(q.users), "sample": sample,
        "algorithm": base_cfg.mab.algorithm, "model_version": load_config("recommend")["version"],
        "measured_at": datetime.now().isoformat(timespec="seconds"), "rows": rows,
    }


def main() -> None:
    cfg = load_config("recommend")["tradeoff"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=cfg["sample"], help="평가 조회 수 (0이면 전부)")
    args = parser.parse_args()
    result = run(list(cfg["exploration_slots"]), args.sample or None)

    out = artifact_dir() / "metrics" / "tradeoff_exploration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    k = result["k"]
    print(f"\n| 탐색 칸 | HitRate@{k} | NDCG@{k} | Coverage | p95 ms |")
    print("|---|---|---|---|---|")
    for r in result["rows"]:
        mark = "" if r["in_spec_range"] else " (범위 밖)"
        print(f"| {r['exploration_slots']}{mark} | {r['hit_rate']:.4f} | {r['ndcg']:.4f} | "
              f"{r['coverage']:.4f} | {r['latency_p95_ms']:.1f} |")
    print(f"[tradeoff] 저장 → {out}")


if __name__ == "__main__":
    main()
