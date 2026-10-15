"""시뮬레이터 배치 생성: python -m src.simulator.run

    python -m src.simulator.run                     # configs/simulator.yaml 규모 (고객 1만 명)
    python -m src.simulator.run --users 500         # 개발용 소규모
    python -m src.simulator.run --engine attribute  # 검색 인덱스 없이 (속성 일치 검색)
    python -m src.simulator.run --recalibrate       # Retailrocket 보정을 강제로 다시

순서 (없는 단계만 새로 만들고, 있으면 data/processed/simulator/ 에서 재사용)
    상품 지도(H&M) → 검색 결과표 → Retailrocket 전이 확률 추정 → 페르소나 보정
    → 고객 생성 → 방문·행동 생성 → 저장 → 검증 리포트
입력: data/processed/products.parquet (src/data/build_catalog.py), data/raw/hm, data/raw/retailrocket
출력 (data/processed)
- users.parquet        : user_id, age_group, gender, signup_at                       (공개, 모델 입력 가능)
- users_hidden.parquet : persona, 선호 카테고리·브랜드, 가격 민감도 등                  (숨김, 모델 입력 금지)
- events.parquet       : event_id, user_id, session_id, timestamp, event_type, product_id, query, source, impression_id
- impressions.parquet  : impression_id, user_id, session_id, timestamp, source, context, product_ids, clicked_ids
"""

from __future__ import annotations

import argparse
import copy
import time
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from src.common.config import ROOT, load_config
from src.common.seed import set_seed
from src.simulator import item_space
from src.simulator.calibrate import calibrated_personas
from src.simulator.catalog import load_catalog
from src.simulator.pipeline import load_inputs, load_world, simulate
from src.simulator.validate import build_report, print_report, save_report


def check_dates(cfg: dict[str, Any], cat_cfg: dict[str, Any]) -> None:
    last = np.datetime64(cfg["period"]["start_date"], "D") + cfg["period"]["days"] - 1
    ref = np.datetime64(cat_cfg["created_at"]["reference_date"], "D")
    if last != ref:
        print(f"  [warn] 로그 마지막 날({last})과 상품 등록 기준일({ref})이 다릅니다 → 신상품 구간이 어긋남")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--users", type=int, default=None, help="고객 수 (기본: scale.n_users)")
    parser.add_argument("--engine", choices=["clip", "attribute"], default=None, help="검색 엔진 (기본: search.engine)")
    parser.add_argument("--rebuild-space", action="store_true", help="상품 지도를 다시 학습")
    parser.add_argument("--recalibrate", action="store_true", help="Retailrocket 보정을 다시")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    cfg = copy.deepcopy(load_config("simulator"))
    cat_cfg = load_config("catalog")
    if args.users:
        cfg["scale"]["n_users"] = args.users
    if args.engine:
        cfg["search"]["engine"] = args.engine
    set_seed(cfg["seed"])
    check_dates(cfg, cat_cfg)
    t0 = time.perf_counter()

    if args.rebuild_space or not item_space.exists(cfg):
        print("[simulator] 상품 지도 학습 (H&M 구매 기록)")
        item_space.build_item_space(cfg, cat_cfg)
    world = load_world(cfg)
    inputs = load_inputs(cfg, cat_cfg)
    personas, calibration = calibrated_personas(cfg, world, inputs, force=args.recalibrate)

    print(f"[simulator] 고객 {cfg['scale']['n_users']:,}명 행동 생성 시작 ({time.perf_counter() - t0:.0f}s)")
    pop, events, impressions = simulate(cfg, world, personas, inputs, progress=True)

    out = ROOT / cfg["paths"]["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    pop.public.to_parquet(out / "users.parquet", index=False)
    pop.hidden.drop(columns=["signup_day"]).to_parquet(out / "users_hidden.parquet", index=False)
    events.to_parquet(out / "events.parquet", index=False)
    pq.write_table(impressions, out / "impressions.parquet")
    print(f"[simulator] 저장 → {out.relative_to(ROOT)} (users, users_hidden, events, impressions) "
          f"({time.perf_counter() - t0:.0f}s)")

    if not args.no_report:
        report = build_report(cfg, load_catalog(cfg), pop.public, pop.hidden, events, impressions.to_pandas(),
                              rr_targets=calibration["targets"])
        save_report(cfg, report)
        print_report(report)


if __name__ == "__main__":
    main()
