"""train / valid / test = 8 / 1 / 1 시간 기반 분할 (최근 데이터 → test): python -m src.evaluation.split

기준
- 로그 기간(configs/simulator.yaml의 period, 90일)을 날짜로 8:1:1 → train 72일 / valid 9일 / test 9일
- 방문(session) 단위로 나눈다. 방문이 시작된 날짜의 구간으로 방문 전체를 보낸다
  (자정을 걸친 방문이 둘로 쪼개지면 세션 기반 모델이 잘린 방문을 배우게 된다)
- 고객 단위로 나누지 않는다. 같은 고객의 과거 행동(train)으로 미래 행동(test)을 맞히는 구조다.
  로그 기간 중에 가입해서 train 이력이 없는 고객은 test에서 신규 고객(cold start)이 된다.
- 무작위 요소가 없어서 같은 로그면 항상 같은 결과다 (seed 불필요).
- users.parquet(공개 정보)은 나누지 않고 그대로 쓴다. users_hidden.parquet은 모델 입력 금지라 여기서 다루지 않는다.

입력: data/processed/events.parquet, impressions.parquet (python -m src.simulator.run)
출력: data/splits/
    {train,valid,test}_events.parquet       행동 로그 (search / view / cart / purchase)
    {train,valid,test}_impressions.parquet  노출 로그 (보여준 상품 목록, 클릭한 상품) → Ranking 음성 데이터
    meta.json                               구간 날짜, 건수와 비율, 신규 고객·처음 보는 상품 수
읽기: from src.evaluation.split import load_split
    train = load_split("train")                    # 행동 로그
    test_imp = load_split("test", "impressions")   # 노출 로그
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.common.config import ROOT, load_config

SPLITS = ("train", "valid", "test")
RATIOS = (0.8, 0.1, 0.1)  # 명세서 고정
TABLES = ("events", "impressions")
OUT_DIR = ROOT / "data" / "splits"


def boundaries(sim_cfg: dict[str, Any]) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """구간마다 [시작, 끝) 날짜. 90일이면 train 72일 / valid 9일 / test 9일."""
    start = pd.Timestamp(sim_cfg["period"]["start_date"])
    days = int(sim_cfg["period"]["days"])
    n_train = round(days * RATIOS[0])
    n_valid = round(days * RATIOS[1])
    edges = [start + pd.Timedelta(days=d) for d in (0, n_train, n_train + n_valid, days)]
    return {name: (edges[i], edges[i + 1]) for i, name in enumerate(SPLITS)}


def session_labels(sessions: pd.Series, timestamps: pd.Series,
                   bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]]) -> pd.Series:
    """방문 → 구간 이름. 방문의 첫 기록(이벤트 또는 노출) 시각이 속한 구간.

    마지막 날 자정 직전에 시작한 방문은 첫 기록이 다음 날 0시 몇 분일 수 있어서, 기간 끝 이후는 마지막 구간(test)에 넣는다.
    """
    first = timestamps.groupby(sessions.to_numpy()).min()
    edges = [bounds[name][0] for name in SPLITS] + [pd.Timestamp.max]
    labels = pd.cut(first, bins=edges, right=False, labels=list(SPLITS))
    if labels.isna().any():
        raise ValueError(f"로그 시작일 전에 시작한 방문 {int(labels.isna().sum())}개 (period 설정 확인)")
    return labels.astype(str)


def split_logs(events: pd.DataFrame, impressions: pa.Table,
               bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]]) -> dict[str, dict[str, Any]]:
    """events(DataFrame)와 impressions(pyarrow Table, 목록 컬럼 유지)를 방문 단위로 나눈다.

    노출만 있고 이벤트가 없는 방문(홈 목록만 보고 나감)도 있어서 두 로그를 합쳐 방문 시작 시각을 정한다.
    """
    imp_sessions = impressions.column("session_id").to_pandas()
    labels = session_labels(
        pd.concat([events["session_id"], imp_sessions], ignore_index=True),
        pd.concat([events["timestamp"], impressions.column("timestamp").to_pandas()], ignore_index=True),
        bounds,
    )
    ev_label = events["session_id"].map(labels).to_numpy()
    imp_label = imp_sessions.map(labels).to_numpy()
    return {
        name: {
            "events": events[ev_label == name].reset_index(drop=True),
            "impressions": impressions.filter(pa.array(imp_label == name)),
        }
        for name in SPLITS
    }


def summarize(parts: dict[str, dict[str, Any]], bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
              users: pd.DataFrame) -> dict[str, Any]:
    n_events = sum(len(p["events"]) for p in parts.values())
    train = parts["train"]["events"]
    train_users = set(train["user_id"])
    train_items = set(train["product_id"].dropna())
    out: dict[str, Any] = {"ratios": dict(zip(SPLITS, RATIOS)), "unit": "session (방문 시작 시각 기준)", "splits": {}}
    for name, p in parts.items():
        ev = p["events"]
        lo, hi = bounds[name]
        users_here = set(ev["user_id"])
        items_here = set(ev["product_id"].dropna())
        info: dict[str, Any] = {
            "from": str(lo.date()),
            "to": str((hi - pd.Timedelta(days=1)).date()),  # 마지막 날 포함
            "days": (hi - lo).days,
            "events": len(ev),
            "event_share": round(len(ev) / max(n_events, 1), 4),
            "events_by_type": {t: int(n) for t, n in ev["event_type"].value_counts().sort_index().items()},
            "sessions": int(ev["session_id"].nunique()),
            "impressions": p["impressions"].num_rows,
            "users": len(users_here),
        }
        if name != "train":
            info["users_without_train_history"] = len(users_here - train_users)   # cold start 대상
            info["items_unseen_in_train"] = len(items_here - train_items)        # 신규 상품 등
        out["splits"][name] = info
    out["users_total"] = len(users)
    return out


def load_split(name: str, table: str = "events") -> pd.DataFrame:
    """분할 결과 읽기. name: train / valid / test, table: events / impressions."""
    if name not in SPLITS or table not in TABLES:
        raise ValueError(f"name은 {SPLITS}, table은 {TABLES} 중 하나")
    path = OUT_DIR / f"{name}_{table}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path.relative_to(ROOT)} 없음 → make split 먼저 실행")
    return pd.read_parquet(path)


def main() -> None:
    sim_cfg = load_config("simulator")
    src = ROOT / sim_cfg["paths"]["output_dir"]
    events = pd.read_parquet(src / "events.parquet")
    impressions = pq.read_table(src / "impressions.parquet")
    users = pd.read_parquet(src / "users.parquet")

    bounds = boundaries(sim_cfg)
    parts = split_logs(events, impressions, bounds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, p in parts.items():
        p["events"].to_parquet(OUT_DIR / f"{name}_events.parquet", index=False)
        pq.write_table(p["impressions"], OUT_DIR / f"{name}_impressions.parquet")
    meta = summarize(parts, bounds, users)
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[split] 행동 로그 {len(events):,}건, 노출 {impressions.num_rows:,}건 → {OUT_DIR.relative_to(ROOT)}")
    for name, info in meta["splits"].items():
        extra = ""
        if name != "train":
            extra = (f", train 이력 없는 고객 {info['users_without_train_history']:,}명"
                     f", train에 없던 상품 {info['items_unseen_in_train']:,}개")
        print(f"  {name:5s} {info['from']} ~ {info['to']} ({info['days']}일)  이벤트 {info['events']:,} "
              f"({info['event_share']:.1%})  방문 {info['sessions']:,}  고객 {info['users']:,}{extra}")


if __name__ == "__main__":
    main()
