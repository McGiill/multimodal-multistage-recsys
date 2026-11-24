"""저장된 평가 artifact를 읽는 대시보드 공통 함수. 모델을 불러오거나 평가를 실행하지 않는다."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / os.getenv("ARTIFACT_DIR", "artifacts")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("평가 결과의 최상위 값은 JSON 객체여야 합니다")
    return payload


def select_result(patterns: list[str], label: str, command: str, key: str,
                  preferred: str | None = None) -> tuple[Path, dict]:
    import streamlit as st

    paths = set()
    for pattern in patterns:
        paths.update((ARTIFACTS / "metrics").glob(pattern))
    ordered = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
    if not ordered:
        st.info("아직 저장된 평가 결과가 없습니다. 평가를 실행하면 이 화면에서 확인할 수 있습니다.")
        st.code(command, language="bash")
        st.stop()
    initial = next((i for i, p in enumerate(ordered) if p.name == preferred), 0)
    path = st.selectbox(label, ordered, index=initial, format_func=lambda p: p.name, key=key)
    try:
        data = read_json(path)
    except (OSError, ValueError, TypeError) as exc:
        st.error(f"평가 파일을 읽지 못했습니다: {exc}")
        st.stop()
    st.caption(f"파일: {path.name} · 측정: {data.get('measured_at', data.get('evaluated_at', '미기록'))}")
    return path, data


def recommendation_runs(payload: dict[str, Any]) -> dict[str, dict]:
    """과거 단일 조회 평가와 조회/구매를 함께 저장한 새 결과 모두 지원."""
    nested = {k: payload[k] for k in ("view", "purchase") if isinstance(payload.get(k), dict)}
    return nested or {"평가 결과": payload}


def recommendation_rows(payload: dict[str, Any]) -> list[dict]:
    return [{"전략": name, **{k: values.get(k) for k in ("hit_rate", "ndcg", "coverage", "latency_p95_ms")}}
            for name, values in payload.items() if isinstance(values, dict) and "hit_rate" in values]


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
