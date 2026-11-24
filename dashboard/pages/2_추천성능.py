import pandas as pd
import plotly.express as px
import streamlit as st
from metrics import (
    ARTIFACTS,
    read_json,
    recommendation_rows,
    recommendation_runs,
    select_result,
)

st.header("추천 성능")
path, payload = select_result(["recommend_final*.json", "recommend_eval*.json"], "최종 추천 평가",
                              "python -m src.evaluation.eval_pipeline --target both", "recommend_quality")
runs = recommendation_runs(payload)
choice = st.radio("정답 기준", list(runs), horizontal=True) if len(runs) > 1 else next(iter(runs))
data = runs[choice]
st.caption(f"모델: {data.get('model_version', payload.get('model_version', '미기록'))} · "
           f"정답: {data.get('target', choice)} · 평가 건수: {data.get('n_queries', '미기록'):,}"
           if isinstance(data.get("n_queries"), int) else f"정답: {data.get('target', choice)}")
rows = recommendation_rows(data)
if not rows:
    st.warning("이 파일에는 최종 추천 지표가 없습니다.")
    st.stop()
frame = pd.DataFrame(rows)
k = data.get("k", 50)
labels = {"hit_rate": f"HitRate@{k}", "ndcg": f"NDCG@{k}", "coverage": "Coverage", "latency_p95_ms": "p95 (ms)"}
st.dataframe(frame.rename(columns=labels), use_container_width=True, hide_index=True)
quality = frame.melt(id_vars=["전략"], value_vars=["hit_rate", "ndcg", "coverage"], var_name="지표", value_name="값")
quality["지표"] = quality["지표"].map(labels)
st.plotly_chart(px.bar(quality, x="전략", y="값", color="지표", barmode="group"), use_container_width=True)
targets = data.get("targets", {})
if targets:
    st.caption("명세 목표: " + " · ".join(f"{labels.get(key, key)} ≥ {value}" for key, value in targets.items()))
st.plotly_chart(px.bar(frame, x="전략", y="latency_p95_ms", labels=labels), use_container_width=True)
st.info("위 지연시간은 오프라인 전략 호출의 처리시간입니다. HTTP·실제 Redis를 포함한 API p95와 구분합니다.")
if not data.get("history_source") and not data.get("evaluation_history"):
    st.caption("기존 artifact는 실시간 이력 격리 여부를 기록하지 않았습니다. 새로운 고정 평가와 구분해 해석하세요.")

version = data.get("model_version", payload.get("model_version"))
if isinstance(version, str) and "/" not in version and "\\" not in version:
    trained = ARTIFACTS / "models" / version / "metrics.json"
    if trained.is_file():
        try:
            metrics = read_json(trained)
        except (OSError, ValueError, TypeError) as exc:
            st.warning(f"학습 평가 결과를 읽지 못했습니다: {exc}")
        else:
            st.subheader("같은 모델 버전의 후보·랭킹 평가")
            a, b = st.columns(2)
            recall, auc = metrics.get("retrieval", {}).get("recall_at_300"), metrics.get("ranking", {}).get("auc")
            a.metric("Recall@300 · 목표 0.30", f"{recall:.4f}" if recall is not None else "미측정")
            b.metric("Ranking AUC · 목표 0.70", f"{auc:.4f}" if auc is not None else "미측정")
            st.caption(metrics.get("label_definition", "학습 정답 정의 미기록"))
with st.expander("평가 재현 조건"):
    st.json({key: data.get(key, payload.get(key)) for key in
             ("model_version", "target", "session", "seed", "history_source", "evaluation_history", "measured_at")})

session_file = ARTIFACTS / "metrics" / "session_encoder.json"
if session_file.is_file():
    with st.expander("세션 인코더 별도 검증"):
        try:
            session = read_json(session_file)
            st.caption(f"세션 모델: {session.get('config', {}).get('model_file', '미기록')} · "
                       "최종 파이프라인 평가와 별도 검증셋 결과")
            st.dataframe(pd.DataFrame(session.get("valid", {})).T, use_container_width=True)
            if session.get("loss"):
                st.line_chart(pd.DataFrame({"학습 손실": session["loss"]}))
        except (OSError, ValueError, TypeError) as exc:
            st.warning(f"세션 평가를 읽지 못했습니다: {exc}")
