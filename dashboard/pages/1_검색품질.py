import pandas as pd
import plotly.express as px
import streamlit as st
from metrics import ARTIFACTS, read_json, select_result

st.header("검색 품질")
path, data = select_result(["search_*.json"], "검색 평가 결과", "make eval-search", "search_quality")
st.caption(f"모델: {data.get('model_name', '미기록')} · split: {data.get('split', '미기록')} · "
           f"정답 기준: {data.get('relevance', '미기록')} · seed: {data.get('seed', '미기록')}")
results = data.get("results", {})
if not results:
    st.warning("이 파일에는 검색 품질 지표가 없습니다.")
    st.stop()
k = data.get("k", 10)
frame = pd.DataFrame([{"방식": name, **metrics} for name, metrics in results.items()])
st.dataframe(frame, use_container_width=True, hide_index=True)
columns = [x for x in ("mrr", f"ndcg@{k}") if x in frame]
if columns:
    long = frame.melt(id_vars=["방식"], value_vars=columns, var_name="지표", value_name="값")
    fig = px.bar(long, x="방식", y="값", color="지표", barmode="group", range_y=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
overall = results.get("all", {})
if overall:
    checks = [{"지표": "MRR", "값": overall.get("mrr"), "명세 목표": .55},
              {"지표": "NDCG@10", "값": overall.get("ndcg@10"), "명세 목표": .50}]
    for row in checks:
        row["판정"] = "미측정" if row["값"] is None else ("통과" if row["값"] >= row["명세 목표"] else "미달")
    st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)
st.caption("서로 다른 정답 기준·평가셋·split의 점수는 같은 조건의 성능 개선으로 비교하지 않습니다.")

latencies = []
for candidate in (ARTIFACTS / "metrics").glob("latency_search*.json"):
    try:
        payload = read_json(candidate)
    except (OSError, ValueError, TypeError):
        continue
    if payload.get("model_name") == data.get("model_name"):
        latencies.append((candidate, payload))
if latencies:
    st.subheader("별도로 측정한 API 응답시간")
    selected = st.selectbox("지연시간 평가", sorted(latencies, key=lambda x: x[0].name),
                            format_func=lambda x: x[0].name)
    timing = selected[1]
    st.caption(f"측정: {timing.get('measured_at', '미기록')} · CPU 제한: {timing.get('cpu_limit', '미기록')} · "
               f"장치: {timing.get('device', '미기록')} · 품질 평가와 별도 실행")
    rows = [{"검색": mode, **timing[mode]} for mode in ("text", "image", "hybrid") if mode in timing]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("이 검색 모델과 일치하는 별도 지연시간 측정 결과가 없습니다.")

with st.expander("평가 조건"):
    st.json({key: data.get(key) for key in ("index", "weights", "eval_set", "mrr_min_grade", "seed")})
