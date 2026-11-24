import pandas as pd
import plotly.express as px
import streamlit as st
from metrics import select_result

st.header("A/B 테스트")
path, data = select_result(["abtest*.json"], "A/B 실험 결과",
                           "python -m src.abtest.simulate", "ab_quality")
if data.get("source") != "simulation":
    st.warning("이 결과는 생성 출처를 확인할 수 없습니다. 아래 메타데이터를 확인하세요.")
else:
    st.info("가상 고객 반응으로 측정한 시뮬레이션입니다. 실제 고객의 클릭·구매율 개선을 입증하는 결과는 아닙니다.")
if data.get("run_scope") == "smoke_test_not_final_performance":
    st.warning("작은 표본으로 동작과 재현성만 확인한 검증 실행입니다. 최종 성능 비교에는 전체 실험을 실행하세요.")
st.caption(f"모델: {data.get('model_version', '미기록')} · seed: {data.get('config', {}).get('seed', '미기록')} · "
           f"기간: {data.get('start_at', '미기록')} ~ {data.get('end_at', '미기록')}")
groups = data.get("groups", {})
if not all(isinstance(groups.get(name), dict) for name in ("control", "treatment")):
    st.warning("두 실험 그룹의 집계가 없습니다.")
    st.stop()
rows = [{"그룹": name, **{k: g.get(k) for k in ("strategy", "assigned_users", "exposed_users", "clicked_users",
         "purchased_users", "assigned_user_cvr", "user_ctr", "user_cvr", "item_ctr")}} for name, g in groups.items()]
frame = pd.DataFrame(rows)
st.dataframe(frame, use_container_width=True, hide_index=True)
rates = frame.melt(id_vars=["그룹"], value_vars=["assigned_user_cvr", "user_ctr", "user_cvr", "item_ctr"],
                   var_name="지표", value_name="비율")
st.plotly_chart(px.bar(rates, x="지표", y="비율", color="그룹", barmode="group"), use_container_width=True)
st.caption("주 지표: 배정된 전체 사용자 중 구매한 사용자 비율(ITT). 노출 사용자 기준 비율은 보조 지표이며, "
           "반복 노출이 있는 item_ctr는 기술 통계로만 표시합니다.")
comparisons = []
for metric, values in data.get("comparisons", {}).items():
    if values.get("status") != "ok":
        st.warning(f"{metric}: 비교할 노출 표본이 부족합니다.")
        continue
    comparisons.append({"지표": metric, **{k: values.get(k) for k in
                        ("control_rate", "treatment_rate", "diff", "lift", "p_value", "ci_low", "ci_high", "significant")}})
    if values.get("small_counts"):
        st.warning(f"{metric}: 전환 또는 비전환 수가 5건 미만인 그룹이 있어 Z 검정·Wald 구간 근사가 불안정합니다.")
if comparisons:
    comparisons = pd.DataFrame(comparisons)
    st.dataframe(comparisons, use_container_width=True, hide_index=True)
    comparisons["상단"] = comparisons["ci_high"] - comparisons["diff"]
    comparisons["하단"] = comparisons["diff"] - comparisons["ci_low"]
    fig = px.scatter(comparisons, x="diff", y="지표", error_x="상단", error_x_minus="하단",
                     labels={"diff": "Treatment − Control 비율 차이"})
    fig.add_vline(x=0, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)
    confidence = 1 - float(data.get("config", {}).get("alpha", .05))
    st.caption(f"오차 막대: 비율 차이의 {confidence:.0%} 신뢰구간. Lift는 상대 변화, diff는 비율 차이입니다.")
daily = [{"날짜": day, "그룹": name, **counts} for name, g in groups.items()
         for day, counts in g.get("daily", {}).items()]
if daily:
    st.subheader("일별 실험 관측량")
    st.plotly_chart(px.line(pd.DataFrame(daily).sort_values("날짜"), x="날짜", y="impressions", color="그룹",
                           labels={"impressions": "노출 목록 수"}), use_container_width=True)
with st.expander("지표 정의와 실험 조건"):
    st.json({key: data.get(key) for key in ("definitions", "config", "history_source", "response_calibration",
                                         "run_scope", "reproducibility_check", "limitations")})
