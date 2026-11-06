"""Streamlit 대시보드: streamlit run dashboard/app.py  (http://localhost:8501)"""

import os

import streamlit as st

API = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Search & RecSys Dashboard", layout="wide")
st.title("멀티모달 검색 · Multi-Stage 추천 대시보드")
st.caption(f"API: {API}")
st.markdown(
    """
- **검색 데모**: 텍스트, 사진, 사진 + 텍스트, 착용 사진 속 옷별 검색 (스타일검색 메뉴)
- **추천 데모**: 사용자별 추천 결과, 추천 이유, 클릭 후 변화 (추천데모 메뉴)
- **검색 품질**: MRR, NDCG@10 (CLIP vs BM25)
- **추천 성능**: HitRate@50, NDCG@50, Recall@300, Coverage, AUC
- **A/B 테스트**: 전환율, p-value, 95% 신뢰구간
"""
)
