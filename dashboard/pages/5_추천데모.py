"""추천 데모: 사용자를 고르면 Multi-Stage 추천 결과를 상품 사진으로 보여주고, 클릭을 바로 반영한다.

- GET  /api/recommend             : 추천 (at, session_id)
- POST /api/events                : 클릭(view) → 세션 클릭 저장, 탐색 칸 상품이면 MAB 클릭 기록
- GET  /api/products/{id}, /image : 상품 정보와 사진

API 서버와 학습된 추천 모델이 필요하다: make docker-train → docker compose up --build
"""

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests
import streamlit as st

API = os.getenv("API_BASE_URL", "http://localhost:8000")
TOP_N = 10
GRID_COLUMNS = 5
HISTORY_SHOWN = 10
# 로그 데이터가 끝나는 시점. 실제 현재 시각을 쓰면 등록 7일 이내 상품이 없어서 신규 상품 칸이 동작하지 않는다.
DEMO_AT = "2026-04-01T00:00:00+00:00"
NEW_USER = "NEW_USER"  # 데이터에 없는 ID → 이력 없는 사용자
USERS = ["U00005", "U00020", "U00036", "U00002", NEW_USER]
DIRECT = "직접 입력"
REASON = {
    "ctcvr": ("구매 확률순", "gray"),
    "ctr": ("클릭 확률순", "gray"),
    "cvr": ("구매 확률순", "gray"),
    "mab_exploration": ("탐색 칸 (MAB)", "orange"),
    "new_item": ("신규 상품 칸", "green"),
    "popular": ("인기", "blue"),
    "trending": ("트렌딩", "violet"),
}
COLD_START = {"popular", "trending"}


@st.cache_data(show_spinner=False)
def product_info(product_id: str) -> dict | None:
    r = requests.get(f"{API}/api/products/{product_id}", timeout=30)
    return r.json() if r.status_code == 200 else None


@st.cache_data(show_spinner=False)
def product_image(product_id: str) -> bytes | None:
    r = requests.get(f"{API}/api/products/{product_id}/image", timeout=30)
    return r.content if r.status_code == 200 else None


def load_products(ids: list[str]) -> dict[str, tuple[dict | None, bytes | None]]:
    with ThreadPoolExecutor(max_workers=8) as pool:
        infos, images = list(pool.map(product_info, ids)), list(pool.map(product_image, ids))
    return dict(zip(ids, zip(infos, images)))


def fetch_recommendation(user_id: str, session_id: str) -> dict:
    params = {"user_id": user_id, "top_n": TOP_N, "at": DEMO_AT, "session_id": session_id}
    r = requests.get(f"{API}/api/recommend", params=params, timeout=60)
    if r.status_code == 503:
        raise RuntimeError("추천 모델이 없어요. `make docker-train`으로 먼저 학습해 주세요.")
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code}: {r.text}")
    return r.json()


def start_visit(user_id: str) -> None:
    st.session_state.update(user_id=user_id, session_id=f"demo-{uuid.uuid4().hex[:12]}", clicks=0, fetched=None,
                            result=None, previous=None, history=[])


def record_click(product_id: str) -> None:
    body = {"event_id": f"demo-{uuid.uuid4().hex}", "user_id": st.session_state.user_id,
            "session_id": st.session_state.session_id, "event_type": "view", "product_id": product_id,
            "timestamp": datetime.now(timezone.utc).isoformat(), "source": "home"}
    try:
        requests.post(f"{API}/api/events", json=body, timeout=30).raise_for_status()
    except requests.RequestException as e:
        st.session_state.click_error = str(e)
        return
    st.session_state.clicks += 1


def score_text(item: dict, is_cold: bool) -> str:
    # cold start는 탐색 칸도 인기·트렌딩 목록에서 고르므로 점수가 구매 확률이 아니다. 신규 상품 칸만 DeepFM 점수.
    if is_cold and item["reason"] != "new_item":
        return f"인기·트렌딩 점수 {item['score']:.2f}"
    kind = "클릭" if item["reason"] == "ctr" else "구매"  # Stage 2 학습 목표 (configs/recommend.yaml ranking.objective)
    return f"{kind} 확률 추정 {item['score']:.2%}"


def product_card(item: dict, product: tuple[dict | None, bytes | None], is_changed: bool, is_cold: bool) -> None:
    info, image = product
    if image:
        st.image(image, width="stretch")
    label, color = REASON.get(item["reason"], (item["reason"], "gray"))
    st.markdown(f":{color}-badge[{label}]" + (" :red-badge[새로 추천]" if is_changed else ""))
    score = score_text(item, is_cold)
    if info:
        st.caption(f"**{info['name']}**  \n{info['category']} · {info['color']} · ${info['price']:.2f}  \n{score}")
    else:
        st.caption(f"**{item['product_id']}**  \n{score}")
    st.button("이 상품 보기", key=f"click_{item['product_id']}", on_click=record_click, args=(item["product_id"],),
              width="stretch")


def thumbnails(product_ids: list[str]) -> None:
    ids = list(reversed(product_ids))[:HISTORY_SHOWN]  # 최근 본 순서
    products = load_products(ids)
    for col, product_id in zip(st.columns(HISTORY_SHOWN), ids):
        with col:
            info, image = products[product_id]
            if image:
                st.image(image, width="stretch")
            st.caption(info["name"] if info else product_id)


st.set_page_config(page_title="추천 데모", layout="wide")
st.header("추천 데모 — 사용자별 추천과 클릭 반영")
st.caption(f"API: {API}/api/recommend · 후보(Two-Tower + 세션 GRU) → 정렬(DeepFM) → 재정렬(다양성·신규 칸·MAB 탐색 칸)")

controls = st.columns([2, 2, 1], vertical_alignment="bottom")
with controls[0]:
    choice = st.selectbox("사용자", [*USERS, DIRECT],
                          format_func=lambda u: "NEW_USER (이력 없는 신규 사용자)" if u == NEW_USER else u)
user_id = choice
if choice == DIRECT:
    with controls[1]:
        user_id = st.text_input("사용자 ID", value="U00001").strip() or "U00001"
if st.session_state.get("user_id") != user_id:
    start_visit(user_id)
with controls[2]:
    st.button("새 방문 시작", on_click=start_visit, args=(user_id,), width="stretch", help="이번 방문의 클릭을 비우고 처음부터 다시 추천받아요.")

if error := st.session_state.pop("click_error", None):
    st.error(f"클릭을 보내지 못했어요: {error}")

if st.session_state.fetched != st.session_state.clicks:
    try:
        with st.spinner("추천 중..."):
            result = fetch_recommendation(user_id, st.session_state.session_id)
    except requests.ConnectionError:
        st.error(f"API 서버에 연결할 수 없어요 ({API}). `docker compose up --build`로 먼저 띄워 주세요.")
        st.stop()
    except RuntimeError as e:
        st.error(f"추천 실패: {e}")
        st.stop()
    st.session_state.update(previous=st.session_state.result, result=result, fetched=st.session_state.clicks)
    if st.session_state.clicks == 0:  # 클릭 전에는 저장된 과거 조회가 온다
        st.session_state.history = result["session_context"]["recent_clicks"]

result = st.session_state.result
items = result["recommendations"]
latency = result["pipeline_latency"]
for col, (label, value) in zip(st.columns(4), (("전체 응답", latency["total_ms"]), ("후보", latency["candidate_ms"]),
                                               ("정렬", latency["ranking_ms"]), ("재정렬", latency["reranking_ms"]))):
    col.metric(label, f"{value:.0f} ms")

previous = st.session_state.previous
previous_ids = {i["product_id"] for i in previous["recommendations"]} if previous else None
changed = [i for i in items if previous_ids is not None and i["product_id"] not in previous_ids]

st.subheader(f"추천 상품 {len(items)}개")
st.caption(f"기준 시각 {DEMO_AT[:10]} (로그 데이터가 끝나는 시점) · 확률 추정값은 정렬용 점수이고, 실제 클릭·구매율로 검증하지 않았어요.")
is_cold = any(i["reason"] in COLD_START for i in items)
if is_cold:
    st.info("조회 이력이 적어 인기·트렌딩 상품으로 대체했어요 (cold start).")
if previous_ids is not None:
    main = sum(not i["is_exploration"] for i in changed)
    st.success(f"직전 추천과 비교: {len(items)}개 중 {len(changed)}개가 새로 들어왔어요 "
               f"(일반 칸 {main}개, 탐색·신규 칸 {len(changed) - main}개). "
               "탐색 칸은 MAB 통계에 따라 클릭이 없어도 요청마다 바뀌므로, 이 변화가 모두 클릭 효과는 아니에요.")

products = load_products([i["product_id"] for i in items])
changed_ids = {i["product_id"] for i in changed}
for start in range(0, len(items), GRID_COLUMNS):
    row = items[start:start + GRID_COLUMNS]
    for col, item in zip(st.columns(GRID_COLUMNS), row):
        with col:
            product_card(item, products[item["product_id"]], item["product_id"] in changed_ids, is_cold)

context = result["session_context"]
if st.session_state.clicks:
    st.subheader("이번 방문에서 본 상품")
    interest = f" · 현재 관심 카테고리: {context['session_interest']}" if context["session_interest"] else ""
    st.caption(f"세션 GRU의 입력이에요. 이 상품들과 비슷한 상품을 후보에 더해요{interest}")
    thumbnails(context["recent_clicks"])

st.subheader("과거 최근 조회")
if st.session_state.history:
    st.caption("Two-Tower의 입력이에요. 저장된 로그(3월 22일까지)에서 읽어서 이번 방문의 클릭은 들어가지 않아요.")
    thumbnails(st.session_state.history)
else:
    st.caption("저장된 조회 기록이 없어요.")
