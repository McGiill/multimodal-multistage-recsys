"""스타일 사진 검색 데모: 패션 사진 (+ 텍스트) → 사진 속 옷마다 비슷한 상품 (POST /api/search/style).

API 서버가 떠 있어야 한다: uvicorn src.serving.main:app --port 8000
상품 사진은 GET /api/products/{product_id}/image로 받아서 보여준다 (대시보드는 H&M 원본 폴더를 직접 읽지 않음).

검색할 영역: 자동(사진 속 옷 전부) / 찾은 옷 중 하나 / 직접 지정(사진 위를 드래그).
원하는 옷을 못 찾았거나 틀렸을 때 사용자가 짚어 주면 그 영역의 옷만 검색한다 (API의 box).
"""

import base64
import hashlib
import io
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

API = os.getenv("API_BASE_URL", "http://localhost:8000")
EXAMPLES = ["", "이런 스타일 옷 추천해줘", "이 옷이랑 어울리는 신발", "이 옷이랑 어울리는 바지", "하의 검은색으로",
            "소매 더 짧게", "상의 빨간색으로"]
AUDIENCE = {"자동": "auto", "여성": "women", "남성": "men", "아동": "kids", "전체": "all"}
INTENT = {"style": "사진 속 옷 전부", "find": "지정한 옷 찾기", "match": "어울리는 옷 찾기"}
PHOTO = {"style": "착용 사진 (옷 검출)", "product": "상품 사진 (사진 전체가 기준)"}
BASIS = {"photo_item": "사진 속 옷과 비슷한 상품", "photo_style": "사진에 없는 종류 → 사진 분위기와 비슷한 상품",
         "whole_photo": "사진 전체와 비슷한 상품"}
SOURCE = {"low_score": "요청한 부위라 점수가 낮은 검출도 사용", "grounding": "Grounding DINO로 찾음",
          "user": "직접 고른 영역"}
FILTER = {"color": "색", "sleeve": "소매", "length": "기장", "rise": "밑위", "fit": "핏"}
ARROW = {"up": "↑", "down": "↓"}
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c"]
AUTO, DRAW = "auto", "draw"
PICKER_SIZE = 700  # 직접 지정 화면에 보낼 사진 크기 (긴 변, 픽셀)
PICKER_WIDTH = 340  # 직접 지정 화면의 표시 너비 (픽셀)


@st.cache_data(show_spinner=False)
def style_search(image_bytes: bytes, query: str, audience: str, top_k: int,
                 box: tuple[int, int, int, int] | None = None) -> dict:
    body = {"image_base64": base64.b64encode(image_bytes).decode(), "query_text": query or None,
            "audience": audience, "top_k": top_k, "box": list(box) if box else None}
    r = requests.post(f"{API}/api/search/style", json=body, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code}: {r.text}")
    return r.json()


@st.cache_data(show_spinner=False)
def product_image(product_id: str) -> bytes | None:
    r = requests.get(f"{API}/api/products/{product_id}/image", timeout=30)
    return r.content if r.status_code == 200 else None


def draw_boxes(image: Image.Image, groups: list[dict]) -> Image.Image:
    shown = image.copy()
    draw = ImageDraw.Draw(shown)
    width = max(2, shown.width // 200)
    for i, g in enumerate(g for g in groups if g["box"]):
        color = COLORS[i % len(COLORS)]
        draw.rectangle(g["box"], outline=color, width=width)
        draw.text((g["box"][0] + width * 2, g["box"][1] + width * 2), g["slot_name"], fill=color)
    return shown


def box_picker(image: Image.Image, key: str) -> tuple[int, int, int, int] | None:
    """사진 위를 드래그해서 영역 지정 (plotly 박스 선택). 원본 사진의 픽셀 좌표로 돌려준다."""
    scale = min(1.0, PICKER_SIZE / max(image.size))
    shown = image.resize((round(image.width * scale), round(image.height * scale))) if scale < 1 else image
    w, h = shown.size
    buf = io.BytesIO()
    shown.save(buf, format="JPEG", quality=85)
    fig = go.Figure(go.Image(source="data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()))
    # 사진만 있으면 plotly가 박스 선택을 받지 않아서, 보이지 않는 점을 촘촘히 깔아 둔다
    xs, ys = np.meshgrid(np.linspace(0, w, 40), np.linspace(0, h, 40))
    invisible = {"marker": {"opacity": 0}}
    fig.add_trace(go.Scatter(x=xs.ravel(), y=ys.ravel(), mode="markers", hoverinfo="none", marker={"opacity": 0},
                             selected=invisible, unselected=invisible))
    fig.update_layout(dragmode="select", showlegend=False, margin={"l": 0, "r": 0, "t": 0, "b": 0},
                      width=PICKER_WIDTH, height=round(PICKER_WIDTH * h / w),  # 사진 비율 그대로 (위아래 빈칸 없이)
                      xaxis={"visible": False, "range": [0, w]},
                      yaxis={"visible": False, "range": [h, 0], "scaleanchor": "x"})
    event = st.plotly_chart(fig, key=key, on_select="rerun", selection_mode="box", width="content",
                            config={"displayModeBar": False})
    boxes = event.selection.box if event else []
    if not boxes:
        return None
    (x1, x2), (y1, y2) = sorted(boxes[-1]["x"]), sorted(boxes[-1]["y"])
    return tuple(round(v / scale) for v in (x1, y1, x2, y2))


def search_or_stop(photo: bytes, box: tuple[int, int, int, int] | None = None) -> dict:
    try:
        with st.spinner("검색 중..."):
            return style_search(photo, query.strip(), AUDIENCE[audience], top_k, box)
    except requests.ConnectionError:
        st.error(f"API 서버에 연결할 수 없어요 ({API}). `uvicorn src.serving.main:app --port 8000`으로 먼저 띄워 주세요.")
    except RuntimeError as e:
        st.error(f"검색 실패: {e}")
    st.stop()


st.set_page_config(page_title="스타일 사진 검색", layout="wide")
st.header("스타일 사진 검색 — 패션 사진 + 요청 → 사진 속 옷과 비슷한 상품")
st.caption(f"API: {API}/api/search/style · 상품은 H&M 카탈로그 50,000개")

left, right = st.columns([1, 2], gap="large")
with left:
    upload = st.file_uploader("패션 사진 (사람이 옷을 입은 사진 또는 상품 사진)", type=["jpg", "jpeg", "png", "webp"])
    example = st.selectbox("예시 요청", EXAMPLES, format_func=lambda t: t or "(요청 없음: 사진 속 옷 전부)")
    query = st.text_input("요청 (선택, 직접 입력 가능)", value=example, key=f"query_{example}",
                          placeholder="예: 이 옷이랑 어울리는 바지 / 소매 더 짧게 / 빨간색으로")
    audience = st.radio("고객층", list(AUDIENCE), horizontal=True, help="자동: 착용 사진이면 사진 속 사람으로 판단")
    top_k = st.slider("부위별 결과 수", 4, 20, 8, step=4)

if upload is None:
    right.info("왼쪽에 패션 사진을 올려 주세요.")
    st.stop()

image = ImageOps.exif_transpose(Image.open(io.BytesIO(upload.getvalue()))).convert("RGB")
buf = io.BytesIO()
image.save(buf, format="JPEG", quality=92)  # 휴대폰 사진의 회전 정보(EXIF)를 반영한 사진으로 보냄
photo = buf.getvalue()
digest = hashlib.md5(photo).hexdigest()[:12]

base = search_or_stop(photo)  # 자동: 사진 속 옷 전부 (또는 요청대로)
detections = base["detections"]


def area_label(option: str | int) -> str:
    if option == AUTO:
        return "자동 (요청대로)" if query.strip() else "자동 (사진 속 옷 전부)"
    if option == DRAW:
        return "직접 지정 (사진 위를 드래그)"
    d = detections[option]
    note = SOURCE.get(d["source"], f"점수 {d['score']:.2f}")
    return f"{d['slot_name']} · {d['kind']} ({note})"


with left:
    area = st.radio("검색할 영역", [AUTO, *range(len(detections)), DRAW], format_func=area_label,
                    key=f"area_{digest}_{query.strip()}",
                    help="원하는 옷을 못 찾았거나 틀렸으면 직접 지정해 주세요. 지정한 영역의 옷만 검색해요.")
    box = None
    if area == DRAW:
        box = box_picker(image, key=f"picker_{digest}")
        if box is None:
            st.caption("사진 위를 드래그해서 찾을 옷을 감싸 주세요. 그 전까지는 자동 결과를 보여줘요.")
    elif area != AUTO:
        box = tuple(detections[area]["box"])

res = search_or_stop(photo, box) if box else base

with left:
    if area != DRAW:
        st.image(draw_boxes(image, res["groups"]), width="stretch")
    st.markdown(
        f"- 요청 해석: **{INTENT.get(res['intent'], res['intent'])}**\n"
        f"- 사진 종류: **{PHOTO.get(res['photo_type'], res['photo_type'])}**\n"
        f"- 고객층 제한: **{res['audience'] or '없음'}**\n"
        f"- 응답시간: **{res['latency_ms']:.0f}ms**"
    )

with right:
    ids = [item["product_id"] for g in res["groups"] for item in g["results"]]
    with ThreadPoolExecutor(max_workers=8) as pool:
        images = dict(zip(ids, pool.map(product_image, ids)))
    if not res["groups"]:
        st.warning("결과가 없어요.")
    for g in res["groups"]:
        st.subheader(f"{g['slot_name']} · {BASIS.get(g['basis'], g['basis'])}")
        notes = [SOURCE[g["box_source"]]] if g["box_source"] in SOURCE else []
        if g["filters"]:  # "color:red"를 그대로 쓰면 Streamlit 마크다운이 ":red"를 색 문법으로 먹어서 바꿔 씀
            notes.append("조건: " + ", ".join(FILTER.get(k, k) + " " + ARROW.get(v, v)
                                              for k, v in (f.split(":", 1) for f in g["filters"])))
        notes.append(f"후보 {g['total_count']:,}개 중 상위 {len(g['results'])}개")
        st.caption(" · ".join(notes))
        if not g["results"]:
            st.info("조건에 맞는 상품이 없어요 (색·디테일 조건을 바꿔 보세요).")
            continue
        cols = st.columns(4)
        for i, item in enumerate(g["results"]):
            with cols[i % 4]:
                if images.get(item["product_id"]):
                    st.image(images[item["product_id"]], width="stretch")
                st.caption(f"**{item['name']}**  \n{item['category']} · {item['color']} · ${item['price']:.2f}  \n"
                           f"유사도 {item['score']:.3f}")
