import base64
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from src.common.schemas import SearchType
from src.search.detector import Detection
from src.search.query_parser import Intent
from src.search.searcher import SearchResult
from src.search.style_search import StyleGroup, StyleSearchResult
from src.serving.main import app
from src.serving.routers.search import (
    get_product_images,
    get_searcher,
    get_style_searcher,
)


class FakeSearcher:
    """모델·인덱스 없이 API 형식만 확인하기 위한 가짜 검색기."""

    def search(self, query_text, image, top_k=10):
        search_type = (
            SearchType.HYBRID if query_text and image is not None else SearchType.IMAGE if image is not None else SearchType.TEXT
        )
        items = [{"product_id": "hm_0108775015", "name": "Strap top", "score": 0.91, "price": 25000}][:top_k]
        return SearchResult(search_type, [i["product_id"] for i in items], [0.91], 37, np.array([0]), items)


class FakeStyleSearcher:
    last_box = None

    def search(self, image, query_text=None, top_k=10, audience=None, box=None):
        FakeStyleSearcher.last_box = box
        item = {"product_id": "hm_0108775015", "name": "Strap top", "score": 0.88, "price": 9.99,
                "category": "Vest top", "color": "Black"}
        det = Detection("tee", "top", "top, t-shirt, sweatshirt", 0.93, (1, 2, 30, 40))
        group = StyleGroup("top", "tee", "photo_item", SearchResult(SearchType.IMAGE, ["hm_0108775015"], [0.88], 120,
                           np.array([0]), [item]), det.box, det.score, ["color:black"], det.source)
        return StyleSearchResult(Intent.FIND, True, "women", [group], [det])


app.dependency_overrides[get_searcher] = FakeSearcher
app.dependency_overrides[get_style_searcher] = FakeStyleSearcher
client = TestClient(app)


def _png_b64() -> str:
    buf = BytesIO()
    Image.new("RGB", (8, 8), "black").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_health() -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_recommend_schema() -> None:
    body = client.get("/api/recommend", params={"user_id": "U1234", "top_n": 10}).json()
    for key in ("user_id", "recommendations", "pipeline_latency", "session_context"):
        assert key in body


def test_search_json_text() -> None:
    body = client.post("/api/search", json={"query_text": "black strap top", "top_k": 10}).json()
    for key in ("search_type", "results", "latency_ms", "total_count"):
        assert key in body
    assert body["search_type"] == "text"
    assert set(body["results"][0]) == {"product_id", "name", "score", "price"}


def test_search_json_hybrid() -> None:
    body = client.post("/api/search", json={"query_text": "black", "image_base64": _png_b64()}).json()
    assert body["search_type"] == "hybrid"


def test_search_multipart_image() -> None:
    buf = BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="JPEG")
    r = client.post("/api/search", files={"image": ("q.jpg", buf.getvalue(), "image/jpeg")}, data={"top_k": "5"})
    assert r.status_code == 200
    assert r.json()["search_type"] == "image"


def test_search_requires_query() -> None:
    assert client.post("/api/search", json={"top_k": 10}).status_code == 422


def test_style_search_schema() -> None:
    body = client.post("/api/search/style", json={"image_base64": _png_b64(), "query_text": "검은 티셔츠"}).json()
    assert {"intent", "photo_type", "audience", "groups", "latency_ms"} <= set(body)
    assert body["photo_type"] == "style"
    group = body["groups"][0]
    assert group["slot"] == "top" and group["slot_name"] == "상의" and group["box"] == [1, 2, 30, 40]
    assert group["box_source"] == "detector"
    assert set(group["results"][0]) == {"product_id", "name", "score", "price", "category", "color"}
    assert body["detections"] == [{"kind": "tee", "slot": "top", "slot_name": "상의", "score": 0.93,
                                   "box": [1, 2, 30, 40], "source": "detector"}]


def test_style_search_box() -> None:
    image = _png_b64()  # 8 x 8
    assert client.post("/api/search/style", json={"image_base64": image, "box": [1, 1, 6, 7]}).status_code == 200
    assert FakeStyleSearcher.last_box == (1, 1, 6, 7)
    client.post("/api/search/style", json={"image_base64": image, "box": [-3, 0.4, 20, 8]})
    assert FakeStyleSearcher.last_box == (0, 0, 8, 8)  # 사진 안으로 자름
    r = client.post("/api/search/style", files={"image": ("q.png", base64.b64decode(image), "image/png")},
                    data={"box": "2,2,7,7"})
    assert r.status_code == 200 and FakeStyleSearcher.last_box == (2, 2, 7, 7)
    client.post("/api/search/style", json={"image_base64": image})
    assert FakeStyleSearcher.last_box is None
    for bad in ([1, 2, 3], "a,b,c,d", [5, 5, 6, 6], [10, 10, 20, 20]):
        assert client.post("/api/search/style", json={"image_base64": image, "box": bad}).status_code == 422


def test_style_search_requires_image_and_valid_audience() -> None:
    assert client.post("/api/search/style", json={"query_text": "바지"}).status_code == 422
    r = client.post("/api/search/style", json={"image_base64": _png_b64(), "audience": "teens"})
    assert r.status_code == 422


def test_product_image(tmp_path) -> None:
    path = tmp_path / "p.jpg"
    Image.new("RGB", (4, 4), "red").save(path, format="JPEG")
    app.dependency_overrides[get_product_images] = lambda: {"hm_0000000001": path}
    try:
        r = client.get("/api/products/hm_0000000001/image")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        assert client.get("/api/products/hm_9999999999/image").status_code == 404  # 카탈로그에 없는 상품
    finally:
        del app.dependency_overrides[get_product_images]
