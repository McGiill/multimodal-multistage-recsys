import base64
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from src.common.schemas import SearchType
from src.search.searcher import SearchResult
from src.serving.main import app
from src.serving.routers.search import get_searcher


class FakeSearcher:
    """모델·인덱스 없이 API 형식만 확인하기 위한 가짜 검색기."""

    def search(self, query_text, image, top_k=10):
        search_type = (
            SearchType.HYBRID
            if query_text and image is not None
            else SearchType.IMAGE
            if image is not None
            else SearchType.TEXT
        )
        items = [
            {
                "product_id": "hm_0108775015",
                "name": "Strap top",
                "score": 0.91,
                "price": 25000,
            }
        ][:top_k]
        return SearchResult(
            search_type,
            [i["product_id"] for i in items],
            [0.91],
            37,
            np.array([0]),
            items,
        )


app.dependency_overrides[get_searcher] = FakeSearcher
client = TestClient(app)


def _png_b64() -> str:
    buf = BytesIO()
    Image.new("RGB", (8, 8), "black").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_health() -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_recommend_missing_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("REC_MODEL_DIR", str(tmp_path / "missing"))
    response = client.get("/api/recommend", params={"user_id": "U1234", "top_n": 10})
    assert response.status_code == 503
    assert "model missing" in response.json()["detail"]


def test_search_json_text() -> None:
    body = client.post(
        "/api/search", json={"query_text": "black strap top", "top_k": 10}
    ).json()
    for key in ("search_type", "results", "latency_ms", "total_count"):
        assert key in body
    assert body["search_type"] == "text"
    assert set(body["results"][0]) == {"product_id", "name", "score", "price"}


def test_search_json_hybrid() -> None:
    body = client.post(
        "/api/search", json={"query_text": "black", "image_base64": _png_b64()}
    ).json()
    assert body["search_type"] == "hybrid"


def test_search_multipart_image() -> None:
    buf = BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="JPEG")
    r = client.post(
        "/api/search",
        files={"image": ("q.jpg", buf.getvalue(), "image/jpeg")},
        data={"top_k": "5"},
    )
    assert r.status_code == 200
    assert r.json()["search_type"] == "image"


def test_search_requires_query() -> None:
    assert client.post("/api/search", json={"top_k": 10}).status_code == 422
