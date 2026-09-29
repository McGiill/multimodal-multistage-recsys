from fastapi.testclient import TestClient

from src.serving.main import app

client = TestClient(app)


def test_health() -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_recommend_schema() -> None:
    body = client.get("/api/recommend", params={"user_id": "U1234", "top_n": 10}).json()
    for key in ("user_id", "recommendations", "pipeline_latency", "session_context"):
        assert key in body


def test_search_schema() -> None:
    body = client.post("/api/search", data={"query_text": "검정 후드티", "top_k": 10}).json()
    for key in ("search_type", "results", "latency_ms", "total_count"):
        assert key in body
