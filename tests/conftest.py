"""API 테스트의 새 런타임 journal은 기존 데모/학습 산출물과 분리한다."""

import pytest


@pytest.fixture(autouse=True)
def isolated_event_journal(tmp_path, monkeypatch):
    from src.serving.event_log import get_event_log
    from src.serving.feature_store import get_feature_store

    monkeypatch.setenv("EVENT_LOG_PATH", str(tmp_path / "runtime.sqlite3"))
    get_event_log.cache_clear()
    get_feature_store.cache_clear()
    yield
    get_event_log.cache_clear()
    get_feature_store.cache_clear()
