"""추천 전략 인터페이스 (전략 패턴).

configs/recommend.yaml 의 `strategy` 값만 바꿔서 로직을 교체한다.
(동료평가: "유사도순 → 최신순 변경을 코드 수정 없이 할 수 있는가?")
담당: D
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.common.schemas import RecommendResponse


class RecommendStrategy(ABC):
    name: str

    @abstractmethod
    def recommend(self, user_id: str, top_n: int) -> RecommendResponse: ...


_REGISTRY: dict[str, type[RecommendStrategy]] = {}


def register(name: str):
    def deco(cls: type[RecommendStrategy]) -> type[RecommendStrategy]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def get_strategy(name: str) -> RecommendStrategy:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. available={list(_REGISTRY)}")
    return _REGISTRY[name]()
