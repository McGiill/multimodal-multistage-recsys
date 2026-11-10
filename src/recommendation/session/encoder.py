"""세션 인코더 (GRU) — 현재 세션 클릭 → 단기 관심 임베딩.

구조
  세션 조회 상품 (최근 max_len개) → 상품 임베딩 → GRU → 세션 벡터 (L2 정규화)
  학습: 세션의 다음 조회 상품 맞히기 (전체 상품 softmax, 점수 = 코사인 / temperature) → train.py
  점수: 세션 벡터 · 상품 벡터 (코사인, -1~1)

장기 선호와 결합 (merge_candidates)
  Stage 1 User Tower 점수도 정규화 벡터의 내적(코사인)이라 같은 범위다.
  최종 = alpha * 장기(User Tower) + (1 - alpha) * 단기(세션)
  User Tower는 과거 이력만 보므로, 이번 세션에서 새로 생긴 관심 상품을 세션 쪽에서 n_candidates개 추가한다.

상품 번호는 팀 Stage 1과 같은 규칙: product_id 정렬 순서 + 1 (0 = padding).
담당: C
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from src.common.config import ROOT, load_config
from src.common.schemas import SessionContext

ENCODERS = {"gru"}


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------

@dataclass
class SessionConfig:
    encoder: str = "gru"
    embedding_dim: int = 64
    hidden_dim: int = 64
    max_len: int = 10
    min_len: int = 2
    long_short_alpha: float = 0.6
    live_history: bool = True       # 최근 조회를 User Tower·DeepFM 이력에 붙임 (pipeline.py)
    blend_candidates: bool = True   # 결합 점수 상위 n_candidates개로 후보를 자름 (pipeline.py)
    temperature: float = 0.1
    epochs: int = 5
    lr: float = 0.001
    batch_size: int = 512
    eval_k: int = 20
    n_candidates: int = 50
    interest_field: str = "category_l2"
    model_file: str = "session_encoder.pt"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.encoder not in ENCODERS:
            raise ValueError(f"지원하지 않는 세션 인코더: {self.encoder} (가능: {sorted(ENCODERS)})")
        if not 0.0 <= self.long_short_alpha <= 1.0:
            raise ValueError("long_short_alpha는 0~1 사이여야 함")
        if self.max_len < 1 or self.min_len < 2:
            raise ValueError("max_len >= 1, min_len >= 2 여야 함")
        if self.temperature <= 0:
            raise ValueError("temperature는 0보다 커야 함")
        if self.interest_field not in {"category_l1", "category_l2", "category_l3"}:
            raise ValueError(f"interest_field 값이 잘못됨: {self.interest_field}")

    @classmethod
    def from_recommend_config(cls, raw: Mapping[str, Any] | None = None) -> SessionConfig:
        """configs/recommend.yaml 의 session 항목을 읽는다."""
        if raw is None:
            raw = load_config("recommend")
        s = raw["session"]
        return cls(**{k: s[k] for k in cls.__dataclass_fields__ if k in s}, seed=raw.get("seed", 42))


def model_path(raw: Mapping[str, Any] | None = None) -> Path:
    """학습된 세션 인코더 파일 (paths.model_dir / session.model_file)."""
    if raw is None:
        raw = load_config("recommend")
    return ROOT / raw["paths"]["model_dir"] / raw["session"]["model_file"]


# ---------------------------------------------------------------------------
# 상품 번호
# ---------------------------------------------------------------------------

class ItemIndex:
    """product_id ↔ 정수 번호. product_id 정렬 순서 + 1, 0은 padding (팀 Stage 1과 같은 규칙)."""

    def __init__(self, product_ids: Sequence[str]) -> None:
        self.product_ids = np.array(sorted(set(product_ids)), dtype=object)
        self._pos = {p: i + 1 for i, p in enumerate(self.product_ids)}

    def __len__(self) -> int:
        """padding 포함 크기 (임베딩 행 수)."""
        return len(self.product_ids) + 1

    def encode(self, product_ids: Sequence[str]) -> list[int]:
        """모르는 상품은 버린다."""
        return [self._pos[p] for p in product_ids if p in self._pos]

    def positions(self, product_ids: Sequence[str]) -> np.ndarray:
        """encode와 달리 길이를 유지한다. 모르는 상품은 0."""
        return np.array([self._pos.get(p, 0) for p in product_ids], dtype=np.int64)

    def decode(self, idx: Sequence[int] | np.ndarray) -> list[str]:
        return list(self.product_ids[np.asarray(idx, dtype=np.int64) - 1])


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------

class SessionEncoder(nn.Module):
    """세션 조회 상품 시퀀스 → 단기 관심 벡터 (L2 정규화)."""

    def __init__(self, n_items: int, embedding_dim: int = 64, hidden_dim: int = 64) -> None:
        super().__init__()
        self.item_emb = nn.Embedding(n_items, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, seq: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        """seq: (B, L) 오른쪽 padding, length: (B,) 1 이상."""
        x = self.item_emb(seq)
        packed = pack_padded_sequence(x, length.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        last = x[torch.arange(len(seq)), length - 1]  # 마지막 클릭을 직접 더해 학습 초반에도 안정적
        return nn.functional.normalize(self.out(h[-1]) + last, dim=-1)

    def item_vectors(self) -> torch.Tensor:
        """(n_items, D) 정규화 상품 벡터. padding 행은 0."""
        w = nn.functional.normalize(self.item_emb.weight, dim=-1)
        return torch.cat([torch.zeros_like(w[:1]), w[1:]])


def pad_sequences(seqs: Sequence[Sequence[int]], max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """각 시퀀스의 최근 max_len개 → (N, max_len) 오른쪽 padding, 길이."""
    out = np.zeros((len(seqs), max_len), dtype=np.int64)
    length = np.zeros(len(seqs), dtype=np.int64)
    for i, s in enumerate(seqs):
        s = list(s)[-max_len:]
        out[i, : len(s)] = s
        length[i] = len(s)
    return out, length


# ---------------------------------------------------------------------------
# 서빙용 래퍼
# ---------------------------------------------------------------------------

class SessionModel:
    """학습된 인코더 + 상품 번호. 추천 API에서 세션 클릭 목록(product_id)으로 호출한다.

    trained: 학습 세션에 한 번이라도 나온 상품 (len(index),) bool. 나머지 상품의 임베딩은 초기 무작위 값이라
    (로그에서 조회된 상품은 카탈로그의 약 52%) 입력에서 버리고, 점수 0, top_k 대상에서 제외한다.
    """

    def __init__(self, encoder: SessionEncoder, index: ItemIndex, config: SessionConfig,
                 trained: np.ndarray | None = None) -> None:
        self.encoder = encoder.eval()
        self.index = index
        self.config = config
        self.trained = np.ones(len(index), dtype=bool) if trained is None else np.array(trained, dtype=bool, copy=True)
        self.trained[0] = False
        with torch.inference_mode():
            self._items = encoder.item_vectors().numpy()
        self._items[~self.trained] = 0.0

    # ---- 저장 / 불러오기 ---------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.encoder.state_dict(),
            "product_ids": list(self.index.product_ids),
            "config": asdict(self.config),
            "trained": np.flatnonzero(self.trained).tolist(),
        }, path)

    @classmethod
    def load(cls, path: Path | None = None) -> SessionModel:
        blob = torch.load(path or model_path(), weights_only=False)
        cfg = SessionConfig(**blob["config"])
        index = ItemIndex(blob["product_ids"])
        encoder = SessionEncoder(len(index), cfg.embedding_dim, cfg.hidden_dim)
        encoder.load_state_dict(blob["state_dict"])
        trained = np.zeros(len(index), dtype=bool)
        trained[blob["trained"]] = True
        return cls(encoder, index, cfg, trained)

    def allowed_mask(self, product_ids: Sequence[str], available: np.ndarray) -> np.ndarray:
        """top_k / merge_candidates의 allowed 인자: product_ids 중 available이 True인 상품만 True.

        Stage 1 번호 순서(product_ids, available)를 이 모델의 번호 순서로 옮긴다. 모르는 상품은 무시한다.
        """
        mask = np.zeros(len(self.index), dtype=bool)
        pos = self.index.positions(product_ids)
        keep = (pos > 0) & np.asarray(available, dtype=bool)
        mask[pos[keep]] = True
        return mask

    # ---- 추론 -------------------------------------------------------------

    def encode(self, clicks: Sequence[str]) -> np.ndarray | None:
        """세션 클릭(오래된 것 → 최근 순) → 세션 벡터. 학습된 상품이 하나도 없으면 None."""
        idx = [i for i in self.index.encode(clicks) if self.trained[i]]
        if not idx:
            return None
        seq, length = pad_sequences([idx], self.config.max_len)
        with torch.inference_mode():
            return self.encoder(torch.from_numpy(seq), torch.from_numpy(length))[0].numpy()

    def score(self, clicks: Sequence[str], product_ids: Sequence[str]) -> np.ndarray:
        """상품별 세션 점수(코사인). 세션이 비었거나 모르는·학습 안 된 상품이면 0."""
        out = np.zeros(len(product_ids), dtype=np.float32)
        vec = self.encode(clicks)
        if vec is None:
            return out
        pos = self.index.positions(product_ids)
        known = pos > 0
        out[known] = self._items[pos[known]] @ vec
        return out

    def top_k(self, clicks: Sequence[str], k: int,
              allowed: np.ndarray | None = None) -> tuple[list[str], np.ndarray]:
        """세션 관심과 가까운 상품 k개. 이미 클릭한 상품과 학습 안 된 상품은 뺀다.

        allowed: (len(index),) bool — 요청 시각 기준 등록된 상품만 True (Stage 1의 created_at <= at과 같은 역할)
        """
        vec = self.encode(clicks)
        if vec is None or k <= 0:
            return [], np.zeros(0, dtype=np.float32)
        s = self._items @ vec
        s[~self.trained] = -np.inf
        s[self.index.encode(clicks)] = -np.inf
        if allowed is not None:
            s[~allowed] = -np.inf
        k = min(k, int(np.isfinite(s).sum()))
        top = np.argpartition(-s, k - 1)[:k] if k else np.zeros(0, dtype=np.int64)
        top = top[np.argsort(-s[top], kind="stable")]
        return self.index.decode(top), s[top]

    def merge_candidates(
        self,
        clicks: Sequence[str],
        long_ids: Sequence[str],
        long_scores: Sequence[float],
        allowed: np.ndarray | None = None,
    ) -> tuple[list[str], np.ndarray]:
        """장기(User Tower) 후보 + 단기(세션) 관심 결합 → 결합 점수 내림차순.

        - 세션 클릭이 없으면 장기 후보를 그대로 돌려준다.
        - 세션 쪽에서 n_candidates개를 추가한다. 이 상품들은 장기 점수가 없으므로 장기 후보 최저 점수를 쓴다.
        - 결합 점수 = alpha * 장기 + (1 - alpha) * 세션 (둘 다 코사인).
        """
        long_ids = list(long_ids)
        long_scores = np.asarray(long_scores, dtype=np.float32)
        if self.encode(clicks) is None:
            return long_ids, long_scores

        seen = set(long_ids)
        extra, _ = self.top_k(clicks, self.config.n_candidates, allowed)
        extra = [p for p in extra if p not in seen]
        floor = float(long_scores.min()) if len(long_scores) else 0.0
        ids = long_ids + extra
        long_all = np.concatenate([long_scores, np.full(len(extra), floor, dtype=np.float32)])
        combined = blend_scores(long_all, self.score(clicks, ids), self.config.long_short_alpha)
        order = np.argsort(-combined, kind="stable")
        return [ids[i] for i in order], combined[order]


@lru_cache(maxsize=1)
def get_session_model() -> SessionModel | None:
    """서빙용 세션 인코더 (서버에서 한 번 로딩). 학습 파일이 없으면 None → Stage 1 후보를 그대로 쓴다."""
    path = model_path()
    if not path.exists():
        print(f"[session] 세션 인코더 없음 ({path}) → python -m src.recommendation.session.train", flush=True)
        return None
    return SessionModel.load(path)


def blend_scores(long: np.ndarray, short: np.ndarray, alpha: float) -> np.ndarray:
    """alpha * 장기 + (1 - alpha) * 단기."""
    return alpha * np.asarray(long, dtype=np.float32) + (1 - alpha) * np.asarray(short, dtype=np.float32)


# ---------------------------------------------------------------------------
# session_context (추천 API 필수 필드)
# ---------------------------------------------------------------------------

def session_interest(clicks: Sequence[str], categories: Mapping[str, str]) -> str | None:
    """세션에서 가장 많이 클릭한 카테고리 라벨. 동률이면 더 최근에 클릭한 쪽. 없으면 None.

    categories: product_id → 카테고리 (예: 카탈로그의 category_l2 Series)
    """
    labels = [c for c in (categories.get(p) for p in clicks) if c is not None]
    if not labels:
        return None
    counts = Counter(labels)
    last_seen = {c: i for i, c in enumerate(labels)}
    return max(counts, key=lambda c: (counts[c], last_seen[c]))


def build_session_context(clicks: Sequence[str], categories: Mapping[str, str],
                          max_len: int = 10) -> SessionContext:
    """recent_clicks: 최근 max_len개 (오래된 것 → 최근 순), session_interest: 그 안의 대표 카테고리."""
    recent = list(clicks)[-max_len:]
    return SessionContext(recent_clicks=recent, session_interest=session_interest(recent, categories))
