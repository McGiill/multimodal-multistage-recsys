"""세션 인코더 학습: python -m src.recommendation.session.train

데이터
  공식 분할 로그(data/splits/{train,valid}_events.parquet)의 view 이벤트를 session_id별 시간순으로 묶는다.
  같은 상품 연속 조회는 1번으로 합치고, 조회가 min_len개 이상인 세션만 쓴다.
  세션 v1..vn → (v1..vi → v(i+1)) 예제 n-1개. 입력은 최근 max_len개.
학습
  다음 조회 상품 맞히기. 점수 = 세션 벡터 · 상품 벡터 / temperature, 전체 상품 softmax cross-entropy.
평가 (valid)
  HitRate@k, MRR@k. 세션에서 이미 본 상품과 학습에 안 나온 상품은 후보에서 뺀다 (서빙 때 top_k와 같은 조건).
  베이스라인: train 인기순 (같은 조건).
출력
  artifacts/models/session_encoder.pt, artifacts/metrics/session_encoder.json
담당: C
"""

from __future__ import annotations

import json
from dataclasses import asdict
import time
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.common.config import artifact_dir, load_config
from src.common.seed import set_seed
from src.recommendation.session.encoder import (
    ItemIndex,
    SessionConfig,
    SessionEncoder,
    SessionModel,
    model_path,
    pad_sequences,
)

Examples = tuple[np.ndarray, np.ndarray, np.ndarray]  # (seq (N, L), length (N,), target (N,))


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------

def view_sessions(events: pd.DataFrame, min_len: int = 2) -> list[list[str]]:
    """view 이벤트 → 세션별 조회 상품 목록 (시간순, 연속 중복 제거, min_len개 이상)."""
    v = events.loc[events["event_type"] == "view", ["session_id", "timestamp", "product_id"]]
    v = v.dropna(subset=["product_id"]).sort_values(["session_id", "timestamp"], kind="stable")
    out: list[list[str]] = []
    for _, items in v.groupby("session_id", sort=False)["product_id"]:
        vals = items.tolist()
        seq = vals[:1] + [p for prev, p in zip(vals, vals[1:]) if p != prev]
        if len(seq) >= min_len:
            out.append(seq)
    return out


def make_examples(sessions: Sequence[Sequence[int]], max_len: int) -> Examples:
    """세션 v1..vn → (v1..vi, v(i+1)) i = 1..n-1. 입력은 최근 max_len개."""
    prefixes: list[Sequence[int]] = []
    targets: list[int] = []
    for s in sessions:
        for i in range(1, len(s)):
            prefixes.append(s[max(0, i - max_len): i])
            targets.append(s[i])
    seq, length = pad_sequences(prefixes, max_len)
    return seq, length, np.array(targets, dtype=np.int64)


def encode_sessions(sessions: Sequence[Sequence[str]], index: ItemIndex, min_len: int) -> list[list[int]]:
    """product_id 세션 → 번호 세션. 카탈로그에 없는 상품을 버린 뒤에도 min_len개 이상인 것만."""
    out = [index.encode(s) for s in sessions]
    return [s for s in out if len(s) >= min_len]


# ---------------------------------------------------------------------------
# 학습 / 평가
# ---------------------------------------------------------------------------

def train_encoder(encoder: SessionEncoder, ex: Examples, cfg: SessionConfig,
                  log: Callable[[str], None] | None = print) -> list[float]:
    """epoch별 평균 loss를 돌려준다."""
    seq, length, target = (torch.from_numpy(a) for a in ex)
    opt = torch.optim.Adam(encoder.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    losses: list[float] = []
    encoder.train()
    for epoch in range(cfg.epochs):
        order = torch.randperm(len(target), generator=gen)
        total, t0 = 0.0, time.perf_counter()
        for start in range(0, len(order), cfg.batch_size):
            b = order[start: start + cfg.batch_size]
            logits = encoder(seq[b], length[b]) @ encoder.item_vectors().T / cfg.temperature
            logits[:, 0] = -torch.inf
            loss = nn.functional.cross_entropy(logits, target[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        losses.append(total / max(len(order), 1))
        if log:
            log(f"  epoch {epoch + 1}/{cfg.epochs}  loss {losses[-1]:.4f}  ({time.perf_counter() - t0:.1f}s)")
    encoder.eval()
    return losses


def rank_metrics(scores: torch.Tensor, seq: torch.Tensor, target: torch.Tensor, k: int) -> dict[str, float]:
    """scores (B, n_items) 한 배치의 HitRate@k, MRR@k 합계 (이미 본 상품·padding 제외)."""
    scores = scores.clone()
    scores.scatter_(1, seq, -torch.inf)  # seq의 0(padding)도 함께 제외됨
    scores[:, 0] = -torch.inf
    t = scores.gather(1, target[:, None])
    rank = (scores > t).sum(1) + 1
    hit = (rank <= k) & torch.isfinite(t[:, 0])
    return {"hit": float(hit.sum()), "rr": float((hit / rank).sum())}


@torch.inference_mode()
def evaluate(score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], ex: Examples,
             k: int, batch_size: int = 1024) -> dict[str, float]:
    """score_fn(seq, length) → (B, n_items) 점수. HitRate@k / MRR@k."""
    seq, length, target = (torch.from_numpy(a) for a in ex)
    hit = rr = 0.0
    for s in range(0, len(target), batch_size):
        m = rank_metrics(score_fn(seq[s: s + batch_size], length[s: s + batch_size]),
                         seq[s: s + batch_size], target[s: s + batch_size], k)
        hit, rr = hit + m["hit"], rr + m["rr"]
    n = max(len(target), 1)
    return {f"hit_rate@{k}": hit / n, f"mrr@{k}": rr / n, "n_examples": len(target)}


def encoder_scores(encoder: SessionEncoder,
                   trained: np.ndarray) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """학습 안 된 상품은 서빙(SessionModel.top_k)과 같이 후보에서 뺀다."""
    items = encoder.item_vectors()
    untrained = torch.from_numpy(~trained)
    return lambda seq, length: (encoder(seq, length) @ items.T).masked_fill(untrained, -torch.inf)


def trained_items(ex: Examples, n_items: int) -> np.ndarray:
    """학습 예제(입력 + 정답)에 한 번이라도 나온 상품."""
    mask = np.zeros(n_items, dtype=bool)
    mask[ex[0][ex[0] > 0]] = True
    mask[ex[2]] = True
    mask[0] = False
    return mask


def popularity_scores(train_ex: Examples, n_items: int) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """베이스라인: train에서 다음 상품으로 많이 나온 순서 (세션과 무관)."""
    pop = torch.from_numpy(np.bincount(train_ex[2], minlength=n_items).astype(np.float32))
    return lambda seq, length: pop.expand(len(seq), -1)


def fit(train_sessions: Sequence[Sequence[str]], valid_sessions: Sequence[Sequence[str]],
        product_ids: Sequence[str], cfg: SessionConfig,
        log: Callable[[str], None] | None = print) -> tuple[SessionModel, dict[str, Any]]:
    """세션 목록으로 학습하고 valid 지표를 돌려준다 (파일 입출력 없음)."""
    set_seed(cfg.seed)
    index = ItemIndex(product_ids)
    train_ex = make_examples(encode_sessions(train_sessions, index, cfg.min_len), cfg.max_len)
    valid_ex = make_examples(encode_sessions(valid_sessions, index, cfg.min_len), cfg.max_len)
    if len(train_ex[2]) == 0:
        raise ValueError("학습 예제가 없음 (view가 min_len개 이상인 세션이 없음)")

    encoder = SessionEncoder(len(index), cfg.embedding_dim, cfg.hidden_dim)
    losses = train_encoder(encoder, train_ex, cfg, log)
    trained = trained_items(train_ex, len(index))
    metrics = {
        "config": asdict(cfg),
        "train_examples": len(train_ex[2]),
        "trained_items": int(trained.sum()),
        "loss": losses,
        "valid": {
            "session_gru": evaluate(encoder_scores(encoder, trained), valid_ex, cfg.eval_k),
            "popularity": evaluate(popularity_scores(train_ex, len(index)), valid_ex, cfg.eval_k),
        },
    }
    return SessionModel(encoder, index, cfg, trained), metrics


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def main() -> None:
    from src.evaluation.split import load_split
    from src.recommendation.reranking.catalog import load_catalog

    raw = load_config("recommend")
    cfg = SessionConfig.from_recommend_config(raw)
    product_ids = list(load_catalog().index)
    print(f"[session] 세션 만들기 (min_len={cfg.min_len}, max_len={cfg.max_len})", flush=True)
    train_sessions = view_sessions(load_split("train"), cfg.min_len)
    valid_sessions = view_sessions(load_split("valid"), cfg.min_len)
    print(f"  train 세션 {len(train_sessions):,} / valid 세션 {len(valid_sessions):,}", flush=True)

    model, metrics = fit(train_sessions, valid_sessions, product_ids, cfg)
    path = model_path(raw)
    model.save(path)
    out = artifact_dir() / "metrics" / "session_encoder.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    k = cfg.eval_k
    for name, m in metrics["valid"].items():
        print(f"  {name:12s} HitRate@{k} {m[f'hit_rate@{k}']:.4f}  MRR@{k} {m[f'mrr@{k}']:.4f}")
    print(f"[session] 저장: {path}, {out}")


if __name__ == "__main__":
    main()
