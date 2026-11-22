"""Request-scoped explicit preferences. Never write behavior or alter model history."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from src.recommendation.reranking.items import Candidate


@dataclass(frozen=True)
class PreferencePolicy:
    max_weight: float
    history_decay: float
    candidate_limit: int

    @classmethod
    def from_config(cls, raw: Mapping[str, Any]) -> PreferencePolicy:
        policy = cls(float(raw['max_weight']), float(raw['history_decay']), int(raw['candidate_limit']))
        if not math.isfinite(policy.max_weight) or not 0 <= policy.max_weight <= 0.2:
            raise ValueError('Preference max_weight must be in [0, 0.2]')
        if not math.isfinite(policy.history_decay) or policy.history_decay <= 0:
            raise ValueError('Preference history_decay must be positive')
        if not 1 <= policy.candidate_limit <= 100:
            raise ValueError('Preference candidate_limit must be in [1, 100]')
        return policy


def parse_weights(payload: Mapping[str, Any], user_id: str) -> dict[str, float]:
    """Only accept a full replacement snapshot for this exact user; empty clears."""
    if (payload.get('schema_version') != 'closet-preference-seed-v1'
            or payload.get('operation') != 'replace'
            or payload.get('event_type') != 'outfit_preference'
            or payload.get('user_id') != user_id):
        raise ValueError('Invalid preference contract or user mismatch')
    rows = payload.get('seed_products')
    if not isinstance(rows, list) or len(rows) > 2000:
        raise ValueError('Invalid preference products')
    weights: dict[str, float] = {}
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Invalid preference row')
        pid, value = row.get('product_id'), row.get('relative_weight')
        if not isinstance(pid, str) or not pid or len(pid) > 128 or pid in seen:
            raise ValueError('Invalid or duplicate preference product')
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('Preference weight must be numeric')
        weight = float(value)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('Preference weight must be finite and nonnegative')
        seen.add(pid)
        if weight > 0:
            weights[pid] = weight
    total = sum(weights.values())
    if not math.isfinite(total):
        raise ValueError('Invalid preference weight total')
    return {pid: w / total for pid, w in weights.items()} if total else {}


def eligible_weights(weights: Mapping[str, float], item_map: Mapping[str, int],
                     created: Sequence[float], now: float, limit: int) -> dict[str, float]:
    """Exact catalog IDs only. Missing/future products never enter the pool."""
    return dict(sorted(((pid, w) for pid, w in weights.items()
                        if pid in item_map and created[item_map[pid]] <= now),
                       key=lambda pair: (-pair[1], pair[0]))[:limit])


def apply_preference_order(main: Sequence[Candidate], candidates: Sequence[Candidate],
                           weights: Mapping[str, float], history_len: int,
                           policy: PreferencePolicy) -> list[Candidate]:
    """Blend reciprocal baseline rank with explicit interest before business rules.

    Candidate.score retains its original meaning; this sorting key is NOT a
    purchase probability. The coefficient is an uncalibrated, bounded heuristic.
    No vectors/checkpoints are needed for this exact-product preference path.
    """
    if not weights or policy.max_weight == 0:
        return list(main)
    pool = list(main)
    seen = {c.product_id for c in pool}
    for candidate in candidates:
        if candidate.product_id in weights and candidate.product_id not in seen:
            pool.append(candidate)
            seen.add(candidate.product_id)
    active = {pid: w for pid, w in weights.items() if pid in seen}
    if not active:
        return list(main)
    retained = min(1.0, sum(active.values()))
    alpha = policy.max_weight * retained / (1 + history_len / policy.history_decay)
    largest = max(active.values())
    baseline = {c.product_id: 1 / (i + 1) for i, c in enumerate(main)}
    ordered = sorted(pool, key=lambda c: -((1-alpha) * baseline.get(c.product_id, 0)
                     + alpha * active.get(c.product_id, 0) / largest))
    return [replace(c, reason='outfit_preference') if c.product_id in active else c for c in ordered]
