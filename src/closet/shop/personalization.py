"""Small, bounded implicit-feedback reranker; no model training or user clusters."""
import time
from collections import defaultdict

MIN_DISTINCT_OUTFITS = 3
MAX_BONUS = 0.06
MIN_COHORT_MEMBERS = 3


def resolve_preferences(store, uid, tags, saved_outfit_ids=None):
    own = infer_preferences(store.recent_outfit_views(uid), tags)
    chosen = store.get(uid).get('survey', {}).get('selected', [])
    saved = set(saved_outfit_ids if saved_outfit_ids is not None else
                (row['outfit_id'] for row in store.saved_outfits(uid)))
    saved = {oid for oid in saved if tags.get(oid, {}).get('style')}
    counts = defaultdict(float)
    # Rebuild from CURRENT state, not cumulative save events. Overlap counts once.
    for oid in set(chosen) | saved:
        style = tags.get(oid, {}).get('style')
        if style:
            counts[style] += 1
    if counts:
        total = sum(counts.values())
        initial = {style: count / total for style, count in counts.items()}
        behavior_share = own['distinct_outfits'] / (own['distinct_outfits'] + 5) if own['ready'] else 0
        weights = {style: (1 - behavior_share) * initial.get(style, 0) + behavior_share * own['weights'].get(style, 0)
                   for style in initial.keys() | own['weights'].keys()}
        source = ('saved_and_views' if behavior_share else 'saved_outfits') if saved else ('survey_and_views' if behavior_share else 'survey_selection')
        return {**own, 'ready': True, 'source': source,
                'weights': weights, 'selected_outfits': len({o for o in chosen if tags.get(o, {}).get('style')}),
                'saved_outfits': len(saved), 'explicit_outfits': int(total)}
    if own['ready']:
        return own
    peers = [infer_preferences(views, tags) for views in store.cohort_views(uid).values()]
    peers = [peer for peer in peers if peer['ready']]
    if len(peers) < MIN_COHORT_MEMBERS:
        return own
    scores = defaultdict(float)
    for peer in peers:
        for style, weight in peer['weights'].items():
            scores[style] += weight / len(peers)
    # A demographic prior is weaker than the user's own observed interests.
    return {**own, 'source': 'registration_cohort', 'ready': True,
            'weights': dict(scores), 'strength': 0.5, 'cohort_members': len(peers)}


def infer_preferences(views, tags, now=None):
    now = time.time() if now is None else now
    scores = defaultdict(float)
    seen = set()
    for view in views:
        oid = view['outfit_id']
        style = tags.get(oid, {}).get('style')
        age = max(0, now - view['viewed_at'])
        if oid in seen or not style or age > 90 * 86400:
            continue
        seen.add(oid)
        scores[style] += 0.5 ** (age / (30 * 86400))
    ready = len(seen) >= MIN_DISTINCT_OUTFITS
    total = sum(scores.values())
    weights = {key: value / total for key, value in scores.items()} if ready else {}
    return {'source': 'outfit_views', 'ready': ready, 'distinct_outfits': len(seen),
            'weights': weights}


def rerank_automatic(outfits, tags, preference):
    result = []
    for position, outfit in enumerate(outfits):
        style = tags.get(outfit['id'], {}).get('style')
        bonus = MAX_BONUS * preference.get('strength', 1) * preference['weights'].get(style, 0)
        result.append({**outfit, 'preference_bonus': bonus,
                       'rank_score': float(outfit['similarity']) + bonus,
                       'preference_reasons': [{'registration_cohort':'가입 정보 기반 초기 추천', 'survey_selection':'선택한 코디 반영', 'survey_and_views':'선택한 코디와 최근 관심 반영', 'saved_outfits':'저장한 코디 반영', 'saved_and_views':'저장한 코디와 최근 관심 반영'}.get(preference['source'], '최근 본 코디 반영')] if bonus else [],
                       'base_position': position})
    # With no evidence or equal bonuses, retain the original presentation order.
    if result and max(o['preference_bonus'] for o in result) - min(o['preference_bonus'] for o in result) > 1e-6:
        result.sort(key=lambda o: (-o['rank_score'], o['base_position']))
    return result
