"""First-visit browsing from already reviewed outfits, without a survey gate."""
from collections import defaultdict, deque
from audience import matches_audience


def discover(pool, audience, preference, viewed=(), limit=12):
    if audience not in ('all', 'women', 'men', 'unisex', 'kids'):
        raise ValueError('쇼핑 대상을 확인해 주세요.')
    # This demo only serves visually screened adult-presenting looks.
    if audience == 'kids':
        return []
    candidates = [p for p in pool if matches_audience(p['audience'], audience)]
    seen_before = set(viewed)
    buckets = defaultdict(deque)
    for item in sorted(candidates, key=lambda p: (p['id'] in seen_before, p['audience'] == 'unknown')):
        buckets[(item['style'], item['slot'], item['audience'])].append(item)
    # Interleave clothing types/styles; behavior only changes bucket priority.
    keys = sorted(buckets, key=lambda k: -preference['weights'].get(k[0], 0))
    result, seen = [], set()
    while keys and len(result) < limit:
        remaining = []
        for key in keys:
            queue = buckets[key]
            while queue and queue[0]['id'] in seen:
                queue.popleft()
            if queue and len(result) < limit:
                item = queue.popleft()
                result.append(item)
                seen.add(item['id'])
            if queue:
                remaining.append(key)
        keys = remaining
    return result
