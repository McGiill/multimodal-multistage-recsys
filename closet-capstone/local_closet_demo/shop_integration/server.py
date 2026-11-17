"""Local shop integration: existing closet API + optional team API, no inference.

8767 serves the copied team shop and persistent local demo profiles.
8765 serves cached retrieval plus reviewed shoe rows. 8000 is the existing team API.
"""
import argparse
import os
import copy
import json
import mimetypes
import sys
from collections import Counter
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, unquote, urlsplit
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_paths import data_demo

from store import ProfileStore
from personalization import resolve_preferences, rerank_automatic
from discovery import discover
from preference_seed import build_preference_seed
from audience import AUDIENCES, REVIEW, is_eligible, preferred_audience, filter_outfits, visible_survey, visible_profile

HERE = Path(__file__).resolve().parent
DEMO = data_demo()
sys.path.insert(0, str(HERE.parent / 'team_integration'))
from closet_selection_helpers import connection, allowed_products

BASE_SELECTION = json.loads((DEMO / 'team_integration/closet_selection.json').read_text(encoding='utf-8'))
SHOES = json.loads((DEMO / 'shoe_extension.json').read_text(encoding='utf-8'))
DRESSES = json.loads((DEMO / 'dress_extension.json').read_text(encoding='utf-8'))
REPLACEMENTS = json.loads((DEMO / 'replacement_extension.json').read_text(encoding='utf-8'))
SELECTION = copy.deepcopy(BASE_SELECTION)
SELECTION['products'].update(copy.deepcopy(SHOES['products']))
SELECTION['connections'].update(copy.deepcopy(SHOES['connections']))
for pid, connections in SELECTION['connections'].items():
    for oid, selected in connections.items():
        footwear = SHOES['footwear'].get(oid)
        if footwear and pid not in SHOES['products']:
            selected['groups'][footwear['garment_id']] = footwear['products']
            selected['products'] = list(dict.fromkeys(selected['products'] + footwear['products']))
SELECTION['products'].update(copy.deepcopy(DRESSES['products']))
SELECTION['outfits'].update(DRESSES['outfits'])
SELECTION['connections'].update(copy.deepcopy(DRESSES['connections']))
for pid, entry in REPLACEMENTS['products'].items():
    if set(entry['outfits']) & set(SELECTION['products'][pid]['outfits']):
        raise ValueError('Replacement pair already exists')
    SELECTION['products'][pid]['outfits'].extend(entry['outfits'])
    SELECTION['connections'][pid].update(copy.deepcopy(REPLACEMENTS['connections'][pid]))
SELECTION['outfits'].update(REPLACEMENTS['outfits'])
SELECTION['review']['stats'].update({
    'input_products': len(SELECTION['products']),
    'product_outfit_connections': sum(len(p['outfits']) for p in SELECTION['products'].values()),
    'reverse_product_connections': sum(len(c['products']) for links in SELECTION['connections'].values() for c in links.values()),
    'shoe_input_products': len(SHOES['products']), 'shoe_outfit_connections': len(SHOES['footwear']),
    'unique_reverse_products': len({p for links in SELECTION['connections'].values() for c in links.values() for p in c['products']}),
    'outfits_per_product': dict(Counter(str(len(p['outfits'])) for p in SELECTION['products'].values())),
    'input_slots': {**BASE_SELECTION['review']['stats']['input_slots'], 'shoes': len(SHOES['products']), 'dress': len(DRESSES['products'])},
    'shoe_alternative_connections': sum(len(item['products']) - 1 for item in SHOES['footwear'].values()),
    'unique_outfit_images': len(SELECTION['outfits']),
    'dress_input_products': len(DRESSES['products']),
    'dress_outfit_connections': sum(len(p['outfits']) for p in DRESSES['products'].values()),
})
TAGS = json.loads((DEMO / 'team_integration/closet_tags.json').read_text(encoding='utf-8'))['outfits']
TAGS.update(DRESSES['tags'])
TAGS.update(REPLACEMENTS['tags'])
TAGS = {oid: tags for oid, tags in TAGS.items() if is_eligible(oid)}
STYLES = sorted({t['style'] for t in TAGS.values() if t.get('style')})
CONFIG = json.loads((DEMO / 'config.json').read_text(encoding='utf-8'))
DATASET_VERSION = CONFIG.get('dataset_version') or CONFIG.get('snapshot_sha256') or 'legacy-unversioned'
QUERIES = json.loads((DEMO / CONFIG['outfits_dir'] / 'queries.json').read_text(encoding='utf-8'))
INPUT_SLOTS = {q['product_id']: q['slot'] for q in QUERIES}
INPUT_SLOTS.update({pid: 'shoes' for pid in SHOES['products']})
INPUT_SLOTS.update({pid: 'dress' for pid in DRESSES['products']})
# Preserve frozen source connections, but expose only currently approved looks.
if set(REVIEW['outfits']) != set(SELECTION['outfits']):
    raise ValueError('The outfit review must cover every demo image before serving.')
ACTIVE_PRODUCTS = {pid: [oid for oid in entry['outfits'] if is_eligible(oid)]
                   for pid, entry in SELECTION['products'].items()}
ACTIVE_PRODUCTS = {pid: oids for pid, oids in ACTIVE_PRODUCTS.items() if oids}
ACTIVE_CONNECTIONS = [c for links in SELECTION['connections'].values()
                      for oid, c in links.items() if is_eligible(oid)]
DEMO_STATS = {
    'source_input_products': len(SELECTION['products']), 'source_outfit_images': len(SELECTION['outfits']),
    'input_products': len(ACTIVE_PRODUCTS),
    'product_outfit_connections': sum(map(len, ACTIVE_PRODUCTS.values())),
    'unique_outfit_images': sum(is_eligible(oid) for oid in SELECTION['outfits']),
    'excluded_outfit_images': sum(not is_eligible(oid) for oid in SELECTION['outfits']),
    'audiences': dict(Counter(AUDIENCES[oid] for oid in SELECTION['outfits'] if is_eligible(oid))),
    'reverse_product_connections': sum(len(c['products']) for c in ACTIVE_CONNECTIONS),
    'unique_reverse_products': len({p for c in ACTIVE_CONNECTIONS for p in c['products']}),
    'outfits_per_product': dict(Counter(str(len(oids)) for oids in ACTIVE_PRODUCTS.values())),
    'input_slots': dict(Counter(INPUT_SLOTS[pid] for pid in ACTIVE_PRODUCTS)),
    'shoe_input_products': sum(pid in ACTIVE_PRODUCTS for pid in SHOES['products']),
    'shoe_outfit_connections': sum(len(ACTIVE_PRODUCTS.get(pid, [])) for pid in SHOES['products']),
    'shoe_alternative_connections': (sum(len(item['products']) - 1 for oid, item in SHOES['footwear'].items() if is_eligible(oid))
                                      + sum(len(c.get('shoe_alternatives', [])) for c in ACTIVE_CONNECTIONS)),
    'dress_input_products': sum(pid in ACTIVE_PRODUCTS for pid in DRESSES['products']),
    'dress_outfit_connections': sum(len(ACTIVE_PRODUCTS.get(pid, [])) for pid in DRESSES['products']),
}
KINDS = json.loads((DEMO / CONFIG['provenance_file']).read_text(encoding='utf-8'))['kinds']
CATEGORY_SLOTS = {}
for kind in KINDS.values():
    for category in kind['l3']:
        CATEGORY_SLOTS.setdefault(category, set()).add(kind['slot'])
CLOSET = os.environ.get('CLOSET_RETRIEVAL_URL', 'http://127.0.0.1:8765')
TEAM = os.environ.get('CLOSET_TEAM_URL', 'http://127.0.0.1:8000')


def fetch(base, path, method='GET', body=None):
    request = Request(base + path, data=body, method=method,
                      headers={'Content-Type': 'application/json'} if body else {})
    with urlopen(request, timeout=30) as response:
        return response.read(), response.headers.get('Content-Type', 'application/json')


def closet(path):
    return json.loads(fetch(CLOSET, path)[0])


@lru_cache(maxsize=1)
def catalog():
    return closet('/api/wardrobe')['products']


@lru_cache(maxsize=2048)
def product(pid):
    item = closet('/api/products/' + quote(pid, safe=''))
    slots = CATEGORY_SLOTS.get(item.get('category'), set())
    if 'slot' not in item and len(slots) == 1:
        item['slot'] = next(iter(slots))
    return item


@lru_cache(maxsize=1)
def discovery_pool():
    # Reuse frozen, reviewed connections and existing image tags; no new inference.
    pool = []
    for pid, outfits in ACTIVE_PRODUCTS.items():
        for oid in outfits:
            pool.append({'id': oid, 'product_id': pid, 'image': '/media/outfit/' + oid,
                         'title': connection(SELECTION, pid, oid)['title'],
                         'style': TAGS.get(oid, {}).get('style', ''), 'slot': INPUT_SLOTS[pid],
                         'audience': AUDIENCES.get(oid, 'unknown')})
    return pool


@lru_cache(maxsize=2048)
def preference_links(oid):
    """Only reviewed pairs/alternatives, before ownership exclusion."""
    groups = {}
    for pid, oids in ACTIVE_PRODUCTS.items():
        if oid not in oids:
            continue
        detail = closet('/api/outfits/' + quote(oid, safe='') + '?product_id=' + quote(pid, safe='')
                        + '&reverse_limit=' + str(SELECTION.get('reverse_candidate_limit', 4)))
        selected = connection(SELECTION, pid, oid)
        for group in detail['groups']:
            candidates = ([{**detail['product'], 'similarity':detail['similarity']}]
                          if group['is_anchor'] else [p for p in group['products']
                              if p['id'] in allowed_products(selected, group['id'])])
            for item in candidates:
                bucket=groups.setdefault(group['id'], {})
                value={'similarity':item['similarity'], 'slot':group['slot'],
                       'relation':'anchor_match' if group['is_anchor'] else 'reviewed_alternative'}
                if item['id'] not in bucket or value['similarity'] > bucket[item['id']]['similarity']:
                    bucket[item['id']]=value
    return groups


class Handler(BaseHTTPRequestHandler):
    store: ProfileStore

    def send(self, data, kind='application/json; charset=utf-8', status=200):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(data)

    def body(self):
        size = int(self.headers.get('Content-Length', 0))
        if not 0 < size <= 8_000_000:
            raise ValueError('요청 크기를 확인해 주세요.')
        return self.rfile.read(size)

    def do_GET(self):
        self.handle_request()

    def do_POST(self):
        self.handle_request()

    def handle_request(self):
        try:
            self.dispatch()
        except HTTPError as exc:
            self.send({'detail': '연결된 상품 또는 코디를 찾을 수 없습니다.'}, status=exc.code)
        except (URLError, TimeoutError, ConnectionError):
            self.send({'detail': '추천 서버에 연결할 수 없습니다. 로컬 실행 상태를 확인해 주세요.'}, status=503)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self.send({'detail': str(exc)}, status=400)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def dispatch(self):
        url = urlsplit(self.path)
        path = unquote(url.path)
        params = parse_qs(url.query)
        if self.command == 'GET' and path in ('/', '/shop/', '/shop/index.html'):
            return self.send((HERE / 'shop/index.html').read_bytes(), 'text/html; charset=utf-8')
        if self.command == 'GET' and path in ('/shop/shop.js', '/shop/shop.css', '/shop/closet.js', '/shop/closet.css', '/shop/cold_start.js', '/shop/moodboard.js', '/shop/moodboard.css'):
            file = HERE / 'shop' / path.rsplit('/', 1)[1]
            return self.send(file.read_bytes(), mimetypes.guess_type(file.name)[0] + '; charset=utf-8')
        if self.command == 'GET' and path == '/health':
            return self.send({'service': 'local-shop-closet-v1'})
        if path.startswith('/api/closet/'):
            return self.closet_request(path, params)
        if self.command == 'GET' and path.startswith('/media/'):
            if path.startswith('/media/outfit/') and not is_eligible(path.removeprefix('/media/outfit/')):
                return self.send({'detail': '시연에서 제외된 코디입니다.'}, status=404)
            data, kind = fetch(CLOSET, self.path)
            return self.send(data, kind)
        if path.startswith('/api/'):
            body = self.body() if self.command == 'POST' else None
            # Preserve the real team behavior. Never fabricate recommendations or events.
            try:
                data, kind = fetch(TEAM, self.path, self.command, body)
                return self.send(data, kind)
            except (URLError, TimeoutError, ConnectionError):
                parts = path.strip('/').split('/')
                if self.command == 'GET' and len(parts) in (3, 4) and parts[:2] == ['api', 'products']:
                    if len(parts) == 3:
                        p = product(parts[2])
                        return self.send({**p, 'product_id': p['id'], 'price': None, 'local_catalog': True})
                    if parts[3] == 'image':
                        data, kind = fetch(CLOSET, '/media/product/' + quote(parts[2], safe=''))
                        return self.send(data, kind)
                return self.send({'detail': '팀 검색·맞춤 추천 API가 실행되지 않았습니다. 마이페이지의 옷장·코디 추천은 이용할 수 있습니다.'}, status=503)
        return self.send({'detail': '찾을 수 없는 주소입니다.'}, status=404)

    def closet_request(self, path, params):
        if path == '/api/closet/catalog' and self.command == 'GET':
            products = [{**p, 'display_name': SELECTION['products'].get(p['id'], {}).get('title', p['name'])} for p in catalog()]
            return self.send({'products': products, 'demo_ids': list(ACTIVE_PRODUCTS),
                              'shoe_demo_ids': [p for p in SHOES['products'] if p in ACTIVE_PRODUCTS],
                              'dress_demo_ids': [p for p in DRESSES['products'] if p in ACTIVE_PRODUCTS], 'styles': STYLES,
                              'demo_stats': DEMO_STATS})
        uid = params.get('user_id', [''])[0]
        self.store.validate_user(uid)
        if path == '/api/closet/preference-seed' and self.command == 'GET':
            profile=self.store.get(uid)
            return self.send(build_preference_seed(uid, profile, DATASET_VERSION, preferred_audience(profile),
                lambda oid, scope: bool(filter_outfits([{'id':oid}], scope)), preference_links))
        if path == '/api/closet/survey':
            if self.command == 'GET':
                profile = self.store.get(uid)
                audience = preferred_audience(profile)
                return self.send({'outfits': discover(discovery_pool(), audience, {'weights':{}}, limit=len(SELECTION['outfits'])), 'audience':audience,
                                  'survey': visible_survey(profile.get('survey', {'selected':[], 'completed':False})), 'max_picks':5})
            if self.command == 'POST':
                body = json.loads(self.body())
                if not isinstance(body, dict):
                    raise ValueError('선택한 코디를 확인해 주세요.')
                selected, skipped = body.get('selected'), body.get('skipped', False)
                if not isinstance(skipped, bool) or not isinstance(selected, list) or not all(isinstance(oid, str) for oid in selected):
                    raise ValueError('선택한 코디를 확인해 주세요.')
                if len(selected) > 5 or len(set(selected)) != len(selected) or (not skipped and not selected):
                    raise ValueError('코디를 1~5장 선택해 주세요.')
                if any(oid not in SELECTION['outfits'] for oid in selected):
                    raise ValueError('지원하지 않는 코디입니다.')
                if len(filter_outfits([{'id':oid} for oid in selected], preferred_audience(self.store.get(uid)))) != len(selected):
                    raise ValueError('현재 추천 범위의 코디를 선택해 주세요.')
                survey = self.store.save_survey(uid, selected, skipped, dataset_version=DATASET_VERSION)
                return self.send({'survey':visible_survey(survey),
                                  'preference_seed_url':'/api/closet/preference-seed?user_id=' + quote(uid, safe='')})
            return self.send({'detail':'지원하지 않는 요청입니다.'}, status=405)
        if path == '/api/closet/registration' and self.command == 'POST':
            profile = self.store.set_registration(uid, json.loads(self.body()))
            return self.send({'registration': profile['registration'],
                              'audience': profile.get('shopping_audience', {'F':'women', 'M':'men'}.get(profile['registration']['gender'], 'all'))})
        if path == '/api/closet/discover' and self.command == 'GET':
            profile = self.store.get(uid)
            account = profile.get('registration', {})
            default_audience = preferred_audience(profile)
            audience = params.get('audience', [default_audience])[0]
            views = self.store.recent_outfit_views(uid)
            preference = resolve_preferences(self.store, uid, TAGS)
            return self.send({'outfits': discover(discovery_pool(), audience, preference,
                                                 [v['outfit_id'] for v in views]),
                              'audience': audience, 'personalization': preference,
                              'owned_count': len(profile['owned']), 'registration': account,
                              'source': 'reviewed_local_outfits', 'audience_tags': 'demo_clothing_review'})
        if path == '/api/closet/events' and self.command == 'POST':
            body = json.loads(self.body())
            if not isinstance(body, dict) or body.get('event_type') != 'outfit_view':
                raise ValueError('지원하지 않는 관심 기록입니다.')
            pid, oid = body.get('product_id'), body.get('outfit_id')
            if not isinstance(pid, str) or not isinstance(oid, str):
                raise ValueError('상품과 코디를 확인해 주세요.')
            if not filter_outfits([{'id': oid}], preferred_audience(self.store.get(uid))):
                return self.send({'detail': '현재 추천 범위에서 제공하지 않는 코디입니다.'}, status=404)
            candidates = closet('/api/products/' + quote(pid, safe='') + '/outfits')['outfits']
            if not any(o['id'] == oid for o in candidates):
                raise ValueError('해당 상품의 코디가 아닙니다.')
            self.store.record_outfit_view(uid, oid)
            return self.send({'recorded': True})
        if path == '/api/closet/profile':
            if self.command == 'GET':
                profile = self.store.get(uid)
                preference = resolve_preferences(self.store, uid, TAGS)
                return self.send({**visible_profile(profile), 'audience':preferred_audience(profile), 'personalization': preference, 'products': [product(pid) for pid in profile['owned']]})
            body = json.loads(self.body())
            action, values = body.get('action'), body.get('values')
            if not isinstance(values, list) or len(values) > 500 or not all(isinstance(v, str) for v in values):
                raise ValueError('저장할 목록을 확인해 주세요.')
            if action == 'styles' and any(v not in STYLES for v in values):
                raise ValueError('지원하지 않는 스타일입니다.')
            if action == 'add':
                for pid in values:
                    product(pid)  # Reject nonexistent IDs before writing anything.
            return self.send(visible_profile(self.store.update(uid, action, values)))
        if self.command != 'GET':
            return self.send({'detail': '지원하지 않는 요청입니다.'}, status=405)
        pid = params.get('product_id', [''])[0]
        supports_full_search = CONFIG.get('demo_mode') != 'curated_snapshot'
        full = supports_full_search and params.get('mode', ['curated'])[0] == 'all'
        profile = self.store.get(uid)
        if path == '/api/closet/outfits':
            data = closet('/api/products/' + quote(pid, safe='') + '/outfits')
            curated = not full and pid in SELECTION['products']
            if curated:
                lookup = {o['id']: o for o in data['outfits']}
                data['outfits'] = [lookup[oid] for oid in SELECTION['products'][pid]['outfits'] if oid in lookup]
            audience = preferred_audience(profile)
            before_count = len(data['outfits'])
            data['outfits'] = filter_outfits(data['outfits'], audience)
            preference = resolve_preferences(self.store, uid, TAGS)
            data['outfits'] = rerank_automatic(data['outfits'], TAGS, preference)
            for outfit in data['outfits']:
                outfit['title'] = connection(SELECTION, pid, outfit['id']).get('title', outfit['style'])
                outfit['style_tag'] = TAGS.get(outfit['id'], {}).get('style')
                outfit['clothing_audience'] = AUDIENCES[outfit['id']]
            # Optional pagination is applied AFTER scope filtering and reranking.
            # Omitted limit preserves the API's complete reviewed-pool response.
            total_count = len(data['outfits'])
            offset = int(params.get('offset', ['0'])[0])
            limit = int(params['limit'][0]) if 'limit' in params else None
            if offset < 0 or (limit is not None and not 1 <= limit <= 40):
                raise ValueError('코디 표시 범위를 확인해 주세요.')
            data['outfits'] = data['outfits'][offset:offset+limit if limit is not None else None]
            return self.send({**data, 'curated': curated, 'supports_full_search': supports_full_search, 'styles': [], 'personalization': preference,
                              'audience':audience, 'excluded_count':before_count - total_count,
                              'total_count':total_count, 'offset':offset,
                              'has_more':offset + len(data['outfits']) < total_count})
        if path == '/api/closet/outfit':
            oid = params.get('outfit_id', [''])[0]
            if not filter_outfits([{'id': oid}], preferred_audience(profile)):
                return self.send({'detail': '현재 추천 범위에서 제공하지 않는 코디입니다.'}, status=404)
            curated = not full and pid in SELECTION['products'] and oid in SELECTION['products'][pid]['outfits']
            limit = '&reverse_limit=' + str(SELECTION.get('reverse_candidate_limit', 4)) if curated else ''
            data = closet('/api/outfits/' + quote(oid, safe='') + '?product_id=' + quote(pid, safe='') + limit)
            selected = connection(SELECTION, pid, oid)
            for group in data['groups']:
                candidates = group['products']
                if curated:
                    allowed = allowed_products(selected, group['id'])
                    candidates = [p for p in candidates if p['id'] in allowed]
                group['owned_count'] = sum(p['id'] in profile['owned'] for p in candidates)
                group['products'] = [p for p in candidates if p['id'] not in profile['owned']]
            shoes = data.get('shoe_alternatives', [])
            data['shoe_alternatives_owned_count'] = sum(p['id'] in profile['owned'] for p in shoes)
            data['shoe_alternatives'] = [p for p in shoes if p['id'] not in profile['owned']]
            return self.send({**data, 'title': selected.get('title', '코디 상세'), 'curated': curated})
        return self.send({'detail': '찾을 수 없는 주소입니다.'}, status=404)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--db', type=Path, default=HERE / 'data/profiles.sqlite3')
    args = parser.parse_args()
    Handler.store = ProfileStore(args.db)
    httpd = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Local: http://127.0.0.1:{args.port}/shop/#/mypage', flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
