"""Integration checks use live existing retrieval, an isolated temporary database/port."""
import json
import hashlib
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from server import Handler, ProfileStore, ThreadingHTTPServer, SELECTION, BASE_SELECTION, SHOES, DRESSES, REPLACEMENTS, TAGS, connection, allowed_products, closet
from runtime_paths import data_demo, media_path
from personalization import infer_preferences, rerank_automatic
from discovery import discover
from audience import AUDIENCES, REVIEW, matches_audience, is_eligible, filter_outfits


class IntegrationTest(unittest.TestCase):
    def test_replacements_restore_depleted_products_with_reviewed_reverse_links(self):
        expected_counts = {'hm_0629596001':3, 'hm_0630081002':3, 'hm_0853740008':3,
                           'hm_0827397003':3, 'hm_0577150001':3, 'hm_0642675003':3, 'hm_0622220004':2}
        self.assertEqual(set(REPLACEMENTS['products']), set(expected_counts))
        self.assertEqual(len(REPLACEMENTS['outfits']), 8)
        checked = 0
        self.update('REPLACEMENT_OWNED', 'add', list(SELECTION['products']))
        for pid, links in REPLACEMENTS['connections'].items():
            outfits = self.request(f'/api/closet/outfits?user_id=REPLACEMENT_NEW&product_id={pid}')['outfits']
            self.assertEqual(len(outfits), expected_counts[pid])
            self.assertFalse(any(not is_eligible(o['id']) for o in outfits))
            for oid, selected in links.items():
                checked += 1
                match = next(o for o in outfits if o['id'] == oid)
                self.assertEqual(match['matched_garment_id'], selected['matched_garment_id'])
                self.assertGreaterEqual(match['similarity'], .55)
                self.assertAlmostEqual(match['similarity'], selected['similarity'], places=5)
                path = f'/api/closet/outfit?product_id={pid}&outfit_id={oid}'
                detail = self.request(path + '&user_id=REPLACEMENT_NEW')
                self.assertEqual(sum(g['is_anchor'] for g in detail['groups']), 1)
                found = [p for g in detail['groups'] for p in g['products']]
                self.assertCountEqual([p['id'] for p in found], selected['products'])
                self.assertTrue(found)
                self.assertCountEqual([p['id'] for p in detail['shoe_alternatives']], selected['shoe_alternatives'])
                for group in detail['groups']:
                    self.assertCountEqual([p['id'] for p in group['products']], selected['groups'].get(group['id'], []))
                for item in found + detail['shoe_alternatives']:
                    self.assertGreaterEqual(item['similarity'], .50)
                    self.assertNotEqual(item['id'], pid)
                    with urlopen(self.base + item['image']) as response:
                        self.assertTrue(response.headers['Content-Type'].startswith('image/'))
                        content = response.read()
                        evidence = next((e for e in selected['reverse_evidence'] if e['product_id'] == item['id']), None)
                        if evidence:
                            self.assertEqual(hashlib.sha256(content).hexdigest(), evidence['served_image_sha256'])
                            self.assertAlmostEqual(item['similarity'], evidence['similarity'], places=5)
                owned = self.request(path + '&user_id=REPLACEMENT_OWNED')
                remaining = [p['id'] for g in owned['groups'] for p in g['products']]
                self.assertTrue(remaining)
                self.assertFalse(set(remaining) & set(SELECTION['products']))
                for scope in ('men', 'women', 'unisex', 'all'):
                    uid = 'REPLACEMENT_SCOPE_' + scope
                    self.update(uid, 'audience', [scope])
                    if matches_audience(AUDIENCES[oid], scope):
                        self.assertEqual(self.request(path + '&user_id=' + uid)['id'], oid)
                    else:
                        with self.assertRaises(HTTPError) as error:
                            self.request(path + '&user_id=' + uid)
                        self.assertEqual(error.exception.code, 404)
        self.assertEqual(checked, 10)
        for oid, review in REPLACEMENTS['audiences'].items():
            self.assertEqual(review['age_review'], 'adult_presenting')
            self.assertEqual(review['status'], 'accepted')
            with urlopen(self.base + '/media/outfit/' + oid) as response:
                self.assertEqual(hashlib.sha256(response.read()).hexdigest(), review['image_sha256'])
        self.assertEqual(self.store.recent_outfit_views('REPLACEMENT_NEW'), [])

    def test_replacements_preserve_automatic_candidates_for_all_107_inputs(self):
        baseline = json.loads((data_demo() / 'shop_integration/replacement_review/automatic-baseline.json').read_text(encoding='utf-8'))
        self.assertEqual(len(baseline), 107)
        additions = 0
        for pid, previous in baseline.items():
            current = closet('/api/products/' + pid + '/outfits')
            self.assertEqual(previous['threshold'], current['threshold'])
            original_ids = {o['id'] for o in previous['outfits']}
            old = [o for o in current['outfits'] if o['id'] in original_ids]
            self.assertEqual([o['id'] for o in old], [o['id'] for o in previous['outfits']])
            for a, b in zip(previous['outfits'], old):
                self.assertEqual(a['matched_garment_id'], b['matched_garment_id'])
                self.assertAlmostEqual(a['similarity'], b['similarity'], places=5)
            for item in current['outfits']:
                if item['id'] not in original_ids:
                    additions += 1
                    self.assertIn(item['id'], REPLACEMENTS['connections'].get(pid, {}))
                    self.assertTrue(item['reviewed_pair'])
                    self.assertGreaterEqual(item['similarity'], item['review_threshold'])
        self.assertEqual(additions, 4)  # Two joggers and two previously unavailable red shoe regions.
        self.assertEqual(closet('/health')['inference_calls'], 0)

    def test_scope_filters_survey_and_closet_before_ranking(self):
        uid = 'SCOPE_DEMO'
        self.request('/api/closet/registration?user_id=' + uid, {'gender':'M','age_group':'20s'})
        self.assertEqual(self.request('/api/closet/profile?user_id=' + uid)['audience'], 'men')
        survey = self.request('/api/closet/survey?user_id=' + uid)
        self.assertEqual(len(survey['outfits']), 61)
        self.assertTrue(all(o['audience'] in ('menswear', 'unisex') for o in survey['outfits']))
        female_oid = next(oid for oid, label in AUDIENCES.items() if label == 'womenswear')
        with self.assertRaises(HTTPError):
            self.request('/api/closet/survey?user_id=' + uid, {'selected':[female_oid]})
        for audience in ('men', 'women', 'unisex', 'all'):
            self.update(uid, 'audience', [audience])
            for pid, entry in SELECTION['products'].items():
                result = self.request(f'/api/closet/outfits?user_id={uid}&product_id={pid}')
                expected = [oid for oid in entry['outfits'] if matches_audience(AUDIENCES[oid], audience)]
                self.assertEqual([o['id'] for o in result['outfits']], expected)
                self.assertEqual(result['excluded_count'], len(entry['outfits']) - len(expected))
        self.update(uid, 'audience', ['men'])
        # Women's styled slim-denim connections must not fill a man's empty result.
        result = self.request('/api/closet/outfits?user_id=' + uid + '&product_id=hm_0562245100')
        self.assertEqual(result['outfits'], [])
        self.update(uid, 'audience', ['all'])
        self.assertEqual(len(self.request('/api/closet/survey?user_id=' + uid)['outfits']), 89)

    def test_ambiguous_does_not_mean_unisex(self):
        pool = [{'id':str(i),'style':'a','slot':'top','audience':label} for i,label in enumerate(['unknown','unisex','menswear','womenswear'])]
        self.assertEqual({o['audience'] for o in discover(pool,'men',{'weights':{}})}, {'unisex','menswear'})
        self.assertEqual({o['audience'] for o in discover(pool,'women',{'weights':{}})}, {'unisex','womenswear'})
        self.assertEqual({o['audience'] for o in discover(pool,'all',{'weights':{}})}, {'unisex','menswear','womenswear'})
        self.assertEqual({o['audience'] for o in discover(pool,'unisex',{'weights':{}})}, {'unisex'})
        self.assertEqual(len(AUDIENCES), 97)
        self.assertEqual(sum(label == 'unknown' for label in AUDIENCES.values()), 8)

    def test_review_covers_exact_images_and_live_catalog_counts(self):
        index = json.loads((data_demo() / 'shop_integration/audience_review/recheck/index.json').read_text(encoding='utf-8'))
        self.assertEqual(len(index), 89)
        self.assertEqual(set(REVIEW['outfits']), set(SELECTION['outfits']))
        for row in index:
            self.assertEqual(hashlib.sha256(media_path(row['path']).read_bytes()).hexdigest(), REVIEW['outfits'][row['id']]['image_sha256'])
        excluded = [r for r in REVIEW['outfits'].values() if r['status'] == 'excluded']
        self.assertEqual(sum(r['age_review'] == 'child_presenting' for r in excluded), 2)
        self.assertEqual(sum(r['age_review'] == 'age_uncertain' for r in excluded), 6)
        catalog = self.request('/api/closet/catalog')
        stats = catalog['demo_stats']
        self.assertEqual((stats['input_products'], stats['unique_outfit_images'], stats['product_outfit_connections']), (37, 89, 107))
        self.assertEqual((stats['reverse_product_connections'], stats['shoe_alternative_connections']), (172, 16))
        self.assertEqual(stats['audiences'], {'menswear':26, 'womenswear':28, 'unisex':35})
        self.assertIn('hm_0622220004', catalog['demo_ids'])
        self.update('RETIRED_SHOE', 'add', ['hm_0622220004'])
        self.assertEqual(self.request('/api/closet/profile?user_id=RETIRED_SHOE')['owned'], ['hm_0622220004'])
        self.assertEqual(len(self.request('/api/closet/outfits?user_id=RETIRED_SHOE&product_id=hm_0622220004')['outfits']), 2)

    def test_excluded_and_unreviewed_photos_cannot_bypass_filters(self):
        uid = 'REVIEW_GUARDS'
        excluded = {oid for oid in SELECTION['outfits'] if not is_eligible(oid)}
        pairs = {oid: pid for pid, entry in SELECTION['products'].items() for oid in entry['outfits'] if oid in excluded}
        first_pid = next(iter(SELECTION['products']))
        raw = closet(f'/api/products/{first_pid}/outfits')['outfits']
        unreviewed = next(o['id'] for o in raw if o['id'] not in REVIEW['outfits'])
        pairs[unreviewed] = first_pid
        for audience, count in [('all',89), ('men',61), ('women',63), ('unisex',35)]:
            self.update(uid, 'audience', [audience])
            survey = self.request('/api/closet/survey?user_id=' + uid)['outfits']
            self.assertEqual(len(survey), count)
            self.assertFalse(excluded & {o['id'] for o in survey})
            discovery = self.request('/api/closet/discover?user_id=' + uid)['outfits']
            self.assertTrue(all(is_eligible(o['id']) and matches_audience(o['audience'], audience) for o in discovery))
            for pid in SELECTION['products']:
                full = self.request(f'/api/closet/outfits?user_id={uid}&product_id={pid}&mode=all')['outfits']
                self.assertTrue(all(is_eligible(o['id']) and matches_audience(AUDIENCES[o['id']], audience) for o in full))
            for oid, pid in pairs.items():
                for mode in ('curated', 'all'):
                    with self.assertRaises(HTTPError) as error:
                        self.request(f'/api/closet/outfit?user_id={uid}&product_id={pid}&outfit_id={oid}&mode={mode}')
                    self.assertEqual(error.exception.code, 404)
                with self.assertRaises(HTTPError):
                    self.view(uid, pid, oid)
                with self.assertRaises(HTTPError):
                    self.request('/api/closet/survey?user_id=' + uid, {'selected':[oid]})
        self.assertEqual(self.store.recent_outfit_views(uid), [])
        for oid in pairs:
            with self.assertRaises(HTTPError) as error:
                urlopen(self.base + '/media/outfit/' + oid)
            self.assertEqual(error.exception.code, 404)
        for oid in SELECTION['outfits']:
            if is_eligible(oid):
                with urlopen(self.base + '/media/outfit/' + oid) as response:
                    self.assertTrue(response.headers['Content-Type'].startswith('image/'))
                    self.assertGreater(len(response.read()), 1000)

    def test_retired_survey_and_history_no_longer_influence_preferences(self):
        uid = 'RETIRED_HISTORY'
        excluded = [oid for oid in SELECTION['outfits'] if not is_eligible(oid)][:3]
        # Seed legacy records directly, as if saved before this review; don't erase them.
        self.store.save_survey(uid, excluded)
        for oid in excluded:
            self.store.record_outfit_view(uid, oid)
        profile = self.request('/api/closet/profile?user_id=' + uid)
        self.assertEqual(profile['survey']['selected'], [])
        self.assertFalse(profile['personalization']['ready'])
        self.assertEqual(profile['personalization']['distinct_outfits'], 0)
        self.assertEqual(self.request('/api/closet/survey?user_id=' + uid)['survey']['selected'], [])
        self.assertEqual(self.request('/api/closet/survey?user_id=' + uid, {'selected':[], 'skipped':True})['survey']['selected'], [])
        self.assertEqual(self.store.get(uid)['survey']['selected'], excluded)
        valid = self.style_examples('클래식')[0][1]
        self.store.save_survey(uid, excluded + [valid])
        self.assertEqual(self.request('/api/closet/profile?user_id=' + uid)['personalization']['selected_outfits'], 1)
        account = {'gender':'M', 'age_group':'60s+'}
        self.store.set_registration(uid, account)
        self.store.save_survey(uid, excluded)
        for i in range(3):
            peer = 'RETIRED_PEER_' + str(i)
            self.store.set_registration(peer, account)
            for oid in excluded:
                self.store.record_outfit_view(peer, oid)
        self.assertFalse(self.request('/api/closet/profile?user_id=' + uid)['personalization']['ready'])

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db = Path(cls.tmp.name) / 'profiles.sqlite3'
        cls.store = Handler.store = ProfileStore(cls.db)
        cls.http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.http.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.http.server_close()
        cls.thread.join()
        cls.tmp.cleanup()

    def request(self, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = Request(self.base + path, data=data, headers={'Content-Type': 'application/json'})
        with urlopen(req, timeout=60) as response:
            return json.load(response)

    def update(self, uid, action, values):
        return self.request('/api/closet/profile?user_id=' + uid, {'action': action, 'values': values})

    def view(self, uid, pid, oid):
        return self.request('/api/closet/events?user_id=' + uid,
                            {'event_type': 'outfit_view', 'product_id': pid, 'outfit_id': oid})

    def style_examples(self, style, audience='all'):
        examples = {}
        for pid, entry in SELECTION['products'].items():
            for oid in entry['outfits']:
                if TAGS.get(oid, {}).get('style') == style and matches_audience(AUDIENCES[oid], audience):
                    examples.setdefault(oid, pid)
        return [(pid, oid) for oid, pid in list(examples.items())[:3]]

    def test_profile_persistence_and_isolation(self):
        pid = next(iter(SELECTION['products']))
        self.update('TEST_A', 'add', [pid, pid])
        self.update('TEST_A', 'styles', ['클래식'])
        self.assertEqual(ProfileStore(self.db).get('TEST_A'), {'owned': [pid], 'styles': ['클래식']})
        self.assertEqual(self.request('/api/closet/profile?user_id=TEST_B')['owned'], [])
        self.update('TEST_A', 'remove', [pid])
        self.assertEqual(self.store.get('TEST_A')['owned'], [])

    def test_unknown_product_not_saved(self):
        with self.assertRaises(HTTPError):
            self.update('INVALID_TEST', 'add', ['not-a-product'])
        self.assertEqual(self.store.get('INVALID_TEST')['owned'], [])
        with self.assertRaises(HTTPError):
            self.update('INVALID_TEST', 'styles', ['invented-style'])

    def test_all_curated_connections(self):
        # Frozen source remains 30/96/136; excluded photos cannot reach the UI.
        SELECTION = BASE_SELECTION
        outfits, links = 0, 0
        self.assertEqual(len(SELECTION['products']), 30)
        self.update('FULL_WARDROBE', 'add', list(SELECTION['products']))
        for pid, entry in SELECTION['products'].items():
            self.assertTrue(3 <= len(entry['outfits']) <= 5)
            self.assertEqual(len(set(entry['outfits'])), len(entry['outfits']))
            result = self.request(f'/api/closet/outfits?user_id=CURATED_TEST&product_id={pid}')
            expected = entry['outfits'] + REPLACEMENTS['products'].get(pid, {}).get('outfits', [])
            self.assertEqual([o['id'] for o in result['outfits']], [oid for oid in expected if is_eligible(oid)])
            for oid in (oid for oid in entry['outfits'] if is_eligible(oid)):
                data = self.request(f'/api/closet/outfit?user_id=CURATED_TEST&product_id={pid}&outfit_id={oid}')
                data['groups'] = [g for g in data['groups'] if g['slot'] != 'shoes']
                self.assertEqual(sum(g['is_anchor'] for g in data['groups']), 1)
                found = [p['id'] for g in data['groups'] for p in g['products']]
                selected = connection(SELECTION, pid, oid)
                self.assertCountEqual(found, selected['products'])
                for group in data['groups']:
                    self.assertCountEqual([p['id'] for p in group['products']], allowed_products(selected, group['id']))
                    self.assertTrue(all(p['similarity'] >= .5 for p in group['products']))
                full = self.request(f'/api/closet/outfit?user_id=FULL_WARDROBE&product_id={pid}&outfit_id={oid}')
                available = [p['id'] for g in full['groups'] for p in g['products']]
                self.assertTrue(available, (pid, oid))
                self.assertFalse(set(available) & set(SELECTION['products']))
                outfits += 1
                links += len(found)
        self.assertEqual((outfits, links), (88, 125))
        self.assertEqual(sum(len(p['outfits']) for p in SELECTION['products'].values()), 96)
        self.assertEqual(sum(len(c['products']) for links in SELECTION['connections'].values() for c in links.values()), 136)
        self.assertEqual(len(SELECTION['outfits']), 87)

    def test_shoes_use_cached_regions_and_complete_both_directions(self):
        catalog = self.request('/api/closet/catalog')
        self.assertEqual(len(catalog['products']), 107)
        self.assertEqual(len(catalog['demo_ids']), 37)
        self.assertCountEqual(catalog['shoe_demo_ids'], SHOES['products'])
        health = closet('/health')
        self.assertEqual(health['inference_calls'], 0)
        self.assertEqual(health['reviewed_shoe_regions'], 9)
        self.update('SHOE_FULL', 'add', catalog['demo_ids'])
        count = 0
        for pid, entry in SHOES['products'].items():
            result = self.request(f'/api/closet/outfits?user_id=SHOE_TEST&product_id={pid}')
            self.assertEqual(result['product']['slot'], 'shoes')
            expected = entry['outfits'] + REPLACEMENTS['products'].get(pid, {}).get('outfits', [])
            self.assertEqual([o['id'] for o in result['outfits']], [oid for oid in expected if is_eligible(oid)])
            self.assertGreaterEqual(result['threshold'], .55)
            for outfit in result['outfits']:
                oid = outfit['id']; count += 1
                self.assertEqual(outfit['slot'], 'shoes')
                self.assertGreaterEqual(outfit['similarity'], result['threshold'])
                detail = self.request(f'/api/closet/outfit?user_id=SHOE_FULL&product_id={pid}&outfit_id={oid}')
                anchors = [g for g in detail['groups'] if g['is_anchor']]
                self.assertEqual(len(anchors), 1)
                self.assertEqual(anchors[0]['slot'], 'shoes')
                self.assertEqual(anchors[0]['products'], [])
                clothes = [p for g in detail['groups'] for p in g['products']]
                shoes = detail['shoe_alternatives']
                self.assertTrue(clothes, (pid, oid))
                self.assertTrue(shoes, (pid, oid))
                self.assertTrue(all(p['similarity'] >= .5 for p in clothes + shoes))
                self.assertFalse({p['id'] for p in clothes + shoes} & set(catalog['demo_ids']))
                allowed = (REPLACEMENTS['connections'][pid][oid]['shoe_alternatives'] if oid in REPLACEMENTS['connections'].get(pid, {})
                           else SHOES['footwear'][oid]['products'])
                self.assertTrue(all(p['id'] in allowed for p in shoes))
                # Saving an alternative excludes it next time, without changing the anchor.
                chosen = shoes[0]['id']; uid = 'SHOE_OWN_' + pid + '_' + str(count)
                self.update(uid, 'add', [chosen])
                owned = self.request(f'/api/closet/outfit?user_id={uid}&product_id={pid}&outfit_id={oid}')
                self.assertNotIn(chosen, [p['id'] for p in owned['shoe_alternatives']])
                self.assertEqual(owned['shoe_alternatives_owned_count'], 1)
        self.assertEqual(count, 9)

    def test_clothing_outfits_also_include_reviewed_footwear(self):
        checked = set()
        for pid, links in BASE_SELECTION['connections'].items():
            for oid in links:
                if oid not in SHOES['footwear'] or not is_eligible(oid):
                    continue
                result = self.request(f'/api/closet/outfit?user_id=SHOE_REVERSE&product_id={pid}&outfit_id={oid}')
                group = next(g for g in result['groups'] if g['slot'] == 'shoes')
                self.assertFalse(group['is_anchor'])
                self.assertCountEqual([p['id'] for p in group['products']], SHOES['footwear'][oid]['products'])
                checked.add(oid)
        self.assertEqual(checked, {oid for oid in SHOES['footwear'] if is_eligible(oid)})

    def test_dress_connections_exclude_unreviewed_detections_and_owned_products(self):
        catalog = self.request('/api/closet/catalog')
        self.assertCountEqual(catalog['dress_demo_ids'], DRESSES['products'])
        self.assertEqual(len(catalog['dress_demo_ids']), 2)
        self.assertEqual(closet('/health')['reviewed_dress_regions'], 2)
        self.update('DRESS_FULL', 'add', catalog['demo_ids'])
        checked = 0
        for pid, entry in DRESSES['products'].items():
            result = self.request(f'/api/closet/outfits?user_id=DRESS_FULL&product_id={pid}')
            self.assertEqual(result['product']['slot'], 'dress')
            self.assertEqual([o['id'] for o in result['outfits']], entry['outfits'])
            raw = closet(f'/api/products/{pid}/outfits')
            self.assertTrue(all(o['id'] in DRESSES['outfits'] and o['slot'] == 'dress' for o in raw['outfits']))
            for outfit in result['outfits']:
                checked += 1; oid = outfit['id']
                self.assertGreaterEqual(outfit['similarity'], result['threshold'])
                path = f'/api/closet/outfit?product_id={pid}&outfit_id={oid}'
                detail = self.request(path + '&user_id=DRESS_FULL')
                anchor = [g for g in detail['groups'] if g['is_anchor']]
                self.assertEqual(len(anchor), 1)
                self.assertEqual(anchor[0]['slot'], 'dress')
                self.assertEqual(anchor[0]['products'], [])
                visible = [p for g in detail['groups'] for p in g['products']]
                self.assertCountEqual([p['id'] for p in visible], DRESSES['connections'][pid][oid]['products'])
                self.assertTrue(visible)
                self.assertTrue(all(p['similarity'] >= .5 for p in visible))
                self.assertTrue(all(g['slot'] in ('outer','shoes') for g in detail['groups'] if g['products']))
                self.update('DRESS_HIDE_' + pid, 'add', [p['id'] for p in visible])
                hidden = self.request(path + '&user_id=DRESS_HIDE_' + pid)
                self.assertFalse([p for g in hidden['groups'] for p in g['products']])
                self.assertEqual(sum(g['owned_count'] for g in hidden['groups']), len(visible))
                for item in [detail['product'], *visible, {'image':detail['image']}]:
                    with urlopen(self.base + item['image'], timeout=10) as image:
                        self.assertTrue(image.headers['Content-Type'].startswith('image/'))
                        self.assertGreater(len(image.read()), 1000)
        self.assertEqual(checked, 2)
        self.assertEqual(self.store.recent_outfit_views('DRESS_FULL'), [])

    def test_dress_scope_and_survey_keep_existing_photos(self):
        uid = 'DRESS_SCOPE'
        all_photos = self.request('/api/closet/survey?user_id=' + uid)['outfits']
        self.assertEqual({o['id'] for o in all_photos}, {oid for oid in SELECTION['outfits'] if is_eligible(oid)})
        for audience in ('women', 'men'):
            self.update(uid, 'audience', [audience])
            for pid in DRESSES['products']:
                result = self.request(f'/api/closet/outfits?user_id={uid}&product_id={pid}')
                self.assertEqual(len(result['outfits']), 1 if audience == 'women' else 0)

    def test_raw_results_and_expanded_candidate_prefix(self):
        # Expanding review depth must not alter ordinary retrieval/ranking.
        pid = next(iter(SELECTION['products']))
        raw = closet(f'/api/products/{pid}/outfits')
        response = self.request(f'/api/closet/outfits?user_id=RAW_TEST&product_id={pid}&mode=all')
        self.assertEqual([o['id'] for o in filter_outfits(raw['outfits'], 'all')], [o['id'] for o in response['outfits']])
        oid = raw['outfits'][0]['id']
        base = f'/api/outfits/{oid}?product_id={pid}'
        default, expanded = closet(base), closet(base + '&reverse_limit=40')
        for a, b in zip(default['groups'], expanded['groups']):
            self.assertEqual(a['products'], b['products'][:4])
        for invalid in ('0', '41', 'not-a-number'):
            with self.assertRaises(HTTPError):
                closet(base + '&reverse_limit=' + invalid)

    def test_same_outfit_different_anchor_keeps_separate_links(self):
        by_outfit = {}
        for pid, entry in SELECTION['products'].items():
            for oid in entry['outfits']:
                by_outfit.setdefault(oid, []).append(pid)
        checked = 0
        for oid, pids in by_outfit.items():
            if len(pids) < 2 or not is_eligible(oid):
                continue
            responses = [self.request(f'/api/closet/outfit?user_id=PAIR_TEST&product_id={pid}&outfit_id={oid}') for pid in pids]
            anchors = {r['matched_garment_id'] for r in responses}
            if len(anchors) < 2:
                continue
            checked += 1
            for pid, response in zip(pids, responses):
                expected = connection(SELECTION, pid, oid)
                for group in response['groups']:
                    self.assertCountEqual([p['id'] for p in group['products']], expected['groups'].get(group['id'], []))
                    if group['is_anchor']:
                        self.assertEqual(group['products'], [])
        self.assertGreater(checked, 0)

    def test_preferences_keep_candidates_and_owned_exclusion(self):
        pid = 'hm_0451380032'
        for uid, style in [('AUTO_CLASSIC', '클래식'), ('AUTO_STREET', '스트리트')]:
            for source, oid in self.style_examples(style):
                self.view(uid, source, oid)
        classic = self.request(f'/api/closet/outfits?user_id=AUTO_CLASSIC&product_id={pid}')['outfits']
        street = self.request(f'/api/closet/outfits?user_id=AUTO_STREET&product_id={pid}')['outfits']
        self.assertCountEqual([o['id'] for o in classic], [o['id'] for o in street])
        self.assertNotEqual([o['id'] for o in classic], [o['id'] for o in street])
        self.assertEqual({o['id']: o['similarity'] for o in classic}, {o['id']: o['similarity'] for o in street})
        self.assertTrue(all(0 <= o['preference_bonus'] <= .06 for o in classic + street))
        pid = next(iter(SELECTION['products']))
        oid = SELECTION['products'][pid]['outfits'][0]
        detail_path = f'/api/closet/outfit?user_id=OWNED_TEST&product_id={pid}&outfit_id={oid}'
        detail = self.request(detail_path)
        chosen = next(p['id'] for g in detail['groups'] for p in g['products'])
        self.update('OWNED_TEST', 'add', [chosen])
        profile = self.request('/api/closet/profile?user_id=OWNED_TEST')
        self.assertEqual(profile['products'][0]['slot'], 'bottom')
        detail = self.request(detail_path)
        self.assertNotIn(chosen, [p['id'] for g in detail['groups'] for p in g['products']])
        self.assertEqual(sum(g['owned_count'] for g in detail['groups']), 1)

    def test_automatic_history_is_explicit_distinct_and_persistent(self):
        uid = 'IMPLICIT_TEST'
        examples = self.style_examples('스트리트')
        pid, oid = examples[0]
        self.update(uid, 'add', list(SELECTION['products']))
        self.update(uid, 'styles', ['클래식'])  # Legacy settings do not control the new ranking.
        result = self.request(f'/api/closet/outfits?user_id={uid}&product_id={pid}')
        self.request(f'/api/closet/outfit?user_id={uid}&product_id={pid}&outfit_id={oid}')
        self.assertEqual(self.store.recent_outfit_views(uid), [])  # GET/auto-added demo is not feedback.
        self.assertTrue(all(o['preference_bonus'] == 0 for o in result['outfits']))
        for _ in range(5):
            self.view(uid, pid, oid)
        self.assertEqual(len(self.store.recent_outfit_views(uid)), 1)
        result = self.request(f'/api/closet/outfits?user_id={uid}&product_id={pid}')
        self.assertFalse(result['personalization']['ready'])
        for source, selected in examples[1:]:
            self.view(uid, source, selected)
        result = self.request(f'/api/closet/outfits?user_id={uid}&product_id={pid}')
        self.assertTrue(result['personalization']['ready'])
        self.assertEqual(result['personalization']['weights'], {'스트리트': 1.0})
        self.assertEqual(len(ProfileStore(self.db).recent_outfit_views(uid)), 3)
        self.assertEqual(self.store.recent_outfit_views('OTHER_AUTO_USER'), [])
        with self.assertRaises(HTTPError):
            self.view(uid, pid, 'not-an-outfit')
        self.assertEqual(len(self.store.recent_outfit_views(uid)), 3)

    def test_automatic_recency_unknown_tags_and_empty_history(self):
        now = 200 * 86400
        tags = {'a': {'style': '클래식'}, 'b': {'style': '스트리트'}, 'c': {'style': '스트리트'}}
        views = [{'outfit_id': 'a', 'viewed_at': now - 60 * 86400},
                 {'outfit_id': 'b', 'viewed_at': now}, {'outfit_id': 'c', 'viewed_at': now}]
        inferred = infer_preferences(views, tags, now)
        self.assertTrue(inferred['ready'])
        self.assertGreater(inferred['weights']['스트리트'], inferred['weights']['클래식'])
        self.assertFalse(infer_preferences(views + [{'outfit_id':'unknown', 'viewed_at':now}], tags, now + 91*86400)['ready'])
        outfits = [{'id':'a', 'similarity':.58}, {'id':'b', 'similarity':.60}]
        cold = rerank_automatic(outfits, tags, infer_preferences([], tags, now))
        self.assertEqual([o['id'] for o in cold], ['a', 'b'])

    def test_discovery_reuses_scope_without_survey_or_fake_views(self):
        uid = 'DISCOVER_TEST'
        path = '/api/closet/discover?user_id=' + uid
        first = self.request(path)
        self.assertEqual(first['audience'], 'all')
        self.assertEqual(len(first['outfits']), 12)
        self.assertEqual(len({o['id'] for o in first['outfits']}), 12)
        self.assertGreater(len({o['style'] for o in first['outfits']}), 1)
        for outfit in first['outfits']:
            self.assertIn(outfit['id'], SELECTION['products'][outfit['product_id']]['outfits'])
        self.update(uid, 'audience', ['women'])
        saved = self.request(path)
        self.assertEqual(saved['audience'], 'women')
        self.assertTrue(all(o['audience'] in ('womenswear', 'unisex') for o in saved['outfits']))
        self.assertEqual(ProfileStore(self.db).get(uid)['shopping_audience'], 'women')
        self.assertEqual(self.store.recent_outfit_views(uid), [])
        self.assertEqual(self.store.get(uid)['owned'], [])
        self.assertEqual(self.request(path + '&audience=kids')['outfits'], [])
        self.assertEqual(self.request('/api/closet/discover?user_id=OTHER_DISCOVER')['audience'], 'all')
        with self.assertRaises(HTTPError):
            self.request(path + '&audience=invalid')

    def test_discovery_keeps_variety_with_behavior(self):
        pool = [{'id':str(i), 'style':style, 'slot':slot, 'audience':'womenswear'}
                for i, (style, slot) in enumerate([('클래식','top'), ('스트리트','bottom'), ('클래식','outer'), ('스트리트','top')])]
        cold = discover(pool, 'women', {'weights':{}}, limit=4)
        warm = discover(pool, 'women', {'weights':{'스트리트':1}}, limit=4)
        self.assertNotEqual(cold[0]['id'], warm[0]['id'])
        self.assertEqual({o['id'] for o in cold}, {o['id'] for o in warm})

    def test_registration_defaults_validation_and_explicit_override(self):
        uid = 'SIGNUP_TEST'
        path = '/api/closet/registration?user_id=' + uid
        self.request(path, {'gender':'F', 'age_group':'20s'})
        result = self.request('/api/closet/discover?user_id=' + uid)
        self.assertEqual(result['audience'], 'women')
        self.assertFalse(result['personalization']['ready'])
        self.assertEqual(ProfileStore(self.db).get(uid)['registration'], {'gender':'F', 'age_group':'20s'})
        self.update(uid, 'audience', ['all'])
        self.request(path, {'gender':'M', 'age_group':'30s'})
        self.assertEqual(self.request('/api/closet/discover?user_id=' + uid)['audience'], 'all')
        self.assertEqual(self.store.recent_outfit_views(uid), [])
        with self.assertRaises(HTTPError):
            self.request(path, {'gender':'M', 'age_group':'invalid'})
        self.assertEqual(self.store.get(uid)['registration']['age_group'], '30s')
        self.request('/api/closet/registration?user_id=SIGNUP_UNKNOWN', {'gender':'unknown','age_group':'unknown'})
        self.assertEqual(self.request('/api/closet/discover?user_id=SIGNUP_UNKNOWN')['audience'], 'all')

    def test_registration_cohort_requires_evidence_then_own_history_wins(self):
        account = {'gender':'F', 'age_group':'40s'}
        target = 'COHORT_TARGET'
        self.request('/api/closet/registration?user_id=' + target, account)
        path = '/api/closet/discover?user_id=' + target
        street = self.style_examples('스트리트', 'women')
        classic = self.style_examples('클래식', 'women')
        self.assertEqual((len(street), len(classic)), (3, 3))
        for number in range(3):
            uid = 'COHORT_PEER_' + str(number)
            self.request('/api/closet/registration?user_id=' + uid, account)
            for pid, oid in street:
                self.view(uid, pid, oid)
            if number < 2:
                self.assertFalse(self.request(path)['personalization']['ready'])
        result = self.request(path)['personalization']
        self.assertEqual(result['source'], 'registration_cohort')
        self.assertEqual(result['cohort_members'], 3)
        self.assertEqual(result['distinct_outfits'], 0)
        self.assertEqual(result['weights'], {'스트리트':1.0})
        recommendations = self.request('/api/closet/outfits?user_id=' + target + '&product_id=hm_0451380032')['outfits']
        self.assertTrue(all(o['preference_bonus'] <= .03 for o in recommendations))
        self.request('/api/closet/registration?user_id=COHORT_OTHER', {'gender':'F', 'age_group':'50s'})
        self.assertFalse(self.request('/api/closet/discover?user_id=COHORT_OTHER')['personalization']['ready'])
        for pid, oid in classic:
            self.view(target, pid, oid)
        own = self.request(path)['personalization']
        self.assertEqual(own['source'], 'outfit_views')
        self.assertEqual(own['weights'], {'클래식':1.0})

    def test_moodboard_choices_persist_without_faking_views_or_ownership(self):
        uid = 'MOOD_TEST'
        path = '/api/closet/survey?user_id=' + uid
        data = self.request(path)
        self.assertEqual(len(data['outfits']), 89)
        self.assertEqual(len({o['id'] for o in data['outfits']}), 89)
        self.assertFalse(data['survey']['completed'])
        chosen = [oid for _, oid in self.style_examples('스트리트')]
        self.request(path, {'selected':chosen, 'skipped':False})
        self.assertEqual(ProfileStore(self.db).get(uid)['survey']['selected'], chosen)
        self.assertEqual(self.store.recent_outfit_views(uid), [])
        self.assertEqual(self.store.get(uid)['owned'], [])
        result = self.request('/api/closet/discover?user_id=' + uid)
        self.assertEqual(result['personalization']['source'], 'survey_selection')
        self.assertEqual(result['personalization']['weights'], {'스트리트':1.0})
        self.assertFalse(self.request('/api/closet/survey?user_id=MOOD_OTHER')['survey']['completed'])
        replacement = [self.style_examples('클래식')[0][1]]
        self.request(path, {'selected':replacement})
        self.assertEqual(self.store.get(uid)['survey']['selected'], replacement)
        self.assertEqual(self.request('/api/closet/discover?user_id=' + uid)['personalization']['weights'], {'클래식':1.0})
        self.request(path, {'selected':[], 'skipped':True})
        self.assertEqual(self.store.get(uid)['survey']['selected'], replacement)

    def test_moodboard_rejects_invalid_picks_and_allows_skip(self):
        path = '/api/closet/survey?user_id=MOOD_INVALID'
        ids = list(SELECTION['outfits'])
        for selected in ([], ids[:6], [ids[0],ids[0]], ['bad-id'], [3], 'not-a-list'):
            with self.assertRaises(HTTPError):
                self.request(path, {'selected':selected})
        self.assertNotIn('survey', self.store.get('MOOD_INVALID'))
        self.request(path, {'selected':[], 'skipped':True})
        self.assertTrue(self.request(path)['survey']['completed'])
        self.assertFalse(self.request('/api/closet/discover?user_id=MOOD_INVALID')['personalization']['ready'])

    def test_moodboard_prior_blends_with_later_behavior(self):
        uid = 'MOOD_BLEND'
        classic = [oid for _, oid in self.style_examples('클래식')]
        self.request('/api/closet/survey?user_id=' + uid, {'selected':classic})
        for pid, oid in self.style_examples('스트리트'):
            self.view(uid, pid, oid)
        pref = self.request('/api/closet/discover?user_id=' + uid)['personalization']
        self.assertEqual(pref['source'], 'survey_and_views')
        self.assertGreater(pref['weights']['스트리트'], 0)
        self.assertGreater(pref['weights']['클래식'], 0)
        self.assertAlmostEqual(sum(pref['weights'].values()), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
