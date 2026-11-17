"""Live API checks for a versioned, reviewed dataset; no source models required."""
import hashlib
import json
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen
import test_integration as original
from runtime_paths import data_root, data_demo
from server import SELECTION, ACTIVE_PRODUCTS, DEMO_STATS, CONFIG, closet
from audience import AUDIENCES, is_eligible, matches_audience


class ReviewedChecks(original.IntegrationTest):
    def test_dataset_counts_images_and_assets(self):
        report = json.loads((data_root()/'FULL_DATASET_REVIEW.json').read_text(encoding='utf-8'))['summary']
        health = closet('/health')
        self.assertEqual(health['mode'], 'curated_snapshot')
        self.assertFalse(health['supports_full_search'])
        self.assertEqual(health['inference_calls'], 0)
        for key, source in [('query_products','input_products'),('outfits','outfits'),('connections','connections')]:
            self.assertEqual(health[key], report[source])
        self.assertEqual(DEMO_STATS['unique_outfit_images'], report['outfits'])
        self.assertEqual(DEMO_STATS['product_outfit_connections'], report['connections'])
        cfg = json.loads((data_demo()/'config.json').read_text(encoding='utf-8'))
        snapshot = json.loads((data_demo()/cfg['snapshot_file']).read_text(encoding='utf-8'))
        for url, entry in snapshot['images'].items():
            with urlopen(self.base+url) as response:
                self.assertTrue(response.headers['Content-Type'].startswith('image/'))
                self.assertEqual(hashlib.sha256(response.read()).hexdigest(), entry['sha256'])
        for file in ('','shop.js','shop.css','closet.js','closet.css','moodboard.js','moodboard.css','cold_start.js'):
            with urlopen(self.base+'/shop/'+file) as response:
                self.assertGreater(len(response.read()),100)

    def test_every_pair_and_reverse_link(self):
        count = 0
        for pid, oids in ACTIVE_PRODUCTS.items():
            data = self.request(f'/api/closet/outfits?user_id=REVIEWED_EMPTY&product_id={pid}')
            self.assertCountEqual([o['id'] for o in data['outfits']], oids)
            self.assertLessEqual(len(oids),CONFIG.get('candidate_pool_limit',5))
            for oid in oids:
                count += 1
                selected = SELECTION['connections'][pid][oid]
                path = f'/api/closet/outfit?user_id=REVIEWED_EMPTY&product_id={pid}&outfit_id={oid}'
                detail = self.request(path)
                self.assertEqual(sum(g['is_anchor'] for g in detail['groups']),1)
                found = [p for g in detail['groups'] for p in g['products']]
                self.assertTrue(found)
                self.assertCountEqual([p['id'] for p in found], selected['products'])
                self.assertTrue(all(p['id'] != pid and p['similarity'] >= .5 for p in found))
                for ev in selected.get('reverse_evidence',[]):
                    item = next(p for p in found + detail['shoe_alternatives'] if p['id']==ev['product_id'])
                    self.assertAlmostEqual(item['similarity'], ev['similarity'],places=5)
        self.assertEqual(count,DEMO_STATS['product_outfit_connections'])
        # Newly added links obey ownership filtering too.
        report=json.loads((data_root()/'FULL_DATASET_REVIEW.json').read_text(encoding='utf-8'))
        for i, pair in enumerate(report['accepted']):
            ids=[r['product_id'] for r in pair['reverse']]
            uid=f'REVIEWED_OWNED_{i}'
            self.update(uid,'add',ids)
            d=self.request(f"/api/closet/outfit?user_id={uid}&product_id={pair['product_id']}&outfit_id={pair['outfit_id']}")
            self.assertFalse(set(ids)&{p['id'] for g in d['groups'] for p in g['products']})

    def test_scopes_survey_and_exclusions(self):
        for scope in ('men','women','unisex','all','kids'):
            uid='REVIEWED_SCOPE_'+scope
            self.update(uid,'audience',[scope])
            survey=self.request('/api/closet/survey?user_id='+uid)
            expected={oid for oid in SELECTION['outfits'] if is_eligible(oid) and matches_audience(AUDIENCES[oid],scope)}
            shown={o['id'] for o in survey['outfits']}
            self.assertTrue(shown <= expected)
            self.assertEqual(len(shown),len(expected))
            for pid,oids in ACTIVE_PRODUCTS.items():
                result=self.request(f'/api/closet/outfits?user_id={uid}&product_id={pid}&mode=all')
                self.assertFalse(result['supports_full_search'])
                self.assertEqual({o['id'] for o in result['outfits']},set(oids)&expected)
        for oid in SELECTION['outfits']:
            if not is_eligible(oid):
                with self.assertRaises(HTTPError) as error:urlopen(self.base+'/media/outfit/'+oid)
                self.assertEqual(error.exception.code,404)
        # A new photo participates in real survey preferences, without fake views.
        report=json.loads((data_root()/'FULL_DATASET_REVIEW.json').read_text(encoding='utf-8'))
        oid=report['accepted'][0]['outfit_id']
        self.request('/api/closet/survey?user_id=REVIEWED_SURVEY',{'selected':[oid]})
        self.assertEqual(self.store.get('REVIEWED_SURVEY')['survey']['selected'],[oid])
        pref=self.request('/api/closet/discover?user_id=REVIEWED_SURVEY')['personalization']
        self.assertEqual(pref['source'],'survey_selection')
        self.assertEqual(self.store.recent_outfit_views('REVIEWED_SURVEY'),[])


def load_tests(loader, standard_tests, pattern):
    names=['test_dataset_counts_images_and_assets','test_every_pair_and_reverse_link',
           'test_scopes_survey_and_exclusions','test_profile_persistence_and_isolation',
           'test_unknown_product_not_saved','test_preferences_keep_candidates_and_owned_exclusion',
           'test_automatic_history_is_explicit_distinct_and_persistent',
           'test_automatic_recency_unknown_tags_and_empty_history',
           'test_discovery_reuses_scope_without_survey_or_fake_views','test_discovery_keeps_variety_with_behavior',
           'test_registration_defaults_validation_and_explicit_override',
           'test_registration_cohort_requires_evidence_then_own_history_wins',
           'test_moodboard_rejects_invalid_picks_and_allows_skip','test_moodboard_prior_blends_with_later_behavior']
    return unittest.TestSuite(ReviewedChecks(name) for name in names)

if __name__=='__main__':unittest.main(verbosity=2)
