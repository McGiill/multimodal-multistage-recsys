"""Exercise the cohort guarantee through the serving API, with an isolated DB."""
import hashlib
import json
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen
import tests.closet.integration.test_integration as original
from src.closet.runtime_paths import data_demo, data_root
from src.closet.shop.server import COHORTS, ACTIVE_PRODUCTS, SELECTION, closet
import src.closet.shop.publisher_models as models
class GenderFiveChecks(original.IntegrationTest):
    expected_products = 37
    expected_cohorts = [('M','MEN',22),('F','WOMEN',15)]
    expected_photos = 226
    expected_verified = 224
    def register(self, uid, gender):
        return self.request('/api/closet/registration?user_id='+uid, {'gender':gender,'age_group':'20s'})

    def test_every_product_has_five_matching_outfits_and_reverse_products(self):
        self.assertEqual(len(COHORTS),self.expected_products)
        for gender,label,count in self.expected_cohorts:
            uid='FIVE_'+gender;self.register(uid,gender)
            catalog=self.request('/api/closet/catalog?user_id='+uid)
            self.update(uid,'add',catalog['demo_ids'])
            self.assertEqual(len(catalog['demo_ids']),count)
            self.assertEqual(set(catalog['demo_ids']),{p for p,t in COHORTS.items() if t['gender']==label})
            self.assertCountEqual([p['id'] for p in catalog['products']],catalog['demo_ids'])
            for pid in catalog['demo_ids']:
                whole=self.request(f'/api/closet/outfits?user_id={uid}&product_id={pid}')
                self.assertGreaterEqual(whole['total_count'],5,pid)
                self.assertEqual(len({o['id'] for o in whole['outfits']}),whole['total_count'])
                first=self.request(f'/api/closet/outfits?user_id={uid}&product_id={pid}&limit=5')
                self.assertEqual(len(first['outfits']),5)
                for outfit in whole['outfits']:
                    self.assertEqual(outfit['publisher_model_gender'],label)
                    detail=self.request(f'/api/closet/outfit?user_id={uid}&product_id={pid}&outfit_id={outfit["id"]}')
                    products=[p for g in detail['groups'] for p in g['products']]
                    self.assertTrue(products,(pid,outfit['id']))
                    self.assertTrue(all(p['id']!=pid for p in products))
                    for p in products:
                        with urlopen(self.base+p['image']) as response:self.assertGreater(len(response.read()),500)

    def test_registration_changes_catalog_without_erasing_owned_items(self):
        uid='FIVE_CHANGE';self.register(uid,'M')
        men=self.request('/api/closet/catalog?user_id='+uid)['demo_ids'];self.update(uid,'add',men)
        self.register(uid,'F')
        women=self.request('/api/closet/catalog?user_id='+uid)['demo_ids']
        self.assertFalse(set(men)&set(women))
        self.assertCountEqual(self.request('/api/closet/profile?user_id='+uid)['owned'],men)
        self.register(uid,'unknown')
        self.assertEqual(len(self.request('/api/closet/catalog?user_id='+uid)['demo_ids']),self.expected_products)
        retired={v['replaces'] for v in COHORTS.values() if 'replaces' in v}
        self.assertFalse(retired&set(self.request('/api/closet/catalog')['demo_ids']))

    def test_visible_wardrobe_switches_cohort_and_keeps_ownership(self):
        uid = 'WARDROBE_SWITCH'
        all_ids = list(COHORTS)
        self.update(uid, 'add', all_ids)
        for gender, label, count in self.expected_cohorts:
            self.register(uid, gender)
            profile = self.request('/api/closet/profile?user_id=' + uid)
            visible = {p['id'] for p in profile['products']}
            self.assertEqual(visible, {p for p, entry in COHORTS.items() if entry['gender'] == label})
            self.assertEqual(len(visible), count)
            self.assertCountEqual(profile['owned'], all_ids)
            self.assertEqual(profile['hidden_owned_count'], self.expected_products - count)
            self.assertEqual(profile['audience'], {'M':'men', 'F':'women'}[gender])
            # Previously selected clothing scope must not survive a gender switch.
            self.update(uid, 'audience', [profile['audience']])
        self.register(uid, 'unknown')
        profile = self.request('/api/closet/profile?user_id=' + uid)
        self.assertEqual(len(profile['products']), self.expected_products)
        self.assertEqual(profile['hidden_owned_count'], 0)
        self.assertEqual(profile['audience'], 'all')
        self.register(uid, 'M')
        self.assertEqual(len(self.request('/api/closet/profile?user_id=' + uid)['products']), self.expected_cohorts[0][2])

    def test_all_images_and_publisher_evidence_are_bound(self):
        cfg=json.loads((data_demo()/'config.json').read_text(encoding='utf-8'))
        snap=json.loads((data_demo()/cfg['snapshot_file']).read_text(encoding='utf-8'))
        self.assertEqual(len(models.ROWS),self.expected_photos)
        self.assertEqual(sum(models.verified(o) for o in models.ROWS),self.expected_verified)
        for url,entry in snap['images'].items():
            if url.startswith('/media/outfit/') and not models.verified(url.rsplit('/',1)[-1]):
                with self.assertRaises(HTTPError):urlopen(self.base+url)
            else:
                with urlopen(self.base+url) as response:self.assertEqual(hashlib.sha256(response.read()).hexdigest(),entry['sha256'])
        self.assertEqual(closet('/health')['inference_calls'],0)

    def test_survey_seed_and_detail_cannot_bypass_gender(self):
        for gender,label in [('M','MEN'),('F','WOMEN')]:
            uid='FIVE_SCOPE_'+gender;self.register(uid,gender)
            survey=self.request('/api/closet/survey?user_id='+uid)
            self.assertTrue(survey['outfits'])
            self.assertTrue(all(models.gender(o['id'])==label for o in survey['outfits']))
            oid=survey['outfits'][0]['id']
            self.request('/api/closet/survey?user_id='+uid,{'selected':[oid]})
            seed=self.request('/api/closet/preference-seed?user_id='+uid)
            self.assertEqual(seed['effective_outfit_ids'],[oid]);self.assertTrue(seed['seed_products'])
            wrong=next(o for o in models.ROWS if models.verified(o) and models.gender(o)!=label)
            pid=next(p for p,oids in ACTIVE_PRODUCTS.items() if wrong in oids)
            with self.assertRaises(HTTPError):self.request(f'/api/closet/outfit?user_id={uid}&product_id={pid}&outfit_id={wrong}')


def load_tests(loader, standard_tests, pattern):
    names=[n for n in GenderFiveChecks.__dict__ if n.startswith('test_')]
    names+=['test_profile_persistence_and_isolation','test_unknown_product_not_saved','test_automatic_recency_unknown_tags_and_empty_history']
    return unittest.TestSuite(GenderFiveChecks(n) for n in names)


if __name__=='__main__':unittest.main(verbosity=2)
