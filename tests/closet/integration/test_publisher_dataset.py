"""Publisher metadata filtering across survey, wardrobe, detail and seed export."""
import hashlib
import json
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen
import tests.closet.integration.test_integration as original
from src.closet.runtime_paths import data_root, data_demo
from src.closet.shop.server import ACTIVE_PRODUCTS, SELECTION, DEMO_STATS, closet
from src.closet.shop.audience import AUDIENCES, matches_audience
import src.closet.shop.publisher_models as models
class PublisherChecks(original.IntegrationTest):
    def register(self, uid, gender):
        return self.request('/api/closet/registration?user_id='+uid, {'gender':gender,'age_group':'20s'})

    def test_review_hashes_and_excluded_media(self):
        self.assertTrue(models.ENABLED)
        self.assertEqual(len(models.ROWS),204)
        self.assertEqual(sum(models.verified(o) for o in models.ROWS),202)
        self.assertEqual(DEMO_STATS['unique_outfit_images'],202)
        self.assertEqual(DEMO_STATS['product_outfit_connections'],231)
        self.assertEqual(len(ACTIVE_PRODUCTS),37)
        cfg=json.loads((data_demo()/'config.json').read_text(encoding='utf-8'))
        self.assertEqual(hashlib.sha256((data_demo()/cfg['publisher_model_review_file']).read_bytes()).hexdigest(),cfg['publisher_model_review_sha256'])
        snap=json.loads((data_demo()/cfg['snapshot_file']).read_text(encoding='utf-8'))
        for url,entry in snap['images'].items():
            if url.startswith('/media/outfit/') and not models.verified(url.rsplit('/',1)[-1]):
                with self.assertRaises(HTTPError) as error:urlopen(self.base+url)
                self.assertEqual(error.exception.code,404)
            else:
                with urlopen(self.base+url) as response:self.assertEqual(hashlib.sha256(response.read()).hexdigest(),entry['sha256'])
        self.assertEqual(closet('/health')['inference_calls'],0)

    def test_all_product_pairs_obey_registered_gender(self):
        for gender,label in [('M','MEN'),('F','WOMEN'),('unknown',None)]:
            uid='PUBLISHER_'+gender;self.register(uid,gender);self.update(uid,'audience',['all'])
            for pid,oids in ACTIVE_PRODUCTS.items():
                base=f'/api/closet/outfits?user_id={uid}&product_id={pid}'
                result=self.request(base)
                expected={o for o in oids if label is None or models.gender(o)==label}
                self.assertEqual({o['id'] for o in result['outfits']},expected)
                self.assertEqual(result['publisher_model_filter']['gender'],label)
                for oid in oids:
                    url=f'/api/closet/outfit?user_id={uid}&product_id={pid}&outfit_id={oid}'
                    if oid in expected:
                        detail=self.request(url);self.assertEqual(detail['publisher_model_gender'],models.gender(oid))
                        allowed=SELECTION['connections'][pid][oid]['products']
                        self.assertCountEqual([p['id'] for g in detail['groups'] for p in g['products']],allowed)
                    else:
                        with self.assertRaises(HTTPError) as error:self.request(url)
                        self.assertEqual(error.exception.code,404)

    def test_discover_survey_and_scope_cannot_bypass_model_filter(self):
        for gender,label in [('M','MEN'),('F','WOMEN'),('unknown',None)]:
            uid='SOURCE_SCOPES_'+gender;self.register(uid,gender)
            for scope in ['all','men','women','unisex','kids']:
                self.update(uid,'audience',[scope])
                expected={o for o in models.ROWS if models.verified(o) and matches_audience(AUDIENCES[o],scope)
                          and (label is None or models.gender(o)==label)}
                survey=self.request('/api/closet/survey?user_id='+uid)
                self.assertEqual({o['id'] for o in survey['outfits']},expected)
                discovery=self.request(f'/api/closet/discover?user_id={uid}&audience={scope}')
                self.assertTrue({o['id'] for o in discovery['outfits']} <= expected)
                self.assertEqual(len(discovery['outfits']),min(12,len(expected)))
                self.assertEqual(self.store.recent_outfit_views(uid),[])

    def test_gender_change_retires_previous_choice_without_erasing_history(self):
        uid='SOURCE_CHANGE';self.register(uid,'M');self.update(uid,'audience',['all'])
        male=next(o for o in models.ROWS if models.gender(o)=='MEN')
        female=next(o for o in models.ROWS if models.gender(o)=='WOMEN')
        self.request('/api/closet/survey?user_id='+uid,{'selected':[male]})
        seed=self.request('/api/closet/preference-seed?user_id='+uid)
        self.assertEqual(seed['effective_outfit_ids'],[male]);self.assertTrue(seed['seed_products'])
        self.register(uid,'F')
        changed=self.request('/api/closet/preference-seed?user_id='+uid)
        self.assertEqual(changed['selected_outfit_ids'],[male]);self.assertEqual(changed['inactive_outfit_ids'],[male])
        self.assertFalse(changed['seed_products']);self.assertNotEqual(seed['snapshot_id'],changed['snapshot_id'])
        self.assertEqual(changed['publisher_model_filter']['gender'],'WOMEN')
        profile=self.request('/api/closet/profile?user_id='+uid)
        self.assertEqual(profile['survey']['selected'],[]);self.assertFalse(profile['personalization']['ready'])
        with self.assertRaises(HTTPError):self.request('/api/closet/survey?user_id='+uid,{'selected':[male]})
        pid=next(p for p,oids in ACTIVE_PRODUCTS.items() if male in oids)
        with self.assertRaises(HTTPError):self.request('/api/closet/events?user_id='+uid,{'event_type':'outfit_view','outfit_id':male,'product_id':pid})
        self.assertEqual(self.store.recent_outfit_views(uid),[])
        self.request('/api/closet/survey?user_id='+uid,{'selected':[female]})
        self.assertEqual(self.request('/api/closet/preference-seed?user_id='+uid)['effective_outfit_ids'],[female])

    def test_held_posts_never_enter_survey_or_seed(self):
        uid='SOURCE_HELD';held=[o for o in models.ROWS if not models.verified(o)]
        self.assertEqual(len(held),2)
        for oid in held:
            with self.assertRaises(HTTPError):self.request('/api/closet/survey?user_id='+uid,{'selected':[oid]})
        # Simulate a pre-migration saved survey. Keep the original history, ignore it in output.
        self.store.save_survey(uid,held)
        payload=self.request('/api/closet/preference-seed?user_id='+uid)
        self.assertEqual(payload['inactive_outfit_ids'],held);self.assertFalse(payload['seed_products'])

    def test_pagination_and_owned_exclusion_after_gender_filter(self):
        uid='SOURCE_PAGE';self.register(uid,'M');self.update(uid,'audience',['all'])
        pid=next(p for p,oids in ACTIVE_PRODUCTS.items() if sum(models.gender(o)=='MEN' for o in oids)>5)
        base=f'/api/closet/outfits?user_id={uid}&product_id={pid}'
        whole=self.request(base);first=self.request(base+'&limit=5');rest=self.request(base+'&offset=5&limit=5')
        self.assertEqual(first['outfits']+rest['outfits'],whole['outfits']);self.assertTrue(first['has_more'])
        oid=whole['outfits'][0]['id'];ids=SELECTION['connections'][pid][oid]['products'];self.update(uid,'add',ids)
        detail=self.request(f'/api/closet/outfit?user_id={uid}&product_id={pid}&outfit_id={oid}')
        self.assertFalse(set(ids)&{p['id'] for g in detail['groups'] for p in g['products']})
        self.assertEqual(self.request(base)['total_count'],whole['total_count'])


def load_tests(loader, standard_tests, pattern):
    names=[n for n in PublisherChecks.__dict__ if n.startswith('test_')]
    names+=['test_profile_persistence_and_isolation','test_unknown_product_not_saved',
            'test_automatic_recency_unknown_tags_and_empty_history']
    return unittest.TestSuite(PublisherChecks(n) for n in names)

if __name__=='__main__':unittest.main(verbosity=2)
