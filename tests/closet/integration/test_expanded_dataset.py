"""Expanded pool, post-filter pagination, and explicit cold-start handoff."""
import unittest
import json
from urllib.error import HTTPError
import tests.closet.integration.test_reviewed_dataset as reviewed
from src.closet.shop.server import ACTIVE_PRODUCTS, SELECTION, DATASET_VERSION, TAGS, INPUT_SLOTS
from src.closet.shop.audience import AUDIENCES, is_eligible


class ExpandedChecks(reviewed.ReviewedChecks):
    def test_display_limits_after_scope_and_taste(self):
        pid=next(p for p,oids in ACTIVE_PRODUCTS.items() if len(oids)>5)
        for scope in ('all','men','women'):
            uid='PAGES_'+scope
            self.update(uid,'audience',[scope])
            base=f'/api/closet/outfits?user_id={uid}&product_id={pid}'
            whole=self.request(base)
            first=self.request(base+'&limit=5')
            rest=self.request(base+'&limit=5&offset=5')
            self.assertEqual(first['outfits']+rest['outfits'],whole['outfits'])
            self.assertEqual(first['total_count'],len(whole['outfits']))
            self.assertEqual(first['excluded_count'],whole['excluded_count'])
            self.assertEqual(first['has_more'],len(whole['outfits'])>5)
            if whole['outfits']:
                chosen=whole['outfits'][-1]['id']
                self.request('/api/closet/survey?user_id='+uid,{'selected':[chosen]})
                warm=self.request(base);page=self.request(base+'&limit=5')
                self.assertEqual(page['outfits'],warm['outfits'][:5])
        for suffix in ('&limit=0','&limit=41','&offset=-1','&limit=no'):
            with self.assertRaises(HTTPError):self.request(base+suffix)

    def test_seed_roundtrip_skip_replace_scope_and_version(self):
        uid='SEED_TEST';url='/api/closet/preference-seed?user_id='+uid
        fresh=self.request(url)
        self.assertEqual(fresh['status'],'not_started');self.assertEqual(fresh['seed_products'],[])
        self.request('/api/closet/survey?user_id='+uid,{'selected':[],'skipped':True})
        skip=self.request(url)
        self.assertEqual(skip['status'],'skipped');self.assertEqual(skip['seed_products'],[])
        self.request('/api/closet/registration?user_id='+uid,{'gender':'F','age_group':'20s'})
        female=next(o for o,a in AUDIENCES.items() if a=='womenswear' and is_eligible(o))
        saved=self.request('/api/closet/survey?user_id='+uid,{'selected':[female]})
        self.assertEqual(saved['survey']['dataset_version'],DATASET_VERSION)
        payload=self.request(url)
        self.assertEqual(payload,self.request(url))
        self.assertEqual(payload['operation'],'replace')
        self.assertEqual(payload['effective_outfit_ids'],[female])
        self.assertEqual(payload['registration'],{'gender':'F','age_group':'20s'})
        self.assertAlmostEqual(sum(s['relative_weight'] for s in payload['seed_products']),1)
        for seed in payload['seed_products']:
            self.assertTrue(seed['product_id'].startswith('hm_'))
            self.assertEqual(len(seed['source_article_id']),10)
            for ev in seed['evidence']:
                self.assertEqual(ev['outfit_id'],female)
                self.assertGreaterEqual(ev['similarity'],.5)
        self.assertFalse(payload['model_applied'])
        self.assertEqual(self.store.recent_outfit_views(uid),[])
        self.assertEqual(self.store.get(uid)['owned'],[])
        self.assertEqual(self.request('/api/closet/preference-seed?user_id=SEED_OTHER')['seed_products'],[])
        self.request('/api/closet/survey?user_id='+uid,{'selected':[],'skipped':True})
        retained=self.request(url)
        self.assertTrue(retained['retained_previous_selection'])
        self.assertEqual(retained['selected_at'],payload['selected_at'])
        self.assertEqual(retained['seed_products'],payload['seed_products'])
        self.update(uid,'audience',['men'])
        filtered=self.request(url)
        self.assertEqual(filtered['seed_products'],[])
        self.assertEqual(filtered['inactive_outfit_ids'],[female])
        self.assertNotEqual(filtered['snapshot_id'],retained['snapshot_id'])
        male=next(o for o,a in AUDIENCES.items() if a=='menswear' and is_eligible(o))
        self.request('/api/closet/survey?user_id='+uid,{'selected':[male]})
        replaced=self.request(url)
        self.assertEqual(replaced['selected_outfit_ids'],[male])
        self.assertEqual(replaced['effective_outfit_ids'],[male])

    def test_style_alias_normalized_and_coverage_report(self):
        self.assertNotIn('street',{t.get('style') for t in TAGS.values()})
        report=json.loads((reviewed.data_root()/'FULL_DATASET_REVIEW.json').read_text(encoding='utf-8'))['summary']
        for row in report['coverage']:
            oids=ACTIVE_PRODUCTS[row['product_id']]
            self.assertEqual(row['total'],len(oids))
            self.assertEqual(row['men'],sum(AUDIENCES[o] in ('menswear','unisex') for o in oids))
            self.assertEqual(row['women'],sum(AUDIENCES[o] in ('womenswear','unisex') for o in oids))


def load_tests(loader, standard_tests, pattern):
    base=reviewed.load_tests(loader,standard_tests,pattern)
    names=[case._testMethodName for case in base]
    names+=['test_display_limits_after_scope_and_taste','test_seed_roundtrip_skip_replace_scope_and_version',
            'test_style_alias_normalized_and_coverage_report']
    return unittest.TestSuite(ExpandedChecks(name) for name in names)

if __name__=='__main__':unittest.main(verbosity=2)
