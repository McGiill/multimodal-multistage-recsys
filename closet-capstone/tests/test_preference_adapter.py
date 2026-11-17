import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_closet_demo/shop_integration'))
from team_bridge.preference_adapter import build_seed_vector, blend_user_query, rank_seed_candidates
from preference_seed import build_preference_seed


class AdapterTests(unittest.TestCase):
    def payload(self):
        return {'schema_version':'closet-preference-seed-v1','seed_products':[
            {'product_id':'hm_0000000001','relative_weight':.75},
            {'product_id':'hm_0000000002','relative_weight':.25}]}

    def test_exact_mapping_partial_catalog_and_mask(self):
        vectors=np.eye(64,dtype=np.float32)[:3]
        payload=self.payload();ids=['P1','P2','P3']
        self.assertIsNone(build_seed_vector(payload,ids,vectors)['vector'])
        partial=build_seed_vector(payload,ids,vectors,{'hm_0000000001':'P1'})
        self.assertEqual(partial['coverage_weight'],.75)
        self.assertEqual(partial['unmapped_product_ids'],['hm_0000000002'])
        self.assertTrue(np.allclose(partial['vector'],vectors[0]))
        ranked=rank_seed_candidates(partial,ids,vectors,[False,True,True])
        self.assertEqual([r['product_id'] for r in ranked],['P2','P3'])

    def test_weak_blend_declines_with_real_events(self):
        vectors=np.eye(64,dtype=np.float32)[:2]
        result=build_seed_vector(self.payload(),['hm_0000000001','hm_0000000002'],vectors)
        cold,alpha=blend_user_query(vectors[1],result)
        warm,beta=blend_user_query(vectors[1],result,20)
        self.assertAlmostEqual(alpha,.2);self.assertAlmostEqual(beta,.04)
        self.assertLess(np.linalg.norm(warm-vectors[1]),np.linalg.norm(cold-vectors[1]))
        self.assertAlmostEqual(float(np.linalg.norm(cold)),1,places=6)

    def test_missing_and_invalid_vectors_are_not_silently_used(self):
        for matrix in (np.zeros((2,512)),np.full((2,64),np.nan)):
            with self.assertRaises(ValueError):build_seed_vector(self.payload(),['A','B'],matrix)
        zero=build_seed_vector(self.payload(),['hm_0000000001','hm_0000000002'],np.zeros((2,64)))
        self.assertIsNone(zero['vector'])
        q=np.eye(64,dtype=np.float32)[0]
        out,weight=blend_user_query(q,zero)
        self.assertTrue(np.array_equal(q,out));self.assertEqual(weight,0)
        p=self.payload();p['seed_products'][0]['relative_weight']=-1
        with self.assertRaises(ValueError):build_seed_vector(p,['A','B'],np.eye(64)[:2])

    def test_outfit_and_garment_weights_do_not_depend_on_alternative_count(self):
        profile={'survey':{'selected':['a','b'],'completed':True,'updated_at':1}}
        links={'a':{'g1':{'x':{'similarity':.7}},'g2':{'y':{'similarity':.6},'z':{'similarity':.8}}},
               'b':{'g3':{'q':{'similarity':.7}}}}
        p=build_preference_seed('U',profile,'v1','all',lambda o,a:True,lambda o:links[o])
        weights={s['product_id']:s['relative_weight'] for s in p['seed_products']}
        self.assertEqual(weights,{'x':.25,'y':.125,'z':.125,'q':.5})
        self.assertEqual(p,build_preference_seed('U',profile,'v1','all',lambda o,a:True,lambda o:links[o]))

if __name__=='__main__':unittest.main(verbosity=2)
