"""All 60 inputs, live API/media checks, gender isolation and preference handoff."""
import json
import unittest
from tests.closet.integration.test_gender_five_dataset import GenderFiveChecks
from src.closet.runtime_paths import data_root


class Expanded60Checks(GenderFiveChecks):
    expected_products = 60
    expected_cohorts = [('M', 'MEN', 30), ('F', 'WOMEN', 30)]
    expected_photos = 310
    expected_verified = 308

    def test_every_cohort_link_survives_all60_owned(self) -> None:
        """Every displayed look must retain an unowned alternative, not just repaired looks."""
        if not (data_root() / 'FINISH_REVIEW.json').exists():
            self.skipTest('Bundle predates the final empty-link repair')
        coverage = json.loads((data_root() / 'QUALITY68_REVIEW.json').read_text(encoding='utf-8'))['coverage']
        uid = 'ALL60_COMPLETE_COVERAGE'
        self.register(uid, 'unknown')
        owned = self.request('/api/closet/catalog?user_id=' + uid)['demo_ids']
        self.update(uid, 'add', owned)
        checked = 0
        for gender, label, _ in self.expected_cohorts:
            self.register(uid, gender)
            for pid in self.request('/api/closet/catalog?user_id=' + uid)['demo_ids']:
                looks = self.request(f'/api/closet/outfits?user_id={uid}&product_id={pid}')['outfits']
                self.assertGreaterEqual(len(looks), 5, pid)
                self.assertCountEqual([o['id'] for o in looks], coverage[pid]['outfit_ids'])
                for look in looks:
                    detail = self.request(f'/api/closet/outfit?user_id={uid}&product_id={pid}&outfit_id={look["id"]}')
                    products = {p['id'] for g in detail['groups'] for p in g['products']}
                    self.assertTrue(products, (pid, look['id']))
                    self.assertFalse(products & set(owned), (pid, look['id']))
                    self.assertEqual(detail['publisher_model_gender'], label)
                    checked += 1
        self.assertEqual(checked, 318)

    def test_saved_outfits_persist_without_fabricating_preferences(self) -> None:
        from src.closet.shop.store import ProfileStore
        from src.closet.shop.server import COHORTS
        from urllib.error import HTTPError
        uid = 'SAVE_LOOK_TEST'
        self.register(uid, 'M')
        pid = next(pid for pid, item in COHORTS.items() if item['gender'] == 'MEN')
        oid = self.request(f'/api/closet/outfits?user_id={uid}&product_id={pid}')['outfits'][0]['id']
        path = '/api/closet/saved-outfits?user_id=' + uid
        payload = {'product_id':pid, 'outfit_id':oid, 'saved':True}
        before = self.store.get(uid)
        seed = self.request('/api/closet/preference-seed?user_id=' + uid)
        self.request(path, payload)
        first = self.request(path)['outfits'][0]
        self.request(path, payload)
        result = self.request(path)
        self.assertEqual(result['total_count'], 1)
        self.assertEqual(result['outfits'][0]['saved_at'], first['saved_at'])
        self.assertEqual(ProfileStore(self.store.path).saved_outfits(uid)[0]['outfit_id'], oid)
        self.assertEqual(self.request('/api/closet/saved-outfits?user_id=SAVE_OTHER')['outfits'], [])
        self.assertEqual(self.store.get(uid), before)
        self.assertEqual(self.store.recent_outfit_views(uid), [])
        saved_seed = self.request('/api/closet/preference-seed?user_id='+uid)
        self.assertEqual(saved_seed['effective_saved_outfit_ids'], [oid])
        self.assertTrue(saved_seed['seed_products'])
        self.assertNotEqual(saved_seed['snapshot_id'], seed['snapshot_id'])
        self.assertEqual(saved_seed['operation'], 'replace')
        self.assertEqual(saved_seed['snapshot_id'], self.request('/api/closet/preference-seed?user_id='+uid)['snapshot_id'])
        detail = self.request(f'/api/closet/outfit?user_id={uid}&product_id={pid}&outfit_id={oid}')
        self.assertTrue(detail['saved'])
        self.register(uid, 'F')
        hidden = self.request(path)
        self.assertEqual(hidden['outfits'], [])
        self.assertEqual(hidden['hidden_count'], 1)
        self.assertEqual(self.request('/api/closet/preference-seed?user_id='+uid)['effective_saved_outfit_ids'], [])
        with self.assertRaises(HTTPError): self.request(path, payload)
        self.register(uid, 'M')
        self.assertEqual(len(self.request(path)['outfits']), 1)
        self.request(path, {**payload, 'saved':False})
        self.request(path, {**payload, 'saved':False})
        self.assertEqual(self.request(path)['total_count'], 0)
        self.assertEqual(self.request('/api/closet/preference-seed?user_id='+uid), seed)
        for invalid in [{**payload, 'outfit_id':'does-not-exist'}, {**payload, 'product_id':'bad'}, {**payload, 'saved':'true'}]:
            with self.assertRaises(HTTPError): self.request(path, invalid)
        self.assertEqual(self.request(path)['total_count'], 0)

    @classmethod
    def setUpClass(cls) -> None:
        review = data_root() / 'QUALITY68_REVIEW.json'
        if review.exists():
            summary = json.loads(review.read_text(encoding='utf-8'))['summary']
            cls.expected_photos = summary['stored_outfit_records']
            cls.expected_verified = summary['publisher_verified_outfits']
        super().setUpClass()

    def test_quality68_replacements_and_retired_links(self) -> None:
        path = data_root() / 'QUALITY68_REVIEW.json'
        if not path.exists():
            self.skipTest('Bundle predates the 68-connection repair')
        report = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(len(report['pairs']), 69)
        self.assertEqual(sum(p['original_status'] != 'ready' for p in report['pairs']), 68)
        uid = 'QUALITY68_ALL_OWNED'
        self.register(uid, 'unknown')
        all_ids = self.request('/api/closet/catalog?user_id=' + uid)['demo_ids']
        self.update(uid, 'add', all_ids)
        for gender, label, _ in self.expected_cohorts:
            self.register(uid, gender)
            for p in (p for p in report['pairs'] if p['gender'] == label):
                live = self.request(f'/api/closet/outfits?user_id={uid}&product_id={p["product_id"]}')
                ids = {o['id'] for o in live['outfits']}
                self.assertIn(p['outfit_id'], ids, p['key'])
                self.assertNotIn(p['old_outfit_id'], ids, p['key'])
                detail = self.request(f'/api/closet/outfit?user_id={uid}&product_id={p["product_id"]}&outfit_id={p["outfit_id"]}')
                self.assertEqual(detail['publisher_model_gender'], label)
                self.assertEqual(detail['matched_garment_id'], p['matched_garment_id'])
                returned = {v['id'] for g in detail['groups'] for v in g['products']}
                self.assertTrue(set(p['reverse_products']) <= returned, p['key'])
                self.assertFalse(returned & set(all_ids), p['key'])
                for expected in p['reverse_groups']:
                    group = next(g for g in detail['groups'] if g['id'] == expected['id'])
                    self.assertEqual(group['box'], expected['box'], p['key'])
                from urllib.error import HTTPError
                with self.assertRaises(HTTPError):
                    self.request(f'/api/closet/outfit?user_id={uid}&product_id={p["old_product_id"]}&outfit_id={p["old_outfit_id"]}')

    def test_dataset_product_migration_preserves_other_history(self) -> None:
        import tempfile
        from pathlib import Path
        from src.closet.shop.store import ProfileStore
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory) / 'profiles.sqlite3')
            store.update('saved', 'add', ['old', 'new', 'unrelated'])
            store.set_registration('saved', {'gender':'M', 'age_group':'20s'})
            store.save_survey('saved', ['retained_outfit'])
            before = store.get('saved')
            self.assertEqual(store.migrate_products({'old':'new'}), 1)
            after = store.get('saved')
            self.assertEqual(after['owned'], ['new', 'unrelated'])
            self.assertEqual({k:v for k,v in after.items() if k != 'owned'},
                             {k:v for k,v in before.items() if k != 'owned'})
            self.assertEqual(store.migrate_products({'old':'new'}), 0)

    def test_reverse_repair_survives_all_owned_filter(self) -> None:
        path = data_root() / 'REVERSE52_REVIEW.json'
        if not path.exists():
            self.skipTest('This older bundle predates the reverse-product repair')
        report = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(len(report['pairs']), report['summary']['repaired_connections'])
        uid = 'REVERSE_REPAIR_ALL_OWNED'
        self.register(uid, 'unknown')
        all_ids = self.request('/api/closet/catalog?user_id=' + uid)['demo_ids']
        self.assertEqual(len(all_ids), 60)
        self.update(uid, 'add', all_ids)
        for gender, label, _ in self.expected_cohorts:
            self.register(uid, gender)
            for pair in (p for p in report['pairs'] if p['gender'] == label):
                detail = self.request(f'/api/closet/outfit?user_id={uid}&product_id={pair["product_id"]}&outfit_id={pair["outfit_id"]}')
                products = [p for g in detail['groups'] for p in g['products']]
                ids = {p['id'] for p in products}
                self.assertIn(pair['new_product_id'], ids, pair['key'])
                self.assertFalse(ids & set(pair['removed_product_ids']), pair['key'])
                self.assertFalse(ids & set(all_ids), pair['key'])
                group = next(g for g in detail['groups'] if g['id'] == pair['new_garment_id'])
                self.assertFalse(group['is_anchor'], pair['key'])
                self.assertEqual(group['box'], pair['box'])
                product = next(p for p in group['products'] if p['id'] == pair['new_product_id'])
                self.assertAlmostEqual(product['similarity'], pair['similarity'], places=6)

    def test_expansion_review_matches_live_connections(self) -> None:
        report = json.loads((data_root() / 'EXPANDED60_REVIEW.json').read_text(encoding='utf-8'))
        self.assertEqual(len(report['added_pairs']), 115)
        self.assertEqual(len({p['product_id'] for p in report['added_pairs']}), 23)
        for gender, label, count in self.expected_cohorts:
            uid = 'EXP60_' + gender
            self.register(uid, gender)
            catalog = self.request('/api/closet/catalog?user_id=' + uid)
            self.update(uid, 'add', catalog['demo_ids'])
            for pid in catalog['demo_ids']:
                live = self.request(f'/api/closet/outfits?user_id={uid}&product_id={pid}')
                self.assertCountEqual([o['id'] for o in live['outfits']], report['coverage'][pid]['outfit_ids'])
                for pair in (p for p in report['added_pairs'] if p['product_id'] == pid):
                    detail = self.request(f'/api/closet/outfit?user_id={uid}&product_id={pid}&outfit_id={pair["outfit_id"]}')
                    ids = {p['id'] for g in detail['groups'] for p in g['products']}
                    self.assertTrue(ids & set(pair['reverse_products']))
                    self.assertFalse(ids & set(catalog['demo_ids']))


def load_tests(loader, standard_tests, pattern):
    names = [n for n in GenderFiveChecks.__dict__ if n.startswith('test_')]
    names += ['test_saved_outfits_persist_without_fabricating_preferences']
    names += ['test_every_cohort_link_survives_all60_owned']
    names += ['test_quality68_replacements_and_retired_links', 'test_dataset_product_migration_preserves_other_history', 'test_expansion_review_matches_live_connections', 'test_reverse_repair_survives_all_owned_filter', 'test_profile_persistence_and_isolation',
              'test_unknown_product_not_saved', 'test_automatic_recency_unknown_tags_and_empty_history']
    return unittest.TestSuite(Expanded60Checks(n) for n in names)


if __name__ == '__main__':
    unittest.main(verbosity=2)
