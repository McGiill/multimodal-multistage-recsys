"""Saving affects ranking; removing it reverses only that source of preference."""
import sys
import tempfile
import unittest
from pathlib import Path

from src.closet.shop.store import ProfileStore
from src.closet.shop.personalization import resolve_preferences, rerank_automatic
from src.closet.shop.preference_seed import build_preference_seed


class SavedPreferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = ProfileStore(Path(self.directory.name) / 'profiles.sqlite3')
        self.tags = {'a':{'style':'classic'}, 'b':{'style':'street'},
                     'c':{'style':'street'}, 'd':{'style':'street'}}

    def preferences(self):
        return resolve_preferences(self.store, 'U', self.tags)

    def test_save_changes_ranking_and_remove_restores_original(self):
        items = [{'id':'a','similarity':.7}, {'id':'b','similarity':.7}]
        base = rerank_automatic(items, self.tags, self.preferences())
        self.store.set_outfit_saved('U', 'b', 'P', True)
        ranked = rerank_automatic(items, self.tags, self.preferences())
        self.assertEqual([v['id'] for v in ranked], ['b', 'a'])
        self.assertEqual(self.preferences()['source'], 'saved_outfits')
        self.assertFalse(resolve_preferences(self.store, 'OTHER', self.tags)['ready'])
        self.store.set_outfit_saved('U', 'b', 'P', False)
        self.assertEqual(rerank_automatic(items, self.tags, self.preferences()), base)

    def test_survey_overlap_is_not_double_counted_and_views_are_preserved(self):
        self.store.save_survey('U', ['a'])
        self.store.set_outfit_saved('U', 'a', 'P', True)
        self.assertEqual(self.preferences()['explicit_outfits'], 1)
        self.store.set_outfit_saved('U', 'a', 'P', False)
        self.assertEqual(self.preferences()['weights'], {'classic':1})
        for oid in ['b', 'c', 'd']: self.store.record_outfit_view('U', oid)
        baseline = self.preferences()
        self.store.set_outfit_saved('U', 'b', 'P', True)
        self.assertGreater(self.preferences()['weights']['street'], baseline['weights']['street'])
        self.store.set_outfit_saved('U', 'b', 'P', False)
        restored = self.preferences()
        for style, weight in baseline['weights'].items(): self.assertAlmostEqual(restored['weights'][style], weight)
        self.assertEqual(len(self.store.recent_outfit_views('U')), 3)
        self.assertEqual(self.store.get('U')['survey']['selected'], ['a'])

    def test_seed_removal_deduplication_and_scope(self):
        profile = {'survey':{'selected':['a'], 'completed':True}}
        links = lambda oid: {'region': {oid + '_product':{'similarity':.7}}}
        def seed(rows):
            return build_preference_seed('U', profile, 'v', 'all', lambda oid, _:oid in {'a','b'}, links,
                                         saved_outfits=rows, saved_eligible=lambda row:row['product_id']=='P')
        baseline = seed([])
        overlap = seed([{'outfit_id':'a', 'product_id':'P'}])
        self.assertEqual(overlap['seed_products'][0]['relative_weight'], 1)
        self.assertEqual(overlap['seed_products'][0]['evidence'][0]['signal_sources'], ['survey_selection','saved_outfit'])
        combined = seed([{'outfit_id':'b', 'product_id':'P'}])
        self.assertEqual([p['relative_weight'] for p in combined['seed_products']], [.5,.5])
        self.assertEqual(seed([]), baseline)
        invalid = seed([{'outfit_id':'b','product_id':'retired'}, {'outfit_id':'hidden','product_id':'P'}])
        self.assertEqual(invalid['seed_products'], baseline['seed_products'])
        self.assertEqual(invalid['effective_saved_outfit_ids'], [])
        profile['survey']['selected'] = []
        empty = seed([])
        self.assertFalse(empty['has_preference_signal'])
        self.assertEqual(empty['seed_products'], [])
        self.assertEqual(empty['empty_seed_action'], 'clear_previous_seed')


if __name__ == '__main__': unittest.main(verbosity=2)
