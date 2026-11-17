"""Small-bundle UI/API checks; full-index research checks belong to the full suite."""
import hashlib
import json
from pathlib import Path
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

import test_integration as original
from runtime_paths import data_root, data_demo
from server import closet, SELECTION, is_eligible


class CompactChecks(original.IntegrationTest):
    def test_compact_all_images_and_counts(self):
        health = closet('/health')
        self.assertEqual(health['mode'], 'curated_snapshot')
        self.assertEqual((health['query_products'], health['outfits'], health['connections']), (37, 89, 107))
        self.assertFalse(health['supports_full_search'])
        config = json.loads((data_demo() / 'config.json').read_text(encoding='utf-8'))
        data = json.loads((data_demo() / config['snapshot_file']).read_text(encoding='utf-8'))
        for url, entry in data['images'].items():
            with urlopen(self.base + url) as response:
                self.assertTrue(response.headers['Content-Type'].startswith('image/'))
                self.assertEqual(hashlib.sha256(response.read()).hexdigest(), entry['sha256'])
        catalog = self.request('/api/closet/catalog')
        self.assertEqual(len(catalog['products']), 37)
        self.assertEqual(len(catalog['shoe_demo_ids']), 5)
        self.assertEqual(len(catalog['dress_demo_ids']), 2)

    def test_compact_blocks_removed_photos_and_full_search(self):
        pid = next(iter(SELECTION['products']))
        result = self.request('/api/closet/outfits?user_id=COMPACT&product_id=' + pid + '&mode=all')
        self.assertFalse(result['supports_full_search'])
        self.assertTrue(result['curated'])
        for oid in SELECTION['outfits']:
            if not is_eligible(oid):
                with self.assertRaises(HTTPError) as error:
                    urlopen(self.base + '/media/outfit/' + oid)
                self.assertEqual(error.exception.code, 404)
        with self.assertRaises(HTTPError):
            self.request('/api/closet/survey?user_id=COMPACT', {'selected':['missing']})

    def test_compact_code_assets_and_new_empty_profile(self):
        for path in ['/shop/', '/shop/shop.js', '/shop/shop.css', '/shop/closet.js',
                     '/shop/closet.css', '/shop/cold_start.js', '/shop/moodboard.js', '/shop/moodboard.css']:
            with urlopen(self.base + path) as response:
                self.assertGreater(len(response.read()), 100)
        self.assertEqual(self.request('/api/closet/profile?user_id=COMPACT_EMPTY')['owned'], [])
        self.assertEqual(len(self.request('/api/closet/survey?user_id=COMPACT_EMPTY')['outfits']), 89)


def load_tests(loader, standard_tests, pattern):
    # These five require the full 107-input index, unreviewed search candidates,
    # or excluded source photos. Compact-specific checks cover its narrower contract.
    full_only = {
        'test_replacements_preserve_automatic_candidates_for_all_107_inputs',
        'test_review_covers_exact_images_and_live_catalog_counts',
        'test_excluded_and_unreviewed_photos_cannot_bypass_filters',
        'test_shoes_use_cached_regions_and_complete_both_directions',
        'test_raw_results_and_expanded_candidate_prefix',
    }
    return unittest.TestSuite(CompactChecks(name) for name in loader.getTestCaseNames(CompactChecks)
                              if name not in full_only)


if __name__ == '__main__':
    unittest.main(verbosity=2)
