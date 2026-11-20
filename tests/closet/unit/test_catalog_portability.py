"""Regression: reverse search must work when only relocated images exist."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.closet.retrieval import Catalog


class RelocatedProductTest(unittest.TestCase):
    def test_reverse_search_uses_image_in_external_data_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            photo = root / 'output/catalog/001.jpg'
            photo.parent.mkdir(parents=True)
            photo.write_bytes(b'file-presence fixture')
            catalog = Catalog.__new__(Catalog)
            catalog.garment_by_id = {'region': (0, {'kind': 'trousers', 'slot': 'bottom'})}
            catalog.kinds = {'trousers': {'l3': ['Trousers']}}
            catalog.category_rows = {'Trousers': [0]}
            catalog.reviewed_region_products = {}
            catalog.pvec = np.array([[1., 0.]])
            catalog.gvec = np.array([[1., 0.]])
            catalog.cfg = {'reverse_floor': .5, 'reverse_limit': 4}
            catalog.products = [{'id': 'hm_001', 'sku': '000001001',
                                 'path': r'Z:\nonexistent-author-machine\output\catalog\001.jpg',
                                 'category_l3': 'Trousers'}]
            catalog.query_by_id = {}
            catalog.product_rows = {'hm_001': 0}
            with patch.dict(os.environ, {'CLOSET_DATA_ROOT': str(root)}):
                results = catalog.similar('region', 'owned', 4)
            self.assertEqual([r['id'] for r in results], ['hm_001'])


if __name__ == '__main__':
    unittest.main()
