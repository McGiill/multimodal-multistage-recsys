"""Portable path checks using synthetic temporary folders, no dataset required."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'local_closet_demo'))
from runtime_paths import data_root, media_path, required_files


class ExternalPathsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='closet-paths-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve() / '외부 데이터'
        self.root.mkdir()
        self.env = patch.dict(os.environ, {'CLOSET_DATA_ROOT': str(self.root)})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_missing_environment_explains_launcher(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'run_demo.py'):
                data_root()

    def test_missing_directory_fails(self):
        with patch.dict(os.environ, {'CLOSET_DATA_ROOT': str(self.root / 'missing')}):
            with self.assertRaises(FileNotFoundError):
                data_root()

    def test_historic_windows_path_maps_to_new_root(self):
        self.assertEqual(media_path(r'C:\Users\old user\project\output\photos\한글.jpg'),
                         self.root / 'output/photos/한글.jpg')

    def test_relative_output_path(self):
        self.assertEqual(media_path('output/photos/a.jpg'), self.root / 'output/photos/a.jpg')

    def test_existing_original_is_never_used(self):
        old = Path(self.tmp.name) / 'old/output/photos/a.jpg'
        old.parent.mkdir(parents=True)
        old.write_bytes(b'path fixture')
        self.assertEqual(media_path(old), self.root / 'output/photos/a.jpg')
        self.assertFalse(media_path(old).exists())

    def test_paths_outside_output_are_rejected(self):
        for value in ['output/../private/a.jpg', '/private/a.jpg']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                media_path(value)

    def test_missing_config_is_reported(self):
        self.assertEqual(required_files(), [self.root / 'local_closet_demo/config.json'])

    def test_metadata_layout_and_escape_rejection(self):
        demo = self.root / 'local_closet_demo'
        demo.mkdir()
        config = demo / 'config.json'
        cfg = dict(outfits_dir='../output/outfits', products_dir='../output/products',
                   provenance_file='../output/provenance.json',
                   shoe_extension_file='shoe_extension.json',
                   dress_extension_file='dress_extension.json',
                   replacement_extension_file='replacement_extension.json')
        config.write_text(json.dumps(cfg), encoding='utf-8')
        paths = required_files()
        self.assertEqual(len(paths), 19)
        self.assertEqual(len(required_files(True)), 21)
        for path in paths:
            self.assertTrue(path.is_relative_to(self.root))
        cfg['outfits_dir'] = '../../outside'
        config.write_text(json.dumps(cfg), encoding='utf-8')
        with self.assertRaises(ValueError):
            required_files()


if __name__ == '__main__':
    unittest.main()
