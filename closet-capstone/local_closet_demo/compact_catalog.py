"""Reviewed demo snapshot, explicitly not a replacement for full vector search."""
import copy
import hashlib
import json
from runtime_paths import config_path, media_path


class CompactCatalog:
    def __init__(self):
        config = config_path()
        cfg = json.loads(config.read_text(encoding='utf-8'))
        path = (config.parent / cfg['snapshot_file']).resolve()
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != cfg['snapshot_sha256']:
            raise ValueError('Compact snapshot checksum mismatch')
        self.data = json.loads(raw)
        if self.data['schema'] != 'closet-curated-snapshot-v1':
            raise ValueError('Unsupported compact snapshot')
        for entry in self.data['images'].values():
            image = media_path(entry['path'])
            if hashlib.sha256(image.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError('Compact image checksum mismatch: ' + str(image))
        self.queries = [{'product_id': pid} for pid in self.data['input_ids']]
        self.default_ids = self.data['input_ids'][:6]
        self.meta = self.data['meta']

    def product(self, pid):
        return copy.deepcopy(self.data['products'][pid])

    def recommend(self, pid):
        return copy.deepcopy(self.data['recommendations'][pid])

    def detail(self, oid, pid, reverse_limit=None):
        if reverse_limit is not None and not 1 <= int(reverse_limit) <= 40:
            raise ValueError('Reverse candidate limit must be between 1 and 40')
        # Preserve reviewed display lists; do not reinterpret a full-search top-k.
        return copy.deepcopy(self.data['details'][pid][oid])

    def media(self, kind, item):
        return media_path(self.data['images'][f'/media/{kind}/{item}']['path'])
