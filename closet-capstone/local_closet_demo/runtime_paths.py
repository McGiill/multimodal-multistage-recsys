"""Resolve external data without rewriting frozen manifests or checksums."""
import os
from pathlib import Path


def data_root():
    value = os.environ.get('CLOSET_DATA_ROOT')
    if not value:
        raise RuntimeError('Use: python run_demo.py --data-root <external-data-folder>')
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError('External data folder does not exist: ' + str(root))
    return root


def data_demo():
    return data_root() / 'local_closet_demo'


def config_path():
    return data_demo() / 'config.json'


def media_path(value):
    """Map historic Windows output paths to the supplied data root, even on Linux.

    Always prefer the external bundle, never a coincidentally existing old path.
    This mapping affects image IO only, not frozen JSON/vector contents.
    """
    root = data_root()
    normalized = str(value).replace('\\', '/')
    marker = normalized.lower().find('/output/')
    if marker >= 0:
        relative = 'output/' + normalized[marker + len('/output/'):]
    elif normalized.startswith('output/'):
        relative = normalized
    else:
        raise ValueError('Unsupported image path; expected a path below output/: ' + normalized)
    target = (root / relative).resolve()
    target.relative_to(root / 'output')
    return target


def required_files(include_test_data=False):
    import json
    demo = data_demo()
    config = config_path()
    if not config.is_file():
        return [config]
    cfg = json.loads(config.read_text(encoding='utf-8'))
    root = data_root()
    def resolve(value):
        p = (demo / value).resolve()
        p.relative_to(root)
        return p
    if cfg.get('demo_mode') == 'curated_snapshot':
        return [config, resolve(cfg['snapshot_file']), resolve(cfg['provenance_file']),
                resolve(cfg['outfits_dir']) / 'queries.json'] + [
            demo / p for p in ['shoe_extension.json', 'dress_extension.json', 'replacement_extension.json',
                              'team_integration/closet_selection.json', 'team_integration/closet_tags.json',
                              'shop_integration/audience_review/recheck/reviewed.json']]
    outfits = resolve(cfg['outfits_dir'])
    products = resolve(cfg['products_dir'])
    paths = [config, resolve(cfg['provenance_file'])]
    paths += [resolve(cfg[key]) for key in ['shoe_extension_file','dress_extension_file','replacement_extension_file']]
    paths += [demo / p for p in ['team_integration/closet_selection.json', 'team_integration/closet_tags.json',
        'shop_integration/audience_review/recheck/reviewed.json']]
    paths += [outfits / p for p in ['protocol.json','queries.json','query_embeddings.npy','embeddings.npy',
        'approved_garments.json','garments.json','outfits.json']]
    paths += [products / p for p in ['product_manifest.json','catalog/manifest.json','product_vectors_fashion_clip.npy','encoding_time_fashion_clip.json']]
    if include_test_data:
        paths += [demo / p for p in ['shop_integration/replacement_review/automatic-baseline.json',
            'shop_integration/audience_review/recheck/index.json']]
    return paths
