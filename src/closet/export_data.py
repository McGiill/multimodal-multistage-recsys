"""Export reviewed results from existing full data. No downloads or inference.

Run from the code folder:
python -m src.closet.export_data --data-root FULL_DATA --destination SMALL_DATA
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()



def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', required=True, type=Path)
    ap.add_argument('--destination', required=True, type=Path)
    args = ap.parse_args()
    source = args.data_root.resolve()
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError('Refusing to overwrite an existing bundle: ' + str(destination))
    os.environ['CLOSET_DATA_ROOT'] = str(source)
    from src.closet import retrieval as search
    from src.closet.shop import server as shop
    full = search.Catalog()
    snapshot = dict(schema='closet-curated-snapshot-v1', input_ids=list(shop.ACTIVE_PRODUCTS),
                    products={}, recommendations={}, details={}, images={}, meta={})
    product_ids = set(snapshot['input_ids'])
    outfit_ids = set()
    pairs = 0
    for pid in snapshot['input_ids']:
        result = copy.deepcopy(full.recommend(pid))
        lookup = {o['id']: o for o in result['outfits']}
        # Rejected IDs retain only their exclusion metadata so the UI's counts stay exact.
        result['outfits'] = [lookup[oid] for oid in shop.SELECTION['products'][pid]['outfits'] if oid in lookup]
        snapshot['recommendations'][pid] = result
        snapshot['details'][pid] = {}
        for oid in shop.ACTIVE_PRODUCTS[pid]:
            detail = copy.deepcopy(full.detail(oid, pid, shop.SELECTION['reverse_candidate_limit']))
            selected = shop.connection(shop.SELECTION, pid, oid)
            for group in detail['groups']:
                allowed = shop.allowed_products(selected, group['id'])
                group['products'] = [p for p in group['products'] if p['id'] in allowed]
                product_ids.update(p['id'] for p in group['products'])
            product_ids.update(p['id'] for p in detail['shoe_alternatives'])
            snapshot['details'][pid][oid] = detail
            outfit_ids.add(oid)
            pairs += 1
        print(f'Exported reviewed product {len(snapshot["details"])}/{len(snapshot["input_ids"])}', flush=True)
    for pid in sorted(product_ids):
        item = full.product(pid)
        item['can_recommend'] = pid in snapshot['input_ids']
        snapshot['products'][pid] = item
    for links in snapshot['details'].values():
        for detail in links.values():
            for item in [detail['product']] + detail['shoe_alternatives'] + [p for g in detail['groups'] for p in g['products']]:
                item['can_recommend'] = item['id'] in snapshot['input_ids']
    destination.mkdir(parents=True)
    (destination / '.gitignore').write_text('*\n!.gitignore\n', encoding='utf-8')
    for kind, ids in [('product', product_ids), ('outfit', outfit_ids)]:
        for item in sorted(ids):
            original = full.media(kind, item)
            relative = Path('output/compact/images') / f'{kind}_{item}{original.suffix.lower()}'
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            checksum = digest(original)
            if digest(target) != checksum:
                raise ValueError('Image copy mismatch')
            snapshot['images'][f'/media/{kind}/{item}'] = dict(path=relative.as_posix(), sha256=checksum)
    snapshot['meta'] = dict(service='local-closet-compact-v1', mode='curated_snapshot',
        inference_calls=0, query_products=len(snapshot['input_ids']), catalog_products=len(product_ids),
        outfits=len(outfit_ids), connections=pairs, supports_full_search=False,
        reviewed_dress_regions=full.meta['reviewed_dress_regions'],
        note='Frozen reviewed search results; live preference ranking and ownership filtering remain in the shop API.')
    data_demo = destination / 'local_closet_demo'
    for relative in ['shoe_extension.json', 'dress_extension.json', 'replacement_extension.json',
                     'team_integration/closet_selection.json', 'team_integration/closet_tags.json',
                     'shop_integration/audience_review/recheck/reviewed.json']:
        target = data_demo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / 'local_closet_demo' / relative, target)
    cfg = dict(demo_mode='curated_snapshot', snapshot_file='../output/compact/snapshot.json',
               outfits_dir='../output/compact', provenance_file='../output/compact/provenance.json')
    snapshot_path = destination / 'output/compact/snapshot.json'
    write(snapshot_path, snapshot)
    cfg['snapshot_sha256'] = digest(snapshot_path)
    write(data_demo / 'config.json', cfg)
    write(destination / 'output/compact/queries.json',
          [{'product_id': pid, 'slot': shop.INPUT_SLOTS[pid]} for pid in snapshot['input_ids']])
    write(destination / 'output/compact/provenance.json', dict(kinds=full.kinds,
          mode='derived_curated_snapshot', source_model=full.meta['model'],
          source_config_sha256=digest(source / 'local_closet_demo/config.json'),
          source_product_vectors_sha256=digest(source / 'output/outfit_to_product_full/product_vectors_fashion_clip.npy'),
          source_selection_sha256=digest(source / 'local_closet_demo/team_integration/closet_selection.json'),
          note='Subset of original API results, not newly extracted features or a trained model.'))
    files = [dict(path=p.relative_to(destination).as_posix(), bytes=p.stat().st_size, sha256=digest(p))
             for p in sorted(destination.rglob('*')) if p.is_file()]
    report = dict(mode='curated_snapshot', input_products=len(snapshot['input_ids']),
                  display_products=len(product_ids), outfit_images=len(outfit_ids), connections=pairs,
                  image_files=len(snapshot['images']), data_bytes=sum(f['bytes'] for f in files),
                  vectors_included=False, models_included=False, original_images_unchanged=True,
                  comparison='All exported connection lists/scores derived from the full Catalog before ownership and taste ranking.',
                  files=files)
    write(destination / 'BUNDLE_MANIFEST.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'files'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
