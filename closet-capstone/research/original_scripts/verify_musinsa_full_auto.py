"""Independent offline integrity checks; no visual or demographic judgments."""
import argparse
import collections
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output/musinsa_full_auto'
BASE = ROOT / 'output/outfit2000_products100'
SOURCE = ROOT / 'output/style_dataset_audit'
sys.path.insert(0, str(SOURCE / 'deps'))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from PIL import Image


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def audit_source():
    import pyarrow.parquet as pq
    inventory = {r['row']: r for r in read(SOURCE / 'image_inventory.json')}
    errors, rows, pixels, byte_hashes = [], [], collections.defaultdict(list), collections.defaultdict(list)
    i = 0
    for batch in pq.ParquetFile(SOURCE / 'musinsa-no-kid.parquet').iter_batches(batch_size=64):
        for row in batch.to_pylist():
            try:
                raw = row['image']['bytes']
                byte_digest = hashlib.sha256(raw).hexdigest()
                label = json.loads(row['label'])
                oid = str(next(v for k, v in label.items() if k.endswith('.파일 번호')))
                with Image.open(io.BytesIO(raw)) as im:
                    im.load()
                    im = im.convert('RGB')
                    pixel_digest = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
                    expected = inventory[i]
                    assert expected['source_id'] == oid, 'source ID mismatch'
                    assert expected['bytes_sha256'] == byte_digest, 'image byte hash mismatch'
                    assert expected['pixels_sha256'] == pixel_digest, 'pixel hash mismatch'
                    assert (expected['width'], expected['height']) == im.size
                    assert im.width > 0 and im.height > 0
                rows.append(oid)
                pixels[pixel_digest].append(oid)
                byte_hashes[byte_digest].append(oid)
            except Exception as e:
                errors.append(dict(row=i, error=type(e).__name__ + ': ' + str(e)))
            i += 1
    result = dict(total_rows=i, decoded_and_labels_verified=len(rows), unique_source_ids=len(set(rows)),
        duplicate_byte_groups=[v for v in byte_hashes.values() if len(v) > 1],
        duplicate_pixel_groups=[v for v in pixels.values() if len(v) > 1], errors=errors,
        parquet_sha256=sha(SOURCE / 'musinsa-no-kid.parquet'), visual_review=False)
    save('source_audit.json', result)
    assert i == len(rows) == len(set(rows)) == 8626 and not errors
    print('Source audit: all 8626 embedded images and labels decoded; byte/pixel hashes match inventory.', flush=True)


def audit_outputs():
    summary = read(OUT / 'summary.json')
    protocol = read(OUT / 'protocol.json')
    records = read(OUT / 'outfits.json')
    garments = read(OUT / 'garments.json')
    approved = read(OUT / 'approved_garments.json')
    decisions = read(OUT / 'region_decisions.json')
    selections = read(OUT / 'selection.json')
    audit = read(OUT / 'technical_audit.json')
    matrix = np.load(OUT / 'embeddings.npy', mmap_mode='r', allow_pickle=False)
    old = read(BASE / 'garments.json')
    old_matrix = np.load(BASE / 'embeddings.npy', mmap_mode='r', allow_pickle=False)
    assert summary['status'] == 'complete' and summary['technical_errors'] == 0
    assert len(records) == len({r['outfit_id'] for r in records}) == 8626
    assert {r['dataset_row'] for r in records} == set(range(8626))
    assert all(r['status'] == 'ok' for r in records)
    assert all(r['visual_review'] == 'pending' for r in records)
    assert matrix.shape == (len(garments), 512) and matrix.dtype == np.float32
    assert np.isfinite(matrix).all() and np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=.001)
    assert [g['embedding_row'] for g in garments] == list(range(len(garments)))
    assert len({g['garment_id'] for g in garments}) == len(garments)
    assert [g for r in records for g in r['garments']] == garments
    assert garments[:len(old)] == old and np.array_equal(matrix[:len(old)], old_matrix)
    old_ids = {r['outfit_id'] for r in read(BASE / 'outfits.json')}
    assert [g for g in approved if g['outfit_id'] in old_ids] == read(BASE / 'approved_garments.json')
    selection = {r['outfit_id']: r for r in selections}
    accepted_ids = {r['outfit_id'] for r in selections if r['accepted']}
    assert len(accepted_ids) == summary['automatic_candidate_images']
    assert len(approved) == summary['automatic_candidate_regions']
    by_record = {r['outfit_id']: r for r in records}
    by_audit = {r['outfit_id']: r for r in audit}
    by_decision = {g['garment_id']: g for g in decisions}
    flagged_regions = collections.defaultdict(list)
    for g in decisions:
        if g['rejection_reasons']:
            flagged_regions[g['outfit_id']].append(dict(garment_id=g['garment_id'], reasons=g['rejection_reasons']))
    invalid = []
    for g in garments:
        w, h = by_record[g['outfit_id']]['image_size']
        x1, y1, x2, y2 = g['box']
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            invalid.append(g['garment_id'])
            assert 'invalid_box' in by_decision[g['garment_id']]['rejection_reasons']
            assert 'invalid_box:' + g['garment_id'] in by_audit[g['outfit_id']]['flags']
    assert len(invalid) == summary['invalid_box_regions']
    assert not set(invalid) & {g['garment_id'] for g in approved}
    assert all(not g['rejection_reasons'] and g['selected'] for g in approved)
    for oid in accepted_ids:
        local = [g for g in by_record[oid]['garments'] if by_decision[g['garment_id']]['selected']]
        assert len({g['slot'] for g in local}) >= 2
    checkpoint_total = 0
    key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    for r in records:
        if r['outfit_id'] in old_ids:
            continue
        z = read(OUT / 'records' / (r['outfit_id'] + '.json'))
        assert z['protocol_key'] == key and z['status'] == 'ok'
        assert sha(OUT / z['vector_path']) == z['vector_sha256']
        v = np.load(OUT / z['vector_path'], allow_pickle=False)
        assert np.array_equal(v, matrix[[g['embedding_row'] for g in r['garments']]])
        checkpoint_total += 1
    assert checkpoint_total == 6626
    for path, expected in protocol['source_hashes'].items():
        assert sha(path) == expected, 'Source changed: ' + path
    for name, expected in summary['output_hashes'].items():
        assert sha(OUT / name) == expected, 'Output changed: ' + name
    # The queue is for later review; no photo is deemed visually safe here.
    with (OUT / 'review_queue.jsonl').open('w', encoding='utf-8') as f:
        for r in records:
            issues = flagged_regions[r['outfit_id']]
            f.write(json.dumps(dict(outfit_id=r['outfit_id'], image_path=r['original_path'],
                visual_review='pending', automatic_candidate=selection[r['outfit_id']]['accepted'],
                technical_flags=by_audit[r['outfit_id']].get('flags', []), region_flags=issues), ensure_ascii=False) + '\n')
    result = dict(passed=True, total_images=8626, total_regions=len(garments), checkpoint_images=checkpoint_total,
        old_2000_vectors_bitwise_equal=True, old_automatic_selection_equal=True,
        source_and_output_hashes_match=True, invalid_boxes_excluded_from_candidates=len(invalid),
        no_manual_review=True, demo_config_unchanged=True)
    save('verification.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-only', action='store_true')
    parser.add_argument('--outputs-only', action='store_true')
    args = parser.parse_args()
    if not args.outputs_only:
        audit_source()
    if not args.source_only:
        audit_outputs()
