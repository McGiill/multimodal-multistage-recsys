"""Offline full-gallery preprocessing; preserve the 2,000-image demo cache.

Run with the existing CUDA Python environment. No download, paid API, visual
review, retraining, or demo configuration changes. Checkpoints are resumable.
"""
import os
import sys
import json
import time
import hashlib
import argparse
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output/musinsa_full_auto'
BASE = ROOT / 'output/outfit2000_products100'
FULL = ROOT / 'output/musinsa_full_200'
OLD = ROOT / 'output/outfit_missing200'
REPO = Path('C:/Users/전민재/Downloads/multimodal-multistage-recsys-main (2)/multimodal-multistage-recsys-main')
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OPENBLAS_NUM_THREADS='1')
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(p, x):
    p = Path(p)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    t.replace(p)


def freeze():
    previous = read(OLD / 'protocol.json')
    for name, expected in previous['source_hashes'].items():
        if sha(REPO / name) != expected:
            raise ValueError('Source changed: ' + name)
    models = ROOT / 'output/team_reuse_500/models'
    for name, provenance in previous['provenance'].items():
        if read(models / name / 'provenance.json') != provenance:
            raise ValueError('Model provenance changed: ' + name)
        for f in provenance['files']:
            if sha(models / name / f['name']) != f['sha256']:
                raise ValueError('Model file changed: ' + f['name'])
    # The existing 2,000-image run recorded this exact full-bank checksum.
    base_protocol = read(BASE / 'protocol.json')
    bank_path = FULL / 'vectors_fashion_clip.npz'
    expected_bank = base_protocol['source_hashes'][str(bank_path)]
    if sha(bank_path) != expected_bank:
        raise ValueError('Full-gallery vector bank changed')
    sources = [BASE / f for f in ['manifest.json', 'outfits.json', 'garments.json', 'embeddings.npy', 'protocol.json', 'approved_garments.json']]
    sources += [FULL / f for f in ['gallery_manifest.json', 'crop_manifest.json', 'vectors_fashion_clip.npz', 'protocol.json']]
    sources += [ROOT / 'output/style_dataset_audit' / f for f in ['musinsa-no-kid.parquet', 'image_inventory.json']]
    sources += [ROOT / 'output/outfit_typecheck200' / f for f in ['text_prompts.json', 'text_prototypes.npy']]
    sources += [ROOT / 'local_closet_demo' / f for f in ['config.json', 'shoe_extension.json', 'dress_extension.json', 'replacement_extension.json']]
    sources += [ROOT / f for f in ['run_outfit2000_cached.py', 'compare_outfit_missing200.py', 'auto_select_outfit200.py', 'preprocess_outfits_local.py']]
    protocol = dict(schema=1, dataset='disl/musinsa-no-kid', rows=8626,
        source_hashes={str(p): sha(p) for p in sources},
        detector=previous['detector'], grounding=previous['grounding'], encoder=previous['encoder'],
        model_provenance=previous['provenance'], code_hashes=previous['source_hashes'],
        runner_sha256=sha(__file__), rules=base_protocol['selection_protocol'],
        reused_images=2000, online=False, manual_review=False, demo_updated=False,
        note='All source rows retained; raw predictions include all configured slots. Strict detections plus consensus for missing top/bottom/outer. Automatic candidates are not adult/audience or aesthetic approval.')
    p = OUT / 'protocol.json'
    if p.exists() and read(p) != protocol:
        raise ValueError('Protocol differs from checkpoint; use a new output directory')
    write(p, protocol)
    return previous, protocol


def valid_vectors(v, n):
    import numpy as np
    if v.shape != (n, 512) or v.dtype != np.float32 or not np.isfinite(v).all():
        raise ValueError('Invalid vector shape/dtype/values')
    if n and not np.allclose(np.linalg.norm(v, axis=1), 1, atol=.001):
        raise ValueError('Non-normalized vectors')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit-new', type=int, default=0, help='Checkpoint a bounded batch; 0 completes all remaining rows')
    parser.add_argument('--retry-errors', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    for name in ['records', 'vectors']:
        (OUT / name).mkdir(exist_ok=True)
    started = time.time()
    previous, protocol = freeze()
    key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    import run_outfit2000_cached as p
    import numpy as np
    from PIL import Image
    from dataclasses import asdict
    p.OUT = OUT
    cfg = p.yaml.safe_load((REPO / 'configs/style_search.yaml').read_text(encoding='utf-8'))
    runtime = p.copy.deepcopy(cfg)
    models = ROOT / 'output/team_reuse_500/models'
    runtime['detector']['model_name'] = str(models / 'yolos')
    runtime['grounding']['model_name'] = str(models / 'grounding_tiny')
    device = 'cuda' if p.torch.cuda.is_available() else 'cpu'
    p.torch.set_num_threads(6)
    p.torch.manual_seed(42)
    base_records = read(BASE / 'outfits.json')
    base_by_id = {r['outfit_id']: r for r in base_records}
    base_vectors = np.load(BASE / 'embeddings.npy', allow_pickle=False)
    base_garments = read(BASE / 'garments.json')
    valid_vectors(base_vectors, len(base_garments))
    gallery = read(FULL / 'gallery_manifest.json')
    full_by_id = {'musinsa_' + r['source_id']: r for r in gallery}
    samples = read(BASE / 'manifest.json')
    for r in sorted(gallery, key=lambda r: r['row']):
        oid = 'musinsa_' + r['source_id']
        if oid in base_by_id:
            continue
        samples.append(dict(outfit_id=oid, dataset_id=r['id'], source_id=r['source_id'],
            dataset_row=r['row'], publisher_style=r.get('publisher_style'),
            original_path=r['path'], input_sha256=r['input_sha256'], split='full_remaining6626'))
    assert len(samples) == len({r['outfit_id'] for r in samples}) == 8626
    assert set(r['dataset_row'] for r in samples) == set(range(8626))
    write(OUT / 'manifest.json', samples)
    jobs = read(FULL / 'crop_manifest.json')['requested']
    with np.load(FULL / 'vectors_fashion_clip.npz', allow_pickle=False) as z:
        bank = z['requested']
    valid_vectors(bank, len(jobs))
    # Full-bank entries must point to this gallery's immutable input paths.
    for j in jobs:
        assert Path(j['path']).resolve() == Path(gallery[j['parent']]['path']).resolve()
    lookup = {(str(Path(j['path']).resolve()), tuple(j['box']), j['margin']): i for i, j in enumerate(jobs)}
    all_records, all_garments, all_vectors, audit = [], [], [], []
    counts = collections.Counter()
    det = gd = enc = None
    image_hashes = collections.defaultdict(list)
    new_count = 0
    print('Prepared 8626 images; preserve 2000; device=' + device, flush=True)
    for index, r in enumerate(samples):
        oid = r['outfit_id']
        dest = OUT / 'records' / (oid + '.json')
        if args.limit_new and new_count >= args.limit_new and oid not in base_by_id and not dest.exists():
            break
        item = dict(r, status='error', garments=[])
        flags = []
        try:
            assert r['input_sha256'] == full_by_id[oid]['input_sha256'], 'Gallery input mismatch'
            assert sha(r['original_path']) == r['input_sha256'], 'Image checksum mismatch'
            with Image.open(r['original_path']) as image:
                image.load()
                im = image.convert('RGB')
            image_hashes[r['input_sha256']].append(oid)
            if oid in base_by_id:
                z = base_by_id[oid]
                assert z['status'] == 'ok' and z['input_sha256'] == r['input_sha256']
                assert list(im.size) == z['image_size']
                gg = z['garments']
                vv = base_vectors[[g['embedding_row'] for g in gg]]
                origin = 'reused_2000'
                counts['reused_2000_images'] += 1
                counts['reused_2000_vectors'] += len(vv)
            else:
                z = read(dest) if dest.exists() else None
                if z:
                    assert z['protocol_key'] == key and z['input_sha256'] == r['input_sha256'], 'Checkpoint mismatch'
                if z and z['status'] == 'ok':
                    vp = OUT / z['vector_path']
                    assert sha(vp) == z['vector_sha256'], 'Checkpoint vector checksum mismatch'
                    vv = np.load(vp, allow_pickle=False)
                    gg = z['garments']
                    origin = 'resumed'
                    counts['resumed_images'] += 1
                elif z and not args.retry_errors:
                    raise ValueError('Previous processing error: ' + z.get('error', 'unknown'))
                else:
                    t = time.perf_counter()
                    if det is None:
                        det = p.FashionDetector(runtime, device=device)
                        gd = p.GroundingDetector(runtime, device=device)
                    raw = det.detect_raw(im)
                    strict = sorted(p.select_detections(raw, im.size, runtime), key=lambda d: (d.slot, d.kind, d.box))
                    selected = [(d, 'baseline', None, j) for j, d in enumerate(strict)]
                    present = {d.slot for d in strict}
                    candidates = []
                    added = 0
                    for slot in p.SLOTS:
                        if slot in present:
                            continue
                        low = p.best_in_slot(raw, im.size, runtime, slot)
                        grounded = gd.find(im, [k for k, v in cfg['kinds'].items() if v['slot'] == slot])
                        overlap = p._iou(low.box, grounded.box) if low and grounded else None
                        accept = overlap is not None and overlap >= .5
                        candidates.append(dict(slot=slot, low=asdict(low) if low else None,
                            grounding=asdict(grounded) if grounded else None, iou=overlap, accepted=accept))
                        if accept:
                            selected.append((low, 'consensus', overlap, added))
                            added += 1
                    gg = [p.region(d, r, origin, j, cfg, confirmation=overlap) for d, origin, overlap, j in selected]
                    vv = np.empty((len(gg), 512), dtype='float32')
                    missing = []
                    for j, (d, _, _, _) in enumerate(selected):
                        cache_key = (str(Path(r['original_path']).resolve()), tuple(d.box), cfg['detector']['crop_margin'])
                        if cache_key in lookup:
                            vv[j] = bank[lookup[cache_key]]
                        else:
                            missing.append(j)
                    if missing:
                        if enc is None:
                            enc = p.build_encoder(str(models / 'fashion_clip'), device=device, pad_square=previous['encoder']['pad_to_square'])
                        vv[missing] = enc.encode_images([p.crop(im, selected[j][0].box, cfg['detector']['crop_margin']) for j in missing], batch_size=16)
                    valid_vectors(vv, len(gg))
                    vp = OUT / 'vectors' / (oid + '.npy')
                    with vp.with_suffix('.npy.tmp').open('wb') as f:
                        np.save(f, vv)
                    vp.with_suffix('.npy.tmp').replace(vp)
                    z = dict(r, status='ok', protocol_key=key, image_size=list(im.size),
                        garments=gg, raw_detections=raw, candidates=candidates,
                        vector_path=vp.relative_to(OUT).as_posix(), vector_sha256=sha(vp),
                        bank_vectors_reused=len(gg)-len(missing), new_vectors_encoded=len(missing),
                        seconds=time.perf_counter()-t)
                    write(dest, z)
                    new_count += 1
                    counts['newly_processed_images'] += 1
                    origin = 'processed'
                counts['remaining_bank_vectors_reused'] += z['bank_vectors_reused']
                counts['remaining_new_vectors_encoded'] += z['new_vectors_encoded']
            valid_vectors(vv, len(gg))
            for g in gg:
                assert g['outfit_id'] == oid
                assert len(g['box']) == 4
                x1, y1, x2, y2 = g['box']
                if not (0 <= x1 < x2 <= im.width and 0 <= y1 < y2 <= im.height):
                    # Preserve raw model outputs and cached vectors; the existing
                    # automatic selection rejects these boxes before admission.
                    flags.append('invalid_box:' + g['garment_id'])
                    counts['invalid_box_regions'] += 1
                assert np.isfinite(g['detection_score']) and 0 <= g['detection_score'] <= 1
            if not gg:
                flags.append('no_detections')
            item = dict(r, status='ok', image_size=list(im.size), garments=[], preprocessing_source=origin,
                visual_review='pending')
            for g, v in zip(gg, vv):
                g = dict(g, embedding_row=len(all_garments))
                all_garments.append(g)
                all_vectors.append(v)
                item['garments'].append(g)
            if flags:
                counts['flagged_images'] += 1
            audit.append(dict(outfit_id=oid, technical_status='flagged' if flags else 'pass', flags=flags))
        except Exception as e:
            error = type(e).__name__ + ': ' + str(e)
            item.update(error=error, visual_review='pending')
            counts['errors'] += 1
            audit.append(dict(outfit_id=oid, technical_status='error', error=error))
            # Preserve any existing cache for diagnosis; create only missing error records.
            if oid not in base_by_id and not dest.exists():
                write(dest, dict(r, status='error', protocol_key=key, error=error))
            print('ERROR', oid, error, flush=True)
        all_records.append(item)
        if (index + 1) % 100 == 0:
            status = dict(processed=index+1, total=8626, elapsed_seconds=round(time.time()-started, 1), counts=dict(counts))
            write(OUT / 'progress.json', status)
            print(json.dumps(status), flush=True)
    write(OUT / 'technical_audit.json', audit)
    if len(all_records) != 8626:
        write(OUT / 'progress.json', dict(status='checkpointed_partial', processed=len(all_records), counts=dict(counts)))
        print('Partial checkpoint saved; rerun without --limit-new to finish.', flush=True)
        return
    matrix = np.asarray(all_vectors, dtype='float32').reshape(-1, 512)
    valid_vectors(matrix, len(all_garments))
    assert len({g['garment_id'] for g in all_garments}) == len(all_garments)
    # Exact prefix preservation protects every existing demo vector and ID.
    assert all_garments[:len(base_garments)] == base_garments
    assert np.array_equal(matrix[:len(base_vectors)], base_vectors)
    np.save(OUT / 'embeddings.npy', matrix)
    write(OUT / 'outfits.json', all_records)
    write(OUT / 'garments.json', all_garments)
    approved, decisions = p.select(all_records, all_garments, matrix)
    old_approved = read(BASE / 'approved_garments.json')
    assert [g for g in approved if g['outfit_id'] in base_by_id] == old_approved
    # Keep exclusion reasons for later visual review; do not delete any source row.
    region_decisions = read(OUT / 'region_decisions.json')
    duplicate_groups = [ids for ids in image_hashes.values() if len(ids) > 1]
    for path, digest in protocol['source_hashes'].items():
        assert sha(path) == digest, 'Source modified during run: ' + path
    result = dict(status='complete' if not counts['errors'] else 'complete_with_errors',
        total_images=8626, technical_errors=counts['errors'],
        no_detection_images=sum('no_detections' in r.get('flags', []) for r in audit),
        flagged_images=counts['flagged_images'], invalid_box_regions=counts['invalid_box_regions'],
        total_regions=len(all_garments), slots=dict(collections.Counter(g['slot'] for g in all_garments)),
        automatic_candidate_images=sum(r['accepted'] for r in decisions), automatic_candidate_regions=len(approved),
        automatic_rejection_reasons=dict(collections.Counter(reason for g in region_decisions for reason in g['rejection_reasons'])),
        duplicate_image_groups=duplicate_groups, counts=dict(counts),
        base_2000_exactly_preserved=True, sources_unchanged=True,
        device=device, gpu=p.torch.cuda.get_device_name(0) if device == 'cuda' else None,
        wall_seconds=round(time.time()-started, 2), manual_review=False, demo_updated=False,
        output_hashes={name: sha(OUT / name) for name in ['embeddings.npy', 'outfits.json', 'garments.json', 'approved_garments.json', 'selection.json', 'technical_audit.json']})
    write(OUT / 'summary.json', result)
    write(OUT / 'progress.json', dict(status=result['status'], processed=8626, total=8626))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if counts['errors']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
