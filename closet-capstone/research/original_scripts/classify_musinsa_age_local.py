"""Offline, abstaining appearance triage using cached full-photo embeddings.

This is not age verification. No image inference, training, downloads, deletions,
or edits to the demo review list. Thresholds are frozen before inspecting counts.
"""
import os
import sys
import json
import csv
import time
import hashlib
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output/musinsa_age_triage6026'
FULL = ROOT / 'output/musinsa_full_auto'
PHOTOS = ROOT / 'output/musinsa_scenes8626'
MODEL = ROOT / 'output/team_reuse_500/models/fashion_clip'
REPO = Path('C:/Users/전민재/Downloads/multimodal-multistage-recsys-main (2)/multimodal-multistage-recsys-main')
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OPENBLAS_NUM_THREADS='1')
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8')

CLASSES = ['adult', 'child', 'adolescent']
SUBJECTS = ['an adult', 'a child', 'a teenager']
TEMPLATES = [
    'a full body photograph of {} wearing clothes',
    'a fashion photograph showing {}',
    'a photograph of {} posing in an outfit',
    'a person who appears to be {}, modeling clothes',
]
RULES = dict(adult_margin=.02, child_margin=.015, minimum_agreeing_prompts=3,
    minimum_adult_pair_margin=.005, maximum_child_pair_margin=.005,
    adolescent_policy='always_uncertain',
    other_policy='uncertain unless a consistent adult or child appearance margin is met',
    known_exclusions='previous child-presenting -> child_suspect; previous other exclusions -> uncertain',
    scores_are_probabilities=False, thresholds_validated=False)


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def save(p, x):
    p = Path(p)
    temp = p.with_suffix(p.suffix + '.tmp')
    temp.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(p)


def decide(scores, rules=RULES):
    """Four prompts x three appearance classes. Abstain on close/conflicting scores."""
    import numpy as np
    scores = np.asarray(scores)
    if scores.shape != (4, 3) or not np.isfinite(scores).all():
        raise ValueError('Expected finite 4x3 prompt similarities')
    means = scores.mean(axis=0)
    votes = np.bincount(scores.argmax(axis=1), minlength=3)
    adult_gap = float(means[0] - max(means[1:]))
    child_gap = float(means[1] - max(means[0], means[2]))
    adult_pairs = scores[:, 0] - scores[:, 1]
    if (adult_gap >= rules['adult_margin'] and votes[0] >= rules['minimum_agreeing_prompts']
            and adult_pairs.min() >= rules['minimum_adult_pair_margin']):
        return 'adult_candidate', ['consistent_adult_appearance_signal']
    if (child_gap >= rules['child_margin'] and votes[1] >= rules['minimum_agreeing_prompts']
            and adult_pairs.max() <= rules['maximum_child_pair_margin']):
        return 'child_suspect', ['consistent_child_appearance_signal']
    reasons = []
    if int(means.argmax()) == 2:
        reasons.append('adolescent_appearance_signal')
    if max(votes) < 3 or (votes[0] and votes[1]):
        reasons.append('prompt_disagreement')
    if adult_gap < rules['adult_margin']:
        reasons.append('insufficient_adult_margin')
    if adult_pairs.min() < rules['minimum_adult_pair_margin']:
        reasons.append('adult_child_boundary_uncertain')
    return 'uncertain', reasons or ['insufficient_consensus']


def main():
    started = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    inputs = [FULL / n for n in ['selection.json', 'outfits.json', 'approved_garments.json', 'verification.json']]
    inputs += [PHOTOS / n for n in ['full_photo_embeddings.npy', 'tags.json', 'protocol.json']]
    inputs += [ROOT / 'output/musinsa_audience_scene8626/protocol.json',
        ROOT / 'local_closet_demo/shop_integration/audience_review/recheck/reviewed.json',
        ROOT / 'local_closet_demo/replacement_extension.json', ROOT / 'local_closet_demo/config.json']
    assert read(FULL / 'verification.json')['passed']
    source_protocol = read(PHOTOS / 'protocol.json')
    provenance = read(MODEL / 'provenance.json')
    assert provenance == source_protocol['model']
    for f in provenance['files']:
        assert sha(MODEL / f['name']) == f['sha256']
    assert sha(REPO / 'src/search/encoder.py') == source_protocol['encoder_source_sha256']
    assert sha(PHOTOS / 'full_photo_embeddings.npy') == read(ROOT / 'output/musinsa_audience_scene8626/protocol.json')['photo_vectors_sha256']
    protocol = dict(version=1, source='6026 automatic candidates from musinsa_full_auto',
        source_hashes={str(f): sha(f) for f in inputs}, templates=TEMPLATES, subjects=SUBJECTS,
        classes=CLASSES, rules=RULES, score='mean cosine over four matched prompt templates',
        model=provenance, encoder_source_sha256=source_protocol['encoder_source_sha256'],
        runner_sha256=sha(__file__), offline=True, image_inference=0,
        new_visual_review=False, demo_updated=False,
        limitations=['Appearance similarity, not verified age or calibrated probability.',
            'FashionCLIP is not a validated age classifier; results can miss children and reject adults.',
            'Prompt agreement from one model is not independent model validation.',
            'Face visibility, multiple people, occlusion and actual age are not separately measured.',
            'Known assistant-reviewed exclusions are vetoes, not human ground truth.'])
    if (OUT / 'protocol.json').exists():
        assert read(OUT / 'protocol.json') == protocol, 'Protocol changed; use a fresh output directory'
    save(OUT / 'protocol.json', protocol)
    # Import only after sources/protocol are fixed. Existing local dependencies.
    from tag_personalize136 import build_encoder
    import numpy as np
    import torch
    torch.set_num_threads(6)
    torch.manual_seed(42)
    texts = [t.format(s) for t in TEMPLATES for s in SUBJECTS]
    text_path = OUT / 'text_embeddings.npy'
    if text_path.exists():
        tv = np.load(text_path, allow_pickle=False)
        text_inference = 0
    else:
        enc = build_encoder(str(MODEL), device='cuda' if torch.cuda.is_available() else 'cpu', pad_square=source_protocol['pad_square'])
        # Deliberately never call encode_images: all photo vectors are cached.
        tv = enc.encode_texts(texts, batch_size=12)
        np.save(text_path, tv)
        text_inference = len(texts)
        del enc
    assert tv.shape == (12, 512) and np.isfinite(tv).all()
    assert np.allclose(np.linalg.norm(tv, axis=1), 1, atol=.001)
    cached = np.load(PHOTOS / 'full_photo_embeddings.npy', allow_pickle=False)
    assert cached.shape == (8626, 512) and np.isfinite(cached).all()
    assert np.allclose(np.linalg.norm(cached, axis=1), 1, atol=.001)
    photo_rows = read(PHOTOS / 'tags.json')
    assert len(photo_rows) == 8626
    assert all(r['embedding_row'] == i for i, r in enumerate(photo_rows))
    photo_map = {r['outfit_id']: r for r in photo_rows}
    assert len(photo_map) == 8626
    records = {r['outfit_id']: r for r in read(FULL / 'outfits.json')}
    ids = [r['outfit_id'] for r in read(FULL / 'selection.json') if r['accepted']]
    assert len(ids) == len(set(ids)) == 6026
    review = read(ROOT / 'local_closet_demo/shop_integration/audience_review/recheck/reviewed.json')['outfits']
    extra = read(ROOT / 'local_closet_demo/replacement_extension.json')['audiences']
    assert not review.keys() & extra.keys()
    review.update(extra)
    selected_rows = []
    for oid in ids:
        p, r = photo_map[oid], records[oid]
        assert p['input_sha256'] == r['input_sha256'] == sha(r['original_path'])
        assert Path(p['original_path']).resolve() == Path(r['original_path']).resolve()
        if oid in review:
            assert review[oid]['image_sha256'] == r['input_sha256']
        selected_rows.append(p['embedding_row'])
    scores = (cached[selected_rows] @ tv.T).reshape(6026, 4, 3)
    np.save(OUT / 'prompt_scores.npy', scores)
    rows, diagnostics = [], []
    for oid, score, embedding_row in zip(ids, scores, selected_rows):
        auto, reasons = decide(score)
        final = auto
        prior = review.get(oid)
        origin = 'local_automatic'
        if prior and prior['status'] != 'accepted':
            final = 'child_suspect' if prior['age_review'] == 'child_presenting' else 'uncertain'
            reasons = ['previous_review_exclusion_preserved'] + reasons
            origin = 'prior_assistant_review_veto'
        means = score.mean(axis=0)
        votes = np.bincount(score.argmax(axis=1), minlength=3)
        rows.append(dict(outfit_id=oid, original_path=records[oid]['original_path'],
            input_sha256=records[oid]['input_sha256'], photo_embedding_row=embedding_row,
            bucket=final, automatic_bucket=auto, decision_source=origin, reasons=reasons,
            mean_cosine={k:float(v) for k,v in zip(CLASSES,means)},
            template_votes={k:int(v) for k,v in zip(CLASSES,votes)},
            adult_margin=float(means[0]-max(means[1:])), child_margin=float(means[1]-max(means[0],means[2])),
            prior_review=prior, status='automatic_unverified', verified_age=None))
        if prior:
            diagnostics.append(dict(outfit_id=oid, prior_status=prior['status'], prior_age_review=prior['age_review'],
                automatic_bucket=auto, final_bucket=final))
    save(OUT / 'classifications.json', rows)
    counts = collections.Counter(r['bucket'] for r in rows)
    for bucket in ['adult_candidate','child_suspect','uncertain']:
        group = [r for r in rows if r['bucket'] == bucket]
        save(OUT / (bucket + '.json'), group)
        with (OUT / (bucket + '.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            w = csv.writer(f)
            w.writerow(['outfit_id','original_path','bucket','decision_source','adult_margin','child_margin','reasons'])
            for r in group:
                w.writerow([r['outfit_id'],r['original_path'],r['bucket'],r['decision_source'],r['adult_margin'],r['child_margin'],';'.join(r['reasons'])])
    keep_ids = {r['outfit_id'] for r in rows if r['bucket'] == 'adult_candidate'}
    kept_regions = [g for g in read(FULL / 'approved_garments.json') if g['outfit_id'] in keep_ids]
    save(OUT / 'remaining_garments.json', kept_regions)
    save(OUT / 'remaining_feature_reference.json', dict(embedding_file=str(FULL / 'embeddings.npy'),
        embedding_sha256=sha(FULL / 'embeddings.npy'), row_field='embedding_row',
        note='References original full preprocessing matrix. No vector copies, no demo activation.'))
    save(OUT / 'prior_review_diagnostics.json', dict(rows=diagnostics,
        warning='Existing assistant judgments are not independent human labels. This is a disagreement diagnostic, not accuracy.'))
    # Report sensitivity without changing or selecting the primary threshold by count.
    sensitivity = {}
    for margin in [.01,.015,.02,.025,.03]:
        tally = collections.Counter()
        for oid, score in zip(ids, scores):
            bucket,_ = decide(score, {**RULES,'adult_margin':margin})
            prior = review.get(oid)
            if prior and prior['status'] != 'accepted':
                bucket = 'child_suspect' if prior['age_review']=='child_presenting' else 'uncertain'
            tally[bucket] += 1
        sensitivity[str(margin)] = dict(tally)
    for path,digest in protocol['source_hashes'].items():
        assert sha(path) == digest, 'Input modified: ' + path
    summary = dict(total=6026, counts={k:counts[k] for k in ['adult_candidate','child_suspect','uncertain']},
        remaining_regions=len(kept_regions), automatic_counts=dict(collections.Counter(r['automatic_bucket'] for r in rows)),
        prior_exclusions_preserved=sum(r['decision_source']=='prior_assistant_review_veto' for r in rows),
        reused_photo_vectors=6026, new_photo_inference=0, text_prompts=len(texts), new_text_inference=text_inference,
        source_files_unchanged=True, deleted_images=0, demo_updated=False, human_review=False,
        age_verified=False, adult_margin_sensitivity=sensitivity, runtime_seconds=round(time.perf_counter()-started,2),
        outputs={name:sha(OUT/name) for name in ['classifications.json','adult_candidate.json','child_suspect.json','uncertain.json','remaining_garments.json','text_embeddings.npy','prompt_scores.npy']})
    save(OUT / 'summary.json', summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
