"""Partition, veto, threshold-boundary and feature alignment verification."""
import json
import hashlib
from pathlib import Path
import numpy as np
from classify_musinsa_age_local import decide, RULES, ROOT, OUT, FULL, PHOTOS, read, sha, save


def main():
    # Guard against accepting a near tie, teenage signal, or conflicting prompt.
    fixtures = [
        (np.tile([.4,.2,.25], (4,1)), 'adult_candidate'),
        (np.tile([.2,.4,.25], (4,1)), 'child_suspect'),
        (np.tile([.3,.299,.29], (4,1)), 'uncertain'),
        (np.tile([.2,.25,.4], (4,1)), 'uncertain'),
        (np.array([[.4,.2,.25]]*3 + [[.2,.45,.25]]), 'uncertain'),
    ]
    for score, expected in fixtures:
        assert decide(score)[0] == expected
    for invalid in [np.full((4,3), np.nan), np.zeros((3,3))]:
        try:
            decide(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid scores accepted')
    rows = read(OUT/'classifications.json')
    summary = read(OUT/'summary.json')
    protocol = read(OUT/'protocol.json')
    ids = [r['outfit_id'] for r in read(FULL/'selection.json') if r['accepted']]
    assert len(rows) == len(ids) == 6026
    assert [r['outfit_id'] for r in rows] == ids
    groups = [read(OUT/(name+'.json')) for name in ['adult_candidate','child_suspect','uncertain']]
    joined = [r['outfit_id'] for group in groups for r in group]
    assert len(joined) == len(set(joined)) == 6026 and set(joined) == set(ids)
    for name, group in zip(['adult_candidate','child_suspect','uncertain'], groups):
        assert len(group) == summary['counts'][name]
        assert all(r['bucket'] == name for r in group)
    score = np.load(OUT/'prompt_scores.npy', allow_pickle=False)
    text = np.load(OUT/'text_embeddings.npy', allow_pickle=False)
    photos = np.load(PHOTOS/'full_photo_embeddings.npy', allow_pickle=False)
    expected = (photos[[r['photo_embedding_row'] for r in rows]] @ text.T).reshape(6026,4,3)
    assert np.allclose(expected, score, atol=1e-6)
    vetoes = 0
    for r, s in zip(rows, score):
        prior = r['prior_review']
        if prior and prior['status'] != 'accepted':
            vetoes += 1
            assert r['bucket'] == ('child_suspect' if prior['age_review']=='child_presenting' else 'uncertain')
            assert r['decision_source'] == 'prior_assistant_review_veto'
        elif r['bucket'] == 'adult_candidate':
            avg = s.mean(axis=0)
            assert avg[0] - max(avg[1:]) >= .02
            assert sum(s.argmax(axis=1) == 0) >= 3
            assert min(s[:,0] - s[:,1]) >= .005
        elif r['bucket'] == 'child_suspect':
            avg = s.mean(axis=0)
            assert avg[1] - max(avg[0], avg[2]) >= .015
            assert sum(s.argmax(axis=1) == 1) >= 3
            assert max(s[:,0] - s[:,1]) <= .005
        assert r['verified_age'] is None and r['status'] == 'automatic_unverified'
    assert vetoes == summary['prior_exclusions_preserved'] == 8
    keep = {r['outfit_id'] for r in groups[0]}
    regions = read(OUT/'remaining_garments.json')
    assert regions == [g for g in read(FULL/'approved_garments.json') if g['outfit_id'] in keep]
    assert len(regions) == summary['remaining_regions']
    for path, digest in protocol['source_hashes'].items():
        assert sha(path) == digest
    for name, digest in summary['outputs'].items():
        assert sha(OUT/name) == digest
    assert summary['new_photo_inference'] == summary['deleted_images'] == 0
    result = dict(passed=True, rows=6026, disjoint_exhaustive_buckets=True,
        prompt_score_alignment=True, threshold_boundary_cases=7, prior_exclusions_preserved=vetoes,
        source_and_demo_hashes_unchanged=True, remaining_regions=len(regions),
        note='Software/data integrity only; not age-classification accuracy.')
    save(OUT/'verification.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
