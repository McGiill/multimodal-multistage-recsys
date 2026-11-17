"""Publisher-declared model metadata bound to reviewed image hashes."""
import hashlib
import json
from runtime_paths import data_demo

CONFIG = json.loads((data_demo()/'config.json').read_text(encoding='utf-8'))
ENABLED = bool(CONFIG.get('publisher_model_review_file'))
ROWS = {}
if ENABLED:
    raw = (data_demo()/CONFIG['publisher_model_review_file']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG.get('publisher_model_review_sha256'):
        raise ValueError('Publisher metadata checksum mismatch')
    report = json.loads(raw)
    ROWS = report['outfits']
    snapshot = json.loads((data_demo()/CONFIG['snapshot_file']).read_text(encoding='utf-8'))
    expected = {oid for details in snapshot['details'].values() for oid in details}
    if set(ROWS) != expected:
        raise ValueError('Publisher review must cover every active snapshot outfit')
    for oid, row in ROWS.items():
        if row['cached_image_sha256'] != snapshot['images']['/media/outfit/'+oid]['sha256']:
            raise ValueError('Publisher metadata belongs to a different image')
        if row.get('status') == 'verified' and (row.get('returned_id') != oid.removeprefix('musinsa_')
                or row.get('publisher_model_gender') not in ('MEN', 'WOMEN')
                or not row.get('matched_media_sha256')):
            raise ValueError('Incomplete publisher verification evidence')


def verified(oid: str) -> bool:
    return not ENABLED or ROWS.get(oid, {}).get('status') == 'verified'


def gender(oid: str) -> str | None:
    return ROWS.get(oid, {}).get('publisher_model_gender') if verified(oid) else None


def matches(oid: str, profile: dict | None) -> bool:
    if not ENABLED:
        return True
    if not verified(oid):
        return False
    target = {'M':'MEN', 'F':'WOMEN'}.get((profile or {}).get('registration', {}).get('gender'))
    return target is None or gender(oid) == target


def context(profile: dict) -> dict:
    target = {'M':'MEN', 'F':'WOMEN'}.get(profile.get('registration', {}).get('gender'))
    return {'enabled':ENABLED, 'gender':target if ENABLED else None,
            'label':({'MEN':'남성 모델', 'WOMEN':'여성 모델'}.get(target, '모델 전체') if ENABLED else ''),
            'source':'publisher.model.gender' if ENABLED else None,
            'metadata_version':CONFIG.get('publisher_model_version')}
