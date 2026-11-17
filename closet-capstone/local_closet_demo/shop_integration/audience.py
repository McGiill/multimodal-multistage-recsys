"""Reviewed clothing presentation, not the photographed person's gender."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_paths import data_demo
import publisher_models

REVIEW = json.loads((data_demo() / 'shop_integration/audience_review/recheck/reviewed.json').read_text(encoding='utf-8'))
REPLACEMENT_REVIEW = json.loads((data_demo() / 'replacement_extension.json').read_text(encoding='utf-8'))
if REVIEW['outfits'].keys() & REPLACEMENT_REVIEW['audiences'].keys():
    raise ValueError('Replacement review must not overwrite previous decisions')
REVIEW['outfits'].update(REPLACEMENT_REVIEW['audiences'])
AUDIENCES = {oid: row['audience'] for oid, row in REVIEW['outfits'].items()}
LABELS = ('menswear', 'womenswear', 'unisex')


def is_eligible(oid):
    row = REVIEW['outfits'].get(oid, {})
    return (row.get('status') == 'accepted' and row.get('age_review') == 'adult_presenting'
            and row.get('audience') in LABELS and publisher_models.verified(oid))


def visible_survey(survey, profile=None):
    # Keep stored history intact; retired images no longer appear or influence ranking.
    return {**survey, 'selected': [oid for oid in survey.get('selected', []) if is_eligible(oid)
            and publisher_models.matches(oid, profile)]}


def visible_profile(profile):
    return {**profile, **({'survey': visible_survey(profile['survey'], profile)} if 'survey' in profile else {})}


def preferred_audience(profile):
    return profile.get('shopping_audience', {'M':'men', 'F':'women'}.get(profile.get('registration', {}).get('gender'), 'all'))


def matches_audience(label, audience):
    if audience not in ('all', 'men', 'women', 'unisex', 'kids'):
        raise ValueError('추천 범위를 확인해 주세요.')
    if audience == 'all':
        return label in LABELS
    if audience == 'kids':
        return False
    if audience == 'unisex':
        return label == 'unisex'
    return label in ({'men':'menswear', 'women':'womenswear'}[audience], 'unisex')


def filter_outfits(outfits, audience, profile=None):
    matches_audience('unknown', audience)  # Validate even when the list is empty.
    return [o for o in outfits if is_eligible(o['id']) and matches_audience(AUDIENCES[o['id']], audience)
            and publisher_models.matches(o['id'], profile)]
