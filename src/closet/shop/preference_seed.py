"""Export explicit outfit choices as weak, replaceable recommendation context.

This is not a click/purchase event and does not call a team service.
"""
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone


def iso_time(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat() if value is not None else None


def build_preference_seed(uid, profile, dataset_version, audience, eligible, links, model_context=None,
                          saved_outfits=None, saved_eligible=None):
    survey = profile.get('survey', {})
    chosen = list(dict.fromkeys(survey.get('selected', [])))
    active_survey = [oid for oid in chosen if eligible(oid, audience)]
    saved_rows = saved_outfits or []
    saved_ids = sorted({r['outfit_id'] for r in saved_rows})
    active_saved = sorted({r['outfit_id'] for r in saved_rows if eligible(r['outfit_id'], audience)
                           and (saved_eligible is None or saved_eligible(r))})
    active = list(dict.fromkeys(active_survey + active_saved))
    sources = {oid: [name for name, ids in [('survey_selection', active_survey), ('saved_outfit', active_saved)] if oid in ids] for oid in active}
    seeds, evidence = defaultdict(float), defaultdict(list)
    for oid in active:
        # Equal mass per outfit and per garment, so a look with many alternatives
        # does not dominate. Cosine similarity is evidence, not a probability.
        groups = {gid: products for gid, products in links(oid).items() if products}
        for gid, products in sorted(groups.items()):
            mass = 1.0 / len(active) / len(groups) / len(products)
            for pid, match in sorted(products.items()):
                seeds[pid] += mass
                evidence[pid].append({'outfit_id':oid, 'garment_id':gid, **match, 'signal_sources':sources[oid]})
    total = sum(seeds.values())
    payload = {
        'schema_version':'closet-preference-seed-v1', 'operation':'replace',
        'user_id':uid, 'event_type':'outfit_preference',
        'status':('not_started' if not survey.get('completed') else 'skipped' if survey.get('skipped') else 'selected'),
        'selection_revision':survey.get('revision', 0),
        'last_action_at':iso_time(survey.get('updated_at')),
        'selected_at':iso_time(survey.get('selected_at', survey.get('updated_at') if chosen else None)),
        'selection_dataset_version':survey.get('dataset_version'),
        'dataset_version':dataset_version, 'shopping_audience':audience,
        'registration':{k:profile.get('registration', {}).get(k, 'unknown') for k in ('gender','age_group')},
        'selected_outfit_ids':chosen, 'effective_outfit_ids':active,
        'effective_survey_outfit_ids':active_survey,
        'inactive_outfit_ids':[oid for oid in chosen if oid not in active_survey],
        'saved_outfit_ids':saved_ids, 'effective_saved_outfit_ids':active_saved,
        'inactive_saved_outfit_ids':[oid for oid in saved_ids if oid not in active_saved],
        'has_preference_signal':bool(seeds),
        'signal_policy_version':'survey-saved-union-v1',
        'empty_seed_action':'clear_previous_seed',
        'retained_previous_selection':bool(chosen and survey.get('skipped')),
        'signal_strength':'weak', 'suggested_max_blend':0.2,
        'seed_products':[
            {'product_id':pid, 'source':'hm',
             'source_article_id':pid[3:] if re.fullmatch(r'hm_\d{10}',pid) else None,
             'relative_weight':weight / total, 'evidence':evidence[pid]}
            for pid, weight in sorted(seeds.items(), key=lambda p:(-p[1],p[0]))
        ],
        'model_applied':False,
        'publisher_model_filter':model_context,
    }
    # Same saved choices + context produce the same ID; consumers replace the
    # user's prior snapshot instead of counting GET/retries as new behavior.
    raw=json.dumps(payload,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()
    payload['snapshot_id']=hashlib.sha256(raw).hexdigest()
    return payload
