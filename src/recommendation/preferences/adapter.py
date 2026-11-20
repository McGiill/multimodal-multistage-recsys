"""Model-neutral adapter for TRAINED Two-Tower 64D item vectors.

No event writes, no fabricated history, no training/model loading. The caller
must provide the exact ordered item IDs/vectors from the same model/index.
"""
import math
import numpy as np


def build_seed_vector(payload, item_ids, item_vectors, id_map=None):
    if payload.get('schema_version') != 'closet-preference-seed-v1':
        raise ValueError('Unsupported preference schema')
    ids=list(item_ids)
    matrix=np.asarray(item_vectors,dtype=np.float32)
    if matrix.shape != (len(ids),64) or len(set(ids)) != len(ids):
        raise ValueError('Provide unique ordered item IDs and matching trained 64D vectors; not raw 512D FashionCLIP')
    if not np.isfinite(matrix).all():
        raise ValueError('Item vectors contain non-finite values')
    lookup={pid:i for i,pid in enumerate(ids)}
    weights={};missing=[];seen=set();source_total=0.0
    for seed in payload.get('seed_products',[]):
        pid=seed['product_id'];weight=float(seed['relative_weight'])
        if pid in seen or not math.isfinite(weight) or weight < 0:
            raise ValueError('Duplicate product or invalid weight')
        seen.add(pid);source_total+=weight
        # Exact IDs by default. If the team uses different IDs, provide a vetted
        # crosswalk; do not turn hm_ IDs into P00001 or row offsets by guessing.
        target=id_map.get(pid) if id_map is not None else pid
        if target not in lookup or np.linalg.norm(matrix[lookup[target]]) < 1e-8:
            missing.append(pid);continue
        weights[target]=weights.get(target,0.0)+weight
    retained=sum(weights.values())
    if retained<=0:
        return {'vector':None,'matched_product_ids':[],'unmapped_product_ids':missing,'coverage_weight':0.0}
    vector=np.zeros(64,dtype=np.float32)
    for pid,weight in weights.items():
        v=matrix[lookup[pid]];vector+=weight/retained*v/np.linalg.norm(v)
    norm=float(np.linalg.norm(vector))
    return {'vector':vector/norm if norm>1e-8 else None,
            'matched_product_ids':sorted(weights),'unmapped_product_ids':missing,
            'coverage_weight':retained/source_total if source_total else 0.0}


def blend_user_query(user_vector, seed_result, real_event_count=0, max_weight=0.2):
    """A bounded heuristic to be evaluated, not a trained/calibrated coefficient."""
    if not 0<=max_weight<=0.2 or isinstance(real_event_count,bool) or not isinstance(real_event_count,int) or real_event_count<0:
        raise ValueError('Use max_weight in [0,.2] and a nonnegative real event count')
    q=np.asarray(user_vector,dtype=np.float32).reshape(-1)
    if q.shape!=(64,) or not np.isfinite(q).all() or np.linalg.norm(q)<1e-8:
        raise ValueError('Expected a nonzero trained 64D user query')
    q=q/np.linalg.norm(q)
    if seed_result['vector'] is None:return q,0.0
    seed=np.asarray(seed_result['vector'],dtype=np.float32)
    coverage=float(seed_result['coverage_weight'])
    if seed.shape!=(64,) or not np.isfinite(seed).all() or not 0<=coverage<=1:
        raise ValueError('Invalid seed result')
    alpha=max_weight*coverage/(1+real_event_count/5)
    mixed=(1-alpha)*q+alpha*seed
    return mixed/np.linalg.norm(mixed),alpha


def rank_seed_candidates(seed_result, item_ids, item_vectors, allowed_mask, limit=20):
    """Optional cold-start candidate source. Caller applies audience/availability.

    Combine with popularity/real history upstream; never create view events from
    these IDs or increase the model's cold-start history length.
    """
    if not isinstance(limit,int) or isinstance(limit,bool) or not 1<=limit<=500:
        raise ValueError('limit must be in 1..500')
    matrix=np.asarray(item_vectors,dtype=np.float32);mask=np.asarray(allowed_mask,dtype=bool)
    if matrix.shape!=(len(item_ids),64) or mask.shape!=(len(item_ids),) or not np.isfinite(matrix).all():
        raise ValueError('ID, vector and availability mask alignment required')
    if seed_result['vector'] is None:return []
    norms=np.linalg.norm(matrix,axis=1);mask=mask & (norms>1e-8)
    scores=(matrix/np.maximum(norms[:,None],1e-8))@seed_result['vector']
    rows=np.flatnonzero(mask);rows=rows[np.argsort(-scores[rows],kind='stable')[:limit]]
    return [{'product_id':item_ids[i],'score':float(scores[i]),'source':'outfit_preference'} for i in rows]
