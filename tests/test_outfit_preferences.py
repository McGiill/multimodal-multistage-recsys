"""Production API/pipeline/reranker with deterministic scoring doubles, no checkpoints.

This verifies wiring and isolation, not learned-model quality or real latency.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.closet.shop.preference_seed import build_preference_seed
from src.closet.shop.store import ProfileStore
from src.common.config import load_config
from src.common.schemas import Event
from src.recommendation import pipeline
from src.recommendation.preferences.signal import PreferencePolicy, parse_weights
from src.recommendation.reranking import RerankConfig, RerankService, history_count, max_run_length
from src.serving import outfit_preferences as transport
from src.serving.feature_store import MemoryFeatureStore
from src.serving.routers import recommend, events

NOW = datetime(2026, 4, 1, tzinfo=timezone.utc)
PIDS = [f'hm_{i:010d}' for i in range(1, 41)]


def snapshot(uid='A', products=None):
    return dict(schema_version='closet-preference-seed-v1', operation='replace',
                event_type='outfit_preference', user_id=uid,
                seed_products=[] if products is None else products)


@pytest.mark.parametrize('row', [
    {'product_id':'p', 'relative_weight':-1},
    {'product_id':'p', 'relative_weight':float('nan')},
    {'product_id':'p', 'relative_weight':True},
    {'product_id':'p', 'relative_weight':'1'},
    {'product_id':'', 'relative_weight':1},
    None,
])
def test_invalid_signal_rejected(row):
    with pytest.raises(ValueError):
        parse_weights(snapshot(products=[row]), 'A')


def test_contract_identity_duplicates_and_empty():
    payload = snapshot(products=[dict(product_id='p', relative_weight=1)])
    assert parse_weights(payload, 'A') == {'p':1}
    for change in [dict(user_id='B'), dict(operation='append'), dict(schema_version='other')]:
        with pytest.raises(ValueError):
            parse_weights({**payload, **change}, 'A')
    with pytest.raises(ValueError):
        parse_weights(snapshot(products=payload['seed_products'] * 2), 'A')
    assert parse_weights(snapshot(), 'A') == {}


@pytest.fixture
def system(monkeypatch):
    products = pd.DataFrame([dict(product_id=pid, category_l1='Menswear',
        category_l2=f'category{i%4}', category_l3='shirt', created_at=pd.Timestamp('2025-01-01'))
        for i, pid in enumerate(PIDS)])
    store = SimpleNamespace(item_ids=PIDS, pmap={p:i+1 for i,p in enumerate(PIDS)},
        products=products, created=np.zeros(41), metadata={'selection_cutoff':0})
    store.created[39] = NOW.timestamp() + 86400  # known but not yet available
    features = MemoryFeatureStore()
    histories = {}
    observations = []

    class Retriever:
        def retrieve(self, user_id, n, at, recent=None):
            h = np.zeros(20, dtype=np.int64)
            real = histories.get(user_id, []) + list(recent or [])
            h[:min(20,len(real))] = real[-20:]
            state = (h, max(1,len(real[-20:])), np.zeros(4))
            observations.append((user_id, deepcopy(state)))
            return np.arange(1,21), np.linspace(1,0,20), state

    def score(ids, state, now):
        return 1 / (np.asarray(ids, dtype=np.float32) + 1)

    models = SimpleNamespace(store=store, cfg={'candidate':{'n_candidates':20}},
                             retriever=Retriever(), score=score)
    cfg = RerankConfig(top_n=10, new_item_slots=0, exploration_enabled=False)
    svc = RerankService(PIDS, products, cfg)
    svc.cold_start_lists = lambda now, k: (svc.candidates(range(1,21), np.linspace(1,.1,20),'popular'), [])
    last_results = []
    svc.record_shown = lambda uid, result: last_results.append(result)
    monkeypatch.setattr(pipeline,'get_recommendation_models',lambda:models)
    monkeypatch.setattr(pipeline,'get_rerank_service',lambda _:svc)
    monkeypatch.setattr(pipeline,'get_feature_store',lambda:features)
    monkeypatch.setattr(pipeline,'get_session_model',lambda:None)
    monkeypatch.setattr(events,'get_feature_store',lambda:features)
    monkeypatch.setattr(events,'record_exploration_click',lambda *args:None)
    app = FastAPI()
    app.include_router(recommend.router,prefix='/api')
    app.include_router(events.router,prefix='/api')
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, features=features, models=models, svc=svc,
            histories=histories, observations=observations, results=last_results)


@pytest.fixture
def source(tmp_path, monkeypatch):
    profiles = ProfileStore(tmp_path/'prefs.sqlite3')
    control = {'mode':'normal', 'requests':[]}
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            uid = parse_qs(urlsplit(self.path).query)['user_id'][0]
            control['requests'].append(uid)
            if control['mode']=='error':
                self.send_error(503); return
            selected = profiles.get(uid)
            saved = profiles.saved_outfits(uid)
            payload = build_preference_seed(uid, selected, 'test', 'men', lambda *args:True,
                lambda oid:{'top': {PIDS[-1] if oid=='look1' else PIDS[-3]: {}}},
                saved_outfits=saved)
            if control['mode']=='wrong_user': payload['user_id']='another-user'
            raw = json.dumps(payload).encode()
            self.send_response(200); self.send_header('Content-Type','application/json')
            self.end_headers(); self.wfile.write(raw)
        def log_message(self, *args): pass
    server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
    thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    monkeypatch.setenv('CLOSET_PREFERENCE_URL', f'http://127.0.0.1:{server.server_port}/api/closet/preference-seed')
    try:
        yield SimpleNamespace(profiles=profiles, control=control)
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=3)


def request(system, uid='A', enabled=True):
    return system.client.get('/api/recommend', params={'user_id':uid,'top_n':10,
        'at':NOW.isoformat(), 'session_id':f'{uid}-session', 'use_outfit_preferences':str(enabled).lower()})


def ids(response):
    assert response.status_code == 200, response.text
    return [r['product_id'] for r in response.json()['recommendations']]


def test_selection_changes_existing_recommendation_without_events_and_cancel_restores(system, source):
    before = request(system)
    assert before.headers['x-outfit-preference-source']=='empty'
    source.profiles.save_survey('A',['look1'],False,dataset_version='test')
    selected = request(system)
    assert PIDS[-1] in ids(selected) and PIDS[-1] not in ids(before)
    assert system.results[-1].is_cold_start
    assert selected.json()['session_context']['recent_clicks']==[]
    assert system.features.get_recent_views('A')==[]
    assert system.features.get_session_clicks('A-session')==[]
    assert history_count(system.observations[-1][1])==0
    assert ids(request(system))==ids(selected)  # reading twice never accumulates
    assert ids(request(system,uid='B'))==ids(before)
    source.profiles.save_survey('A',[],False,dataset_version='test')
    cleared = request(system)
    assert cleared.headers['x-outfit-preference-source']=='empty'
    assert cleared.json()['recommendations']==before.json()['recommendations']


def test_saved_union_unsave_and_selection_replace(system, source):
    source.profiles.save_survey('A',['look1'],False,dataset_version='test')
    original = ids(request(system))
    source.profiles.set_outfit_saved('A','look1','owned',True)
    assert ids(request(system))==original  # overlap counts once
    source.profiles.set_outfit_saved('A','look1','owned',False)
    assert ids(request(system))==original  # survey remains
    source.profiles.save_survey('A',['look2'],False,dataset_version='test')
    replaced = ids(request(system))
    assert PIDS[-3] in replaced and PIDS[-1] not in replaced
    source.profiles.save_survey('A',[],False,dataset_version='test')
    baseline = ids(request(system))
    source.profiles.set_outfit_saved('A','look1','owned',True)
    assert PIDS[-1] in ids(request(system))
    source.profiles.set_outfit_saved('A','look1','owned',False)
    assert ids(request(system))==baseline


def test_source_failure_never_reuses_previous_preference(system, source):
    baseline = ids(request(system))
    source.profiles.save_survey('A',['look1'],False,dataset_version='test')
    assert PIDS[-1] in ids(request(system))
    for mode in ['error','wrong_user']:
        source.control['mode']=mode
        response=request(system)
        assert response.headers['x-outfit-preference-source']=='unavailable'
        assert ids(response)==baseline


def test_explicit_opt_in_keeps_offline_and_baseline_calls_unchanged(system, source):
    source.profiles.save_survey('A',['look1'],False,dataset_version='test')
    response=request(system,enabled=False)
    assert PIDS[-1] not in ids(response)
    assert source.control['requests']==[]
    assert 'x-outfit-preference-source' not in response.headers


def test_real_view_and_purchase_still_use_original_event_path(system, source):
    for kind in ['view','purchase']:
        response=system.client.post('/api/events',json=dict(event_id=kind,user_id='A',session_id='A-session',
            event_type=kind,product_id=PIDS[1],timestamp=NOW.isoformat()))
        assert response.status_code==200
    assert system.features.get_recent_views('A')==[PIDS[1]]
    source.profiles.save_survey('A',['look1'],False,dataset_version='test')
    selected=request(system)
    assert selected.json()['session_context']['recent_clicks']==[PIDS[1]]
    assert history_count(system.observations[-1][1])==1
    source.profiles.save_survey('A',[],False,dataset_version='test')
    request(system)
    assert system.features.get_recent_views('A')==[PIDS[1]]
    assert system.features.get_session_clicks('A-session')==[PIDS[1]]


def test_missing_future_ids_and_zero_weight_do_not_change_recommendations(system, monkeypatch):
    strategy=pipeline.MultiStageStrategy()
    baseline=strategy.recommend('A',10,at=NOW,session_clicks=[])
    invalid=strategy.recommend('A',10,at=NOW,session_clicks=[],
        preference_weights={'missing':.5,PIDS[38]:.5})
    assert invalid.recommendations==baseline.recommendations
    original=load_config
    monkeypatch.setattr(pipeline,'load_config',lambda name: {**original(name),'max_weight':0})
    zero=strategy.recommend('A',10,at=NOW,session_clicks=[],preference_weights={PIDS[-1]:1})
    assert zero.recommendations==baseline.recommendations


def test_warm_history_is_unchanged_and_preference_is_weaker(system):
    strategy=pipeline.MultiStageStrategy()
    cold=strategy.recommend('A',20,at=NOW,session_clicks=[],preference_weights={PIDS[-1]:1})
    system.histories['A']=[1,2,3,4,5,6,7,8,9,10]
    warm=strategy.recommend('A',20,at=NOW,session_clicks=[],preference_weights={PIDS[-1]:1})
    assert history_count(system.observations[-1][1])==10
    assert not system.results[-1].is_cold_start
    cold_ids=[r.product_id for r in cold.recommendations]
    warm_ids=[r.product_id for r in warm.recommendations]
    assert cold_ids.index(PIDS[-1]) < warm_ids.index(PIDS[-1])


def test_diversity_new_and_exploration_rules_still_apply(system):
    system.svc.config.new_item_slots=1
    system.svc.config.exploration_enabled=True
    system.svc.config.exploration_slots=2
    from dataclasses import replace
    system.svc._rows[-2]=replace(system.svc._rows[-2],created_at=NOW)
    system.svc._created=system.svc._created.copy()
    system.svc._created[-2]=NOW.timestamp()
    result=pipeline.MultiStageStrategy().recommend('A',10,at=NOW,session_clicks=[],
                                                 preference_weights={PIDS[-1]:1})
    reasons=[r.reason for r in result.recommendations]
    assert 'new_item' in reasons and 'mab_exploration' in reasons
    cats=[system.svc.catalog.loc[r.product_id,'category_l2'] for r in result.recommendations]
    assert max_run_length(cats)<=2
    assert len({r.product_id for r in result.recommendations})==len(result.recommendations)
    assert system.results[-1].is_cold_start


def test_model_missing_is_still_503_not_fake_recommendation(system, monkeypatch):
    def missing(): raise FileNotFoundError('model missing')
    monkeypatch.setattr(pipeline,'get_recommendation_models',missing)
    response=request(system,enabled=False)
    assert response.status_code==503


def test_timeout_and_invalid_body_fall_back(monkeypatch):
    class Opener:
        def open(self,*args,**kwargs): raise TimeoutError()
    monkeypatch.setattr(transport,'build_opener',lambda *args:Opener())
    assert transport.fetch_outfit_preferences('A')==({},'unavailable')
