"""Nested 500 -> 2000 outfit audit, 30 retained -> 100 queries. Offline only."""
from preprocess_outfits_local import *
from dataclasses import asdict
from compare_outfit_missing200 import region,REPO,SLOTS
from src.search.detector import FashionDetector,GroundingDetector,best_in_slot,select_detections,crop,_iou
from src.search.encoder import build_encoder

OLD=ROOT/'output/outfit_missing200'
FULL=ROOT/'output/musinsa_full_200'
PREF=ROOT/'output/outfit_preferences136'
PREV=ROOT/'output/outfit500_products30'
OUT=ROOT/'output/outfit2000_products100'
SEED=2026110805

def prepare():
 for d in [OUT,OUT/'records',OUT/'vectors']:d.mkdir(exist_ok=True)
 previous=read(OLD/'protocol.json')
 for name,digest in previous['source_hashes'].items():assert sha(REPO/name)==digest,(name,'source changed')
 for name in ['src/search/encoder.py','src/search/detector.py']:
  assert sha(ROOT/'output/team_reuse_500/team_source'/name)==previous['source_hashes'][name]
 models=ROOT/'output/team_reuse_500/models'
 for name,prov in previous['provenance'].items():
  assert read(models/name/'provenance.json')==prov
  for f in prov['files']:assert sha(models/name/f['name'])==f['sha256']
 old=read(PREV/'manifest.json');samples=list(old);ids={r['source_id'] for r in old};hashes={r['input_sha256'] for r in old}
 pool=sorted(read(FULL/'gallery_manifest.json'),key=lambda r:r['row']);random.Random(SEED).shuffle(pool)
 for r in pool:
  if r['source_id'] in ids:continue
  digest=sha(r['path'])
  if digest in hashes:continue
  samples.append(dict(outfit_id='musinsa_'+r['source_id'],dataset_id=r['id'],source_id=r['source_id'],dataset_row=r['row'],publisher_style=r.get('publisher_style'),original_path=r['path'],input_sha256=digest,split='new1500'))
  ids.add(r['source_id']);hashes.add(digest)
  if len(samples)==2000:break
 assert len(samples)==len(ids)==len(hashes)==2000
 queries=read(PREV/'queries.json');qids={q['query_id'] for q in queries};fullq=read(FULL/'query_manifest.json')
 for slot in SLOTS:
  need={'top':34,'bottom':33,'outer':33}[slot]-sum(q['slot']==slot for q in queries)
  for q in [q for q in fullq if q['group']==slot and q['id'] not in qids][:need]:
   queries.append(dict(query_id=q['id'],product_id=q['catalog']['product_id'],slot=slot,name=q['catalog']['name'],category=q['catalog']['category_l3'],color=q['catalog']['color'],image_path=q['path'],input_sha256=q['input_sha256']))
   qids.add(q['id'])
 assert len(queries)==len(qids)==100
 for r in samples:assert sha(r['original_path'])==r['input_sha256']
 for q in queries:assert sha(q['image_path'])==q['input_sha256']
 protocol=dict(seed=SEED,raw_outfits=2000,retained_raw=500,retained_approved=349,queries=100,retained_queries=30,query_slots=dict(collections.Counter(q['slot'] for q in queries)),sampling='Keep all original 500 raw photos; append 1500 fixed-seed shuffled, distinct source IDs and exact image hashes. Keep 30 old queries; append first unused queries to reach top34/bottom33/outer33. This is a development pool, not a random population estimate. No relevance labels used.',previous_protocol_sha256=sha(OLD/'protocol.json'),encoder=previous['encoder'],selection_protocol=read(ROOT/'output/outfit_autoselect200/protocol.json'),retrieval=read(PREF/'protocol.json')['retrieval'],floors=[.65,.60,.55,.50],limitations=['Coverage and cosine are not human relevance accuracy.','Same development queries/data; no independent held-out evaluation.','No taste reranking in this gallery-size test.','Timing excludes image encoding, model loading and network.'],source_hashes={str(p):sha(p) for p in [PREV/'manifest.json',PREV/'queries.json',PREV/'protocol.json',PREV/'embeddings.npy',PREV/'query_embeddings.npy',PREV/'rankings.json',FULL/'query_manifest.json',FULL/'vectors_fashion_clip.npz',ROOT/'output/outfit_typecheck200/text_prototypes.npy']},runner_sha256=sha(__file__))
 if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol,'Changed protocol; use fresh output directory'
 else:write(OUT/'protocol.json',protocol)
 write(OUT/'manifest.json',samples);write(OUT/'queries.json',queries)
 print('Frozen 2000 photos / 100 queries; no labels loaded',flush=True)
 return samples,queries,previous

def preprocess(samples,previous):
 cfg=yaml.safe_load((REPO/'configs/style_search.yaml').read_text(encoding='utf-8'))
 models=ROOT/'output/team_reuse_500/models';device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(6);torch.manual_seed(42)
 runtime=copy.deepcopy(cfg);runtime['detector']['model_name']=str(models/'yolos');runtime['grounding']['model_name']=str(models/'grounding_tiny')
 fullmanifest=read(FULL/'crop_manifest.json');bank=np.load(FULL/'vectors_fashion_clip.npz',allow_pickle=False)
 bank_vectors=bank['requested'];lookup={(str(Path(r['path']).resolve()),tuple(r['box']),r['margin']):i for i,r in enumerate(fullmanifest['requested'])}
 assert len(bank_vectors)==len(fullmanifest['requested']) and bank_vectors.shape[1]==512
 det=gd=enc=None;allrows=[];vectors=[];records=[];fresh_images=0;encoded=0;bank_reuse=0;old_reuse=0;resumed=0;prior500_vectors=0
 start=time.perf_counter()
 for i,r in enumerate(samples):
  oid=r['outfit_id'];oldpath=OLD/'records'/(oid+'.json');dest=OUT/'records'/(oid+'.json')
  if oldpath.exists():
   z=read(oldpath);assert z['input_sha256']==r['input_sha256'] and z['status']=='ok'
   parts=[]
   for key in ['baseline','added']:
    p=OLD/z[key+'_vector_path'];assert sha(p)==z[key+'_vector_sha256'];v=np.load(p,allow_pickle=False);assert len(v)==len(z[key]);parts.append(v);old_reuse+=len(v)
   vv=np.concatenate(parts);gg=z['baseline']+z['added'];status='ok'
  elif (PREV/'records'/(oid+'.json')).exists():
   z=read(PREV/'records'/(oid+'.json'));assert z['status']=='ok' and z['input_sha256']==r['input_sha256']
   p=PREV/z['vector_path'];assert sha(p)==z['vector_sha256'];vv=np.load(p,allow_pickle=False);gg=z['garments'];status='ok';prior500_vectors+=len(vv)
  elif dest.exists() and read(dest)['status']=='ok':
   z=read(dest);assert z['input_sha256']==r['input_sha256'];p=OUT/z['vector_path'];assert sha(p)==z['vector_sha256'];vv=np.load(p,allow_pickle=False);gg=z['garments'];status='ok';resumed+=1
  else:
   z=dict(r,status='pending',garments=[]);gg=[];vv=np.empty((0,512),dtype='float32');status='error';tick=time.perf_counter()
   try:
    if det is None:det=FashionDetector(runtime,device=device);gd=GroundingDetector(runtime,device=device)
    with Image.open(r['original_path']) as im:im=im.convert('RGB')
    raw=det.detect_raw(im);strict=sorted(select_detections(raw,im.size,runtime),key=lambda d:(d.slot,d.kind,d.box));selected=[(d,'baseline',None,j) for j,d in enumerate(strict)]
    present={d.slot for d in strict};candidates=[];added=0
    for slot in SLOTS:
     if slot in present:continue
     low=best_in_slot(raw,im.size,runtime,slot)
     grounded=gd.find(im,[k for k,v in cfg['kinds'].items() if v['slot']==slot])
     overlap=_iou(low.box,grounded.box) if low and grounded else None
     accept=overlap is not None and overlap>=.5
     candidates.append(dict(slot=slot,low=asdict(low) if low else None,grounding=asdict(grounded) if grounded else None,iou=overlap,accepted=accept))
     if accept:selected.append((low,'consensus',overlap,added));added+=1
    gg=[region(d,r,origin,j,cfg,confirmation=overlap) for d,origin,overlap,j in selected]
    vv=np.empty((len(gg),512),dtype='float32');missing=[]
    for j,(d,_,_,_) in enumerate(selected):
     key=(str(Path(r['original_path']).resolve()),tuple(d.box),cfg['detector']['crop_margin'])
     if key in lookup:vv[j]=bank_vectors[lookup[key]];bank_reuse+=1
     else:missing.append(j)
    if missing:
     if enc is None:enc=build_encoder(str(models/'fashion_clip'),device=device,pad_square=previous['encoder']['pad_to_square'])
     v=enc.encode_images([crop(im,selected[j][0].box,cfg['detector']['crop_margin']) for j in missing],batch_size=16)
     vv[missing]=v;encoded+=len(missing)
    assert np.isfinite(vv).all() and (not len(vv) or np.allclose(np.linalg.norm(vv,axis=1),1,atol=.001))
    vp=OUT/'vectors'/(oid+'.npy');np.save(vp,vv);z.update(status='ok',image_size=list(im.size),garments=gg,candidates=candidates,vector_path=vp.relative_to(OUT).as_posix(),vector_sha256=sha(vp));status='ok'
   except Exception as e:z.update(status='error',error=f'{type(e).__name__}: {e}');gg=[];vv=np.empty((0,512),dtype='float32')
   sync();z['seconds']=time.perf_counter()-tick;write(dest,z);fresh_images+=1
  record=dict(r,status=status,image_size=z.get('image_size'),garments=[])
  if status=='error':record['error']=z.get('error');print('ERROR',oid,record['error'],flush=True)
  for g,v in zip(gg,vv):
   item=dict(g,embedding_row=len(allrows));allrows.append(item);vectors.append(v);record['garments'].append(item)
  records.append(record)
  if (i+1)%50==0:print(f'Processed {i+1}/2000 | old vectors {old_reuse+prior500_vectors}, bank reuse {bank_reuse}, new vectors {encoded}',flush=True)
 matrix=np.asarray(vectors,dtype='float32').reshape(-1,512);np.save(OUT/'embeddings.npy',matrix);write(OUT/'garments.json',allrows);write(OUT/'outfits.json',records)
 info=dict(old_images_reused=500,prior500_additional_vectors_reused=prior500_vectors,fresh_images=fresh_images,resumed_images=resumed,old_vectors_reused=old_reuse,fullbank_vectors_reused=bank_reuse,new_vectors_encoded=encoded,text_vectors_encoded=0,total_regions=len(allrows),errors=sum(r['status']!='ok' for r in records),wall_seconds=time.perf_counter()-start,device=device,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None)
 write(OUT/'runtime.json',info);print(json.dumps(info),flush=True)
 return records,allrows,matrix,bank['queries'],fullmanifest['queries']

def select(records,garments,vectors):
 from auto_select_outfit200 import iou,area
 proto=np.load(ROOT/'output/outfit_typecheck200/text_prototypes.npy',allow_pickle=False);concepts=read(ROOT/'output/outfit_typecheck200/text_prompts.json')['concepts']
 byout=collections.defaultdict(list);photos={r['outfit_id']:r for r in records};all_scores=vectors@proto.T
 for i,g in enumerate(garments):
  reasons=[];slot=g['slot'];w,h=photos[g['outfit_id']]['image_size'];x1,y1,x2,y2=g['box']
  scores={s:float(max(all_scores[i,j] for j,c in enumerate(concepts) if c['slot']==s)) for s in {c['slot'] for c in concepts}}
  margin=scores[slot]-max(v for s,v in scores.items() if s!=slot) if slot in ['top','bottom','outer','dress'] else None
  if margin is None:reasons.append('slot_out_of_scope')
  elif margin<.02:reasons.append('type_score_conflict_or_ambiguous')
  if not 0<=x1<x2<=w or not 0<=y1<y2<=h:reasons.append('invalid_box')
  if area(g['box'])/(w*h)<.01:reasons.append('box_too_small')
  if g['detection_source']=='baseline' and g['detection_score']<.5:reasons.append('detector_score_low')
  if g['detection_source']=='consensus' and (g['detection_score']<.1 or (g.get('confirmation_iou') or 0)<.5):reasons.append('agreement_insufficient')
  byout[g['outfit_id']].append(dict(g,type_margin=margin,rejection_reasons=reasons,selected=False))
 approved=[];decisions=[];region_decisions=[]
 for r in records:
  local=byout[r['outfit_id']];candidates=[g for g in local if not g['rejection_reasons']];candidates.sort(key=lambda g:(-g['type_margin'],-g['detection_score'],g['garment_id']));chosen=[]
  for g in candidates:
   if any(g['slot']==p['slot'] for p in chosen):g['rejection_reasons'].append('duplicate_slot')
   elif any(iou(g['box'],p['box'])>=.7 for p in chosen):g['rejection_reasons'].append('overlapping_region_duplicate')
   else:g['selected']=True;chosen.append(g)
  eligible=len({g['slot'] for g in chosen})>=2
  if eligible:approved.extend(chosen)
  decisions.append(dict(outfit_id=r['outfit_id'],accepted=eligible,status=r['status'],selected_slots=[g['slot'] for g in chosen]));region_decisions.extend(local)
 oldids={r['outfit_id'] for r in read(OLD/'manifest.json')};old_kept=read(ROOT/'output/outfit_autoselect200/conservative/garments.json');assert {g['garment_id'] for g in approved if g['outfit_id'] in oldids}=={g['garment_id'] for g in old_kept}
 oldvectors=np.load(ROOT/'output/outfit_autoselect200/conservative/embeddings.npy',allow_pickle=False);lookup={g['garment_id']:g for g in approved}
 for g in old_kept:assert np.array_equal(oldvectors[g['embedding_row']],vectors[lookup[g['garment_id']]['embedding_row']])
 write(OUT/'selection.json',decisions);write(OUT/'region_decisions.json',region_decisions);write(OUT/'approved_garments.json',approved)
 print('Approved',sum(r['accepted'] for r in decisions),'photos;',len(approved),'regions; old136/280 exactly reproduced',flush=True)
 return approved,decisions

def evaluate(queries,records,garments,vectors,bank_queries,query_jobs,approved,decisions):
 fullq=read(FULL/'query_manifest.json');lookup={q['id']:i for i,q in enumerate(fullq)};oldvec=np.load(PREV/'query_embeddings.npy',allow_pickle=False);qv=[]
 for i,q in enumerate(queries):
  j=lookup[q['query_id']];assert str(Path(query_jobs[j]['path']).resolve())==str(Path(q['image_path']).resolve())
  if i<30:
   assert float(oldvec[i]@bank_queries[j])>.9999
   qv.append(oldvec[i])
  else:qv.append(bank_queries[j])
 qv=np.asarray(qv);np.save(OUT/'query_embeddings.npy',qv)
 oldids={r['outfit_id'] for r in read(OLD/'manifest.json')};previous_ids={r['outfit_id'] for r in read(PREV/'manifest.json')};banks={'old136':[g for g in approved if g['outfit_id'] in oldids],'old500_selected':[g for g in approved if g['outfit_id'] in previous_ids],'new2000_selected':approved,'new2000_unfiltered':[g for g in garments if g['slot'] in SLOTS]}
 summary={};rankings={};timing={}
 for name,items in banks.items():
  summary[name]={};rankings[name]={};timing[name]={}
  for floor in [.65,.60,.55,.50]:
   runs=[];times=[]
   # One warmup batch; per-query timings below are in-memory retrieval, excluding scoring judgments.
   for slot in SLOTS:
    ix=[g['embedding_row'] for g in items if g['slot']==slot]
    if ix:qv[0]@vectors[ix].T
   for q,v in zip(queries,qv):
    begin=time.perf_counter();best={}
    for g in items:
     if g['slot']!=q['slot']:continue
     s=float(v@vectors[g['embedding_row']]);oid=g['outfit_id']
     if oid not in best or s>best[oid]['product_similarity']:best[oid]=dict(outfit_id=oid,garment_id=g['garment_id'],box=g['box'],product_similarity=s)
    ordered=sorted(best.values(),key=lambda r:(-r['product_similarity'],r['outfit_id']));threshold=max(floor,ordered[0]['product_similarity']-.06) if ordered else floor
    passing=[r for r in ordered if r['product_similarity']>=threshold];candidates=passing[:20];times.append((time.perf_counter()-begin)*1000)
    runs.append(dict(query_id=q['query_id'],threshold=threshold,candidate_count=len(candidates),passing_before_cap=len(passing),top5=candidates[:5],ungated_top5=ordered[:5]))
   key=f'{floor:.2f}';rankings[name][key]=runs;summary[name][key]={}
   for n in [20,30,100]:
    part=runs[:n];counts=[r['candidate_count'] for r in part]
    summary[name][key][str(n)]=dict(queries=n,zero=sum(x==0 for x in counts),at_least_one=sum(x>=1 for x in counts),at_least_three=sum(x>=3 for x in counts),at_least_five=sum(x>=5 for x in counts),mean_candidates=float(np.mean(counts)),candidate_count_distribution=dict(collections.Counter(counts)))
   timing[name][key]=dict(median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),scope='100 serial in-memory queries; includes Python filtering/dedup/sorting, excludes model/image encoding and loading')
 oldr=read(PREF/'sensitivity_rankings.json')
 for floor,runs in oldr.items():
  for a,b in zip(rankings['old136'][floor][:20],runs):
   assert a['candidate_count']==b['candidate_count']
   assert [r['outfit_id'] for r in a['top5']]==[r['outfit_id'] for r in b['profiles']['baseline'][:5]]
 old500=read(PREV/'rankings.json')
 for name in ['old136','old500_selected']:
  source='new500_selected' if name=='old500_selected' else name
  for floor,runs in old500[source].items():
   for a,b in zip(rankings[name][floor][:30],runs):
    assert a['candidate_count']==b['candidate_count']
    assert [r['outfit_id'] for r in a['top5']]==[r['outfit_id'] for r in b['top5']]
 prevgarments=read(PREV/'approved_garments.json');prevvectors=np.load(PREV/'embeddings.npy',allow_pickle=False);gmap={g['garment_id']:g for g in banks['old500_selected']}
 assert set(gmap)=={g['garment_id'] for g in prevgarments}
 for g in prevgarments:assert np.array_equal(prevvectors[g['embedding_row']],vectors[gmap[g['garment_id']]['embedding_row']])
 # Freeze outputs before any relevance review.
 write(OUT/'rankings.json',rankings);write(OUT/'summary.json',dict(raw_outfits=2000,approved_outfits=sum(d['accepted'] for d in decisions),approved_regions=len(approved),queries=100,results=summary,timing=timing,accuracy='Not measured: no new independent relevance labels.'))
 write(OUT/'validation.json',dict(old136_selected_ids_exact=True,old280_vectors_bitwise_identical=True,old20_rankings_all_four_thresholds_reproduced=True,old500_rankings_all_four_thresholds_reproduced=True,old719_vectors_bitwise_identical=True,query_vectors_reused=100,new_query_encodings=0,same_encoder_and_crop_source_hashes=True,source_photo_hashes_verified=True,unique_source_ids=2000,unique_photo_hashes=2000))
 print(json.dumps({name:{floor:v['100'] for floor,v in rows.items()} for name,rows in summary.items()},ensure_ascii=False),flush=True)

if __name__=='__main__':
 samples,queries,previous=prepare()
 records,garments,vectors,bq,qjobs=preprocess(samples,previous)
 approved,decisions=select(records,garments,vectors)
 evaluate(queries,records,garments,vectors,bq,qjobs,approved,decisions)
