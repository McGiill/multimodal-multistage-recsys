"""Fixed-rule, resumable local 200-image comparison; no network or paid APIs."""
from preprocess_outfits_local import *
from dataclasses import asdict

OUT=ROOT/'output/outfit_missing200'
REPO=Path('C:/Users/전민재/Downloads/multimodal-multistage-recsys-main (2)/multimodal-multistage-recsys-main')
sys.path.insert(0,str(REPO))
from src.search.detector import FashionDetector,GroundingDetector,best_in_slot,select_detections,crop,_iou
from src.search.encoder import build_encoder
SLOTS=['top','bottom','outer']

def sample_manifest():
 if (OUT/'manifest.json').exists():return read(OUT/'manifest.json')
 old=read(ROOT/'output/outfit_preprocess_20/manifest.json')
 samples=[dict(r,split='pilot20') for r in old]
 ids={r['source_id'] for r in samples};hashes={r['input_sha256'] for r in samples}
 pool=sorted(read(ROOT/'output/musinsa_full_200/gallery_manifest.json'),key=lambda r:r['row'])
 random.Random(20261106).shuffle(pool)
 for r in pool:
  if r['source_id'] in ids:continue
  digest=sha(r['path'])
  if digest in hashes:continue
  samples.append(dict(outfit_id='musinsa_'+r['source_id'],dataset_id=r['id'],source_id=r['source_id'],dataset_row=r['row'],publisher_style=r.get('publisher_style'),original_path=str(Path(r['path']).resolve()),input_sha256=digest,split='new180'))
  ids.add(r['source_id']);hashes.add(digest)
  if len(samples)==200:break
 assert len(samples)==len(ids)==len(hashes)==200
 write(OUT/'manifest.json',samples);return samples

def region(d,r,origin,idx,cfg,confirmation=None):
 return dict(garment_id=r['outfit_id']+f'_{origin}_{idx:02d}',outfit_id=r['outfit_id'],dataset_id=r['dataset_id'],kind=d.kind,slot=d.slot,label=d.label,box=list(d.box),detection_score=d.score,detection_source=origin,detection_status='unreviewed_model_prediction',crop_margin=cfg['detector']['crop_margin'],embedding_local_row=idx,confirmation_iou=confirmation)

def run():
 for sub in ['records','vectors','photos']:(OUT/sub).mkdir(parents=True,exist_ok=True)
 cfg=yaml.safe_load((REPO/'configs/style_search.yaml').read_text(encoding='utf-8'))
 search=yaml.safe_load((REPO/'configs/search.yaml').read_text(encoding='utf-8'))
 models=ROOT/'output/team_reuse_500/models'
 provenance={k:read(models/k/'provenance.json') for k in ['yolos','grounding_tiny','fashion_clip']}
 assert provenance['yolos']['repo']==cfg['detector']['model_name']
 assert provenance['grounding_tiny']['repo']==cfg['grounding']['model_name']
 assert provenance['fashion_clip']['repo']==search['clip']['model_name']
 torch.set_num_threads(6);torch.manual_seed(42)
 device='cuda' if torch.cuda.is_available() else 'cpu'
 protocol=dict(schema=1,seed=20261106,sample='Original pilot20 plus 180 new exact-hash/source-ID-disjoint photos; no result-based replacements.',slots=SLOTS,detector=cfg['detector'],grounding=cfg['grounding'],kinds=cfg['kinds'],encoder=search['clip'],provenance=provenance,source_hashes={p:sha(REPO/p) for p in ['src/search/detector.py','src/search/encoder.py','src/common/faiss_utils.py','configs/style_search.yaml']},rule='Keep all original strict YOLOS detections. For each absent top/bottom/outer slot, find YOLOS best >=0.1 and Grounding DINO >=0.2. Append YOLOS box only when both have same slot and IoU>=0.5. No manual filtering in automatic results.',consensus_iou=0.5,torch=torch.__version__,device=device,offline=True)
 key=hashlib.sha256(json.dumps(protocol,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
 if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
 else:write(OUT/'protocol.json',protocol)
 samples=sample_manifest();pending=[];records=[]
 for r in samples:
  assert sha(r['original_path'])==r['input_sha256']
  path=OUT/'records'/(r['outfit_id']+'.json')
  if path.exists():
   z=read(path);assert z['protocol_key']==key
   records.append(z)
  else:pending.append(r)
 loadtime=0;fresh=[]
 if pending:
  runtime=copy.deepcopy(cfg);runtime['detector']['model_name']=str(models/'yolos');runtime['grounding']['model_name']=str(models/'grounding_tiny')
  t=time.perf_counter();det=FashionDetector(runtime,device=device);gd=GroundingDetector(runtime,device=device);enc=build_encoder(str(models/'fashion_clip'),device=device,pad_square=search['clip']['pad_to_square'])
  with Image.open(pending[0]['original_path']) as im:warm=im.convert('RGB')
  det.detect_raw(warm);gd.find(warm,['shirt','tee']);enc.encode_images([warm]);sync();loadtime=time.perf_counter()-t
  if device=='cuda':torch.cuda.reset_peak_memory_stats()
  for i,r in enumerate(pending):
   z=dict(r,protocol_key=key,status='pending',baseline=[],added=[],candidates=[],timing={})
   try:
    with Image.open(r['original_path']) as im:im=im.convert('RGB')
    z['image_size']=list(im.size)
    photo=OUT/'photos'/(r['outfit_id']+'.jpg');shutil.copyfile(r['original_path'],photo);z['display_path']=photo.relative_to(OUT).as_posix()
    sync();t=time.perf_counter();raw=det.detect_raw(im);strict=sorted(select_detections(raw,im.size,runtime),key=lambda d:(d.slot,d.kind,d.box));sync();z['timing']['baseline_detect']=time.perf_counter()-t
    t=time.perf_counter();bvec=enc.encode_images([crop(im,d.box,cfg['detector']['crop_margin']) for d in strict],batch_size=16);sync();z['timing']['baseline_encode']=time.perf_counter()-t
    z['raw_detections']=raw;z['baseline']=[region(d,r,'baseline',idx,cfg) for idx,d in enumerate(strict)]
    present={d.slot for d in strict};extra=[];t=time.perf_counter()
    for slot in SLOTS:
     if slot in present:continue
     low=best_in_slot(raw,im.size,runtime,slot);kinds=[k for k,v in cfg['kinds'].items() if v['slot']==slot]
     grounded=gd.find(im,kinds);overlap=_iou(low.box,grounded.box) if low and grounded else None
     accept=overlap is not None and overlap>=.5
     z['candidates'].append(dict(candidate_id=r['dataset_id']+':'+slot,slot=slot,low=asdict(low) if low else None,grounding=asdict(grounded) if grounded else None,iou=overlap,accepted=accept))
     if accept:extra.append((low,overlap))
    sync();z['timing']['extra_detect']=time.perf_counter()-t
    t=time.perf_counter();avec=enc.encode_images([crop(im,d.box,cfg['detector']['crop_margin']) for d,_ in extra],batch_size=16);sync();z['timing']['extra_encode']=time.perf_counter()-t
    z['added']=[region(d,r,'consensus',idx,cfg,confirmation=overlap) for idx,(d,overlap) in enumerate(extra)]
    for name,v in [('baseline',bvec),('added',avec)]:
     assert v.shape[1]==512 and np.isfinite(v).all()
     assert not len(v) or np.allclose(np.linalg.norm(v,axis=1),1,atol=.001)
     vp=OUT/'vectors'/(r['outfit_id']+'_'+name+'.npy');np.save(vp,v)
     z[name+'_vector_path']=vp.relative_to(OUT).as_posix();z[name+'_vector_sha256']=sha(vp)
    z['status']='ok'
   except Exception as e:z['status']='error';z['error']=f'{type(e).__name__}: {e}'
   write(OUT/'records'/(r['outfit_id']+'.json'),z);fresh.append(z);records.append(z)
   if (i+1)%25==0 or z['status']=='error':print(f'{i+1}/{len(pending)} processed; latest={r["dataset_id"]} {z["status"]}',flush=True)
 lookup={r['outfit_id']:r for r in records};records=[lookup[r['outfit_id']] for r in samples]
 vectors=[];baseline_rows=[];baseline_matrix=[];outfits=[]
 for r in records:
  if r['status']!='ok':continue
  combined=[]
  for name in ['baseline','added']:
   vp=OUT/r[name+'_vector_path'];assert sha(vp)==r[name+'_vector_sha256'];v=np.load(vp,allow_pickle=False);assert len(v)==len(r[name])
   for idx,g in enumerate(r[name]):
    g['embedding_row']=len(vectors);vectors.append(v[idx]);combined.append(g)
    if name=='baseline':
     bg=dict(g,embedding_row=len(baseline_matrix));baseline_rows.append(bg);baseline_matrix.append(v[idx])
  outfits.append(dict(outfit_id=r['outfit_id'],dataset_id=r['dataset_id'],split=r['split'],original_path=r['original_path'],display_path=r['display_path'],image_size=r['image_size'],status=r['status'],garments=combined))
 matrix=np.asarray(vectors,dtype='float32').reshape(-1,512);bm=np.asarray(baseline_matrix,dtype='float32').reshape(-1,512)
 np.save(OUT/'baseline_embeddings.npy',bm);np.save(OUT/'consensus_embeddings.npy',matrix)
 write(OUT/'baseline_garments.json',baseline_rows);write(OUT/'consensus_garments.json',[g for r in outfits for g in r['garments']]);write(OUT/'outfits.json',outfits);write(OUT/'results.json',records)
 assert len({g['garment_id'] for r in outfits for g in r['garments']})==len(matrix)
 summary={}
 for split in ['all200','pilot20','new180']:
  group=[r for r in records if split=='all200' or r['split']==split];ok=[r for r in group if r['status']=='ok']
  counts={};times={k:sum(r['timing'].get(k,0) for r in ok) for k in ['baseline_detect','baseline_encode','extra_detect','extra_encode']}
  for mode in ['baseline','consensus']:
   detections=[g for r in ok for g in r['baseline']+(r['added'] if mode=='consensus' else [])]
   counts[mode]=dict(regions=len(detections),major_regions=sum(g['slot'] in SLOTS for g in detections),slots=dict(collections.Counter(g['slot'] for g in detections)),zero_detection_images=sum(not(r['baseline']+(r['added'] if mode=='consensus' else [])) for r in ok))
  summary[split]=dict(images=len(group),successful=len(ok),errors=len(group)-len(ok),counts=counts,images_with_additions=sum(bool(r['added']) for r in ok),added_regions=sum(len(r['added']) for r in ok),missing_slot_probes=sum(len(r['candidates']) for r in ok),timing_seconds=times,baseline_compute_seconds=times['baseline_detect']+times['baseline_encode'],consensus_compute_seconds=sum(times.values()),note='Counts are model outputs, not accuracy. No annotated ground-truth recall or product retrieval test.')
 runinfo=dict(fresh_images=len(fresh),reused_images=len(records)-len(fresh),setup_and_warmup_seconds=loadtime,wall_seconds=time.perf_counter()-BOOT,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,gpu_peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2 if device=='cuda' and fresh else 0,ram_peak_mib=ram_peak())
 write(OUT/'summary.json',summary);write(OUT/f'run_{time.time_ns()}.json',runinfo);write(OUT/'latest_run.json',runinfo)
 make_html(records,summary)
 print(json.dumps(dict(summary=summary,run=runinfo),ensure_ascii=False),flush=True)

def make_html(records,summary):
 cards=[];added_cards=[];review_idx=0
 for r in records:
  if r['status']!='ok':continue
  w,h=r['image_size'];panels=[]
  for label,items in [('기존',r['baseline']),('보완',r['baseline']+r['added'])]:
   boxes=[]
   for g in items:
    if g['slot'] not in SLOTS:continue
    x1,y1,x2,y2=g['box'];c='#d92d20' if g['detection_source']=='consensus' else '#176b9c'
    boxes.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="{c}" stroke-width="3"/><text x="{x1}" y="{max(14,y1-3)}" font-size="15" fill="{c}" stroke="white" stroke-width="2" paint-order="stroke">{g["slot"]}</text>')
   panels.append(f'<div><b>{label}</b><div class="photo"><img loading="lazy" src="{r["display_path"]}"><svg viewBox="0 0 {w} {h}">'+''.join(boxes)+'</svg></div></div>')
  cards.append(f'<section data-added="{bool(r["added"])}" data-split="{r["split"]}"><h2>{r["dataset_id"]} · {r["split"]}</h2><div class="pair">'+''.join(panels)+f'</div><p>전체 영역 {len(r["baseline"])} → {len(r["baseline"])+len(r["added"])}</p></section>')
  for g in r['added']:
   review_idx+=1;x1,y1,x2,y2=g['box']
   added_cards.append(f'<section data-review="{review_idx}" data-split="{r["split"]}"><h2>#{review_idx} {r["dataset_id"]} · {g["slot"]}</h2><div class="photo"><img src="{r["display_path"]}"><svg viewBox="0 0 {w} {h}"><rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="#d92d20" stroke-width="3"/></svg></div><p>{r["split"]} · {g["kind"]} · IoU {g["confirmation_iou"]:.2f}</p></section>')
 css='body{font:14px/1.5 system-ui;background:#eef2f5;color:#182b40;margin:20px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(450px,1fr));gap:16px}section{background:white;padding:12px}h2{font-size:17px;margin:0 0 8px}.pair{display:flex;gap:12px}.photo{width:210px;position:relative}img{width:100%;display:block}svg{position:absolute;inset:0;width:100%;height:100%}body.hide svg{display:none}button{padding:8px;margin:5px}section.hidden{display:none}'
 totals=summary['all200'];head=f'<h1>무신사 200장 · 기존 / 두 모델 합의 보완</h1><p>기존 20 + 신규 180. YOLOS 0.5 / 보완 YOLOS 0.1 + DINO 0.2, IoU 0.5. 상의·하의·아우터만 보완. 파랑 기존 · 빨강 추가.</p><p>영역 {totals["counts"]["baseline"]["regions"]} → {totals["counts"]["consensus"]["regions"]}. 검출 개수는 정확도가 아닙니다. 이 화면은 수동 선별 전 자동 결과입니다.</p>'
 controls='<label><input type="checkbox" checked onchange="document.body.classList.toggle(\'hide\',!this.checked)"> 박스 표시</label><button onclick="filterCards(false)">전체 200장</button><button onclick="filterCards(true)">추가된 사진만</button><script>function filterCards(only){document.querySelectorAll(\'section\').forEach(s=>s.classList.toggle(\'hidden\',only&&s.dataset.added!==\'True\'))}</script>'
 (OUT/'comparison.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>200장 비교</title><style>'+css+'</style>'+head+controls+'<div class="grid">'+''.join(cards)+'</div></html>',encoding='utf-8')
 (OUT/'additions_review.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>추가 영역 검수</title><style>'+css+'.grid{grid-template-columns:repeat(5,1fr);gap:12px}.photo{width:220px}h2{font-size:15px}section{padding:8px}</style>'+head+'<div class="grid">'+''.join(added_cards)+'</div></html>',encoding='utf-8')

if __name__=='__main__':run()
