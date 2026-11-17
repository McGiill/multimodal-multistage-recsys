"""Label-free selection of outfits from frozen model outputs and cached features.

No human/assistant visual labels are loaded. No image or text re-inference.
Thresholds are engineering heuristics, not calibrated confidence estimates.
"""
from pathlib import Path
import collections, hashlib, html, json, time
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'output/outfit_missing200'
TEXT=ROOT/'output/outfit_typecheck200'
OUT=ROOT/'output/outfit_autoselect200'
SLOTS=['top','bottom','outer','dress']
PROFILES={'conservative':0.02,'type_match_only':0.0}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
 temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def iou(a,b):
 intersection=area([max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])])
 union=area(a)+area(b)-intersection
 return intersection/union if union else 0.0

def main():
 start=time.perf_counter();OUT.mkdir(exist_ok=True)
 source_names=['outfits.json','consensus_garments.json','consensus_embeddings.npy','protocol.json']
 text_names=['text_prototypes.npy','text_prompts.json']
 detection_protocol=read(BASE/'protocol.json')
 protocol=dict(schema=1,profiles=PROFILES,primary='conservative',allowed_slots=SLOTS,
  minimum_box_area_fraction=.01,maximum_duplicate_iou=.7,max_regions_per_slot=1,minimum_distinct_slots=2,
  baseline_min_detector_score=.5,supplement_min_detector_score=.1,supplement_min_agreement_iou=.5,
  inputs={str(BASE/name):sha(BASE/name) for name in source_names}|{str(TEXT/name):sha(TEXT/name) for name in text_names},
  label_files_read=[],manual_decisions_used=0,rule_note='Model evidence only. Positive margin means the target slot outranks all other text-prototype slots. Profile thresholds fixed before seeing outfit counts.',
  scope_note='Preprocessing admission candidates, not verified product links, aesthetic approval, or detection accuracy. Shoes/accessories excluded from the two-garment requirement.',
  limitations=['No segmentation or true occlusion measurement.','Two different slots can still be wrong.','Thresholds/prompts were informed by prior development work; this is not independent held-out validation.'])
 if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
 else:save(OUT/'protocol.json',protocol)
 outfits=read(BASE/'outfits.json');garments=read(BASE/'consensus_garments.json')
 vectors=np.load(BASE/'consensus_embeddings.npy',allow_pickle=False)
 prototypes=np.load(TEXT/'text_prototypes.npy',allow_pickle=False);prompt_data=read(TEXT/'text_prompts.json');concepts=prompt_data['concepts']
 assert len(outfits)==200 and len(garments)==len(vectors)==976
 assert vectors.shape[1]==prototypes.shape[1]==512 and len(concepts)==len(prototypes)
 assert np.isfinite(vectors).all() and np.isfinite(prototypes).all()
 assert np.allclose(np.linalg.norm(vectors,axis=1),1,atol=.001)
 assert np.allclose(np.linalg.norm(prototypes,axis=1),1,atol=.001)
 photos={r['outfit_id']:r for r in outfits};score_rows=[]
 for g in garments:
  photo=photos[g['outfit_id']];w,h=photo['image_size'];box=g['box'];sim=vectors[g['embedding_row']]@prototypes.T
  scores={slot:float(max(sim[i] for i,c in enumerate(concepts) if c['slot']==slot)) for slot in {c['slot'] for c in concepts}}
  margin=scores[g['slot']]-max(v for k,v in scores.items() if k!=g['slot']) if g['slot'] in SLOTS else None
  best=int(np.argmax(sim));x1,y1,x2,y2=box;reasons=[]
  if g['slot'] not in SLOTS:reasons.append('slot_out_of_scope')
  if not (0<=x1<x2<=w and 0<=y1<y2<=h):reasons.append('invalid_box')
  fraction=area(box)/(w*h)
  if fraction<.01:reasons.append('box_too_small')
  if g['detection_source']=='baseline':
   if g['detection_score']<.5:reasons.append('detector_score_low')
  elif g['detection_source']=='consensus':
   if g['detection_score']<.1 or (g.get('confirmation_iou') or 0)<.5:reasons.append('agreement_insufficient')
  else:reasons.append('unknown_detection_source')
  score_rows.append(dict(garment_id=g['garment_id'],outfit_id=g['outfit_id'],dataset_id=g['dataset_id'],slot=g['slot'],box=box,
   source_embedding_row=g['embedding_row'],detector_score=g['detection_score'],detection_source=g['detection_source'],area_fraction=fraction,
   type_scores=scores,type_margin=margin,predicted_slot=concepts[best]['slot'],best_concept=concepts[best]['text'],common_rejection_reasons=reasons))
 save(OUT/'region_scores.json',score_rows)
 byout=collections.defaultdict(list)
 for g in score_rows:byout[g['outfit_id']].append(g)
 all_stats={};runs={}
 for name,threshold in PROFILES.items():
  decisions=[];region_decisions=[]
  for photo in outfits:
   candidates=[];local=[]
   for g in byout[photo['outfit_id']]:
    row=dict(g,rejection_reasons=list(g['common_rejection_reasons']),selected=False)
    if g['slot'] in SLOTS and g['type_margin']<threshold:row['rejection_reasons'].append('type_score_conflict_or_ambiguous')
    local.append(row)
    if not row['rejection_reasons']:candidates.append(row)
   # A uniform model-score ordering, not a hand-picked override per photo.
   candidates.sort(key=lambda g:(-g['type_margin'],-g['detector_score'],g['garment_id']))
   chosen=[]
   for g in candidates:
    if any(g['slot']==p['slot'] for p in chosen):g['rejection_reasons'].append('duplicate_slot')
    elif any(iou(g['box'],p['box'])>=.7 for p in chosen):g['rejection_reasons'].append('overlapping_region_duplicate')
    else:g['selected']=True;chosen.append(g)
   slots=sorted({g['slot'] for g in chosen});eligible=len(slots)>=2
   decision=dict(outfit_id=photo['outfit_id'],dataset_id=photo['dataset_id'],split=photo['split'],original_path=photo['original_path'],
    display_path='../outfit_missing200/'+photo['display_path'],image_size=photo['image_size'],accepted=eligible,
    selected_slots=slots,selected_garment_ids=[g['garment_id'] for g in chosen],reason='two_distinct_slots_available' if eligible else 'fewer_than_two_distinct_slots')
   decisions.append(decision);region_decisions+=local
  accepted_ids={r['outfit_id'] for r in decisions if r['accepted']}
  kept=[r for r in region_decisions if r['selected'] and r['outfit_id'] in accepted_ids]
  exported=[];export_vectors=[];source_byid={g['garment_id']:g for g in garments}
  for r in kept:
   export_vectors.append(vectors[r['source_embedding_row']]);exported.append(dict(source_byid[r['garment_id']],
    embedding_row=len(export_vectors)-1,source_embedding_row=r['source_embedding_row'],type_margin=r['type_margin'],
    selection_status='automatic_candidate',review_status='not_manually_reviewed'))
  matrix=np.asarray(export_vectors,dtype='float32').reshape(-1,512)
  target=OUT/name;target.mkdir(exist_ok=True)
  np.save(target/'embeddings.npy',matrix);save(target/'garments.json',exported)
  save(target/'outfits.json',[r for r in decisions if r['accepted']]);save(target/'all_outfit_decisions.json',decisions);save(target/'all_region_decisions.json',region_decisions)
  for r in exported:assert np.array_equal(matrix[r['embedding_row']],vectors[r['source_embedding_row']])
  for r in decisions:
   if r['accepted']:assert len(set(r['selected_slots']))>=2
  all_stats[name]=dict(outfits_total=200,accepted_outfits=len(accepted_ids),held_outfits=200-len(accepted_ids),
   accepted_by_split=dict(collections.Counter(r['split'] for r in decisions if r['accepted'])),
   outfit_selected_slot_count=dict(collections.Counter(len(r['selected_slots']) for r in decisions)),
   accepted_slot_combinations=dict(collections.Counter('+'.join(r['selected_slots']) for r in decisions if r['accepted'])),
   exported_regions=len(exported),exported_slots=dict(collections.Counter(r['slot'] for r in exported)),
   rejected_region_reasons=dict(collections.Counter(reason for r in region_decisions for reason in r['rejection_reasons'])),
   embedding_shape=list(matrix.shape))
  runs[name]=(decisions,region_decisions)
 save(OUT/'summary.json',all_stats)
 render(runs['conservative'],all_stats)
 assert all(sha(Path(p))==v for p,v in protocol['inputs'].items())
 runtime=dict(wall_seconds=time.perf_counter()-start,images_redetected=0,image_encodings=0,text_encodings=0,
  cached_image_vectors=len(vectors),cached_text_concepts=len(prototypes),manual_label_reads=0)
 save(OUT/'runtime.json',runtime)
 save(OUT/'validation.json',dict(unique_outfits=len(photos),unique_region_ids=len({g['garment_id'] for g in garments}),
  vector_mappings_verified=True,normalization_verified=True,minimum_two_slots_verified=True,source_files_unchanged=True,
  manual_label_files_not_loaded=True,feature_space_model=detection_protocol['encoder']['model_name']))
 print(json.dumps(dict(summary=all_stats,runtime=runtime),ensure_ascii=False))

def render(run,stats):
 decisions,regions=run;regions_byout=collections.defaultdict(list)
 for g in regions:regions_byout[g['outfit_id']].append(g)
 cards=[]
 names={'top':'상의','bottom':'하의','outer':'아우터','dress':'원피스·점프수트'}
 for r in decisions:
  w,h=r['image_size'];boxes=[];details=[]
  for g in regions_byout[r['outfit_id']]:
   if g['selected']:
    x1,y1,x2,y2=g['box'];boxes.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="#087b58" stroke-width="3"/><text x="{x1}" y="{max(15,y1-3)}" font-size="15" fill="#087b58" stroke="white" stroke-width="2" paint-order="stroke">{names[g["slot"]]}</text>')
   if g['slot'] in SLOTS:details.append(f'{names[g["slot"]]}: '+('통과' if g['selected'] else ', '.join(g['rejection_reasons']))+f' / 종류 점수차 {g["type_margin"]:.3f}')
  label='등록 후보' if r['accepted'] else '보류'
  cards.append(f'<section data-accepted="{r["accepted"]}"><h2>{r["dataset_id"]} · {label}</h2><p>{", ".join(names[s] for s in r["selected_slots"]) or "통과 영역 없음"}</p><div class="photo"><img loading="lazy" src="{r["display_path"]}"><svg viewBox="0 0 {w} {h}">'+''.join(boxes)+'</svg></div><details><summary>영역별 판정 이유</summary><p>'+'</p><p>'.join(html.escape(x) for x in details)+'</p></details></section>')
 s=stats['conservative'];b=stats['type_match_only']
 doc='<!doctype html><html lang="ko"><meta charset="utf-8"><title>200장 자동 선별 결과</title><style>body{font:14px/1.6 system-ui;background:#edf2f6;color:#192e40;margin:24px}h1{font-size:25px}h2{font-size:17px;margin:0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}section{background:white;padding:14px}.photo{position:relative;width:220px}img{width:100%;display:block}svg{position:absolute;inset:0;width:100%;height:100%}button{padding:9px;margin:8px 8px 12px 0}section.hidden{display:none}details{font-size:12px}</style><h1>사람 검수 없이 200장 자동 선별</h1><p>기본 기준: 검출 점수 + 종류 점수차 ≥ 0.02 + 박스 면적 ≥ 1% + 중복 제거 + 서로 다른 의류 부위 2개 이상.</p><p><b>등록 후보 '+str(s['accepted_outfits'])+'장 / 보류 '+str(s['held_outfits'])+'장 / 저장 영역 '+str(s['exported_regions'])+'개</b></p><p>비교: 종류 점수차 기준을 0으로 낮추면 '+str(b['accepted_outfits'])+'장. 어떤 기준도 정확도를 보장하지 않으며, 이번 결과는 상품 연결·스타일 선호 평가를 포함하지 않습니다. 기존 시각 검토 라벨은 읽지 않았습니다. 신발·잡화는 두 부위 조건에 포함하지 않습니다.</p><button onclick="filter(\'all\')">전체 200장</button><button onclick="filter(\'True\')">등록 후보만</button><button onclick="filter(\'False\')">보류만</button><div class="grid">'+''.join(cards)+'</div><script>function filter(v){document.querySelectorAll(\'section\').forEach(s=>s.classList.toggle(\'hidden\',v!==\'all\'&&s.dataset.accepted!==v))}filter(\'True\')</script></html>'
 (OUT/'comparison.html').write_text(doc,encoding='utf-8')

if __name__=='__main__':main()
