"""Offline full-photo tags and bounded preference reranking of real product queries."""
from preprocess_outfits_local import *

BASE=ROOT/'output/outfit_autoselect200/conservative'
OUT=ROOT/'output/outfit_preferences136'
REPO=Path('C:/Users/전민재/Downloads/multimodal-multistage-recsys-main (2)/multimodal-multistage-recsys-main')
sys.path.insert(0,str(REPO))
from src.search.encoder import build_encoder

STYLES={
 'casual':['a person wearing a casual everyday outfit','an outfit of relaxed everyday casual clothes'],
 'minimal':['a person wearing a minimalist outfit with simple clean lines and neutral colors','a simple understated minimalist fashion outfit'],
 'street':['a person wearing an urban streetwear outfit with oversized layers','a streetwear fashion outfit with baggy clothes and bold styling'],
 'formal':['a person wearing a formal tailored outfit with a suit or blazer','an elegant business formal fashion outfit'],
 'romantic':['a person wearing a romantic feminine outfit with lace or ruffles','a soft romantic fashion outfit with delicate decorative details'],
 'sport':['a person wearing athletic sportswear for exercise','a sporty activewear outfit with performance sports clothes'],
 'workwear':['a person wearing a rugged workwear outfit with utility pockets and heavy fabrics','a practical workwear fashion outfit with cargo or utility garments']}
SCENES={
 'studio':['a fashion photograph taken in a photo studio with a plain seamless backdrop','a person posing for a studio photoshoot against a simple solid background'],
 'outdoor':['a fashion photograph taken outdoors on a street with buildings and pavement','a person photographed outdoors with outdoor scenery in the background'],
 'indoor':['a fashion photograph taken inside an ordinary room with furniture and interior details','a person photographed inside a home shop or cafe with a visible furnished interior']}
NAMES={'casual':'캐주얼','minimal':'미니멀','street':'스트리트','formal':'포멀','romantic':'로맨틱','sport':'스포티','workwear':'워크웨어','studio':'스튜디오','outdoor':'야외','indoor':'일상 실내','unknown':'불명'}
PROFILES={
 'baseline':dict(style=None,scene=None,name='상품 매칭만'),
 'casual_studio':dict(style='casual',scene='studio',name='캐주얼 · 스튜디오'),
 'casual_outdoor':dict(style='casual',scene='outdoor',name='캐주얼 · 야외'),
 'street_studio':dict(style='street',scene='studio',name='스트리트 · 스튜디오'),
 'street_outdoor':dict(style='street',scene='outdoor',name='스트리트 · 야외'),
 'minimal_studio':dict(style='minimal',scene='studio',name='미니멀 · 스튜디오'),
 'formal_indoor':dict(style='formal',scene='indoor',name='포멀 · 실내')}

def main():
 start=time.perf_counter();OUT.mkdir(exist_ok=True)
 for sub in ['photos','queries']:(OUT/sub).mkdir(exist_ok=True)
 outfits=read(BASE/'outfits.json');garments=read(BASE/'garments.json');regions=np.load(BASE/'embeddings.npy',allow_pickle=False)
 assert len(outfits)==136 and regions.shape==(280,512)
 allq=read(ROOT/'output/musinsa_full_200/query_manifest.json');queries=[]
 for slot,n in [('top',6),('bottom',8),('outer',6)]:
  candidates=[q for q in allq if q['group']==slot]
  assert len(candidates)>=n
  for q in candidates[:n]:queries.append(dict(query_id=q['id'],product_id=q['catalog']['product_id'],slot=slot,name=q['catalog']['name'],category=q['catalog']['category_l3'],color=q['catalog']['color'],image_path=q['path'],input_sha256=sha(q['path'])))
 model=ROOT/'output/team_reuse_500/models/fashion_clip';prov=read(model/'provenance.json');previous=read(ROOT/'output/outfit_missing200/protocol.json')
 assert prov==previous['provenance']['fashion_clip']
 assert sha(REPO/'src/search/encoder.py')==previous['source_hashes']['src/search/encoder.py']
 cfg=yaml.safe_load((REPO/'configs/search.yaml').read_text(encoding='utf-8'))['clip'];assert cfg==previous['encoder']
 protocol=dict(version=1,source_hashes={str(BASE/f):sha(BASE/f) for f in ['outfits.json','garments.json','embeddings.npy']},
  input_images={r['outfit_id']:sha(r['original_path']) for r in outfits},queries=queries,model=prov,
  styles=STYLES,scenes=SCENES,tagging=dict(style_margin=.01,scene_margin=.015,similarity_probability=False,unknown_policy='Below top1-top2 margin threshold => unknown; no preference bonus.'),
  retrieval=dict(same_slot=True,absolute_min_cosine=.65,maximum_gap_from_best=.06,max_candidates=20,dedupe='best matching region per outfit'),
  preferences=dict(profiles=PROFILES,style_bonus_max=.03,scene_bonus_max=.015,confidence='clip(tag_margin / 0.04, 0, 1); heuristic, not calibrated probability',rule='Only same fixed product candidate pool is reranked; unknown tags give zero bonus.'),
  limitations=['Automatic visual tags, no independent ground truth.','Product relevance threshold is a development heuristic, not a validated quality guarantee.','Ranking changes demonstrate mechanism, not user satisfaction or recommendation accuracy.','No gender or audience tagging in this experiment.'],offline=True)
 if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
 else:write(OUT/'protocol.json',protocol)
 for r in outfits:shutil.copyfile(r['original_path'],OUT/'photos'/(r['outfit_id']+'.jpg'))
 for q in queries:shutil.copyfile(q['image_path'],OUT/'queries'/(q['query_id']+'.jpg'))
 bank=[dict(axis=axis,key=k,text=t) for axis,mapping in [('style',STYLES),('scene',SCENES)] for k,texts in mapping.items() for t in texts]
 cached=(OUT/'features.npz').exists();loadtime=encode_seconds=0
 if cached:
  data=np.load(OUT/'features.npz',allow_pickle=False);photo_vectors=data['photos'];query_vectors=data['queries'];text_vectors=data['texts']
 else:
  for f in prov['files']:assert sha(model/f['name'])==f['sha256']
  torch.set_num_threads(6);torch.manual_seed(42);device='cuda' if torch.cuda.is_available() else 'cpu'
  t=time.perf_counter();enc=build_encoder(str(model),device=device,pad_square=cfg['pad_to_square']);sync();loadtime=time.perf_counter()-t
  t=time.perf_counter()
  def encode_paths(paths):
   chunks=[]
   for i in range(0,len(paths),16):
    images=[]
    for p in paths[i:i+16]:
     with Image.open(p) as im:images.append(im.convert('RGB'))
    chunks.append(enc.encode_images(images,batch_size=16))
   return np.concatenate(chunks)
  photo_vectors=encode_paths([r['original_path'] for r in outfits]);query_vectors=encode_paths([q['image_path'] for q in queries]);text_vectors=enc.encode_texts([r['text'] for r in bank],batch_size=32);sync();encode_seconds=time.perf_counter()-t
  np.savez_compressed(OUT/'features.npz',photos=photo_vectors,queries=query_vectors,texts=text_vectors)
 for matrix,count in [(photo_vectors,136),(query_vectors,20),(text_vectors,len(bank)),(regions,280)]:
  assert matrix.shape==(count,512) and np.isfinite(matrix).all() and np.allclose(np.linalg.norm(matrix,axis=1),1,atol=.001)
 prototypes={}
 for axis,mapping in [('style',STYLES),('scene',SCENES)]:
  prototypes[axis]={}
  for k in mapping:
   v=text_vectors[[i for i,b in enumerate(bank) if b['axis']==axis and b['key']==k]].mean(axis=0);prototypes[axis][k]=v/np.linalg.norm(v)
 tags=[]
 for r,v in zip(outfits,photo_vectors):
  axes={}
  for axis,threshold in [('style',.01),('scene',.015)]:
   scores={k:float(v@p) for k,p in prototypes[axis].items()};ranked=sorted(scores,key=scores.get,reverse=True);gap=scores[ranked[0]]-scores[ranked[1]];label=ranked[0] if gap>=threshold else 'unknown'
   axes[axis]=dict(label=label,top_candidate=ranked[0],second_candidate=ranked[1],margin=gap,reliability=float(np.clip(gap/.04,0,1)) if label!='unknown' else 0,scores=scores)
  scene=axes['scene']['label'];tags.append(dict(outfit_id=r['outfit_id'],dataset_id=r['dataset_id'],photo_path='photos/'+r['outfit_id']+'.jpg',
   style=axes['style'],scene=axes['scene'],location='outdoor' if scene=='outdoor' else 'indoor' if scene in ['studio','indoor'] else 'unknown',shooting_type='studio' if scene=='studio' else 'unknown',tag_source='FashionCLIP full-photo/text similarity',status='automatic_unverified'))
 write(OUT/'tags.json',tags);tagmap={r['outfit_id']:r for r in tags};write(OUT/'queries.json',queries)
 t=time.perf_counter();rankings=[]
 for q,v in zip(queries,query_vectors):
  best={}
  for g in garments:
   if g['slot']!=q['slot']:continue
   score=float(v@regions[g['embedding_row']]);oid=g['outfit_id']
   if oid not in best or score>best[oid]['product_similarity']:best[oid]=dict(outfit_id=oid,dataset_id=g['dataset_id'],matched_garment_id=g['garment_id'],matched_box=g['box'],product_similarity=score)
  ordered=sorted(best.values(),key=lambda x:(-x['product_similarity'],x['outfit_id']))
  threshold=max(.65,ordered[0]['product_similarity']-.06) if ordered else .65
  candidates=[r for r in ordered if r['product_similarity']>=threshold][:20];profile_results={}
  for name,pref in PROFILES.items():
   items=[]
   for r in candidates:
    tag=tagmap[r['outfit_id']]
    style_bonus=.03*tag['style']['reliability'] if pref['style'] and tag['style']['label']==pref['style'] else 0
    scene_bonus=.015*tag['scene']['reliability'] if pref['scene'] and tag['scene']['label']==pref['scene'] else 0
    items.append(dict(r,style_bonus=style_bonus,scene_bonus=scene_bonus,total_score=r['product_similarity']+style_bonus+scene_bonus,style=tag['style']['label'],scene=tag['scene']['label']))
   items.sort(key=lambda x:(-x['total_score'],-x['product_similarity'],x['outfit_id']))
   profile_results[name]=items
  rankings.append(dict(query=q,threshold=threshold,candidate_count=len(candidates),profiles=profile_results,
   candidate_scenes=dict(collections.Counter(tagmap[r['outfit_id']]['scene']['label'] for r in candidates)),candidate_styles=dict(collections.Counter(tagmap[r['outfit_id']]['style']['label'] for r in candidates))))
 ranktime=time.perf_counter()-t;write(OUT/'rankings.json',rankings)
 metrics={}
 for name,pref in PROFILES.items():
  active=[r for r in rankings if r['candidate_count']];eligible=[r for r in active if r['candidate_count']>=2]
  changed1=changed5=changedset=0;drops=[];matches=[];base_matches=[]
  for r in eligible:
   b=r['profiles']['baseline'];p=r['profiles'][name]
   changed1+=b[0]['outfit_id']!=p[0]['outfit_id'];changed5+=[x['outfit_id'] for x in b[:5]]!=[x['outfit_id'] for x in p[:5]]
   changedset+=set(x['outfit_id'] for x in b[:5])!=set(x['outfit_id'] for x in p[:5]);drops.append(b[0]['product_similarity']-p[0]['product_similarity'])
   for items,target in [(b,base_matches),(p,matches)]:
    target.append(sum((not pref['style'] or x['style']==pref['style']) and (not pref['scene'] or x['scene']==pref['scene']) for x in items[:5])/len(items[:5]))
  metrics[name]=dict(queries_with_candidates=len(active),queries_with_at_least_two=len(eligible),top1_changed=changed1,top5_order_changed=changed5,top5_members_changed=changedset,mean_top1_product_cosine_drop=float(np.mean(drops)) if drops else 0,
   mean_top5_full_preference_match_before=float(np.mean(base_matches)) if base_matches else 0,mean_top5_full_preference_match_after=float(np.mean(matches)) if matches else 0,
   queries_with_no_matching_scene=sum(pref['scene'] is not None and not r['candidate_scenes'].get(pref['scene'],0) for r in active))
 pair={}
 for a,b in [('casual_studio','casual_outdoor'),('street_studio','street_outdoor')]:
  pair[a+'__'+b]=dict(top1_different=sum(bool(r['profiles'][a]) and r['profiles'][a][0]['outfit_id']!=r['profiles'][b][0]['outfit_id'] for r in rankings),top5_order_different=sum([x['outfit_id'] for x in r['profiles'][a][:5]]!=[x['outfit_id'] for x in r['profiles'][b][:5]] for r in rankings))
 summary=dict(outfits=136,tag_counts={axis:dict(collections.Counter(r[axis]['label'] for r in tags)) for axis in ['style','scene']},query_count=20,query_slots=dict(collections.Counter(q['slot'] for q in queries)),candidate_count_distribution=dict(collections.Counter(r['candidate_count'] for r in rankings)),profiles=metrics,scene_only_profile_comparison=pair,limitations=protocol['limitations'])
 write(OUT/'summary.json',summary)
 validations=[]
 for r in rankings:
  ids={x['outfit_id'] for x in r['profiles']['baseline']}
  for name,items in r['profiles'].items():
   assert {x['outfit_id'] for x in items}==ids
   for x in items:
    assert x['product_similarity']>=r['threshold']-1e-7
    assert 0<=x['style_bonus']+x['scene_bonus']<=.0450001
    if tagmap[x['outfit_id']]['style']['label']=='unknown':assert x['style_bonus']==0
    if tagmap[x['outfit_id']]['scene']['label']=='unknown':assert x['scene_bonus']==0
 write(OUT/'validation.json',dict(outfits=136,unique_outfit_ids=len(tagmap),query_count=20,fixed_candidate_pools_verified=True,same_slot_gate_verified=True,similarity_threshold_preserved=True,unknown_tags_no_bonus=True,preference_bonus_bounded=True,source_files_unchanged=all(sha(Path(p))==v for p,v in protocol['source_hashes'].items())))
 render(tags,rankings,summary)
 runtime=dict(cached=cached,model_load_seconds=loadtime,image_and_text_encoding_seconds=encode_seconds,ranking_seconds=ranktime,wall_seconds=time.perf_counter()-start,full_photos_encoded=0 if cached else 136,products_encoded=0 if cached else 20,garment_vectors_reused=280)
 write(OUT/f'run_{time.time_ns()}.json',runtime);write(OUT/'latest_run.json',runtime)
 print(json.dumps(dict(tag_counts=summary['tag_counts'],candidates=summary['candidate_count_distribution'],profiles=metrics,scene_pairs=pair,runtime=runtime),ensure_ascii=False),flush=True)

def render(tags,rankings,summary):
 payload=json.dumps(dict(tags=tags,rankings=rankings,summary=summary,profiles=PROFILES,names=NAMES),ensure_ascii=False).replace('</','<\\/')
 shell='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>136장 자동 태그·취향 추천 실험</title><style>body{font:14px/1.6 system-ui;background:#eef2f6;color:#192d40;margin:24px}h1{font-size:25px}.controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap}select,button{padding:9px}#queryphoto{height:170px;width:140px;object-fit:contain;background:white}.rows{display:grid;grid-template-columns:1fr 1fr;gap:20px}.cards{display:flex;gap:10px;overflow:auto}article{background:white;padding:10px;min-width:190px;max-width:220px}.photo{position:relative;width:170px}img{width:100%;display:block}article p{font-size:12px;margin:6px 0}svg{position:absolute;inset:0;width:100%;height:100%}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}table{border-collapse:collapse;background:white}td,th{padding:7px 13px;border:1px solid #c9d4df}.notice{padding:12px;background:#fff2d0}summary{cursor:pointer}#taggallery article{max-width:none}</style><h1>136장 자동 태그 · 같은 상품에서 취향별 순서 비교</h1><p class="notice">상품 매칭 후보를 고정한 뒤 취향 점수만 더합니다. 태그·상품 유사도 기준은 자동 추정이며 정답 판정이 아닙니다. 사진 속 인물의 성별은 이번 실험에서 추정하지 않았습니다.</p><div id="counts"></div><div class="controls"><label>상품 <select id="query"></select></label><label>취향 <select id="profile"></select></label><button id="example">순서가 달라지는 사례</button><img id="queryphoto"><span id="qinfo"></span></div><p id="status"></p><div class="rows"><div><h2>상품 매칭만</h2><div class="cards" id="base"></div></div><div><h2 id="personalTitle">취향 반영</h2><div class="cards" id="personal"></div></div></div><details><summary>136장 자동 태그 전체 보기</summary><p>점수 차이가 작으면 불명. 촬영 장소와 옷 스타일은 별도 태그입니다.</p><div class="gallery" id="taggallery"></div></details><script>const DATA=__DATA__;
const tags=new Map(DATA.tags.map(t=>[t.outfit_id,t]));const names=DATA.names;const qsel=document.querySelector('#query'),psel=document.querySelector('#profile');
DATA.rankings.forEach((r,i)=>qsel.add(new Option(r.query.query_id+' · '+r.query.category+' · '+r.query.color,i)));
Object.entries(DATA.profiles).filter(([k])=>k!=='baseline').forEach(([k,v])=>psel.add(new Option(v.name,k)));
document.querySelector('#counts').textContent='촬영 분위기: '+Object.entries(DATA.summary.tag_counts.scene).map(([k,v])=>names[k]+' '+v+'장').join(' / ')+' | 스타일: '+Object.entries(DATA.summary.tag_counts.style).map(([k,v])=>names[k]+' '+v+'장').join(' / ');
function card(x,i){const t=tags.get(x.outfit_id);return `<article><b>${i+1}. ${t.dataset_id}</b><div class="photo"><img src="${t.photo_path}" loading="lazy"></div><p>${names[t.style.label]} · ${names[t.scene.label]}</p><p>상품 유사도 ${x.product_similarity.toFixed(3)}<br>취향 가산 ${(x.style_bonus+x.scene_bonus).toFixed(3)}</p></article>`}
function update(){const r=DATA.rankings[+qsel.value],p=psel.value;document.querySelector('#queryphoto').src='queries/'+r.query.query_id+'.jpg';document.querySelector('#qinfo').textContent=r.query.name;document.querySelector('#status').textContent='고정 후보 '+r.candidate_count+'장 · 상품 유사도 기준 '+r.threshold.toFixed(3)+' · 이 후보 안에서만 순서 조정';document.querySelector('#base').innerHTML=r.profiles.baseline.slice(0,5).map(card).join('')||'조건을 통과한 코디 없음';document.querySelector('#personal').innerHTML=r.profiles[p].slice(0,5).map(card).join('')||'조건을 통과한 코디 없음';document.querySelector('#personalTitle').textContent=DATA.profiles[p].name;}
document.querySelector('#example').onclick=()=>{let found=false;for(let i=0;i<DATA.rankings.length;i++){const r=DATA.rankings[i],a=r.profiles.baseline,b=r.profiles[psel.value];if(a.length&&b.length&&a[0].outfit_id!==b[0].outfit_id){qsel.value=i;found=true;break}}update();if(!found)document.querySelector('#status').textContent+=' · 이 취향에서 1위가 바뀌는 사례 없음'};
qsel.onchange=update;psel.onchange=update;
document.querySelector('#taggallery').innerHTML=DATA.tags.map(t=>`<article><b>${t.dataset_id}</b><div class="photo"><img loading="lazy" src="${t.photo_path}"></div><p>${names[t.style.label]} · ${names[t.scene.label]}</p><p>스타일 차이 ${t.style.margin.toFixed(3)} / 장소 차이 ${t.scene.margin.toFixed(3)}</p></article>`).join('');update();</script></html>'''
 (OUT/'comparison.html').write_text(shell.replace('__DATA__',payload),encoding='utf-8')

if __name__=='__main__':main()
