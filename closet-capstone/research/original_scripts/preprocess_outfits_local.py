"""Resumable local outfit preprocessing pilot. No network, training, or API calls."""
from pathlib import Path
import os,sys,time,json,hashlib,random,copy,argparse,collections,ctypes,csv,html,shutil
BOOT=time.perf_counter()
ROOT=Path(__file__).resolve().parent
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['OPENBLAS_NUM_THREADS']='1'
sys.dont_write_bytecode=True
sys.stdout.reconfigure(encoding='utf-8')
LEGACY=Path('C:/Users/전민재/Documents/Codex/2026-10-05/stage-1-candidate-generation-two-tower-2')
for p in [LEGACY/'work/packages',LEGACY/'work/integration_packages',Path('C:/Users/전민재/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages')]:sys.path.append(str(p))
import numpy as np
from PIL import Image
import torch,yaml

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):
 temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(p)
def sync():
 if torch.cuda.is_available():torch.cuda.synchronize()
def ram_peak():
 class Counters(ctypes.Structure):
  _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
 c=Counters();c.cb=ctypes.sizeof(c);kernel=ctypes.windll.kernel32;kernel.GetCurrentProcess.restype=ctypes.c_void_p
 ok=ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(kernel.GetCurrentProcess()),ctypes.byref(c),c.cb)
 return round(c.PeakWorkingSetSize/1024**2,2) if ok else None

def gallery(out,records,summary):
 colors={'top':'#087f8c','bottom':'#a34a00','outer':'#6c42a1','dress':'#c13b6d','shoes':'#176b39','bag':'#5f6a32','hat':'#884455','accessory':'#555555','legwear':'#506b91'}
 sections=[]
 for r in records:
  w,h=r.get('image_size',[1,1]);boxes=[];details=[]
  for g in r.get('garments',[]):
   x1,y1,x2,y2=g['box'];c=colors.get(g['slot'],'#555555')
   boxes.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="{c}" stroke-width="2"/><text x="{x1+2}" y="{max(12,y1-3)}" fill="{c}" font-size="12" stroke="white" stroke-width="2" paint-order="stroke">{html.escape(g["slot"]+" / "+g["kind"])}</text>')
   details.append(f'<li>{html.escape(g["slot"])} · {html.escape(g["kind"])} · {g["detection_score"]:.3f} · 특징값 {g["embedding_row"]}</li>')
  sections.append(f'<section><h2>{html.escape(r["dataset_id"])} · {html.escape(r["status"])}</h2><div class="photo" style="aspect-ratio:{w}/{h}"><img src="{r["display_path"]}" alt="원본 코디"><svg viewBox="0 0 {w} {h}">'+''.join(boxes)+f'</svg></div><p>원본 {w}×{h} · 검출 {len(r.get("garments",[]))}개</p><ul>'+''.join(details)+'</ul></section>')
 doc='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>무신사 20장 로컬 전처리</title><style>body{font:14px/1.6 system-ui,"Malgun Gothic",sans-serif;background:#f3f5f7;color:#192d3f;margin:24px}h1{font-size:24px}h2{font-size:17px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:20px}section{background:white;padding:16px}section p{margin:8px 0}ul{padding-left:18px}.photo{width:220px;max-width:100%;position:relative}.photo img{width:100%;height:100%;display:block}.photo svg{position:absolute;inset:0;width:100%;height:100%}label{display:block;margin:16px 0}body.hide-boxes svg{display:none}</style><h1>무신사 20장 · 검출 및 특징값 저장 확인</h1><p>현재 팀 YOLOS + FashionCLIP을 로컬 GPU에서 실행. 검출 기준 0.5, 전체 사진 대체·부위 강제 요청 없음. 원본 파일은 변경하지 않았고 박스는 화면에만 겹쳐 표시합니다.</p><p>처리 '+str(summary['outfits'])+'장 · 검출 '+str(summary['garments'])+'개 · 특징값 '+str(summary['embedding_shape'])+'</p><p>박스는 모델 예측이며 정답 판정이 아닙니다. 검출 누락·옷 가림·잘못된 분류는 화면에서 따로 확인해야 합니다.</p><label><input type="checkbox" id="boxes" checked> 검출 박스 표시</label><div class="grid">'+''.join(sections)+'</div><script>document.getElementById("boxes").addEventListener("change",e=>document.body.classList.toggle("hide-boxes",!e.target.checked));</script></html>'
 (out/'review.html').write_text(doc,encoding='utf-8')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',default='C:/Users/전민재/Downloads/multimodal-multistage-recsys-main (2)/multimodal-multistage-recsys-main');ap.add_argument('--limit',type=int,default=20);ap.add_argument('--out',default='output/outfit_preprocess_20');ap.add_argument('--retry-failed',action='store_true');args=ap.parse_args()
 if not 1<=args.limit<=20:raise ValueError('This pilot is limited to 20 images; full run needs a separate explicit configuration.')
 repo=Path(args.repo);out=(ROOT/args.out).resolve();out.relative_to(ROOT)
 for sub in ['records','vectors','photos']:(out/sub).mkdir(parents=True,exist_ok=True)
 sys.path.insert(0,str(repo))
 from src.search.detector import FashionDetector,select_detections,crop
 from src.search.encoder import build_encoder
 from dataclasses import asdict
 cfg=yaml.safe_load((repo/'configs/style_search.yaml').read_text(encoding='utf-8'))
 search=yaml.safe_load((repo/'configs/search.yaml').read_text(encoding='utf-8'))
 models=ROOT/'output/team_reuse_500/models';det_dir=models/'yolos';enc_dir=models/'fashion_clip'
 det_prov=read(det_dir/'provenance.json');enc_prov=read(enc_dir/'provenance.json')
 for expected,provenance in [(cfg['detector']['model_name'],det_prov),(search['clip']['model_name'],enc_prov)]:
  if provenance.get('repo')!=expected:raise ValueError('Local model does not match configured model: '+expected)
 device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(6);torch.manual_seed(42)
 source_files=['src/search/detector.py','src/search/encoder.py','src/common/faiss_utils.py','configs/style_search.yaml']
 protocol=dict(schema_version=1,seed=20261105,limit=args.limit,source_repo=str(repo),source_hashes={p:sha(repo/p) for p in source_files},detector_id=cfg['detector']['model_name'],encoder_id=search['clip']['model_name'],detector_provenance=det_prov,encoder_provenance=enc_prov,pad_to_square=search['clip']['pad_to_square'],detector_settings=cfg['detector'],selection='Current strict select_detections, all configured kinds; no requested-slot or whole-image fallback',feature='float32, L2-normalized; crop margin from detector settings',device=device,torch=torch.__version__,offline=True)
 protocol_key=hashlib.sha256(json.dumps(protocol,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
 pp=out/'protocol.json'
 if pp.exists() and read(pp)!=protocol:raise ValueError('Protocol changed; use a different output folder')
 if not pp.exists():write(pp,protocol)
 mf=out/'manifest.json'
 if mf.exists():samples=read(mf)
 else:
  g=read(ROOT/'output/musinsa_full_200/gallery_manifest.json');chosen=random.Random(protocol['seed']).sample(sorted(g,key=lambda x:x['row']),args.limit)
  samples=[dict(outfit_id='musinsa_'+r['source_id'],dataset_id=r['id'],source_id=r['source_id'],dataset='disl/musinsa-no-kid',dataset_row=r['row'],publisher_style=r.get('publisher_style'),original_path=str(Path(r['path']).resolve()),input_sha256=sha(r['path'])) for r in chosen]
  write(mf,samples)
 if len(samples)!=args.limit:raise ValueError('Sample size changed')
 pending=[];records=[]
 for r in samples:
  if sha(r['original_path'])!=r['input_sha256']:raise ValueError('Input image changed: '+r['dataset_id'])
  f=out/'records'/(r['outfit_id']+'.json')
  if f.exists():
   saved=read(f)
   if saved['protocol_key']!=protocol_key or saved['input_sha256']!=r['input_sha256']:raise ValueError('Checkpoint mismatch')
   if saved['status']=='error' and args.retry_failed:pending.append(r)
   else:records.append(saved)
  else:pending.append(r)
 model_load=warmup=0.;fresh=[]
 if pending:
  runtime=copy.deepcopy(cfg);runtime['detector']['model_name']=str(det_dir);runtime['detector']['device']=device
  start=time.perf_counter();detector=FashionDetector(runtime,device=device);encoder=build_encoder(str(enc_dir),device=device,pad_square=search['clip']['pad_to_square']);sync();model_load=time.perf_counter()-start
  with Image.open(pending[0]['original_path']) as im:warm=im.convert('RGB')
  start=time.perf_counter();detector.detect_raw(warm);encoder.encode_images([warm]);sync();warmup=time.perf_counter()-start
  if device=='cuda':torch.cuda.reset_peak_memory_stats()
  for r in pending:
   z=dict(r,protocol_key=protocol_key,status='pending',garments=[]);start=time.perf_counter()
   copied=out/'photos'/(r['outfit_id']+'.jpg');shutil.copyfile(r['original_path'],copied);z['display_path']=copied.relative_to(out).as_posix()
   try:
    with Image.open(r['original_path']) as im:image=im.convert('RGB')
    z['image_size']=list(image.size);sync();t=time.perf_counter();raw=detector.detect_raw(image);selected=select_detections(raw,image.size,runtime);selected=sorted(selected,key=lambda d:(d.slot,d.kind,d.box));sync();z['detect_seconds']=time.perf_counter()-t
    t=time.perf_counter();vectors=encoder.encode_images([crop(image,d.box,cfg['detector']['crop_margin']) for d in selected],batch_size=16);sync();z['encode_seconds']=time.perf_counter()-t
    if not np.isfinite(vectors).all() or (len(vectors) and not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=.001)):raise ValueError('Invalid embedding values')
    for i,d in enumerate(selected):
     x1,y1,x2,y2=d.box
     if not(0<=x1<x2<=image.width and 0<=y1<y2<=image.height):raise ValueError('Invalid box')
     z['garments'].append(dict(garment_id=f'{r["outfit_id"]}_{i:02d}_{d.slot}_{d.kind}',outfit_id=r['outfit_id'],kind=d.kind,slot=d.slot,label=d.label,box=list(d.box),detection_score=d.score,detection_source=d.source,detection_status='detected',crop_margin=cfg['detector']['crop_margin'],embedding_local_row=i,embedding_status='saved'))
    vp=out/'vectors'/(r['outfit_id']+'.npy');np.save(vp,vectors);z['vector_path']=vp.relative_to(out).as_posix();z['vector_sha256']=sha(vp);z['raw_detection_count']=len(raw);z['status']='ok' if selected else 'no_detections';z['observed_slots']=sorted({d.slot for d in selected})
   except Exception as e:z.update(status='error',error=f'{type(e).__name__}: {e}',garments=[])
   z['processing_seconds']=time.perf_counter()-start;write(out/'records'/(r['outfit_id']+'.json'),z);records.append(z);fresh.append(z)
   print(r['dataset_id'],z['status'],len(z['garments']),round(z['processing_seconds'],3),'seconds',flush=True)
 ordered={r['outfit_id']:r for r in records};records=[ordered[r['outfit_id']] for r in samples]
 vectors=[];garments=[]
 for r in records:
  if r['status']=='error':continue
  vp=out/r['vector_path']
  if sha(vp)!=r['vector_sha256']:raise ValueError('Saved vector checksum mismatch')
  v=np.load(vp,allow_pickle=False)
  if len(v)!=len(r['garments']):raise ValueError('Row count mismatch')
  for i,g in enumerate(r['garments']):g['embedding_row']=len(garments);garments.append(g);vectors.append(v[i])
 matrix=np.stack(vectors).astype('float32') if vectors else np.empty((0,512),dtype='float32')
 np.save(out/'garment_embeddings.npy',matrix);write(out/'outfits.json',records);write(out/'garments.json',garments)
 with (out/'garments.csv').open('w',encoding='utf-8-sig',newline='') as f:
  fields=['garment_id','outfit_id','kind','slot','label','box','detection_score','detection_source','detection_status','crop_margin','embedding_row','embedding_status'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(garments)
 times=[r['processing_seconds'] for r in records if r['status']!='error'];elapsed=sum(times)
 summary=dict(outfits=len(records),successful=sum(r['status']=='ok' for r in records),no_detections=sum(r['status']=='no_detections' for r in records),errors=sum(r['status']=='error' for r in records),garments=len(garments),slots=dict(collections.Counter(g['slot'] for g in garments)),embedding_shape=list(matrix.shape),device=device,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,inference_seconds_saved=round(elapsed,3),mean_seconds_per_image=round(float(np.mean(times)),4) if times else None,p95_seconds_per_image=round(float(np.percentile(times,95)),4) if times else None,estimated_8626_processing_minutes=round(elapsed/len(times)*8626/60,1) if times else None,estimate_note='Linear extrapolation from 20 images using the same strict YOLOS-only configuration. Excludes setup, review, retries, and additional models; not a measured full run.',input_bytes=sum(Path(r['original_path']).stat().st_size for r in samples),embedding_bytes=(out/'garment_embeddings.npy').stat().st_size,protocol_key=protocol_key)
 summary['recorded_processing_seconds']=summary.pop('inference_seconds_saved')
 write(out/'summary.json',summary)
 run=dict(fresh_images=len(fresh),reused_checkpoints=len(records)-len(fresh),model_load_seconds=round(model_load,3),warmup_seconds=round(warmup,3),fresh_processing_seconds=round(sum(r['processing_seconds'] for r in fresh),3),wall_seconds_including_imports=round(time.perf_counter()-BOOT,3),gpu_peak_allocated_mib=round(torch.cuda.max_memory_allocated()/1024**2,2) if device=='cuda' and fresh else 0,gpu_peak_reserved_mib=round(torch.cuda.max_memory_reserved()/1024**2,2) if device=='cuda' and fresh else 0,process_peak_working_set_mib=ram_peak())
 stamp=str(time.time_ns());write(out/f'run_{stamp}.json',run);write(out/'latest_run.json',run);gallery(out,records,summary)
 print(json.dumps({'summary':summary,'run':run},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
