"""1800-second joint QAT continuation of the completed Delta LCCC SFT."""
import argparse,json,os,sys,time,signal,hashlib,datetime,copy
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['TORCHINDUCTOR_COMPILE_THREADS']='2'
HERE=Path(__file__).resolve().parent;RUN=HERE.parent
os.environ['TORCHINDUCTOR_CACHE_DIR']=str(RUN/'inductor-cache');os.environ['TRITON_CACHE_DIR']=str(RUN/'triton-cache')
sys.path.insert(0,str(HERE))
import torch,numpy as np
from training.train import load_checkpoint,atomic_save,rng_state,restore_rng
from training.model import ModelConfig
from training.data import TokenCorpus
from qat_model import QATModel
from sft_data import ChatCorpus
from sft_graph import Engine
PARENT_SHA='7ca29f788efaf7cbdd6300e8c326d887617373b6dabc4483a6fa034afdad90aa'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def json_atomic(p,obj):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=float,default=1800);parser.add_argument('--max-updates',type=int,default=0);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    root=RUN.parents[1];parent_path=root/'training-runs/delta-lccc-sft-2epoch-20261029/sft/last.pt'
    assert sha(parent_path)==PARENT_SHA
    torch.set_num_threads(8);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(12*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.manual_seed(11032932);np.random.seed(11032932)
    parent=load_checkpoint(parent_path);cfg=ModelConfig.from_dict(parent['model_config']);model=QATModel(cfg).cuda()
    missing=model.load_state_dict(parent['model'],strict=False)
    assert not missing.unexpected_keys and all(k.startswith('delta.grids.') for k in missing.missing_keys)
    assert sum(p.numel() for p in model.parameters())==9289608 and all(p.requires_grad for p in model.parameters())
    data_path=root/'research/delta-lccc-sft-20261029/data/manifest.json'
    corpus=ChatCorpus(data_path,'train',256,132);val=TokenCorpus(data_path,'val',256,132)
    named=list(model.named_parameters());optimizer=torch.optim.AdamW([{'params':[p for n,p in named if p.ndim>=2 and not n.endswith('.alpha')],'weight_decay':.1},{'params':[p for n,p in named if p.ndim<2 or n.endswith('.alpha')],'weight_decay':0.}],lr=1e-5,betas=(.9,.95),eps=1e-8)
    optimizer.load_state_dict(parent['optimizer'])
    for group in optimizer.param_groups:group['lr']=1e-5
    assert all(int(v['step'])==123974 for v in optimizer.state.values())
    out=RUN/'train';out.mkdir(exist_ok=True)
    step=123974;cursor=0;updates=0;targets=0;elapsed=0.
    contract={'version':'new-sft-joint-delta-qat-v1','parent_sha256':PARENT_SHA,'seconds':args.seconds,'optimizer':'inherited SFT AdamW, endpoint lr held at 1e-5','all_parameters_trainable':True,'fake_quant_scope':'Delta weights/activations/recursive state only; backbone/head floating then deployment PTQ','arithmetic':'FP32 TF32 off','effective_batch':256,'data_fingerprint':corpus.manifest['fingerprint'],'source_manifest_sha256':sha(RUN/'source-manifest.json'),'allocator_limit_gib':12,'data_seed':110307,'new_calibration':True}
    if args.resume:
        saved=load_checkpoint(out/'full-last.pt');assert saved['contract']==contract
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
        step,cursor,updates,targets,elapsed=[saved[k] for k in ('step','cursor','updates','targets','elapsed')]
    else:
        assert not (out/'full-last.pt').exists()
        # Stratified train-only full documents, no val/test calibration.
        raw=json.loads(data_path.read_text());rng=np.random.default_rng(11032932);documents=[]
        model.delta.mode='observe'
        with torch.no_grad():
            for source in ('original-sft','core133','extended240'):
                candidates=[s for s in raw['shards'] if s.get('source')==source and s['splits']['train']['documents']]
                for j in range(16):
                    shard=candidates[j%len(candidates)]['splits']['train'];files=shard['files']
                    index=np.memmap(next(n for n in files if n.endswith('.idx')),dtype='<u8',mode='r');tokens=np.memmap(next(n for n in files if n.endswith('.bin')),dtype='<u2',mode='r')
                    doc=int(rng.integers(shard['documents']));ids=np.asarray(tokens[int(index[doc]):int(index[doc+1])-1],dtype=np.int64)
                    x=torch.from_numpy(ids.copy()).unsqueeze(0).cuda();v=torch.ones_like(x,dtype=torch.bool)
                    model.features(x,v)
                    documents.append({'source':source,'split':'train','index_file':next(n for n in files if n.endswith('.idx')),'doc':doc,'ids':ids.tolist()})
                    if len(documents)%8==0:print(json.dumps({'event':'calibrating','documents':len(documents)}),flush=True)
        model.delta.freeze_grids();model.delta.mode='qat'
        ranges={n:{'scale':g.scale.cpu().tolist(),'zp':g.zp.cpu().tolist(),'minimum':g.minimum.cpu().tolist(),'maximum':g.maximum.cpu().tolist()} for n,g in model.delta.grids.items()}
        assert all(torch.isfinite(v).all() for v in model.state_dict().values() if v.dtype.is_floating_point)
        json_atomic(RUN/'calibration.json',{'documents':documents,'ranges':ranges,'split':'train','val_used':False})
        print(json.dumps({'event':'calibration_complete','state_in':ranges['state_in'],'state_out':ranges['state_out']}),flush=True)
    model.delta.use_compiled_step();model.train();warm_rng=rng_state();warmup=time.perf_counter();engine=Engine(model);optimizer.zero_grad(set_to_none=False);restore_rng(warm_rng)
    print(json.dumps({'event':'graph_ready','seconds':time.perf_counter()-warmup}),flush=True)
    order=corpus.epoch_order(110307+cursor//15868671);assert len(order)==15868671
    stopped={'signal':None};signal.signal(signal.SIGTERM,lambda s,f:stopped.update(signal=s));signal.signal(signal.SIGINT,lambda s,f:stopped.update(signal=s))
    log=(out/'metrics.jsonl').open('a',buffering=1);started=None
    def duration():return elapsed+(time.monotonic()-started if started is not None else 0.)
    def emit(row):
        row.update(step=step,updates=updates,targets=targets,cursor=cursor,elapsed_seconds=duration());log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
        json_atomic(out/'status.json',{'status':'running','updated_utc':now(),'contract':contract,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,**row})
    def save():
        torch.cuda.synchronize();state=model.state_dict();rng=rng_state()
        atomic_save({'format':'delta-sft-joint-qat-v1','model':state,'model_config':cfg.to_dict(),'delta_config':{'heads':4,'dimension':32,'normalize':'bounded'},'optimizer':optimizer.state_dict(),'rng':rng,'step':step,'cursor':cursor,'updates':updates,'targets':targets,'elapsed':duration(),'contract':contract,'tokenizer_sha256':parent['tokenizer_sha256'],'chat_template':'minimind3_nothink_v1'},out/'full-last.pt')
        backbone={k:v for k,v in state.items() if not k.startswith('delta.')};delta={k[6:]:v for k,v in state.items() if k.startswith('delta.')}
        atomic_save({'model':backbone,'model_config':cfg.to_dict(),'tokenizer_sha256':parent['tokenizer_sha256'],'chat_template':'minimind3_nothink_v1','step':step,'contract':contract},out/'backbone-last.pt')
        atomic_save({'format':'delta-only-fp-qat-v1','delta':delta,'model_config':cfg.to_dict(),'delta_config':{'heads':4,'dimension':32,'normalize':'bounded'},'parent_checkpoint':str(out/'backbone-last.pt'),'parent_sha256':sha(out/'backbone-last.pt'),'tokenizer_sha256':parent['tokenizer_sha256'],'chat_template':'minimind3_nothink_v1','step':step,'contract':contract},out/'qat-last.pt')
    @torch.no_grad()
    def evaluate():
        sums={'qat':0.,'float':0.};count=0
        for i in range(2):
            x,v,y=[t.cuda() for t in val.batch(8,110329000+i)];mask=y!=-100;count+=int(mask.sum())
            for mode in sums:
                model.delta.mode='qat' if mode=='qat' else 'fp'
                logits=model(x,v,132,supervised_mask=mask);sums[mode]+=float(torch.nn.functional.cross_entropy(logits,y[mask],reduction='sum'))
        model.delta.mode='qat'
        return {'event':'validation','loss':{k:v/count for k,v in sums.items()},'validation_targets':count}
    emit(evaluate());started=time.monotonic();last_save=last_eval=started;last_targets=targets;last_log=started
    try:
        while duration()<args.seconds and not stopped['signal']:
            offset=cursor%15868671
            if offset==0:order=corpus.epoch_order(110307+cursor//15868671)
            stop=min(15868671,offset+256);packed,n=corpus.prepare_macro(order[offset:stop]);optimizer.zero_grad(set_to_none=False)
            loss=engine.backward(packed,n);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            step+=1;updates+=1;cursor+=stop-offset;targets+=n
            if updates<=3 or updates%20==0:
                current=time.monotonic();emit({'event':'train','loss':loss/n,'gradient_norm':float(norm),'targets_per_second':(targets-last_targets)/max(current-last_log,.001)})
                last_targets=targets;last_log=current
            if time.monotonic()-last_eval>=300:emit(evaluate());last_eval=time.monotonic()
            if time.monotonic()-last_save>=120:save();last_save=time.monotonic()
            if args.max_updates and updates>=args.max_updates:break
        training_seconds=duration();save();emit(evaluate())
        assert sha(parent_path)==PARENT_SHA and all(int(v['step'])==step for v in optimizer.state.values())
        result={'status':'complete' if not stopped['signal'] else 'stopped','finished_utc':now(),'step':step,'updates':updates,'new_supervised_targets':targets,'cursor':cursor,'training_seconds':training_seconds,'parent_preserved':True,'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,'artifact_sha256':{n:sha(out/n) for n in ('full-last.pt','backbone-last.pt','qat-last.pt')},'contract':contract}
        json_atomic(out/'completion.json',result);json_atomic(out/'status.json',result);print(json.dumps(result),flush=True)
    finally:log.close()
if __name__=='__main__':main()
