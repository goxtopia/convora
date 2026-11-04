"""Same-input full validation CE and the previous 48 CPU FP32 diagnostics."""
import os,sys,json,hashlib,time,gc,math,datetime
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='0'
run=Path(sys.argv[1]);root=run.parents[1];sys.path.insert(0,str(run/'source'));sys.path.insert(0,str(run))
import torch
from training.train import load_checkpoint
from training.model import ModelConfig,TCNLanguageModel
from training.data import TokenCorpus
from training.streaming import CachedDecoder
from sft_model import SFTModel
from diagnostic_cases import cases,prefix,correct,repetition
from tokenizers import Tokenizer
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for data in iter(lambda:f.read(4*1024**2),b''):h.update(data)
    return h.hexdigest()
old=root/'training-runs/sawtooth8-lccc-sft';baseline=old/'mixed-sft/last.pt';new=run/'sft/last.pt'
assert sha(baseline)=='3720565dc4f08f9c5a2da11a729b212adbe50ba83d451c1b2c4d5de4bede68f6'
datasets={'original':root/'training-data/minimind3/sft/manifest.json','lccc-core':root/'training-data/lccc-light-tokens-20261027/core133/manifest.json','lccc-extended':root/'training-data/lccc-light-tokens-20261027/extended240/manifest.json','web':root/'training-data/sawtooth8-web-sft-1to1/manifest.json'}
historical={'original':(2246242,3.02775016578384),'lccc-core':(1493389,3.707585326439666),'lccc-extended':(86999,3.8684409323760187),'web':(73542,4.205754609185908)}
torch.set_num_threads(4);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(12*1024**3/torch.cuda.get_device_properties(0).total_memory)
torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=True
result={'status':'running','scope':'End-to-end two-epoch LCCC SFT after each own pretrain; same data/seed/effective batch/schedule. New Delta has better pretrain and longer train prefix, not an isolated SFT-only layer effect. Reused diagnostics are not blind tests.','checkpoints':{'old':sha(baseline),'delta':sha(new)},'validation':{},'generation':{}}
with torch.inference_mode():
    for name,path in (('old',baseline),('delta',new)):
        cp=load_checkpoint(path);cfg=ModelConfig.from_dict(cp['model_config']);model=(TCNLanguageModel(cfg) if name=='old' else SFTModel(cfg)).cuda().eval();model.load_state_dict(cp['model']);del cp
        if name=='delta':model.delta.backend='fused'
        scores={}
        for label,manifest in datasets.items():
            corpus=TokenCorpus(manifest,'val',256,cfg.context);order=corpus.epoch_order(110305);total=0.;count=0
            for start in range(0,len(order),128):
                x,v,y=[t.cuda() for t in corpus.batch_windows(order[start:start+128])];mask=y!=-100
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=model(x,v,cfg.context,supervised_mask=mask)
                    loss=torch.nn.functional.cross_entropy(logits.float(),y[mask],reduction='sum')
                total+=float(loss);count+=int(mask.sum())
            assert count==historical[label][0]
            scores[label]={'loss':total/count,'ppl':math.exp(total/count),'targets':count,'windows':len(order),'precision':'bf16','input_context':132}
            if name=='old':assert abs(total/count-historical[label][1])<.003,(label,total/count)
            print(json.dumps({'model':name,'dataset':label,**scores[label]}),flush=True)
        result['validation'][name]=scores
        (run/'comparison.partial.json').write_text(json.dumps(result,indent=2))
        del model,logits,loss,x,v,y,mask;gc.collect();torch.cuda.empty_cache()
tokenizer_path=root/'training-data/minimind3/tokenizer/tokenizer.json';assert sha(tokenizer_path)=='71f32c68cf63a15355a8fc171b7594b3d41870fe0ddb54fc6aefa55f73a4a668'
tokenizer=Tokenizer.from_file(str(tokenizer_path));tokenizer.no_padding();tokenizer.no_truncation();torch.set_num_threads(4)
class DeltaCache:
    def __init__(self,model):
        assert isinstance(model.final_norm,torch.nn.Identity)
        self.model=model;self.head=model.head;model.head=torch.nn.Identity();self.tcn=CachedDecoder(model)
        self.states=[torch.zeros(1,32,1,32) for _ in range(4)]
    def step(self,token):
        features=self.tcn.step(token).reshape(1,256,1,1)
        out,*self.states=self.model.delta.layer.reference(features,self.states)
        return self.head(out.reshape(1,256,1)).squeeze(2)
    def full(self,history):
        self.model.head=self.head
        try:return self.model(torch.tensor([history]),output_start=len(history)-1)[:,-1]
        finally:self.model.head=torch.nn.Identity()
    def close(self):self.model.head=self.head
questions=cases();(run/'cases.json').write_text(json.dumps(questions,ensure_ascii=False,indent=2))
with torch.inference_mode():
    for name,path in (('old',baseline),('delta',new)):
        cp=load_checkpoint(path);cfg=ModelConfig.from_dict(cp['model_config']);model=(TCNLanguageModel(cfg) if name=='old' else SFTModel(cfg)).eval();model.load_state_dict(cp['model']);del cp
        if name=='delta':model.delta.backend='reference'
        predictions=[];checks=[];begin=time.perf_counter()
        for row in questions:
            cache=CachedDecoder(model) if name=='old' else DeltaCache(model)
            history=prefix(tokenizer,row['messages']);generated=[]
            for token in history:logits=cache.step(token)
            history=list(history)
            for step in range(80):
                if row['id'] in ('greeting','blue','name_recall') and step in (0,7):
                    reference=model(torch.tensor([history[-133:]]))[:,-1] if name=='old' else cache.full(history)
                    difference=float((reference-logits).abs().max());same=int(reference.argmax())==int(logits.argmax())
                    assert difference<.0003 and same,(name,row['id'],difference)
                    checks.append({'id':row['id'],'step':step,'max_difference':difference,'same_argmax':same})
                token=int(logits.argmax());generated.append(token);history.append(token)
                if token==2:break
                logits=cache.step(token)
            if name=='delta':cache.close()
            text=tokenizer.decode(generated,skip_special_tokens=True)
            predictions.append(dict(row,text=text,generated_ids=generated,eos=generated[-1]==2,new_tokens=len(generated),strict_correct=correct(row,text),**repetition(generated)))
        single=[r for r in predictions if r['category']=='strict'];multi=[r for r in predictions if r['category']=='multi_strict'];casual=[r for r in predictions if r['category']=='casual']
        result['generation'][name]={'strict_single':{'correct':sum(r['strict_correct'] for r in single),'total':len(single)},'strict_multi':{'correct':sum(r['strict_correct'] for r in multi),'total':len(multi)},'eos_answers':sum(r['eos'] for r in predictions),'answers':len(predictions),'mean_new_tokens':sum(r['new_tokens'] for r in predictions)/len(predictions),'casual_mean_duplicate_4gram_fraction':sum(r['duplicate_4gram_fraction'] for r in casual)/len(casual),'cached_full_checks':checks,'seconds':time.perf_counter()-begin,'precision':'cpu_fp32'}
        (run/f'{name}-predictions.json').write_text(json.dumps(predictions,ensure_ascii=False,indent=2))
        print(json.dumps({'generation_model':name,**result['generation'][name]}),flush=True)
        del model;gc.collect()
result['validation_delta_minus_old']={k:result['validation']['delta'][k]['loss']-result['validation']['old'][k]['loss'] for k in datasets}
result.update(status='complete',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(run/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
