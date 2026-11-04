"""Freeze the historical epoch trainer with the validated FP Delta execution path."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=ROOT/'training-runs/sawtooth8-projected-1epoch'
SPEED=ROOT/'research/delta-pretrain-1epoch-20261028'
HELPERS=['delta_model.py','delta_layer.py','qat_delta.py','fused_delta.py','micro_graph.py','prefix_data.py']
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,data):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
def replace_once(text,old,new):
    assert text.count(old)==1,(old[:120],text.count(old))
    return text.replace(old,new,1)
def prepare(run,steps=0,eval_batches=20):
    assert not run.exists(),f'Refusing to overwrite {run}'
    report=json.loads((SPEED/'report.json').read_text())
    for name in HELPERS:assert sha(SPEED/name)==report['source_sha256'][name],name
    assert sha(BASE/'pretrain/last.pt')=='2e65402ab5f7633c089dfba559078c2f450e59ece1cadb34c9436f7df6283533'
    run.mkdir();shutil.copytree(BASE/'source',run/'source')
    shutil.copy2(HERE/'launch.py',run/'launcher-frozen.py')
    shutil.copy2(HERE/'audit.py',run/'audit.py')
    atomic(run/'control-manifest.json',{n:sha(run/n) for n in ('launcher-frozen.py','audit.py')})
    for name in HELPERS:shutil.copy2(SPEED/name,run/'source'/name)
    source=run/'source/training/train.py';text=source.read_text()
    original_sha=sha(source)
    text=replace_once(text,'import argparse\n','import argparse\nimport hashlib\n')
    text=replace_once(text,'from training.model import ModelConfig, TCNLanguageModel\n',
        'from training.model import ModelConfig, TCNLanguageModel\nfrom delta_model import DeltaModel\nfrom prefix_data import FullPrefixCorpus\nfrom micro_graph import MicroGraph\n')
    text=replace_once(text,'        torch.cuda.set_device(device)\n',
        '        torch.cuda.set_device(device)\n        torch.cuda.set_per_process_memory_fraction(12*1024**3/torch.cuda.get_device_properties(device).total_memory, device)\n')
    text=replace_once(text,'    model = TCNLanguageModel(config).to(device)\n',
        '''    assert args.precision == "bf16" and args.epochs == 1 and args.teacher is None and args.init_from is None
    assert args.micro_batch == 128 and args.accumulation == 2 and args.block_size == 256
    model = DeltaModel(config).to(device)
    model.delta.backend = 'fused'
    def state_sha(values):
        digest = hashlib.sha256()
        for name, value in sorted(values.items()):
            digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    initial_backbone_sha = state_sha({k:v for k,v in model.state_dict().items() if not k.startswith('delta.')})
    assert initial_backbone_sha == '6f92802fc867f2da027833a7be355dd20eb04d208a7fea12a7f942a8e7801c37'
    assert state_sha(model.state_dict()) == '29390dcd2747fcd10c443cb25b3c7c5420897d967c70feb0287f048fa8776373'
    assert sum(p.numel() for p in model.parameters()) == 9289608
''')
    text=replace_once(text,'    train = TokenCorpus(args.data, "train", args.block_size, config.context)\n',
        '    train = FullPrefixCorpus(args.data, "train", args.block_size, config.context)\n    assert train.manifest["fingerprint"] == "316a97a0f816eeacbf7a9ca79646e90dd971c2be842d849bdea1f8c9970ab078"\n')
    text=replace_once(text,'    contract["epochs"] = args.epochs\n',
        '''    contract["epochs"] = args.epochs
    contract.update(architecture='tcn-engram-fp-delta-lite-4x32', delta_backend='fused', qat=False,
                    full_prefix_width=768, compiled_loss=True, validation_micro_batch=64,
                    source_manifest_sha256=hashlib.sha256((Path(__file__).resolve().parents[2]/'source-manifest.json').read_bytes()).hexdigest())
''')
    text=replace_once(text,'    total_windows = epoch_windows*args.epochs\n',
        '''    total_windows = epoch_windows*args.epochs
    assert epoch_windows == 2411917
    assert hashlib.sha256(order.tobytes()).hexdigest() == 'f3b6d81f19038fdf9b90b250080865fc66b237d7b0bc6c164dcc0a562ed3b64c'
''')
    text=replace_once(text,'return {"format_version": 2, "model_config": config.to_dict(), "model": model.state_dict(),',
        '''return {"format_version": 3, "architecture": "tcn-engram-fp-delta-lite-4x32",
                "delta_config": {"heads":4,"dimension":32,"normalize":"bounded","qat":False},
                "training_elapsed_seconds": prior_elapsed+time.monotonic()-started,
                "model_config": config.to_dict(), "model": model.state_dict(),''')
    text=replace_once(text,'val.batch(args.micro_batch, args.seed+10**12+i)','val.batch(64, args.seed+10**12+i)')
    text=replace_once(text,'        return {"loss": sums["normal"]/count,',
        '        assert count == 230424 if args.eval_batches == 20 else count > 0\n        return {"loss": sums["normal"]/count,')
    text=replace_once(text,'    last_save = started = last_log = time.monotonic()\n',
        '''    rng_before_graph = rng_state(device)
    graph_started = time.monotonic()
    engine = MicroGraph(model, args.micro_batch, compiled_loss=True)
    optimizer.zero_grad(set_to_none=False)
    restore_rng(rng_before_graph, device)
    prior_elapsed = checkpoint.get('training_elapsed_seconds',0.) if checkpoint else 0.
    print(json.dumps({'event':'graph_ready','seconds':time.monotonic()-graph_started,
                      'resumed_step':step,'initial_backbone_sha256':initial_backbone_sha}),flush=True)
    last_save = started = last_log = time.monotonic()
''')
    text=text.replace('optimizer.zero_grad(set_to_none=True)','optimizer.zero_grad(set_to_none=False)')
    text=replace_once(text,'            for x, valid, y in batches:\n                x, valid, y = (t.to(device) for t in (x, valid, y))\n',
        '''            for x, valid, y in batches:
                x, valid, y = (t.to(device) for t in (x, valid, y))
                if teacher is None:
                    loss_sum += engine.backward(x, valid, y, update_tokens)
                    continue
''')
    text=replace_once(text,'        metrics_file.write(json.dumps(record) + "\\n")\n',
        '''        metrics_file.write(json.dumps(record) + "\\n")
        status = dict(status='running' if record.get('event') not in ('finished','stopped') else record['event'],
                      updated_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
                      epoch_cursor=epoch_cursor,epoch_windows=epoch_windows,
                      training_elapsed_seconds=prior_elapsed+time.monotonic()-started,
                      graph_replays=engine.replays,
                      allocated_peak_gib=torch.cuda.max_memory_allocated()/1024**3,
                      reserved_peak_gib=torch.cuda.max_memory_reserved()/1024**3,**record)
        temporary=args.out/'status.json.tmp';temporary.write_text(json.dumps(status,indent=2));temporary.replace(args.out/'status.json')
''')
    source.write_text(text)
    config=json.loads((BASE/'train-config.json').read_text())
    config.update(data=str(ROOT/'training-data/minimind3/pretrain/manifest.json'),out=str(run/'pretrain'),model_config=str(run/'source/training/configs/arch-sawtooth8.json'),micro_batch=128,accumulation=2,resume='none',stop_after_updates=steps,eval_batches=eval_batches,log_every=50,deterministic=bool(steps))
    atomic(run/'train-config.json',config)
    manifest={str(p.relative_to(run/'source')):sha(p) for p in sorted((run/'source').rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    atomic(run/'source-manifest.json',manifest)
    atomic(run/'preparation.json',{'created_utc':now(),'historical_source_train_sha256':original_sha,'source_files':len(manifest),'from_scratch':True,'qat':False,'full_prefix':True,'expected_targets':426194486,'expected_windows':2411917,'expected_updates':9422,'source_manifest_sha256':sha(run/'source-manifest.json'),'historical_last_sha256':sha(BASE/'pretrain/last.pt')})
    return config
def execute(run,resume=False,steps=None):
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=str(run/'source'),TORCHINDUCTOR_COMPILE_THREADS='2',TORCHINDUCTOR_CACHE_DIR=str(SPEED/'inductor-cache'),TRITON_CACHE_DIR=str(SPEED/'triton-cache'))
    cmd=[sys.executable,'-u','-m','training.train','--config',str(run/'train-config.json')]
    if resume:cmd+=['--resume','auto']
    if steps is not None:cmd+=['--stop-after-updates',str(steps)]
    with (run/'pipeline.log').open('a',buffering=1) as log:
        child=subprocess.Popen(cmd,cwd=run/'source',env=env,stdout=log,stderr=subprocess.STDOUT)
        atomic(run/'pipeline-status.json',{'status':'running','pid':child.pid,'started_utc':now(),'command':cmd,'ssh_foreground_required':True})
        try:
            while child.poll() is None:
                time.sleep(15)
                status=run/'pretrain/status.json'
                if status.exists():
                    data=json.loads(status.read_text());print(json.dumps({k:data.get(k) for k in ('status','step','tokens','loss','epochs_completed','tokens_per_second','elapsed_seconds')}),flush=True)
            result=child.returncode
        except BaseException:
            child.terminate()
            try:child.wait(timeout=45)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise
    atomic(run/'pipeline-status.json',{'status':'exited' if result==0 else 'failed','returncode':result,'ended_utc':now()})
    if result:
        print((run/'pipeline.log').read_text()[-10000:],flush=True)
        raise SystemExit(result)
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--steps',type=int,default=0);p.add_argument('--eval-batches',type=int,default=20);p.add_argument('--resume',action='store_true');a=p.parse_args()
    assert '/' not in a.name and '..' not in a.name
    run=ROOT/'training-runs'/a.name
    if not a.resume:prepare(run,a.steps,a.eval_batches)
    execute(run,a.resume,a.steps if a.resume else None)
    if not a.steps:
        result=subprocess.run([sys.executable,'-u',str(run/'audit.py'),str(run)]).returncode
        if result:raise SystemExit(result)
if __name__=='__main__':main()
