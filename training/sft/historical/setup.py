import base64,hashlib,json,os,shutil,sys,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=ROOT/'training-runs/sawtooth8-lccc-sft';PARENT=ROOT/'training-runs/delta-fp-pretrain-1epoch-fast-20261029'
PARENT_SHA='0fc4252d0245a516a76dca6e23395c1609b85800844bc1b753ea9a1be5d9d6d6'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def atomic(p,x):
    p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def relocate_data():
    original=json.loads((OLD/'mixed-data/manifest.json').read_text());m=json.loads(json.dumps(original));missing=[]
    for shard in m['shards']:
        for info in shard['splits'].values():
            info['files']={name.replace('/home/wuyupeng/project/tcnengram/',str(ROOT)+'/'):size for name,size in info['files'].items()}
            for name,size in info['files'].items():
                assert Path(name).exists() and Path(name).stat().st_size==size,name
    assert m['fingerprint']=='a0b7193c2f8f391f93dfe2f461f79cfd5d825fb8480686d89c64f3149af2e4c8'
    (HERE/'data').mkdir(exist_ok=True);path=HERE/'data/manifest.json'
    if path.exists():assert json.loads(path.read_text())==m
    else:atomic(path,m)
    return path
def once(text,a,b):
    assert text.count(a)==1,(a[:90],text.count(a))
    return text.replace(a,b,1)
def prepare(name,stop=0,diagnostic=False):
    assert '/' not in name and '..' not in name
    manifest=relocate_data();run=ROOT/'training-runs'/name;assert not run.exists(),str(run)
    assert sha(PARENT/'pretrain/last.pt')==PARENT_SHA
    assert sha(OLD/'mixed-sft/last.pt')=='3720565dc4f08f9c5a2da11a729b212adbe50ba83d451c1b2c4d5de4bede68f6'
    run.mkdir();shutil.copytree(OLD/'source',run/'source')
    for file in ('delta_model.py','delta_layer.py','qat_delta.py','fused_delta.py'):
        shutil.copy2(PARENT/'source'/file,run/'source'/file)
    for file in ('sft_model.py','sft_data.py','sft_graph.py'):
        shutil.copy2(HERE/file,run/'source'/file)
    for file in ('launch.py','setup.py','audit.py','compare.py','diagnostic_cases.py','preflight-report.json','graph-inspection.json','resume-validation.json','generation-cache-validation.json'):
        if (HERE/file).exists():shutil.copy2(HERE/file,run/file)
    if not stop:assert (HERE/'data/asset-sha256.json').exists()
    f=run/'source/training/train.py';text=f.read_text()
    text=once(text,'import argparse\n','import argparse\nimport hashlib\n')
    text=once(text,'from training.model import ModelConfig, TCNLanguageModel\n','from training.model import ModelConfig, TCNLanguageModel\nfrom sft_model import SFTModel\nfrom sft_data import ChatCorpus\nfrom sft_graph import Engine\n')
    text=once(text,'        torch.cuda.set_device(device)\n','        torch.cuda.set_device(device)\n        torch.cuda.set_per_process_memory_fraction(12*1024**3/torch.cuda.get_device_properties(device).total_memory,device)\n')
    text=once(text,'    model = TCNLanguageModel(config).to(device)\n',
        '''    assert args.epochs==2 and args.precision=='bf16' and args.teacher is None
    assert args.micro_batch==256 and args.accumulation==1 and args.block_size==256
    model = SFTModel(config).to(device);model.delta.backend='fused'
    assert sum(p.numel() for p in model.parameters())==9289608
    assert hashlib.sha256(Path(args.init_from).read_bytes()).hexdigest()=='0fc4252d0245a516a76dca6e23395c1609b85800844bc1b753ea9a1be5d9d6d6'
''')
    text=once(text,'    train = TokenCorpus(args.data, "train", args.block_size, config.context)\n','    train = ChatCorpus(args.data, "train", args.block_size, config.context)\n')
    text=once(text,'    contract["epochs"] = args.epochs\n',
        '''    contract["epochs"] = args.epochs
    contract.update(architecture='tcn-engram-fp-delta-lite-4x32',qat=False,full_document_prefix=True,
                    data_manifest_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                    source_manifest_sha256=hashlib.sha256((Path(__file__).resolve().parents[2]/'source-manifest.json').read_bytes()).hexdigest(),
                    parent_sha256='0fc4252d0245a516a76dca6e23395c1609b85800844bc1b753ea9a1be5d9d6d6',
                    asset_manifest_sha256=hashlib.sha256((Path(args.data).parent/'asset-sha256.json').read_bytes()).hexdigest())
''')
    text=once(text,'    if args.init_from and resume_path and resume_path.exists():\n        raise ValueError("Choose resume OR init-from, not both; use --resume none for a new stage")\n',
        '    if args.init_from and resume_path and resume_path.exists():\n        pass  # Parent SHA is bound above; resume restores the full SFT package.\n')
    text=once(text,'    total_windows = epoch_windows*args.epochs\n',
        '''    total_windows = epoch_windows*args.epochs
    assert epoch_windows==15868671 and train.target_count==378138028
    assert hashlib.sha256(order.tobytes()).hexdigest()=='2bb5675962c86d5c82c587a3c621a5ac28ddb6a896bf4494adbddd0dfaf2c887'
''')
    text=once(text,'return {"format_version": 2, "model_config": config.to_dict(), "model": model.state_dict(),',
        '''return {"format_version":3,"architecture":"tcn-engram-fp-delta-lite-4x32",
                "delta_config":{"heads":4,"dimension":32,"normalize":"bounded","qat":False},
                "training_elapsed_seconds":prior_elapsed+time.monotonic()-started,
                "model_config":config.to_dict(),"model":model.state_dict(),''')
    text=once(text,'    last_save = started = last_log = time.monotonic()\n',
        '''    rng_before=rng_state(device);warmup_started=time.monotonic()
    engine=Engine(model);optimizer.zero_grad(set_to_none=False);restore_rng(rng_before,device)
    prior_elapsed=checkpoint.get('training_elapsed_seconds',0.) if checkpoint else 0.
    print(json.dumps({'event':'graph_ready','seconds':time.monotonic()-warmup_started,'resumed_step':step}),flush=True)
    last_save = started = last_log = time.monotonic()
''')
    text=once(text,'                batches = [train.batch_windows(order[i:min(i+args.micro_batch, stop)])\n                           for i in range(offset, stop, args.micro_batch)]\n',
        '                batches, update_tokens = train.prepare_macro(order[offset:stop])\n')
    text=once(text,'            update_tokens = sum(int((y != -100).sum()) for _, _, y in batches)\n','            update_tokens = sum(len(batch[-1]) for _,batch in batches)\n')
    start=text.index('            for x, valid, y in batches:\n');end=text.index('            scaler.unscale_(optimizer)\n',start)
    text=text[:start]+'            loss_sum = engine.backward(batches, update_tokens)\n'+text[end:]
    text=text.replace('optimizer.zero_grad(set_to_none=True)','optimizer.zero_grad(set_to_none=False)')
    text=once(text,'        metrics_file.write(json.dumps(record) + "\\n")\n',
        '''        metrics_file.write(json.dumps(record) + "\\n")
        status=dict(status='running' if record.get('event') not in ('finished','stopped') else record['event'],
                    epoch_cursor=epoch_cursor,epoch_windows=epoch_windows,
                    training_elapsed_seconds=prior_elapsed+time.monotonic()-started,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,**record)
        tmp=args.out/'status.json.tmp';tmp.write_text(json.dumps(status,indent=2));tmp.replace(args.out/'status.json')
''')
    f.write_text(text);compile(text,str(f),'exec')
    cfg=json.loads((OLD/'mixed-sft-config.json').read_text())
    cfg.update(data=str(manifest),out=str(run/'sft'),init_from=str(PARENT/'pretrain/last.pt'),model_config=str(run/'source/training/configs/arch-sawtooth8.json'),resume='none',stop_after_updates=stop,deterministic=diagnostic)
    if stop:cfg.update(eval_batches=1,log_every=1)
    atomic(run/'config.json',cfg)
    sources={str(p.relative_to(run/'source')):sha(p) for p in sorted((run/'source').rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    atomic(run/'source-manifest.json',sources)
    atomic(run/'control-manifest.json',{n:sha(run/n) for n in ('launch.py','setup.py','audit.py','compare.py','diagnostic_cases.py') if (run/n).exists()})
    atomic(run/'preparation.json',{'parent':str(PARENT/'pretrain/last.pt'),'parent_sha256':PARENT_SHA,'baseline':str(OLD/'mixed-sft/last.pt'),'baseline_sha256':sha(OLD/'mixed-sft/last.pt'),'epochs':2,'windows_per_epoch':15868671,'targets_per_epoch':378138028,'expected_updates':123974,'expected_targets':756276056,'source_files':len(sources),'no_web_added':True,'no_qat':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    return run
