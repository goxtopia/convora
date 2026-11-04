import json,hashlib,sys,math,datetime
from pathlib import Path
import torch
run=Path(sys.argv[1]);root=run.parents[1]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for data in iter(lambda:f.read(4*1024**2),b''):h.update(data)
    return h.hexdigest()
for name in ('source-manifest.json','control-manifest.json'):
    values=json.loads((run/name).read_text());base=run/'source' if name.startswith('source') else run
    assert all(sha(base/n)==v for n,v in values.items()),name
cfg=json.loads((run/'config.json').read_text());assets_path=Path(cfg['data']).parent/'asset-sha256.json';assets=json.loads(assets_path.read_text())
assert all(sha(n)==v for n,v in assets['files'].items())
cp=torch.load(run/'sft/last.pt',map_location='cpu',weights_only=False)
assert cp['step']==123974 and cp['tokens_seen']==756276056 and cp['epoch_cursor']==31737342 and cp['epoch_windows']==15868671
assert cp['initialization']['sha256']=='0fc4252d0245a516a76dca6e23395c1609b85800844bc1b753ea9a1be5d9d6d6'
assert sha(cp['initialization']['path'])==cp['initialization']['sha256']
assert cp['architecture']=='tcn-engram-fp-delta-lite-4x32' and cp['delta_config']['qat'] is False
assert torch.equal(cp['model']['embedding.weight'],cp['model']['head.weight'])
assert all(torch.isfinite(v).all() for v in cp['model'].values() if torch.is_tensor(v) and v.dtype.is_floating_point)
assert all(int(v['step'])==123974 for v in cp['optimizer']['state'].values())
assert cp['contract']['source_manifest_sha256']==sha(run/'source-manifest.json')
assert cp['contract']['asset_manifest_sha256']==sha(assets_path)
records=[json.loads(x) for x in (run/'sft/metrics.jsonl').read_text().splitlines()]
assert records[-1]['event']=='finished' and records[-1]['budget_reached'] is True
assert all(math.isfinite(x['loss']) for x in records if 'loss' in x)
e1=torch.load(run/'sft/epoch-0001.pt',map_location='cpu',weights_only=False)
assert e1['step']==61987 and e1['tokens_seen']==378138028 and e1['epoch_cursor']==15868671
e2=torch.load(run/'sft/epoch-0002.pt',map_location='cpu',weights_only=False)
assert e2['step']==123974 and all(torch.equal(cp['model'][k],e2['model'][k]) for k in cp['model'])
best=torch.load(run/'sft/best.pt',map_location='cpu',weights_only=False)
validation=[r for r in records if r['event'] in ('validation','validation_final')]
minimum=min(validation,key=lambda r:r['loss']);assert best['step']==minimum['step']
baseline=root/'training-runs/sawtooth8-lccc-sft/mixed-sft/last.pt'
assert sha(baseline)=='3720565dc4f08f9c5a2da11a729b212adbe50ba83d451c1b2c4d5de4bede68f6'
report={'status':'train_complete','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run':str(run),'step':cp['step'],'tokens_seen':cp['tokens_seen'],'epochs':2,'epoch_cursor':cp['epoch_cursor'],'training_elapsed_seconds':cp['training_elapsed_seconds'],'average_targets_per_second_including_validation_save':cp['tokens_seen']/cp['training_elapsed_seconds'],'source_files_verified':len(json.loads((run/'source-manifest.json').read_text())),'data_files_verified':len(assets['files']),'all_optimizer_counts_verified':True,'all_weights_and_recorded_losses_finite':True,'parent':cp['initialization'],'best_monitor_ce':minimum['loss'],'best_monitor_step':minimum['step'],'final_monitor':validation[-1],'artifact_sha256':{n:sha(run/'sft'/n) for n in ('last.pt','best.pt','epoch-0001.pt','epoch-0002.pt')},'peak_allocated_gib':max(x.get('cuda_peak_gib',0) for x in records),'baseline_sha256':sha(baseline),'qat':False,'no_web_added':True,'comparison_pending':True}
(run/'train-completion-audit.json').write_text(json.dumps(report,indent=2));(run/'sft/completion.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
