"""CPU-only completion and artifact audit; frozen inside the formal run."""
import json,sys,hashlib,math,datetime
from pathlib import Path
import torch
run=Path(sys.argv[1]);root=run.parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((run/'source-manifest.json').read_text())
assert all(sha(run/'source'/name)==value for name,value in manifest.items())
controls=json.loads((run/'control-manifest.json').read_text())
assert all(sha(run/name)==value for name,value in controls.items())
cp=torch.load(run/'pretrain/last.pt',map_location='cpu',weights_only=False)
assert cp['step']==9422 and cp['tokens_seen']==426194486 and cp['epoch_cursor']==cp['epoch_windows']==2411917
assert cp['initialization'] is None and cp['architecture']=='tcn-engram-fp-delta-lite-4x32' and cp['delta_config']['qat'] is False
assert cp['contract']['source_manifest_sha256']==sha(run/'source-manifest.json')
assert torch.equal(cp['model']['embedding.weight'],cp['model']['head.weight'])
assert all(torch.isfinite(v).all() for v in cp['model'].values() if torch.is_tensor(v) and v.dtype.is_floating_point)
assert all(int(v['step'])==9422 for v in cp['optimizer']['state'].values())
records=[json.loads(line) for line in (run/'pretrain/metrics.jsonl').read_text().splitlines()]
assert all(math.isfinite(r['loss']) for r in records if 'loss' in r)
assert records[-1]['event']=='finished' and records[-1]['budget_reached'] is True
final=next(r for r in reversed(records) if r['event']=='validation_final');assert final['supervised_tokens']==230424
vals=[r for r in records if r['event'] in ('validation','validation_final')]
bestcp=torch.load(run/'pretrain/best.pt',map_location='cpu',weights_only=False)
bestval=min(vals,key=lambda r:r['loss']);assert bestcp['step']==bestval['step']
epochcp=torch.load(run/'pretrain/epoch-0001.pt',map_location='cpu',weights_only=False)
assert epochcp['step']==9422 and all(torch.equal(cp['model'][k],epochcp['model'][k]) for k in cp['model'])
baseline=root/'training-runs/sawtooth8-projected-1epoch/pretrain/last.pt'
assert sha(baseline)=='2e65402ab5f7633c089dfba559078c2f450e59ece1cadb34c9436f7df6283533'
baseline_ce=3.1489927857856603
report={'status':'complete','audited_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'step':cp['step'],'tokens_seen':cp['tokens_seen'],'epoch_windows':cp['epoch_cursor'],'epochs':1,'from_scratch':True,'qat':False,'training_elapsed_seconds':cp['training_elapsed_seconds'],'average_targets_per_second_including_validation_and_periodic_save':cp['tokens_seen']/cp['training_elapsed_seconds'],'source_files_verified':len(manifest),'control_files_verified':controls,'all_model_values_finite':True,'all_recorded_losses_finite':True,'all_optimizer_steps':9422,'initialization':None,'validation_targets':230424,'baseline_ce':baseline_ce,'final_ce':final['loss'],'final_ce_minus_baseline':final['loss']-baseline_ce,'best_ce':bestval['loss'],'best_step':bestval['step'],'best_ce_minus_baseline':bestval['loss']-baseline_ce,'validation':vals,'peak_allocated_gib':max(r.get('cuda_peak_gib',0) for r in records),'artifact_sha256':{name:sha(run/'pretrain'/name) for name in ('last.pt','best.pt','epoch-0001.pt')},'comparison_scope':'Same fixed historical validation inputs and labels, 20 batches x64; single seed and one epoch; new Delta train keeps full document prefix whereas old finite-context train used width388. CE only, not a task/NPU/QAT result.'}
(run/'completion-audit.json').write_text(json.dumps(report,indent=2));(run/'pretrain/completion.json').write_text(json.dumps(report,indent=2))
(run/'pipeline-status.json').write_text(json.dumps({'status':'complete','audited_utc':report['audited_utc'],'returncode':0},indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='validation'}),flush=True)
