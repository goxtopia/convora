"""Recompute final evidence from raw files, with no fitting or recalibration."""
from pathlib import Path
import json,hashlib,numpy as np
from analyze_trace import metric
H=Path(__file__).resolve().parent;S=H.parent/'delta-sft-qat-board-20261029'
def read(p,width):return np.fromfile(p,'<f4').reshape(-1,width)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
original=read(S/'board-results/qat-qdq/capture/output-feeds.f32',12288)
baseline=read(H/'board-baseline/baseline/actual.f32',12288)
baseline2=read(H/'board-finalprobes/baseline/actual.f32',12288)
sizes=[int(np.prod(s)) for s in json.loads((H/'trace/manifest.json').read_text())['outputs']]
expected=read(H/'trace/expected.f32',sum(sizes))[:,:12288]
single={}
for variant,path in [('original','board-baseline/baseline'),('saturating-conv','board-conv/saturating-conv'),('layout-only','board-layout/layout-only'),('layout-conv','board-layout/layout-conv')]:
    actual=read(H/path/'actual.f32',12288);ref=expected if variant=='original' else read(H/variant/'expected.f32',12288)
    single[variant]={'logits':metric(actual[:,:6400],ref[:,:6400]),'state':metric(actual[:,8192:],ref[:,8192:]),'per_head':[metric(actual[:,8192+i*1024:9216+i*1024],ref[:,8192+i*1024:9216+i*1024]) for i in range(4)]}
fp_logits=read(S/'qat-qdq/data/reference.f32',6400);fp_states=read(S/'qat-qdq/data/state-reference.f32',4096)
closed={}
for variant,path in [('original',S/'board-results/qat-qdq/steps-2048'),('saturating-conv',H/'board-closed-conv'),('layout-conv',H/'board-closed-layout-conv')]:
    logits=read(path/'logits.f32',6400);states=read(path/'states.f32',4096)
    if variant=='original':ref_l=read(S/'qat-qdq/data/qdq-reference.f32',6400);ref_s=read(S/'qat-qdq/data/state-qdq-reference.f32',4096)
    else:ref_l=read(H/variant/'closed-logits.f32',6400);ref_s=read(H/variant/'closed-states.f32',4096)
    metrics=json.loads((path/'metrics.json').read_text())
    closed[variant]={'vs_fp':{'logits':metric(logits,fp_logits),'state':metric(states,fp_states)},'vs_own_qdq':{'logits':metric(logits,ref_l),'state':metric(states,ref_s)},'software_vs_fp':{'logits':metric(ref_l,fp_logits),'state':metric(ref_s,fp_states)},'top1_agreement_with_fp':int((logits.argmax(1)==fp_logits.argmax(1)).sum()),'positions':260,'benchmark':{k:metrics[k] for k in ['benchmark_steps_per_repeat','repeats','replay_steps_per_second','closed_loop_steps_per_second','major_faults','active','after_cleanup']}}
micro={}
for variant,path in [('micro','board-micro2/micro'),('micro-signed','board-signed/micro-signed'),('micro-opset10','board-alternatives/micro-opset10'),('micro-hsig','board-alternatives/micro-hsig')]:
    a=read(H/path/'actual.f32',256);b=read(H/variant/'expected.f32',256)
    micro[variant]=[{**r,**metric(x,y,r['scale']),'actual_min':float(x.min()),'actual_max':float(x.max())} for r,x,y in zip(json.loads((H/variant/'manifest.json').read_text())['records'],a,b)]
sim=np.load(H/'micro/simulator.npz');micro['simulator_original']=[]
for r,v in zip(json.loads((H/'micro/manifest.json').read_text())['records'],sim.values()):
    i=len(micro['simulator_original']);dq=(v.astype(np.float32)-r['zp'])*r['scale'];ref=read(H/'micro/expected.f32',256)[i]
    micro['simulator_original'].append({'label':r['label'],**metric(dq.reshape(-1),ref,r['scale']),'actual_min':float(dq.min()),'actual_max':float(dq.max())})
a=np.fromfile(H/'board-finalprobes/micro-broadcast/actual.f32','<f4').reshape(8,2,32)
b=np.fromfile(H/'micro-broadcast/expected.f32','<f4').reshape(8,2,32)
broadcast={label:metric(a[:,i],b[:,i],.04466351121664047) for i,label in enumerate(['wide','channels'])}
qkv_scale=.16518276929855347;raw=(np.arange(256,dtype=np.float32)-137)*np.float32(qkv_scale)
endpoint=[]
for label,bound,scale,zp,factor in [('query_key',1.,.0013864838983863592,128,1/np.sqrt(32)),('value',4.,.0313725508749485,127,1.)]:
    old=np.clip(np.rint(np.clip(raw,-bound,bound)*np.float32(factor)/np.float32(scale))+zp,0,255)
    new=np.clip(np.rint(raw*np.float32(factor)/np.float32(scale))+zp,0,255)
    endpoint.append({'label':label,'input_codes':256,'max_output_code_change':float(np.abs(new-old).max()),'changed_codes':int((new!=old).sum())})
result={'status':'investigation_complete_with_validated_candidate','date':'2026-10-29 Asia/Shanghai','scope':'Same formal QAT trained checkpoint, no training/calibration/SDK/runtime changes. Diagnostic inputs already used, not a blind quality evaluation.','checkpoint_sha256':'9db7a2fa06b2687bfaa8091b9814bbf6ce52c7aba64d45bd66af933a902adef5','toolkit':'1.6.0+81f21f4d','driver':'v0.9.2','installed_runtime_sha256':'d6259f5f8537acc15b0c3686010109bf4182e654281aad180a44bfd73ce929cc','baseline_replay_vs_capture':metric(baseline,original),'baseline_replay_repeat':metric(baseline2,baseline),'single_step':single,'closed_loop':closed,'micro':micro,'broadcast_micro':broadcast,'normalization_endpoint_audit':endpoint,'layout_software_checks':{v:json.loads((H/v/'manifest.json').read_text()) for v in ['layout-only','layout-conv']},'candidate':{'variant':'layout-conv','onnx_sha256':sha(H/'layout-conv/step.onnx'),'rknn_sha256':sha(H/'layout-conv/model.rknn'),'learned_initializers_unchanged':True,'normalization_exact_equivalence':False,'normalization_semantics':'signed quantization saturation replaces bounds; max one-code endpoint change in exhaustive input-code audit','service_deployed':False,'long_memory_task_passed':False,'old_strict_logits_gate_passed':False},'final_service_state':'Stopped at user request; no automatic restore or model switch','remaining':['Residual int8 quantization and backbone/head PTQ error remain; candidate state-vs-FP about 14%, not zero.','No new QAT training or unseen memory/task evaluation performed.','No attribution to a specific proprietary compiler/runtime source-code line; measured operator/shape mismatch only.']}
(H/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps({'single_step':single,'closed_loop':closed,'broadcast':broadcast,'endpoint':endpoint,'baseline_repeat':result['baseline_replay_repeat']},indent=2))
