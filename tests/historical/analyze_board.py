import argparse,json,hashlib,re
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def metric(actual,ref):
    a=np.asarray(actual,np.float64).reshape(len(actual),-1);r=np.asarray(ref,np.float64).reshape(len(ref),-1)
    aa=np.linalg.norm(a,axis=1);rr=np.linalg.norm(r,axis=1);cos=np.zeros(len(a));valid=(aa>0)&(rr>0)
    cos[valid]=(a[valid]*r[valid]).sum(1)/(aa[valid]*rr[valid]);cos[(aa==0)&(rr==0)]=1
    return {'nrmse':float(np.linalg.norm(a-r)/max(np.linalg.norm(r),1e-30)),'min_cosine':float(cos.min()),'mean_cosine':float(cos.mean()),'max_absolute':float(np.abs(a-r).max()),'actual_zero_fraction':float(np.mean(a==0)),'reference_rms':float(np.sqrt(np.mean(r*r))),'positions':len(a)}
report={'status':'complete','variants':{},'scope':'Same260 input IDs, independent FP recurrence for each weight set and its newly exported row-quantized Engram table. Replay and fixed-length argmax rates are not end-to-end chat throughput.','calibration':'48 stratified train documents only; val260 diagnostic not used for calibration.'}
for name in ('fp-ptq','qat-weights-ptq','qat-qdq'):
    root=HERE/name;board=HERE/'board-results'/name/'steps-2048';meta=json.loads((root/'report.json').read_text());m=json.loads((board/'metrics.json').read_text())
    logits=np.fromfile(board/'logits.f32','<f4').reshape(260,6400);state=np.fromfile(board/'states.f32','<f4').reshape(260,4,32,32)
    ref=np.fromfile(root/'data/reference.f32','<f4').reshape(logits.shape);sref=np.fromfile(root/'data/state-reference.f32','<f4').reshape(state.shape)
    assert np.isfinite(logits).all() and np.isfinite(state).all() and m['after_cleanup']['dma_used_kib']==12
    lm=metric(logits,ref);sm=metric(state,sref)
    np.testing.assert_allclose(lm['nrmse'],m['logits_vs_fp32']['nrmse'],rtol=2e-5);np.testing.assert_allclose(sm['nrmse'],m['state_vs_fp32']['nrmse'],rtol=2e-5)
    logs=(board/'board.log').read_text();scales=[];zeros=[];types=[];grids=[]
    for h in range(4):
        for side,index,tensor,prefix in (('input',10+h,'delta'+str(h),'state_in'),('output',8+h,'next'+str(h),'state_out')):
            match=re.search(rf'{side.upper()} {index} {tensor} .*?type=(\d+).*?zp=(-?\d+) scale=([\d.eE+-]+)',logs);assert match,(name,side,h)
            dtype=int(match[1]);zp=int(match[2]);scale=float(match[3]);row={'head':h,'side':side,'scale':scale,'zero_point':zp,'native_dtype':dtype}
            if name=='qat-qdq':
                expected=meta['state_grids'][prefix];expected_zp=int(expected['zp'][h])+(128 if dtype==3 else 0);row['matches_training']=zp==expected_zp and abs(scale-expected['scale'][h])<max(1e-9,scale*1e-6);assert row['matches_training'],row
            grids.append(row)
            if side=='output':scales.append(scale);zeros.append(zp);types.append(dtype)
    scale=np.asarray(scales).reshape(1,4,1,1);zp=np.asarray(zeros).reshape(1,4,1,1)
    assert len(set(types))==1 and types[0] in (2,3)
    limits=(0,255) if types[0]==3 else (-128,127)
    direct=(np.clip(np.rint(sref/scale)+zp,*limits)-zp)*scale
    r={'export':meta,'board':m,'logits_all260':lm,'logits_warm128':metric(logits[132:],ref[132:]),'top1_agreement_all260':int(np.sum(logits.argmax(-1)==ref.argmax(-1))),'state_all260':sm,'state_direct_rounding':metric(direct,sref),'state_per_head':[metric(state[:,h],sref[:,h]) for h in range(4)],'native_state_grids':grids,'rknn_sha256':hashlib.sha256((root/'models/tcn_engram.rknn').read_bytes()).hexdigest()}
    if name=='qat-qdq':
        r['interface']=json.loads((root/'interface-check.json').read_text());r['software']={}
        for label in ('delta-qat-reference','qdq-reference'):
            f=np.fromfile(root/'data'/f'{label}.f32','<f4').reshape(ref.shape);s=np.fromfile(root/'data'/f'state-{label}.f32','<f4').reshape(sref.shape)
            r['software'][label]={'logits':metric(f,ref),'state':metric(s,sref)}
    report['variants'][name]=r
before=json.loads((HERE/'board-results/original-service.json').read_text(encoding='utf-8-sig'));after=json.loads((HERE/'board-results/restored-service.json').read_text(encoding='utf-8-sig'))
assert after['ready'] and before['board']['checkpoint_sha256']==after['checkpoint_sha256'] and before['board']['model']==after['model'] and before['board']['context_tokens']==after['context_tokens']
report['original_service']={'before':before,'after':after,'restored':True};(HERE/'board-report.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'status':report['status'],'variants':{n:{'logits':v['logits_all260'],'state':v['state_all260'],'top1':v['top1_agreement_all260'],'replay_steps_per_second':v['board']['replay_steps_per_second'],'closed_loop_steps_per_second':v['board']['closed_loop_steps_per_second'],'memory':v['board']['active']} for n,v in report['variants'].items()},'original_service_restored':True},indent=2))
