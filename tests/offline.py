"""Recompute historical metrics and test model loading/cache and ONNX numerics.

No network, ADB, training, or chat messages. Default runs all 260 recorded feeds.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'modeling'))
import numpy as np
def read(name,width):return np.fromfile(ROOT/'tests/fixtures'/name,'<f4').reshape(-1,width)
def metric(a,b):return float(np.linalg.norm(a.astype('float64')-b)/np.linalg.norm(b.astype('float64')))
def recorded_metrics():
    ref=read('expected.f32',12288);actual=read('npu-single-step.f32',12288)
    logits=read('npu-closed-logits.f32',6400);states=read('npu-closed-states.f32',4096)
    fp_l=read('fp-closed-logits.f32',6400);fp_s=read('fp-closed-states.f32',4096)
    q_l=read('qdq-closed-logits.f32',6400);q_s=read('qdq-closed-states.f32',4096)
    values={'single_step_logits':metric(actual[:,:6400],ref[:,:6400]),'single_step_state':metric(actual[:,8192:],ref[:,8192:]),'closed_logits_vs_fp':metric(logits,fp_l),'closed_state_vs_fp':metric(states,fp_s),'closed_logits_vs_qdq':metric(logits,q_l),'closed_state_vs_qdq':metric(states,q_s),'top1_same_as_fp':int((logits.argmax(1)==fp_l.argmax(1)).sum()),'positions':len(logits)}
    old=json.loads((ROOT/'reports/compatibility/report.json').read_text())
    for key,a,b in [('single_step_logits','single_step','logits'),('single_step_state','single_step','state')]:assert abs(values[key]-old[a]['layout-conv'][b]['nrmse'])<1e-10
    for key,reference,kind in [('closed_logits_vs_fp','vs_fp','logits'),('closed_state_vs_fp','vs_fp','state'),('closed_logits_vs_qdq','vs_own_qdq','logits'),('closed_state_vs_qdq','vs_own_qdq','state')]:assert abs(values[key]-old['closed_loop']['layout-conv'][reference][kind]['nrmse'])<1e-10
    assert values['positions']==260 and values['top1_same_as_fp']==228
    return values
def onnx_replay(count):
    import onnx,onnxruntime as ort
    from onnx import numpy_helper as nh
    path=ROOT/'models/onnx/compatible.onnx';m=onnx.load(str(path));onnx.checker.check_model(m)
    const={i.name:nh.to_array(i) for i in m.graph.initializer};const.update({n.output[0]:nh.to_array(next(a.t for a in n.attribute if a.name=='value')) for n in m.graph.node if n.op_type=='Constant'})
    prod={o:n for n in m.graph.node for o in n.output};affines=[]
    for out in m.graph.output:
        node=prod[out.name]
        if node.op_type=='Identity':node=prod[node.input[0]]
        assert node.op_type=='QuantizeLinear';affines.append((const[node.input[1]],const[node.input[2]]))
    opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess=ort.InferenceSession(str(path),sess_options=opt,providers=['CPUExecutionProvider'])
    inputs=read('inputs.f32',8704);expected=read('expected.f32',12288);maximum=0.
    for row,ref in zip(inputs[:count],expected[:count]):
        offset=0;feeds={}
        for inp in sess.get_inputs():
            n=int(np.prod(inp.shape));feeds[inp.name]=row[offset:offset+n].reshape(inp.shape);offset+=n
        assert offset==8704
        actual=np.concatenate([((v.astype(np.float32)-z)*s).reshape(-1) for v,(s,z) in zip(sess.run(None,feeds),affines)])
        maximum=max(maximum,float(np.abs(actual-ref).max()))
        np.testing.assert_allclose(actual,ref,rtol=0,atol=0)
    return {'positions':min(count,len(inputs)),'max_abs_difference':maximum}
def float_check():
    import torch
    from load import load_float,FloatDecoder
    from sft_model import SFTModel
    from training.model import ModelConfig
    torch.set_num_threads(2);result={}
    ids=np.fromfile(ROOT/'tests/fixtures/sequence.i32','<i4').astype('int64')
    for name in ('sft-before-qat.pt','qat-final.pt'):
        b,l,s=load_float(ROOT/'models/float'/name);assert all(torch.isfinite(v).all() for v in b.state_dict().values())
        if s['kind']=='qat':
            from qat_delta import SequenceDelta
            sequence=SequenceDelta(l);sequence.load_state_dict(s['delta'],strict=True)
            for i in range(4):
                assert sequence.grids['state_in'].scale[i]==sequence.grids['state_out'].scale[i]
                assert sequence.grids['state_in'].zp[i]==sequence.grids['state_out'].zp[i]
        whole=SFTModel(ModelConfig.from_dict(s['model_config'])).eval()
        state={**s['backbone'],**{'delta.layer.'+k:v for k,v in s['layer'].items()}};whole.load_state_dict(state,strict=True)
        decoder=FloatDecoder(b,l);checks=[]
        with torch.no_grad():
            for i,token in enumerate(ids):
                y=decoder.step(int(token))
                if i+1 in (1,32,133,260):
                    ref=whole(torch.from_numpy(ids[:i+1].copy()).unsqueeze(0))[:,-1]
                    diff=float((y-ref).abs().max());assert diff<3e-4 and int(y.argmax())==int(ref.argmax())
                    checks.append({'prefix':i+1,'max_abs_difference':diff})
        decoder.reset();first=decoder.step(int(ids[0]));fresh=FloatDecoder(*load_float(ROOT/'models/float'/name)[:2]);torch.testing.assert_close(first,fresh.step(int(ids[0])),rtol=0,atol=0)
        result[name]={'cached_vs_full':checks,'reset_exact':True,'source_checkpoint_sha256':s['source_checkpoint_sha256']}
    # A second pass pairs the final FP model with the actual dequantized CPU table.
    b,l,_=load_float(deployment_tables=True);decoder=FloatDecoder(b,l);ys=[];ss=[]
    for token in ids:
        ys.append(decoder.step(int(token)).numpy().reshape(-1));ss.append(np.stack([s.numpy() for s in decoder.states]).reshape(-1))
    err_l=float(np.abs(np.stack(ys)-read('fp-closed-logits.f32',6400)).max());err_s=float(np.abs(np.stack(ss)-read('fp-closed-states.f32',4096)).max())
    assert err_l<3e-4 and err_s<3e-4
    result['fp_vs_shipped_reference']={'logits_max_abs':err_l,'state_max_abs':err_s}
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--positions',type=int,default=260);p.add_argument('--metrics-only',action='store_true');p.add_argument('--out',type=Path);a=p.parse_args();assert 1<=a.positions<=260
    artifacts=json.loads((ROOT/'ARTIFACTS.json').read_text())
    missing=[name for name,info in artifacts['files'].items() if info['distribution']=='huggingface' and (not a.metrics_only or info['category']=='diagnostics') and not (ROOT/name).is_file()]
    if missing:raise SystemExit('External model/diagnostic files are not checked into Git. Download models/** and tests/fixtures/** from https://huggingface.co/'+artifacts['model_repository']+'; see README.md. Missing: '+', '.join(missing))
    result={'recorded_metrics':recorded_metrics()}
    if not a.metrics_only:result.update(float=float_check(),onnx=onnx_replay(a.positions))
    result['passed']=True;serialized=json.dumps(result,indent=2);print(serialized)
    if a.out:a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(serialized+'\n')
if __name__=='__main__':main()
