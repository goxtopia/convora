"""CPU-only RKNN preparation for the newly trained FP/QAT model, fresh tables."""
import argparse,sys,json,hashlib,shutil,math
from pathlib import Path
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent
sys.path[:0]=[str(PROJECT/'modeling'),str(HERE/'support')]
import numpy as np,torch,onnx,onnxruntime as ort
from training.model import ModelConfig,TCNLanguageModel
from training.train import load_checkpoint
from delta_layer import DeltaLayer
from qat_delta import SequenceDelta
from prepare_probe import HybridStep,Runtime
from qdq_layer import QDQDelta,QDQHybrid
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--calibration',type=Path,required=True);p.add_argument('--qat',action='store_true');a=p.parse_args()
root=a.out.resolve();root.mkdir(parents=True,exist_ok=False)
for n in ('data','models','calibration','results'):(root/n).mkdir()
torch.set_num_threads(2);saved=load_checkpoint(a.weights);cfg=ModelConfig.from_dict(saved['model_config'])
assert cfg.num_passes==1 and cfg.head_norm=='none' and not getattr(cfg,'prompt_anchor',False)
model=TCNLanguageModel(cfg).eval();model.load_state_dict(saved['backbone'])
layer=DeltaLayer(dimension=32,normalize='bounded').eval();layer.load_state_dict(saved['layer'])
seq=None
if a.qat:
    assert saved['kind']=='qat'
    seq=SequenceDelta(layer).eval();seq.load_state_dict(saved['delta']);quant_delta=QDQDelta(seq).eval()
else:assert saved['kind'] in ('fp','qat')
names=['x']+[f'state{i}' for i in range(8)]+['memory']+[f'delta{i}' for i in range(4)]
outputs=['logits']+[f'h{i}' for i in range(7)]+[f'next{i}' for i in range(4)]
examples=[torch.zeros(1,256,1,1)]+[torch.zeros(1,256,1,2) for _ in range(8)]+[torch.zeros(1,256,1,1)]+[torch.zeros(1,32,1,32) for _ in range(4)]
def export(module,path):
    torch.onnx.export(module,tuple(examples),str(path),opset_version=13,input_names=names,output_names=outputs,do_constant_folding=True)
    onnx.checker.check_model(onnx.load(str(path)))
float_hybrid=HybridStep(model,layer).eval();export(float_hybrid,root/'float.onnx')
if a.qat:export(HybridStep(model,quant_delta).eval(),root/'step.onnx')
else:shutil.copyfile(root/'float.onnx',root/'step.onnx')
(model.embedding.weight.detach()*model.embedding_scale).numpy().astype('<f4').tofile(root/'data/embedding.f32')
table=torch.cat([x.weight.detach() for x in model.engram.tables]).numpy();scales=np.maximum(np.abs(table).max(1)/127,1e-8).astype('<f4')
np.clip(np.rint(table/scales[:,None]),-127,127).astype('i1').tofile(root/'data/memory.i8');scales.tofile(root/'data/memory_scale.f32')
ids=np.fromfile(PROJECT/'tests/fixtures/sequence.i32','<i4').tolist();assert len(ids)==260
np.asarray(ids,'<i4').tofile(root/'data/sequence.i32')
runtime=Runtime(root,cfg,layer);initial_session=runtime.session
plain_options=ort.SessionOptions();plain_options.intra_op_num_threads=2;plain_options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
if a.qat:
    shutil.copyfile(root/'step.onnx',root/'delta-qat.onnx')
    initial_session=ort.InferenceSession(str(root/'step.onnx'),sess_options=plain_options,providers=['CPUExecutionProvider']);runtime.session=initial_session
parent_qdq=QDQHybrid(model,quant_delta).eval() if a.qat else None
documents=json.loads(a.calibration.read_text())['documents'];assert len(documents)==48
lines=[]
with torch.no_grad():
    for doc in documents:
        assert doc['split']=='train';runtime.reset();sequence=doc['ids']
        for t,token in enumerate(sequence):
            _,sample,_=runtime.step(token,t%max(1,len(sequence)//8)==0)
            if sample is not None:
                if a.qat:parent_qdq(*[torch.from_numpy(sample[n]) for n in names])
                paths=[]
                for n in names:
                    f=root/'calibration'/f'{len(lines):04d}-{n}.npy';np.save(f,sample[n]);paths.append(str(f))
                lines.append(' '.join(paths))
(root/'calibration/dataset.txt').write_text('\n'.join(lines)+'\n')
if a.qat:
    (root/'parent-ranges.json').write_text(json.dumps(parent_qdq.ranges,indent=2));parent_qdq.observe=False;parent_qdq.quantized=True;export(parent_qdq,root/'step.onnx')
options=ort.SessionOptions();options.intra_op_num_threads=2
float_session=ort.InferenceSession(str(root/'float.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
sessions=[(float_session,'reference')]
if a.qat:
    sessions += [(initial_session,'delta-qat-reference'),(ort.InferenceSession(str(root/'step.onnx'),sess_options=plain_options,providers=['CPUExecutionProvider']),'qdq-reference')]
references={}
for session,label in sessions:
    runtime.session=session;runtime.reset();logits=[];states=[]
    for token in ids:
        y,_,state=runtime.step(token);logits.append(y);states.append(np.stack(state).reshape(-1))
    logits=np.asarray(logits,'<f4');states=np.asarray(states,'<f4');logits.tofile(root/'data'/f'{label}.f32');states.tofile(root/'data'/f'state-{label}.f32');references[label]=(logits,states)
# Independent whole-sequence parent and matrix delta recurrence on the identical
# dequantized CPU table, not an old model's embedding/Engram lookup.
with torch.no_grad():
    offset=0
    for emb in model.engram.tables:
        n=len(emb.weight);emb.weight.copy_(torch.from_numpy(runtime.table[offset:offset+n]));offset+=n
    captured={};hook=model.head.register_forward_pre_hook(lambda m,x:captured.update(h=x[0].detach()))
    model(torch.tensor([ids]));hook.remove();features=captured['h'];states=[torch.zeros(1,32,1,32) for _ in range(4)];check_logits=[];check_states=[];delta_outputs=[]
    for t in range(len(ids)):
        h,*states=layer.reference(features[:,:,t:t+1].unsqueeze(-1),states)
        delta_outputs.append(h.numpy().reshape(-1))
        check_logits.append(torch.nn.functional.conv2d(h,float_hybrid.head_weight).numpy().reshape(-1));check_states.append(np.stack([s.numpy() for s in states]).reshape(-1))
    fp,sp=references['reference'];check_logits=np.stack(check_logits);check_states=np.stack(check_states)
    np.testing.assert_allclose(fp,check_logits,rtol=3e-4,atol=3e-4);np.testing.assert_allclose(sp,check_states,rtol=3e-4,atol=1e-4)
header=['#pragma once',f'constexpr int C=256,V=6400,L=8,R=133,N={len(ids)},FIRST=0,ROWS={len(ids)},LOOPS=2048;',
        'constexpr int D[L]={1,2,4,8,16,32,1,2};','constexpr int DH=4,DK=32,DE=4096;',
        'constexpr int HN=4,HD=64;','constexpr int BUCKETS[HN]={19997,20011,20021,20023};','constexpr int ORDERS[HN]={2,2,3,3};',
        'constexpr uint64_t HASH_MULT[HN][3]={{841577377,1427818425,1713831177},{1403703963,542880493,52321687},{799045305,480294803,756998953},{1028136769,665279961,1782731651}};']
(root/'engram_config.h').write_text('\n'.join(header)+'\n')
report={'status':'export_verified','kind':saved['kind'],'trained_delta':True,'source_checkpoint_sha256':saved['source_checkpoint_sha256'],'deployment_weights_sha256':sha(a.weights),'calibration_documents':48,'calibration_samples':len(lines),'calibration_split':'train','reference_positions':260,'new_embedding_sha256':sha(root/'data/embedding.f32'),'new_memory_sha256':sha(root/'data/memory.i8'),'new_memory_scales_sha256':sha(root/'data/memory_scale.f32'),'fp_onnx_vs_matrix_logits_max':float(np.abs(fp-check_logits).max()),'fp_onnx_vs_matrix_state_max':float(np.abs(sp-check_states).max()),'qat_grids_preserved':a.qat,'parent_qat':False,'parent_quantization':'PTQ of updated backbone/head, not trained fake quant','state_grids':{n:{'scale':g.scale.tolist(),'zp':g.zp.tolist()} for n,g in seq.grids.items()} if a.qat else None,'onnx_sha256':sha(root/'step.onnx'),'float_onnx_sha256':sha(root/'float.onnx')}
report['kind']='qat-weights-ptq' if saved['kind']=='qat' and not a.qat else saved['kind']
report['diagnostic_only']=bool(saved.get('diagnostic_only',False))
report['software_qdq_reference_optimization']='DISABLE_ALL for explicit Q/DQ semantics' if a.qat else None
raw=features.numpy()[0].T
report['delta_residual_relative_l2']=float(np.linalg.norm(np.stack(delta_outputs)-raw)/np.linalg.norm(raw))
(root/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
