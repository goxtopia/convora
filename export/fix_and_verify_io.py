import argparse,sys,shutil,json,hashlib,subprocess
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from onnx import numpy_helper
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent
sys.path[:0]=[str(PROJECT/'modeling'),str(HERE/'support')]
from training.model import ModelConfig
from training.train import load_checkpoint
from prepare_probe import Runtime
from delta_layer import DeltaLayer
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
if not (root/'step.signed-float-output.onnx').exists():
    shutil.copyfile(root/'step.onnx',root/'step.signed-float-output.onnx')
    subprocess.run([sys.executable,str(HERE/'support/unsigned_activations.py'),'--root',str(root)],check=True)
    shutil.copyfile(root/'step.onnx',root/'step.u8-float-output.onnx')
    subprocess.run([sys.executable,str(HERE/'support/quant_outputs.py'),'--root',str(root)],check=True)
before=onnx.load(str(root/'step.u8-float-output.onnx'));after=onnx.load(str(root/'step.onnx'))
assert {x.name:x.SerializeToString() for x in before.graph.initializer}=={x.name:x.SerializeToString() for x in after.graph.initializer}
constants={x.name:numpy_helper.to_array(x) for x in after.graph.initializer}
constants.update({n.output[0]:numpy_helper.to_array(next(v.t for v in n.attribute if v.name=='value')) for n in after.graph.node if n.op_type=='Constant'})
prod={v:n for n in after.graph.node for v in n.output};params=[]
for out in after.graph.output:
    q=prod[prod[out.name].input[0]];params.append((constants[q.input[1]],constants[q.input[2]]))
options=ort.SessionOptions();options.intra_op_num_threads=2;options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
session=ort.InferenceSession(str(root/'step.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
class Adapter:
    def run(self,names,feeds):return [(x.astype(np.float32)-zp.astype(np.float32))*scale for x,(scale,zp) in zip(session.run(None,feeds),params)]
saved=load_checkpoint(a.weights);cfg=ModelConfig.from_dict(saved['model_config']);runtime=Runtime(root,cfg,DeltaLayer(dimension=32,normalize='bounded'));runtime.session=Adapter();runtime.reset()
logits=[];states=[]
for token in np.fromfile(root/'data/sequence.i32','<i4'):
    y,_,s=runtime.step(int(token));logits.append(y);states.append(np.stack(s).reshape(-1))
logits=np.stack(logits);states=np.stack(states)
signed_session=ort.InferenceSession(str(root/'step.signed-float-output.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
runtime.session=signed_session;runtime.reset();reference=[];sref=[]
for token in np.fromfile(root/'data/sequence.i32','<i4'):
    y,_,s=runtime.step(int(token));reference.append(y);sref.append(np.stack(s).reshape(-1))
reference=np.stack(reference);sref=np.stack(sref)
np.testing.assert_allclose(logits,reference,rtol=1e-6,atol=2e-5);np.testing.assert_allclose(states,sref,rtol=1e-6,atol=2e-5)
for name in ('qdq-reference.f32','state-qdq-reference.f32'):
    path=root/'data'/name
    if not (root/'data'/('ort-optimized-'+name)).exists():shutil.copyfile(path,root/'data'/('ort-optimized-'+name))
reference.astype('<f4').tofile(root/'data/qdq-reference.f32');sref.astype('<f4').tofile(root/'data/state-qdq-reference.f32')
r={'status':'passed','raw_output_type':'UINT8','initializer_bytes_unchanged_by_output_fix':True,'closed_loop_positions':len(logits),'software_logits_max_difference':float(np.abs(logits-reference).max()),'software_state_max_difference':float(np.abs(states-sref).max()),'ort_reference_optimization':'DISABLE_ALL to evaluate explicit Q/DQ semantics; default ORT quantized fusion can change rounding','native_internal_arithmetic_verified':False,'output_affines':[{'scale':float(s),'uint8_zero_point':int(z)} for s,z in params]}
(root/'interface-check.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
