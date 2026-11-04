import sys,json,argparse
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from onnx import numpy_helper as nh
H=Path(__file__).resolve().parent;P=H.parents[1];S=H.parent/'delta-sft-qat-board-20261029'
sys.path[:0]=[str(P/'lccc-evaluation-artifacts/source'),str(P/'research/deltanet-npu-20261028')]
from training.model import ModelConfig
from training.train import load_checkpoint
from delta_layer import DeltaLayer
from prepare_probe import Runtime
saved=load_checkpoint(S/'qat-deploy.pt');cfg=ModelConfig.from_dict(saved['model_config'])
p=argparse.ArgumentParser();p.add_argument('--variant',default='saturating-conv');a=p.parse_args()
root=H/a.variant;m=onnx.load(str(root/'step.onnx'))
const={i.name:nh.to_array(i) for i in m.graph.initializer};const.update({n.output[0]:nh.to_array(n.attribute[0].t) for n in m.graph.node if n.op_type=='Constant'})
prod={o:n for n in m.graph.node for o in n.output};params=[]
for v in m.graph.output:
    n=prod[v.name];n=prod[n.input[0]] if n.op_type=='Identity' else n;params.append((const[n.input[1]],const[n.input[2]]))
opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
session=ort.InferenceSession(str(root/'step.onnx'),sess_options=opt,providers=['CPUExecutionProvider'])
class Adapter:
    def run(self,names,feeds):return [(v.astype(np.float32)-z)*s for v,(s,z) in zip(session.run(None,feeds),params)]
r=Runtime(S/'qat-qdq',cfg,DeltaLayer(dimension=32,normalize='bounded'));r.session=Adapter();r.reset();logits=[];states=[]
for token in np.fromfile(S/'qat-qdq/data/sequence.i32','<i4'):
    y,_,s=r.step(int(token));logits.append(y);states.append(np.stack(s).reshape(-1))
np.asarray(logits,'<f4').tofile(root/'closed-logits.f32');np.asarray(states,'<f4').tofile(root/'closed-states.f32')
print('CLOSED_REFERENCE_COMPLETE')
