"""Exhaust all 256 input codes around the exact deployed normalization grids."""
import json,math
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from onnx import helper as h,numpy_helper as nh,TensorProto as T
HERE=Path(__file__).resolve().parent;root=HERE/'micro';root.mkdir(exist_ok=True)
nodes=[];init=[];outputs=[];records=[];shape=[1,32,1,8]
def c(name,value,dtype=np.float32):init.append(nh.from_array(np.asarray(value,dtype=dtype),name));return name
def op(kind,args,name,**kw):nodes.append(h.make_node(kind,args,[name],name=name,**kw));return name
def qdq(x,s,z,name):
    sn=c(name+'_s',s);zn=c(name+'_z',z,np.uint8)
    q=op('QuantizeLinear',[x,sn,zn],name+'_q');return op('DequantizeLinear',[q,sn,zn],name+'_dq')
sq=.16518276929855347;zq=137
x=qdq('x',sq,zq,'input')
def output(x,s,z,label):
    n=op('QuantizeLinear',[x,c(label+'_s',s),c(label+'_z',z,np.uint8)],label)
    outputs.append(h.make_tensor_value_info(n,T.UINT8,shape));records.append({'label':label,'scale':s,'zp':z})
for bound in (1.,4.):
    tag='b'+str(int(bound));clipped=op('Clip',[x,c(tag+'_lo',-bound),c(tag+'_hi',bound)],tag+'_clip')
    if bound==1:
        y=op('Div',[clipped,c(tag+'_divisor',math.sqrt(32))],tag+'_div')
        output(y,.0013864838983863592,128,'query_clip')
        direct=op('Div',[x,c('direct_divisor',math.sqrt(32))],'direct_div')
        output(direct,.0013864838983863592,128,'query_saturate')
    else:output(clipped,.0313725508749485,127,'value_clip')
m=h.make_model(h.make_graph(nodes,'micro',[h.make_tensor_value_info('x',T.FLOAT,shape)],outputs,init),opset_imports=[h.make_opsetid('',13)]);m.ir_version=8
onnx.checker.check_model(m);onnx.save(m,str(root/'step.onnx'))
inp=(np.arange(256,dtype=np.float32)-zq)*sq;inp.tofile(root/'inputs.f32')
(root/'shapes.txt').write_text(f'1 {len(outputs)} 1\n'+('32 8\n'*(1+len(outputs))))
opt=ort.SessionOptions();opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
s=ort.InferenceSession(str(root/'step.onnx'),sess_options=opt,providers=['CPUExecutionProvider'])
with (root/'expected.f32').open('wb') as f:
    for r,v in zip(records,s.run(None,{'x':inp.reshape(shape)})):((v.astype(np.float32)-r['zp'])*r['scale']).tofile(f)
(root/'manifest.json').write_text(json.dumps({'records':records,'shape':shape},indent=2))
print('MICRO_PREPARED',flush=True)
