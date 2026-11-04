"""Diagnostic: express bounded projection as signed, saturating 1x1 convolution.

This is a candidate deployment arithmetic, not an exactly equivalent graph rewrite.
Original learned initializers are byte-preserved; added identity/scaling kernels
and at most a one-code endpoint change are audited separately.
"""
import copy,json,math,shutil,hashlib
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from onnx import helper as h,numpy_helper as nh,TensorProto as T
H=Path(__file__).resolve().parent;S=H.parent/'delta-sft-qat-board-20261029'
root=H/'saturating-conv';root.mkdir(exist_ok=True)
m=onnx.load(str(S/'qat-qdq/step.onnx'))
original={i.name:i.SerializeToString() for i in m.graph.initializer}
const={i.name:nh.to_array(i) for i in m.graph.initializer}
const.update({n.output[0]:nh.to_array(n.attribute[0].t) for n in m.graph.node if n.op_type=='Constant'})
nodes=[];scaled=set();replaced=[]
for n in m.graph.node:
    if n.name.startswith('/delta/') and n.op_type=='Clip' and len(n.input)==3 and n.input[2] and float(const[n.input[1]]) in (-1.,-4.):
        bound=float(const[n.input[2]]);factor=1/math.sqrt(32) if bound==1 else 1.
        name=n.name+'_saturating_conv'
        def c(s,v):
            k=name+s;m.graph.initializer.append(nh.from_array(v,k));return k
        weight=c('_w',np.full((32,1,1,1),factor,np.float32))
        scale=c('_s',np.full(32,factor/127,np.float32));zp=c('_z',np.zeros(32,np.int8))
        nodes.append(h.make_node('QuantizeLinear',[weight,scale,zp],[name+'_wq'],axis=0,name=name+'_wq'))
        nodes.append(h.make_node('DequantizeLinear',[name+'_wq',scale,zp],[name+'_wdq'],axis=0,name=name+'_wdq'))
        nodes.append(h.make_node('Conv',[n.input[0],name+'_wdq'],list(n.output),group=32,kernel_shape=[1,1],name=name))
        if bound==1:scaled.add(n.output[0])
        replaced.append({'clip':n.name,'bound':bound,'factor':factor})
    elif n.op_type=='Div' and n.input[0] in scaled:
        nodes.append(h.make_node('Identity',[n.input[0]],list(n.output),name=n.name+'_already_scaled'))
    else:nodes.append(n)
del m.graph.node[:];m.graph.node.extend(nodes)
# Signed grids around normalized q/k/v. Only zero-point representation changes.
names=set()
for n in m.graph.node:
    if n.op_type in ('QuantizeLinear','DequantizeLinear') and n.input[2] in const and const[n.input[2]].dtype==np.uint8:
        if float(const.get(n.input[1],0)) in (.0013864838983863592,.0313725508749485):names.add(n.input[2])
for n in m.graph.node:
    if n.op_type=='Constant' and n.output[0] in names:n.attribute[0].t.CopyFrom(nh.from_array((nh.to_array(n.attribute[0].t).astype(np.int16)-128).astype(np.int8)))
assert original=={i.name:i.SerializeToString() for i in m.graph.initializer if i.name in original}
onnx.checker.check_model(m);onnx.save(m,str(root/'step.onnx'))
shutil.copyfile(H/'trace/inputs.f32',root/'inputs.f32')
manifest=json.loads((H/'trace/manifest.json').read_text());shapes=manifest['inputs'];outs=manifest['outputs'][:12]
(root/'shapes.txt').write_text('14 12 260\n'+''.join(f'{s[1]} {s[3]}\n' for s in shapes+outs))
opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
session=ort.InferenceSession(str(root/'step.onnx'),sess_options=opt,providers=['CPUExecutionProvider'])
prod={o:n for n in m.graph.node for o in n.output};params=[]
for v in m.graph.output:
    n=prod[v.name];n=prod[n.input[0]] if n.op_type=='Identity' else n
    params.append((const[n.input[1]],const[n.input[2]]))
inputs=np.fromfile(root/'inputs.f32','<f4').reshape(260,8704);rows=[]
for row in inputs:
    feeds={};offset=0
    for v,shape in zip(m.graph.input,shapes):
        size=int(np.prod(shape));feeds[v.name]=row[offset:offset+size].reshape(shape);offset+=size
    values=session.run(None,feeds);rows.append(np.concatenate([((v.astype(np.float32)-z)*s).reshape(-1) for v,(s,z) in zip(values,params)]))
np.stack(rows).astype('<f4').tofile(root/'expected.f32')
(root/'manifest.json').write_text(json.dumps({'replaced':replaced,'original_learned_initializers_byte_identical':True,'scope':'Diagnostic candidate arithmetic, not exact QDQ equivalence; requires independent acceptance before training/deployment','inputs':shapes,'outputs':outs,'positions':260},indent=2))
print('SATURATING_CONV_PREPARED',len(replaced),flush=True)
