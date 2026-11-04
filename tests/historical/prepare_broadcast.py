import json
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from onnx import helper as h,numpy_helper as nh,TensorProto as T
H=Path(__file__).resolve().parent;root=H/'micro-broadcast';root.mkdir(exist_ok=True)
nodes=[];constants=[]
def c(name,x,dtype=np.float32):constants.append(nh.from_array(np.asarray(x,dtype),name));return name
def op(kind,args,name,**kw):nodes.append(h.make_node(kind,args,[name],name=name,**kw));return name
def qdq(x,s,z,name):
    s=c(name+'_s',s);z=c(name+'_z',z,np.uint8)
    return op('DequantizeLinear',[op('QuantizeLinear',[x,s,z],name+'_q'),s,z],name+'_dq')
a=qdq('alpha',.0038823531940579414,0,'a');v=qdq('vector',.045114655047655106,122,'v')
outputs=[]
for label,aligned in [('wide',False),('channels',True)]:
    x=op('Reshape',[v,c('shape',[1,32,1,1],np.int64)],'vector_channels') if aligned else v
    x=op('Mul',[a,x],label+'_mul')
    x=op('QuantizeLinear',[x,c(label+'_s',.04466351121664047),c(label+'_z',122,np.uint8)],label)
    outputs.append(h.make_tensor_value_info(label,T.UINT8,[1,32,1,1] if aligned else [1,1,1,32]))
m=h.make_model(h.make_graph(nodes,'broadcast',[h.make_tensor_value_info('alpha',T.FLOAT,[1,1,1,1]),h.make_tensor_value_info('vector',T.FLOAT,[1,1,1,32])],outputs,constants),opset_imports=[h.make_opsetid('',13)]);m.ir_version=8
onnx.checker.check_model(m);onnx.save(m,str(root/'step.onnx'))
vectors=(np.arange(256,dtype=np.float32)-122)*np.float32(.045114655047655106)
opt=ort.SessionOptions();opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
session=ort.InferenceSession(str(root/'step.onnx'),sess_options=opt,providers=['CPUExecutionProvider'])
with (root/'inputs.f32').open('wb') as f,(root/'expected.f32').open('wb') as e:
    for row in vectors.reshape(8,32):
        alpha=np.array([[[[np.float32(.0038823531940579414)*255]]]],np.float32)
        alpha.tofile(f);row.tofile(f)
        values=session.run(None,{'alpha':alpha,'vector':row.reshape(1,1,1,32)})
        assert np.array_equal(values[0].reshape(-1),values[1].reshape(-1))
        for v in values:((v.astype(np.float32)-122)*np.float32(.04466351121664047)).tofile(e)
(root/'shapes.txt').write_text('2 2 8\n1 1\n1 32\n1 32\n32 1\n')
(root/'manifest.json').write_text(json.dumps({'input_codes':256,'scalar':.99,'software_layout_outputs_exactly_equal':True,'scale':.04466351121664047},indent=2))
print('BROADCAST_PREPARED')
