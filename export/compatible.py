"""Apply the two audited deployment changes to the original formal QDQ graph.

Learned initializers are immutable. Saturating Conv has the documented V endpoint
difference; vector layout changes preserve software outputs exactly.
"""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np,onnx
from onnx import helper as h,numpy_helper as nh
ROOT=Path(__file__).resolve().parents[1]
def rewrite(source,destination):
    m=onnx.load(str(source));original={i.name:i.SerializeToString() for i in m.graph.initializer}
    const={i.name:nh.to_array(i) for i in m.graph.initializer}
    const.update({n.output[0]:nh.to_array(n.attribute[0].t) for n in m.graph.node if n.op_type=='Constant'})
    nodes=[];scaled=set();replaced=[]
    for n in m.graph.node:
        if n.name.startswith('/delta/') and n.op_type=='Clip' and len(n.input)==3 and n.input[2] and float(const[n.input[1]]) in (-1.,-4.):
            bound=float(const[n.input[2]]);factor=1/math.sqrt(32) if bound==1 else 1.;name=n.name+'_saturating_conv'
            def c(s,v):
                k=name+s;m.graph.initializer.append(nh.from_array(v,k));return k
            weight=c('_w',np.full((32,1,1,1),factor,np.float32));scale=c('_s',np.full(32,factor/127,np.float32));zp=c('_z',np.zeros(32,np.int8))
            nodes.append(h.make_node('QuantizeLinear',[weight,scale,zp],[name+'_wq'],axis=0,name=name+'_wq'))
            nodes.append(h.make_node('DequantizeLinear',[name+'_wq',scale,zp],[name+'_wdq'],axis=0,name=name+'_wdq'))
            nodes.append(h.make_node('Conv',[n.input[0],name+'_wdq'],list(n.output),group=32,kernel_shape=[1,1],name=name))
            if bound==1:scaled.add(n.output[0])
            replaced.append(n.name)
        elif n.op_type=='Div' and n.input[0] in scaled:
            nodes.append(h.make_node('Identity',[n.input[0]],list(n.output),name=n.name+'_already_scaled'))
        else:nodes.append(n)
    del m.graph.node[:];m.graph.node.extend(nodes);names=set()
    for n in m.graph.node:
        if n.op_type in ('QuantizeLinear','DequantizeLinear') and n.input[2] in const and const[n.input[2]].dtype==np.uint8:
            if float(const.get(n.input[1],0)) in (.0013864838983863592,.0313725508749485):names.add(n.input[2])
    for n in m.graph.node:
        if n.op_type=='Constant' and n.output[0] in names:n.attribute[0].t.CopyFrom(nh.from_array((nh.to_array(n.attribute[0].t).astype(np.int16)-128).astype(np.int8)))
    m.graph.initializer.append(nh.from_array(np.array([1,32,1,1],np.int64),'aligned_vector_shape'))
    m.graph.initializer.append(nh.from_array(np.array([1,1,1,32],np.int64),'outer_vector_shape'))
    nodes=[];changes=[]
    for n in m.graph.node:
        for head in range(4):
            requests=[]
            if n.name=='/delta/Mul_'+str(5+7*head):requests=[(1,'aligned_vector_shape')]
            if n.name=='/delta/Sub'+('_'+str(head) if head else ''):requests=[(0,'aligned_vector_shape')]
            if n.name=='/delta/Mul_'+str(8+7*head):requests=[(1,'outer_vector_shape')]
            for index,shape in requests:
                alias=n.name+'_aligned_input_'+str(index);nodes.append(h.make_node('Reshape',[n.input[index],shape],[alias],name=alias));changes.append(alias);n.input[index]=alias
        nodes.append(n)
    del m.graph.node[:];m.graph.node.extend(nodes);del m.graph.value_info[:]
    assert len(replaced)==12 and len(changes)==12
    assert original=={i.name:i.SerializeToString() for i in m.graph.initializer if i.name in original}
    onnx.checker.check_model(m);destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True);onnx.save(m,str(destination))
    return {'bounded_projections':len(replaced),'reshapes':len(changes),'original_initializers_byte_identical':True,'strict_graph_equivalence':False,'onnx_sha256':hashlib.sha256(destination.read_bytes()).hexdigest()}
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'models/onnx/qat-original.onnx');p.add_argument('--out',type=Path,default=ROOT/'build/compatible.onnx');a=p.parse_args()
    if not a.source.is_file():raise SystemExit('Download models/** from https://huggingface.co/heyanzhuo/convora first; see README.md.')
    assert a.source.resolve()!=a.out.resolve();result=rewrite(a.source,a.out)
    if a.source.resolve()==(ROOT/'models/onnx/qat-original.onnx').resolve():
        expected=hashlib.sha256((ROOT/'models/onnx/compatible.onnx').read_bytes()).hexdigest();assert result['onnx_sha256']==expected
        result['identical_to_shipped_final_graph']=True
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
