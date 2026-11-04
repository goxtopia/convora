"""Expose quantized cache/state tensors, retaining internal DQ for float ONNX ops."""
import argparse
import json
from pathlib import Path
import onnx
from onnx import helper
p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args();path=a.root/"step.onnx";m=onnx.load(str(path));nodes=list(m.graph.node);prod={v:n for n in nodes for v in n.output};changes=[]
for value in m.graph.output:
    old=value.name;dq=prod[old];assert dq.op_type=="DequantizeLinear",(old,dq.op_type)
    q=dq.input[0];floating=old+"_internal_float"
    dq.output[0]=floating
    for n in nodes:
        for i,s in enumerate(n.input):
            if s==old:n.input[i]=floating
    nodes.append(helper.make_node("Identity",[q],[old],name="expose_quantized_"+old));value.type.tensor_type.elem_type=onnx.TensorProto.UINT8
    changes.append(old)
del m.graph.node[:];m.graph.node.extend(nodes);onnx.checker.check_model(m);onnx.save(m,str(path));(a.root/"quantized-outputs.json").write_text(json.dumps({"outputs":changes,"warning":"ONNX outputs are raw UINT8 Q tensors; comparisons require their fixed affine dequantization, native runtime may expose signed INT8."},indent=2));print(changes)
