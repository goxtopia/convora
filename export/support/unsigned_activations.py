"""Equivalent unsigned activation Q/DQ convention recommended for RKNN QAT."""
import argparse
from pathlib import Path
import json
import numpy as np
import onnx
from onnx import numpy_helper
p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);a=p.parse_args();path=a.root/"step.onnx";model=onnx.load(str(path))
constants={n.output[0]:n for n in model.graph.node if n.op_type=="Constant"};initializers={n.name:n for n in model.graph.initializer}
changed=set();count=0
for node in model.graph.node:
    if node.op_type!="QuantizeLinear" or any(attr.name=="axis" for attr in node.attribute):continue
    key=node.input[2]
    if key not in changed:
        tensor=initializers.get(key)
        if tensor is None:tensor=next(attr.t for attr in constants[key].attribute if attr.name=="value")
        value=numpy_helper.to_array(tensor)
        assert value.dtype==np.int8,(key,value.dtype)
        tensor.CopyFrom(numpy_helper.from_array((value.astype(np.int16)+128).astype(np.uint8),tensor.name))
        changed.add(key)
    count+=1
onnx.checker.check_model(model);onnx.save(model,str(path));(a.root/"activation-convention.json").write_text(json.dumps({"activation_qdq_nodes":count,"zero_points_shifted":len(changed),"change":"signed int8 activations -> equivalent uint8 activation Q/DQ, same scale; zp+=128; weights remain signed per-channel"},indent=2))
print(count,len(changed))
