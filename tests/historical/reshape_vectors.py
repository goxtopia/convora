"""Exact QDQ-preserving layout rewrite for scalar times 32-element vectors."""
import copy,json,shutil
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from onnx import helper as h,numpy_helper as nh
H=Path(__file__).resolve().parent;S=H.parent/'delta-sft-qat-board-20261029'
opt=ort.SessionOptions();opt.intra_op_num_threads=2;opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
for source,variant in [(S/'qat-qdq/step.onnx','layout-only'),(H/'saturating-conv/step.onnx','layout-conv')]:
    root=H/variant;root.mkdir(exist_ok=True);m=onnx.load(str(source))
    original={i.name:i.SerializeToString() for i in m.graph.initializer}
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
                alias=n.name+'_aligned_input_'+str(index)
                nodes.append(h.make_node('Reshape',[n.input[index],shape],[alias],name=alias))
                changes.append({'node':n.name,'index':index,'source':n.input[index],'shape':shape});n.input[index]=alias
        nodes.append(n)
    del m.graph.node[:];m.graph.node.extend(nodes);del m.graph.value_info[:]
    assert len(changes)==12
    assert original=={i.name:i.SerializeToString() for i in m.graph.initializer if i.name in original}
    onnx.checker.check_model(m);onnx.save(m,str(root/'step.onnx'))
    for name in ('inputs.f32','shapes.txt'):shutil.copyfile(H/'saturating-conv'/name,root/name)
    ref=ort.InferenceSession(str(source),sess_options=opt,providers=['CPUExecutionProvider']);test=ort.InferenceSession(str(root/'step.onnx'),sess_options=opt,providers=['CPUExecutionProvider'])
    shapes=json.loads((H/'trace/manifest.json').read_text())['inputs'];inputs=np.fromfile(root/'inputs.f32','<f4').reshape(260,8704);maximum=0
    for row in inputs:
        feeds={};offset=0
        for v,shape in zip(m.graph.input,shapes):
            size=int(np.prod(shape));feeds[v.name]=row[offset:offset+size].reshape(shape);offset+=size
        before=ref.run(None,feeds);after=test.run(None,feeds)
        maximum=max(maximum,max(float(np.abs(a.astype(float)-b.astype(float)).max()) for a,b in zip(before,after)))
    assert maximum==0,maximum
    if variant=='layout-conv':shutil.copyfile(H/'saturating-conv/expected.f32',root/'expected.f32')
    else:
        sizes=[int(np.prod(s)) for s in json.loads((H/'trace/manifest.json').read_text())['outputs']]
        np.fromfile(H/'trace/expected.f32','<f4').reshape(260,sum(sizes))[:,:12288].copy().tofile(root/'expected.f32')
    (root/'manifest.json').write_text(json.dumps({'scope':'Exact QDQ layout rewrite on 260 identical captured feeds','raw_software_output_max_difference':maximum,'original_initializers_byte_unchanged':True,'changes':changes},indent=2))
    print(variant,'SOFTWARE_IDENTICAL',maximum,flush=True)
