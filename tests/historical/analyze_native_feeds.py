"""Compare NPU to software on the identical actual NPU inputs, no recalibration."""
import argparse,sys,json
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
from onnx import numpy_helper
HERE=Path(__file__).resolve().parent
def metric(a,r):
    a=np.asarray(a,np.float64).reshape(len(a),-1);r=np.asarray(r,np.float64).reshape(len(r),-1)
    an=np.linalg.norm(a,axis=1);rn=np.linalg.norm(r,axis=1);cos=np.zeros(len(a));valid=(an>0)&(rn>0);cos[valid]=(a[valid]*r[valid]).sum(1)/(an[valid]*rn[valid]);cos[(an==0)&(rn==0)]=1
    return {'nrmse':float(np.linalg.norm(a-r)/max(np.linalg.norm(r),1e-30)),'min_cosine':float(cos.min()),'mean_cosine':float(cos.mean()),'mean_bias':float((a-r).mean()),'max_absolute':float(np.abs(a-r).max())}
root=HERE/'qat-qdq';capture=HERE/'board-results/qat-qdq/capture'
positions=np.fromfile(capture/'feed-positions.i32','<i4');assert np.array_equal(positions,np.arange(260))
inputs=np.fromfile(capture/'input-feeds.f32','<f4').reshape(260,8704);actual=np.fromfile(capture/'output-feeds.f32','<f4').reshape(260,12288)
model=onnx.load(str(root/'step.onnx'));constants={x.name:numpy_helper.to_array(x) for x in model.graph.initializer};constants.update({n.output[0]:numpy_helper.to_array(next(a.t for a in n.attribute if a.name=='value')) for n in model.graph.node if n.op_type=='Constant'});prod={v:n for n in model.graph.node for v in n.output}
params=[]
for output in model.graph.output:
    q=prod[prod[output.name].input[0]];params.append((constants[q.input[1]],constants[q.input[2]]))
options=ort.SessionOptions();options.intra_op_num_threads=2
plain_options=ort.SessionOptions();plain_options.intra_op_num_threads=2;plain_options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sessions={'float':ort.InferenceSession(str(root/'float.onnx'),sess_options=options,providers=['CPUExecutionProvider']),'qdq':ort.InferenceSession(str(root/'step.onnx'),sess_options=plain_options,providers=['CPUExecutionProvider'])}
shapes=[('x',(1,256,1,1))]+[(f'state{i}',(1,256,1,2)) for i in range(8)]+[('memory',(1,256,1,1))]+[(f'delta{i}',(1,32,1,32)) for i in range(4)]
expected={key:[] for key in sessions}
for row in inputs:
    feeds={};offset=0
    for name,shape in shapes:
        count=int(np.prod(shape));feeds[name]=row[offset:offset+count].reshape(shape);offset+=count
    assert offset==8704
    for key,session in sessions.items():
        values=session.run(None,feeds)
        if key=='qdq':values=[(x.astype(np.float32)-zp.astype(np.float32))*s for x,(s,zp) in zip(values,params)]
        expected[key].append(np.concatenate([x.reshape(-1) for x in values]))
result={'status':'complete','positions':260,'calibration_modified':False,'scope':'Actual NPU input feeds passed once to FP and exported Q/DQ, measures single-step operator differences separately from feedback accumulation.','reference':{}}
for key,rows in expected.items():
    rows=np.stack(rows);result['reference'][key]={'logits':metric(actual[:,:6400],rows[:,:6400]),'state':metric(actual[:,8192:],rows[:,8192:]),'tcn_layers':[metric(actual[:,6400+i*256:6400+(i+1)*256],rows[:,6400+i*256:6400+(i+1)*256]) for i in range(7)]}
(HERE/'native-feeds-report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
