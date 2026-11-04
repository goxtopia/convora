from pathlib import Path
import numpy as np,json
from analyze_trace import metric
H=Path(__file__).resolve().parent;m=json.loads((H/'micro/manifest.json').read_text())
a=np.fromfile(H/'board-micro2/micro/actual.f32','<f4').reshape(-1,256)
b=np.fromfile(H/'micro/expected.f32','<f4').reshape(a.shape)
x=np.fromfile(H/'micro/inputs.f32','<f4')
result=[]
for r,actual,expected in zip(m['records'],a,b):
    result.append({**r,**metric(actual,expected,r['scale']),'actual_range':[float(actual.min()),float(actual.max())],'expected_range':[float(expected.min()),float(expected.max())],'positive_inputs':int((x>0).sum()),'positive_input_outputs_all_zero':bool((actual[x>0]==0).all())})
(H/'micro-report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
for i in [0,1,2]: print(m['records'][i]['label'],[(float(x[k]),float(a[i,k]),float(b[i,k])) for k in [0,100,130,136,137,138,143,150,160,200,255]])
