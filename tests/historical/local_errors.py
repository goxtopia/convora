"""Use actual predecessor outputs to isolate each operator's local discrepancy."""
from pathlib import Path
import json,numpy as np
from analyze_trace import metric
H=Path(__file__).resolve().parent
inputs=np.fromfile(H/'trace/inputs.f32','<f4').reshape(260,8704)
result={}
for variant,path,bounded in [('original','board-trace/trace',True),('conv','board-traceconv/trace-conv',False)]:
    m=json.loads((H/('trace' if bounded else 'trace-conv')/'manifest.json').read_text())
    sizes=[int(np.prod(s)) for s in m['outputs']];a=np.fromfile(H/path/'actual.f32','<f4').reshape(260,sum(sizes));data={};grids={};offset=12288
    for t,size in zip(m['traces'],sizes[12:]):
        data[t['label']]=a[:,offset:offset+size].reshape((260,)+tuple(t['shape'][1:]));grids[t['label']]=(t['scale'],t['zp']);offset+=size
    def fq(label,x):
        s,z=grids[label];lo,hi=(0,255) if z>=0 and bounded else (-128,127)
        # Original grids all U8; candidate non-normalization grids remain U8 too.
        if not bounded and label not in ('query','key','value'):lo,hi=0,255
        return (np.clip(np.rint(x/np.float32(s))+z,lo,hi)-z)*np.float32(s)
    d=data;state=inputs[:,4608:5632].reshape(260,32,1,32)
    q,k,v=np.split(d['qkv'],3,axis=1);q=q[:,:32];k=k[:,:32];v=v[:,:32]
    expected={
        'query':fq('query',(np.clip(q,-1,1) if bounded else q)/np.float32(np.sqrt(32))),
        'key':fq('key',(np.clip(k,-1,1) if bounded else k)/np.float32(np.sqrt(32))),
        'value':fq('value',(np.clip(v,-4,4) if bounded else v).reshape(260,1,1,32)),
        'alpha':fq('alpha',np.float32(.95)+np.float32(.04)*np.clip(d['alpha_linear']/np.float32(6)+np.float32(.5),0,1)),
        'beta':fq('beta',np.clip(d['beta_linear']/np.float32(6)+np.float32(.5),0,1)),
        'old_product':fq('old_product',state*d['key']),
        'old_sum':fq('old_sum',d['old_product'].sum(1,keepdims=True)),
        'alpha_old':fq('alpha_old',d['alpha']*d['old_sum']),
        'value_error':fq('value_error',d['value']-d['alpha_old']),
        'beta_error':fq('beta_error',d['beta']*d['value_error']),
        'decayed':fq('decayed',d['alpha']*state),
        'outer':fq('outer',d['key']*d['beta_error']),
        'read_product':fq('read_product',a[:,8192:9216].reshape(260,32,1,32)*d['query']),
        'read_sum':fq('read_sum',d['read_product'].sum(1,keepdims=True)),
    }
    s=.052299510687589645;z=117
    raw=d['decayed']+d['outer'];state_expected=(np.clip(np.rint(raw/s)+z,0,255)-z)*s
    state_actual=a[:,8192:9216].reshape(260,32,1,32)
    rows={label:metric(d[label],e,grids[label][0]) for label,e in expected.items()}
    rows['state_add']=metric(state_actual,state_expected,s)
    # Unrounded analog residual, in units of the output state quantizer.
    rows['state_add']['analog_bias_lsb']=float(((state_actual-raw)/s).mean())
    result[variant]=rows
(H/'local-errors.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
