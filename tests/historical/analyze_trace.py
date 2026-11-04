from pathlib import Path
import json,numpy as np
HERE=Path(__file__).resolve().parent
def metric(a,b,scale=None):
    e=a.astype(np.float64)-b
    r={'nrmse':float(np.linalg.norm(e)/max(np.linalg.norm(b.astype(np.float64)),1e-30)), 'bias':float(e.mean()),'max_abs':float(np.abs(e).max()),'exact_fraction':float((e==0).mean())}
    if scale:r.update(mean_lsb=float(e.mean()/scale),max_lsb=float(np.abs(e).max()/scale))
    return r
if __name__=='__main__':
    m=json.loads((HERE/'trace/manifest.json').read_text());sizes=[int(np.prod(s)) for s in m['outputs']];total=sum(sizes)
    expected=np.fromfile(HERE/'trace/expected.f32','<f4').reshape(260,total)
    actual=np.fromfile(HERE/'board-trace/trace/actual.f32','<f4').reshape(260,total)
    original=np.fromfile(HERE.parent/'delta-sft-qat-board-20261029/board-results/qat-qdq/capture/output-feeds.f32','<f4').reshape(260,12288)
    result={'trace_vs_original_output':metric(actual[:,:12288],original),'trace_vs_original_state':metric(actual[:,8192:12288],original[:,8192:]),'trace_vs_qdq_state':metric(actual[:,8192:12288],expected[:,8192:12288]),'nodes':[]}
    offset=12288
    for t,size in zip(m['traces'],sizes[12:]):
        result['nodes'].append({**t,**metric(actual[:,offset:offset+size],expected[:,offset:offset+size],t['scale'])});offset+=size
    (HERE/'trace-report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
