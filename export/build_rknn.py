import argparse,json,hashlib
from pathlib import Path
from rknn.api import RKNN
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--qat',action='store_true');a=p.parse_args();root=a.root.resolve()
r=RKNN(verbose=False)
try:
    assert r.config(target_platform='rv1103',optimization_level=0 if a.qat else 3,quantized_algorithm='normal',quantized_method='channel')==0
    assert r.load_onnx(model=str(root/'step.onnx'))==0
    assert r.build(do_quantization=not a.qat,**({} if a.qat else {'dataset':str(root/'calibration/dataset.txt')}))==0
    assert r.export_rknn(str(root/'models/tcn_engram.rknn'))==0
finally:r.release()
report=json.loads((root/'report.json').read_text());report.update(status='compile_passed',rknn_sha256=hashlib.sha256((root/'models/tcn_engram.rknn').read_bytes()).hexdigest(),rknn_bytes=(root/'models/tcn_engram.rknn').stat().st_size,do_quantization=not a.qat,onnx_sha256=hashlib.sha256((root/'step.onnx').read_bytes()).hexdigest())
(root/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
