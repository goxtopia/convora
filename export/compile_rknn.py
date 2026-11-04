"""Compile the final explicit Q/DQ graph with the historical RKNN 1.6 contract."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--onnx',type=Path,default=ROOT/'models/onnx/compatible.onnx');p.add_argument('--out',type=Path,default=ROOT/'build/model.rknn');a=p.parse_args()
    if not a.onnx.is_file():raise SystemExit('Download models/** from https://huggingface.co/heyanzhuo/convora first; see README.md.')
    from rknn.api import RKNN
    a.out.parent.mkdir(parents=True,exist_ok=True);assert not a.out.exists(),'Refusing to overwrite output'
    r=RKNN(verbose=True,verbose_file=str(a.out.with_suffix('.compiler.log')))
    try:
        assert r.config(target_platform='rv1103',optimization_level=0,quantized_algorithm='normal',quantized_method='channel')==0
        assert r.load_onnx(model=str(a.onnx))==0
        assert r.build(do_quantization=False)==0
        assert r.export_rknn(str(a.out))==0
    finally:r.release()
    print(json.dumps({'rknn_sha256':hashlib.sha256(a.out.read_bytes()).hexdigest(),'do_quantization':False,'optimization_level':0,'target':'rv1103'}))
if __name__=='__main__':main()
