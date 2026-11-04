import argparse,subprocess,shutil,json,hashlib
from pathlib import Path
from rknn.api import RKNN
HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1]
p=argparse.ArgumentParser();p.add_argument('variant');a=p.parse_args();root=HERE/a.variant
r=RKNN(verbose=True,verbose_file=str(root/'compiler.log'))
try:
    assert r.config(target_platform='rv1103',optimization_level=0,quantized_algorithm='normal',quantized_method='channel')==0
    assert r.load_onnx(model=str(root/'step.onnx'))==0
    assert r.build(do_quantization=False)==0
    assert r.export_rknn(str(root/'model.rknn'))==0
finally:r.release()
compiler='/home/wuyupeng/luckfox-minimind/luckfox-sdk/tools/linux/toolchain/arm-rockchip830-linux-uclibcgnueabihf/bin/arm-rockchip830-linux-uclibcgnueabihf-g++'
subprocess.run([compiler,'-O2','-std=c++17','-static-libstdc++','-static-libgcc',str(HERE/'runner.cpp'),'-I',str(PROJECT/'pico-chat/board'),'-L',str(PROJECT/'pico-chat/board'),'-lrknnmrt','-Wl,-rpath,/oem/usr/lib','-lpthread','-ldl','-lm','-o',str(HERE/'runner')],check=True)
(root/'build.json').write_text(json.dumps({'onnx_sha256':hashlib.sha256((root/'step.onnx').read_bytes()).hexdigest(),'rknn_sha256':hashlib.sha256((root/'model.rknn').read_bytes()).hexdigest(),'toolkit':'1.6.0+81f21f4d','do_quantization':False,'optimization_level':0},indent=2))
print('BUILD_COMPLETE',a.variant,flush=True)
