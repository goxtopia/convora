# 导出与工具链

本GitHub仓库只存代码。运行下列命令前，按根README从 [Hugging Face](https://huggingface.co/heyanzhuo/convora) 下载 `models/**`（I/O闭环核验另需`tests/fixtures/**`），并用 `python scripts/verify_release.py --artifacts` 校验。最终已板测RKNN保存在HF，不进入Git历史。

在独立 Linux/WSL CPU 环境使用 Torch 1.13.1、NumPy 1.23.5、ONNX 1.14.1、ORT 1.16.0。RKNN 转换需原 `rknn-toolkit2 1.6.0+81f21f4d` 环境；运行库与 driver 的历史条件见报告。没有升级 SDK，也不假设新版兼容性。

## 已有最终图

```bash
# 从原正式 QDQ 图重做兼容改写；核验与已交付兼容图 SHA 完全相同
python export/compatible.py

# 不做第二次 PTQ；沿用显式量化参数，optimization_level=0
python export/compile_rknn.py
```

输出在 `.gitignore` 已忽略的 `build/`。原学习 initializer 保持字节不变，新增固定 depthwise 核和12个reshape。饱和 Conv 改写不具备严格全图等价性，见测试总结。`compile_rknn.py` 不包含板端上传或服务切换。

本次重编译成功，原 RKNN 和新 RKNN 同为 4,694,317 字节，仅文件前64字节有14个字节不同，其后全部字节相同；未解释专有头部字段的语义。**正式发布仍保留原已板测 SHA6942f512…文件**，新编译文件只作为转换入口验证，不宣称已重新上板。

## 从 float 权重重导出

```bash
python export/prepare_deploy.py \
  --weights models/float/qat-final.pt \
  --out build/reexport-qat \
  --calibration training/qat/historical/calibration.json --qat

python export/fix_and_verify_io.py \
  --root build/reexport-qat --weights models/float/qat-final.pt

python export/compatible.py \
  --source build/reexport-qat/step.onnx --out build/reexport-compatible.onnx

python export/compile_rknn.py \
  --onnx build/reexport-compatible.onnx --out build/reexport-model.rknn
```

`prepare_deploy.py` 用这48个 train 文档重新收集骨干/head PTQ范围，导出431组校准输入，并从该次权重重导 CPU 表。Delta网格来自正式QAT。不要复制旧骨干的表；不要使用val作校准。由于环境、图序列化或范围收集变化，重导出产物需重新核验，不能仅凭权重SHA冒称原图。`fix_and_verify_io.py` 把 int8 激活等价转 u8，并公开量化输出；软件参考必须关闭 ORT graph optimization，避免默认量化融合的舍入差异。

本次以上准备、I/O修复和兼容改图均已实跑；Q/DQ I/O软件260步修复前后 logits/state 最大差0，重新生成的 CPU 表与正式表 SHA 相同。重导出的图数值合同和最终图对比记录在 `reports/package-build-validation.json`。

普通 PTQ 对照可用 `sft-before-qat.pt` 或 `qat-final.pt` 不加 `--qat` 准备，再执行 `build_rknn.py --root build/<variant>`；它使用 `do_quantization=True` / normal / channel / level3。该路线只作基线，不包含最终两处兼容修复，不能当最终部署版。不要对显式 QDQ 图再次启用 PTQ。
