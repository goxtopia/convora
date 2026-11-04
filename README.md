# Convora · 卷语

面向多种 NPU 的卷积语言模型，采用 TCN + Engram + Delta Lite。目前已在 Rockchip RV1103 / Luckfox Pico 上验证。

[模型权重](https://huggingface.co/heyanzhuo/convora) · [测试结果](TEST_SUMMARY.md)

## 模型

| 项目 | 配置 |
|---|---|
| 参数量 | 9.29M |
| 骨干 | 8 层 TCN，256 通道，6400 词表 |
| 记忆 | Engram + Delta Lite（4 头 × 32 维） |
| TCN 感受野 / 聊天输入上限 | 133 / 512 tokens |

## 快速开始

Linux / WSL，Python 3.10：

```bash
pip install -r requirements-cpu.txt huggingface_hub
hf download heyanzhuo/convora --local-dir . \
  --include 'models/**' 'tests/fixtures/**'

python scripts/verify_release.py --artifacts
python scripts/infer_float.py --prompt '你好' --max-new-tokens 80
python tests/offline.py --out outputs/offline.json
```

仅做源码检查可运行 `python scripts/verify_release.py`，无需下载权重。模型文件的 SHA256 见 [ARTIFACTS.json](ARTIFACTS.json)。

## 代码

| 目录 | 内容 |
|---|---|
| [modeling/](modeling/) | 模型实现、缓存推理、fakeQ 和 Triton 递推 |
| [training/](training/README.md) | 预训练、SFT、QAT 代码与配置 |
| [export/](export/README.md) | ONNX / RKNN 导出与兼容改图 |
| [tests/](tests/README.md) | 代码检查、数值回归和板测程序 |
| [runtime/pico-chat/](runtime/pico-chat/) | 聊天网页、桥接服务和板端 daemon 源码 |
| [reports/](reports/) | 实验结果与审计记录 |

## 当前结果

RV1103 上 NPU 定长闭环约 **286 steps/s**。260-token 闭环对浮点参考的 logits / state NRMSE 为 **4.40% / 14.48%**。严格量化精度和长记忆验收尚未通过，回答可能跑题或重复；详见[测试总结](TEST_SUMMARY.md)。

权重与原始诊断数据存放在 Hugging Face。RKNN 必须配合对应 CPU 表使用，Rockchip 运行库需从匹配 SDK 获取。
