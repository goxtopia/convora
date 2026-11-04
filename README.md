# Convora · 卷语

**面向多种 NPU 的卷积语言模型。** Convora 使用 CNN 友好的算子组合 TCN、Engram 和 Delta Lite，探索小型语言模型在资源受限设备上的训练与部署。

当前已经验证 Rockchip RV1103 / Luckfox Pico。其他 NPU 是适配目标，尚未完成实测。项目名称不绑定具体芯片；历史部署身份保留以核验已有权重与实板结果。

本仓库包含模型实现、三阶段训练代码、导出/测试/Chat 源码、配置和实验报告。**权重、ONNX/RKNN、CPU 表与大型原始诊断数据不进入 Git 历史**，在 [Hugging Face：heyanzhuo/convora](https://huggingface.co/heyanzhuo/convora) 提供。归档实验截至 2026-10-29，整理发布于 2026-11-04（Asia/Shanghai）。

**最终版本是 `delta4x32-lccc-sft-qat-compatible-20261029`。** 它解决了两处已复现的 RKNN 算子/布局问题，仍有量化误差；严格 logits 门槛和长记忆任务尚未通过。完整测试结论见 [TEST_SUMMARY.md](TEST_SUMMARY.md)。历史报告的“未部署/停服/board_pending”是当时的阶段快照，后续部署以 `reports/compatibility/chat-deployment/` 为准。此处不把历史 health 当作当前在线状态。

| 目录 | 内容 |
|---|---|
| `models/` | tokenizer、模型身份/权重 SHA 元数据和 C++ 配置；大文件从 Hugging Face 下载到对应位置 |
| `modeling/` | TCN / Engram / Delta、fakeQ、CUDA Triton 递推、CPU 加载和缓存推理 |
| `training/` | pretrain / SFT / QAT 三阶段的实际冻结源码子集、配置与原始控制脚本 |
| `export/` | 从权重导出 ONNX / QDQ、修复 I/O、兼容改图及 RKNN 转换 |
| `tests/` | 无权重代码检查、离线数值回归、板测 C++ 及历史调查脚本；原始 fixture 从 Hugging Face 下载 |
| `runtime/pico-chat/` | 网页、Windows 桥接/FIFO、板端 daemon 源码；SDK 库与生成的二进制不提交 |
| `reports/` | 训练审计、CE/指令诊断、量化/板测/局域网验收，以及本次离线复核 |
| `provenance/` | 原始完整源码清单和归档复制来源；`MANIFEST.json` 校验代码，`ARTIFACTS.json` 校验外部大文件 |

## 最终文件身份

下表文件托管在 Hugging Face，下载后使用这些相对路径；本 GitHub 仓库仅保存 SHA 元数据。

| 文件 | SHA256 |
|---|---|
| `models/float/qat-final.pt` | `c87ba126a71f6bdb34515dde18455d28439e9c6ef62d925f051e85cada36ad06` |
| `models/float/sft-before-qat.pt` | `d8d8946f094c48b700d4a5b1dcb1cf8b57a11f1150ae6936b47b168069ba259c` |
| `models/rknn/model.rknn` | `6942f512bbc74c4d1ddd309fbd8cc580d959520d772c56c6098e0e38d8291175` |
| `models/onnx/compatible.onnx` | `0dd4bf93a825a87c90914f3135626aab6de5b3655090de9c2b2c1215861ecf94` |

`qat-final.pt` 是完整模型的精简部署包：包含本次更新后的 backbone、Delta 学习参数和 QAT 网格。以 FP32 运算加载即为最终同权重 float 参考；不是兼容 Conv 图的逐值等价浮点版本。其原始完整训练 checkpoint SHA 为 `9db7a2fa06b2687bfaa8091b9814bbf6ce52c7aba64d45bd66af933a902adef5`。`sft-before-qat.pt` 来源于 SFT 完整 checkpoint `7ca29f788efaf7cbdd6300e8c326d887617373b6dabc4483a6fa034afdad90aa`。

两个 `.pt` 均去掉了 optimizer/RNG/游标，适合推理、重导出和参数检查；**不能用来原样断点续训**。原始训练完整恢复包仍在服务器，路径和 SHA 见训练文档；本次没有下载或改写它们。

## 快速验证与推理

只校验文件与 Python 语法，任何现代 Python 均可：

```powershell
python scripts/verify_release.py
# 安装 tokenizers 后可运行，无权重、无网络和板端请求
python tests/code_checks.py
```

安装 Hugging Face CLI（`huggingface_hub`），在仓库根目录下载需要的外部文件：

```bash
hf download heyanzhuo/convora --local-dir . \
  --include 'models/**' 'tests/fixtures/**'
python scripts/verify_release.py --artifacts
```

仅需推理时可只下载 `models/**`。保留 `--include`，避免用模型卡覆盖本仓库 README。Rockchip 运行库须另从匹配 SDK 获取。

Linux / WSL CPU 环境使用 Python 3.10，独立安装 `requirements-cpu.txt`；RKNN 使用独立的原工具链环境，CUDA 训练另用独立环境。下载外部文件后可运行以下命令；归档版本已在 Torch 1.13.1 / NumPy 1.23.5 / ONNX 1.14.1 / ORT 1.16.0 CPU 环境实跑：

```bash
python tests/offline.py --out outputs/offline.json
python export/compatible.py
python scripts/infer_float.py --prompt '你好' --max-new-tokens 80
```

`tests/offline.py` 不需要 tokenizer、RKNN 或板子，会重算历史指标、加载两份浮点模型、比较四个前缀的缓存/完整前向、检查 reset、回放全部 260 组 ONNX 输入。`--metrics-only` 只需 NumPy。文字推理需要 `tokenizers`；也支持 `--ids-json` 读取 JSON token 列表，省去文字编码/解码依赖。聊天模板与 `runtime/pico-chat/server.py` 相同。

## 训练、导出与板端

复现流程和数据前提见 [training/README.md](training/README.md)，完整导出命令见 [export/README.md](export/README.md)，板测与聊天运行见 [tests/README.md](tests/README.md)。整理目录本身不会训练、连接板端或启动聊天。

正式 RKNN 必须与本目录 `models/rknn/` 的三份 CPU 表同时使用。512 是聊天输入限制，TCN 感受野为 133；Delta 状态能递归并不等于已验证长记忆能力。SDK 1.6.0+81f21f4d、driver 0.9.2、12 MiB CMA 的历史测试条件没有变化。

## Git 提交

`.gitignore` 排除所有模型权重、ONNX/RKNN、浮点/量化 CPU 表、原始数值 fixture、SDK 二进制、构建输出、训练数据、凭证与缓存。大型文件由 Hugging Face 保存，`ARTIFACTS.json` 记录完整 SHA256；本仓库不包含它们的 Git LFS 指针。

`scripts/verify_release.py` 默认验证代码清单和冻结训练源码；`--artifacts` 额外要求所有 Hugging Face 大文件存在且 SHA 正确。已有本地完整归档继续保留，不因本次分仓发布删除权重。

本目录未为既有权重、数据来源或 Rockchip SDK 重新指定开源许可证；厂商头文件/运行库保留其原始声明。
