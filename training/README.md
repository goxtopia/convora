# 三阶段训练源码

`pretrain/source/`、`sft/source/`、`qat/source/` 是对应正式 run 的依赖子集，每个文件都与原训练 SHA 清单核验。pretrain/SFT 的 `training/train.py` 从原始基线和当时已保存的变换脚本恢复，字节 SHA 与正式清单相同；QAT 的分桶参数及 FP32 graph 也恢复到其实际版本。`source-manifest.json` 列出交付子集，`provenance/*-historical-source-manifest.json` 保留当时完整清单。

`historical/` 保留实际控制脚本和配置，含当时服务器目录。它们用于审阅原流程；直接运行历史 launcher 需要原目录树。便携入口是 `scripts/prepare_training.py`，只准备新的 run，输出实际训练命令，不自行训练。它修改路径和源码清单合同，**不能冒充原历史 run 的原样 resume**。

## 数据和权重前提

原始语料/token shard 没有放入 Git；本目录包含 tokenizer、校准 48 文档的 token ID 和验证 fixtures。完整训练需另外提供原数据及 manifest，manifest 中各 shard 的绝对路径必须存在。不能把 48 校准文档或 260 验证 token 当训练集。SFT 数据叶文件原 SHA 账本在 `reports/sft/data/asset-sha256.json`；数据指纹、完整预算和审计见 `reports/`。

三阶段使用的完整 checkpoint：

| 用途 | 原服务器文件 | SHA256 |
|---|---|---|
| SFT 父权重 | `/workspace/tcnengram/training-runs/delta-fp-pretrain-1epoch-fast-20261029/pretrain/last.pt` | `0fc4252d0245a516a76dca6e23395c1609b85800844bc1b753ea9a1be5d9d6d6` |
| QAT 父权重 | `/workspace/tcnengram/training-runs/delta-lccc-sft-2epoch-20261029/sft/last.pt` | `7ca29f788efaf7cbdd6300e8c326d887617373b6dabc4483a6fa034afdad90aa` |
| 最终可恢复 QAT | `/workspace/tcnengram/training-runs/delta-sft-qat-30m-20261029/train/full-last.pt` | `9db7a2fa06b2687bfaa8091b9814bbf6ce52c7aba64d45bd66af933a902adef5` |

这些完整文件仍留在原服务器。本目录两份 float 是去 optimizer 的部署包，不可代替上表的完整父断点；尤其 QAT 继承 SFT AdamW/step 123974。QAT 分拆 `backbone-last.pt` 和 `qat-last.pt` 必须成对；本目录部署包已包含整套匹配参数。

## 新 run 准备示例

在 Linux CUDA 环境安装匹配 GPU 的 PyTorch/Triton，以及 `requirements-train.txt`。当时没有完整 CUDA pip lock，本次没有 GPU 重训；源码和训练配方准确归档，不声称在任意新版 PyTorch 上位级复现。

```bash
python scripts/prepare_training.py pretrain \
  --data-manifest /absolute/data/pretrain/manifest.json

python scripts/prepare_training.py sft \
  --data-manifest /absolute/data/mixed-sft/manifest.json \
  --asset-manifest reports/sft/data/asset-sha256.json \
  --parent-checkpoint /absolute/checkpoints/pretrain-last.pt

python scripts/prepare_training.py qat \
  --data-manifest /absolute/data/mixed-sft/manifest.json \
  --parent-checkpoint /absolute/checkpoints/sft-last.pt
```

命令会核验数据指纹、shard 字节数和父 checkpoint SHA，创建 `work/training-runs/release-<stage>/` 并输出命令与 cwd。新 run 不覆盖已有目录；QAT 会在该 work 中复制完整父断点到原训练代码要求的位置。准备后由操作者显式执行打印的命令。数据清单使用绝对路径，复制或重新映射数据后需重新做数据 SHA 审计。

原训练配方：

| 阶段 | 精度 / batch | 优化器 / 预算 |
|---|---|---|
| pretrain | BF16，micro128 × 2，有效256，block256，完整前缀宽768 | 从零 AdamW，lr8e-4，1 epoch |
| SFT | BF16，有效256，同更新内按160/256/512/768分桶、监督头稀疏化 | 新建 AdamW，lr1e-4，旧混合数据2 epoch |
| QAT | FP32 / TF32 off，有效256，新分桶micro128/32/16/8 | 继承 SFT AdamW，固定lr1e-5，累计1800秒 |

各阶段均全参数解冻，12 GiB allocator 上限。前两阶段 3% warmup、cosine 到0.1；betas(.9,.95)、eps1e-8、weight decay0.1（bias/gain无decay）、clip1。QAT 只 fakeQ Delta，骨干/head 后续 PTQ。最终兼容 Conv 的 V endpoint 语义尚未接入训练 fakeQ；不能继续沿旧失败导出路径盲目追加 QAT 来宣称修复。
