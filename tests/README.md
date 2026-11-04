# 测试入口

## 离线测试

```bash
python scripts/verify_release.py
python tests/code_checks.py
# 以下需要先从 Hugging Face 下载模型与原始 fixture
python tests/offline.py --out outputs/offline.json
python tests/offline.py --metrics-only
```

`code_checks.py` 只需tokenizers，验证模板golden、FIFO排队/取消和冻结源码，不加载模型或访问网络。完整离线测试还需从 [Hugging Face](https://huggingface.co/heyanzhuo/convora) 下载 `models/**` 与 `tests/fixtures/**`（根README有命令）。fixture包含最终版本260步序列、实际NPU单步输入/输出和FP/QDQ/NPU闭环输出；本Git仓库不存放这些大文件。NRMSE按全体元素的L2误差/参考L2计算，报告JSON中的值为比例，测试总结表转换成百分比。这些已用诊断不能充当盲测题库。

`historical/` 是原始调查和三模式板测脚本，保留原文件，供审阅当时的最小图、穷举、观测与指标计算方法。它们依赖原研究目录的全部中间产物和历史机器路径，**不属于便携直接运行入口**。复核已选最终结果用 `offline.py`；板端源码在 `board/`。

## 编译板测程序

需要 Luckfox ARM uClibc hard-float 工具链，设置实际 `CXX`：

```bash
export CXX=/absolute/sdk/toolchain/bin/arm-rockchip830-linux-uclibcgnueabihf-g++
bash tests/board/build.sh
```

输出 `build/board/{delta-bench,delta-dump,runner}`。头文件 `rknn_api.h` 和正式 `models/rknn/engram_config.h` 已提交；`librknnmrt.so` 须从匹配的Rockchip SDK获取，可用 `RKNN_LIBDIR=/absolute/sdk/lib` 指定目录。`bench.cpp` 的 native状态读写、量化查表LUT与原实板测试一致。

离线编译不访问板子。运行前需单独准备一个全新板目录，避免覆盖原模型或测试证据。程序相对cwd读写文件：

| 程序 | 输入 | 输出 |
|---|---|---|
| `delta-bench 2048` | `models/tcn_engram.rknn`；`data/{embedding.f32,memory.i8,memory_scale.f32,sequence.i32,reference.f32,state-reference.f32}` | `results/metrics.json`、logits/state原始输出等 |
| `delta-dump` | 同上，另需 `data/diag-sequence.i32`、`data/diag-flags.i32`（首项3，其余2） | `results/input-feeds.f32` / `output-feeds.f32` / 位置 |
| `runner` | cwd下`model.rknn`、`inputs.f32`、`shapes.txt` | `actual.f32` |

下载后对应来源：模型/表取 `models/rknn/`；`sequence.i32`取fixtures同名文件；FP参考取`fp-closed-logits.f32`和`fp-closed-states.f32`并按上表重命名。runner所需模型与输入在Hugging Face，shape文本在本Git仓库。

`delta-bench`不负责停止聊天守护进程；板测必须独占NPU，检查实际服务身份/忙碌状态，并在结束时恢复原服务和governor。历史 `run_board.sh` 展示调频/恢复和采集顺序，保留在 `board/`；它会改governor并覆盖当前测试目录的results，不能在正式服务目录执行。本次没有运行任何板测或调整板端状态。

## Chat 功能测试

聊天代码在 `runtime/pico-chat/`。在该目录创建自己的 `.venv` 并安装 `tokenizers==0.23.3`；server使用Python标准库HTTP/SSE。`start.ps1` / `stop.ps1` 通过 `-AdbPath`、`-Serial` 指定实际设备，或使用 `ANDROID_SERIAL`；`board/build.sh`使用实际 `CXX` 和可选 `RKNN_LIBDIR`。板端 `S98picochat` 指向原部署目录，不会因复制此目录而自动安装。预编译daemon和厂商运行库不在Git中。

板端已经部署且转发/桥接运行后，可以手动运行：

```powershell
$env:PICO_CHAT_HOST='192.168.x.x'
$env:PICO_CHAT_PORT='9110'
runtime/pico-chat/.venv/Scripts/python.exe runtime/pico-chat/verify_lan.py
```

该验证会向当前聊天/NPU发送真实请求，测试排队、取消和历史隔离，不是离线检查；输出到`outputs/lan-verification/`。默认目标是127.0.0.1，跨设备LAN需配置实际地址。它会与保存的独立最终NPU输出比较前缀首token。历史验收快照在`reports/compatibility/chat-deployment/`，没有在此次整理中重跑。`verify.py`也是该最终验收的包装入口。
