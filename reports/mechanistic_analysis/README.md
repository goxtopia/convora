# Mechanistic Analysis Artifacts

本目录存放 [`ANALYSIS_REPORT.md`](../../ANALYSIS_REPORT.md)（仓库根目录）所引用的全部取证产物。

| 文件 | 内容 |
|------|------|
| `figures/fig1_architecture_and_params.png` | 架构、三套记忆系统数据流、CPU/NPU 参数分区 |
| `figures/fig2_logit_lens_and_dla.png` | 10 阶段 Logit Lens 与直接日志归因（DLA） |
| `figures/fig3_tcn_tap_collapse_and_erf.png` | TCN tap 能量塌缩与输入梯度有效感受野 |
| `figures/fig4_engram_forensics_and_collision.png` | Engram 桶范数取证、门控能量分配、剪枝消融 |
| `figures/fig5_delta_gate_saturation_and_echo.png` | Delta 门控饱和、状态范数膨胀、回声室行为学 |
| `figures/fig6_npu_quantization_and_deadzone.png` | INT8 衰减死区、状态冻结率、闭环 NRMSE |
| `reproduce_analysis.py` | 复现核心取证数字（P 正交化 / tap 能量 / Engram 考古 / IEEE-754 边界） |
| `reproduce_output.json` | 上述脚本的一次实际运行输出 |

运行复现：

```bash
python3 reports/mechanistic_analysis/reproduce_analysis.py
```

完整报告（含因果消融、ERF、行为干预与全部结论）见仓库根目录的 **[`ANALYSIS_REPORT.md`](../../ANALYSIS_REPORT.md)**。
