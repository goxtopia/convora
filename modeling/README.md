# 模型实现索引

`training/model.py` 实现 TCN 和 Engram：256通道、8个残差层、dilation `[1,2,4,8,16,32,1,2]`，第3层后注入Engram；6400词表。Engram包含4张64维哈希表，bucket `[19997,20011,20021,20023]`，n-gram阶数 `[2,2,3,3]`。

`delta_layer.py` 定义4头×32维的部署适配Delta层。每头状态为32×32，按 `[1,K,1,V]` 布局：先从旧状态读取value，用alpha衰减旧读出，按beta写入误差外积，再用query读取新状态。Q/K有界并除以sqrt(32)，V有界至±4；gate使用HardSigmoid。`delta_model.py` 是全序列浮点实现，`sft_model.py` 支持监督位置稀疏head。

`qat_delta.py` 提供fakeQ、校准网格和逐步递归参考；`qat_model.py` 将state in/out从当前已训练模型重新校准，不能只看`qat_delta.py`内的早期默认常量。`fused_delta.py` 是用于FP训练的Triton前向/解析反向；FP32累积状态，**不模拟量化状态**。QAT实际走自己的逐步fakeQ路径。

`load.py`严格加载交付部署包，`FloatDecoder`逐token维护TCN ring、Engram前词及4个Delta状态；reset同时清空三者。默认使用float Engram表；`load_float(deployment_tables=True)`改用与NPU配对的int8反量化表以便同输入比较。`models/float/qat-final.pt`包含网格，但此加载入口明确执行FP运算。

这些模型核心文件来自正式QAT源码清单，便携加载器为本次新增。`training/train.py`仅提供既有可信checkpoint兼容加载工具；实际三阶段训练入口位于顶层`training/`。兼容Conv/向量布局是部署图变换，训练fakeQ尚未使用该Conv endpoint语义。两者误差、限制和实测数据见测试总结。
