# P014 12B 行为 SFT 实跑：base 4/4，adapter 3/4，无新增修复

2026-10-20，北京时间。作业 `ccjjllt/p014-12b-behavior-pair-v1` COMPLETE，8/8 attempts 完成。运行 `run-014-12b-behavior-pair-v1-20261020T121707Z-0691ca19`，原始展开结果在 `/tmp/p014-12b-output-v1/<run>/`，根代理另保留下载 ZIP。

**结论：这份已完成的 P013 adapter 在本轮没有增加修复，Click 出现一次明确代码回退。暂停原配方继续训练与扩 epochs；31B 正式提交不等待 12B。** 四题为预先冻结的探索性开发子集，不能据一次净负断言 LoRA 总体无效；也不能把 base 4/4 外推为比赛成绩或模型大小优势。

## 实际修复结果与运行限制

| 问题 | base | adapter | 工具数 base/adapter | LLM calls base/adapter |
|---|---|---|---:|---:|
| Click 2604，命令名去后缀 | 通过 | **失败：UnboundLocalError** | 21 / 21 | 21 / 21 |
| MarkupSafe 402，缺少 re 导入 | 通过 | 通过，但陷入重复验证 | 26 / 28 | 28 / 80 |
| pydicom 1938，空 SQ 的 VM | 通过 | 通过 | 28 / 14 | 29 / 15 |
| pydicom 1867，CLI 空文件匹配 | 通过 | 通过 | 24 / 28 | 27 / 32 |
| 合计 | **4/4** | **3/4** | **99 / 91** | **105 / 148** |

W=0、L=1、净差 −1。8 个独立裁判均确认导入各自新工作树源码；不是系统旧安装包导致的差异。此次仅冻结 F2P 机制判据，`p2p_nodes=[]`、`p2p_pass=null`，没有独立回归套件证据。多个正确补丁在 Agent 超时/上下文错误后仍由 runner 回收并通过判据，因此不能用是否正常结束替代修复数。

原件：[逐题结果](evidence/analysis-rows.json)、[运行汇总](evidence/summary.json)、[运行条件](evidence/experiment-manifest.json)。同一 12B CT v2、同一两阶段 Agent 与预算，只给 adapter 臂 repair 挂 LoRA；最多4个隔离上下文共享 vLLM，属于并发研究条件。

## Click 的具体回退：改写闭包参数，验证命令又没有执行成功

原实现 `command(name=...)` 内部的 `decorator(f)` 读取外层参数 `name`。adapter 在内层函数插入：

```python
name = name or f.__name__.lower().replace("_", "-")
```

Python 将赋值目标 `name` 判为内层局部变量，右侧读取的是尚未赋值的局部变量，装饰函数时立即触发 `UnboundLocalError`。base 使用新的 `name_str` 和 `cmd_name`，保持外层 `name` 只读，通过判据。

adapter 修改还会对显式传入的命令名统一去后缀，存在额外契约风险；本轮裁判首先在闭包错误处失败，不能声称该额外风险已实际运行验证。

关键轨迹说明它为什么没有纠正：

- adapter 首次生产修改在 **第19个工具提议、297.54秒**；base 在第8个、166.33秒。此前 adapter 多次读取偏离目标区域的源码，重复 `grep`，并产生 shell 引号错误。
- adapter 开头的 `python3 -c '...; @command()...'` 已因装饰器语法出错。它转用文件复现过原始问题，但修改后在397.73秒再次发出同类非法装饰器单行命令，仍然是 `SyntaxError`。
- 401.33秒重写 `reproduce_issue.py`，却没能在8分钟内执行这个修后脚本。因此 Agent 没有观察到本可直接暴露闭包错误的反馈，也没有进入 reviewer。
- base 同样浪费过复现命令、经历一次编辑匹配失败，但读取真实区域后纠正，406.29秒的复现已输出四个正确名称。它随后清理本轮脚本，最终回收的生产补丁通过。

这次差异的直接机制是**错误的闭包修改 + 修后有效验证缺失**，不能只归为“训练不足”或泛称结束纪律。adapter 留下了 `reproduce_issue.py`，base 最终无该残留。

证据：[adapter 实际评分补丁](evidence/click-adapter-scored-patch.txt)、[base 实际评分补丁](evidence/click-base-scored-patch.txt)、[新工作树裁判异常](evidence/click-adapter-criterion-execution.json)。

## 实际 LoRA 路由正确，仍有严重预算浪费

服务命令明确包含 `--enable-lora --max-loras 1 --max-lora-rank 8 --lora-modules p013_behavior_lora=...`。最终权重和 config 与完成训练的指纹一致。四题 adapter 修复轨迹的 `model_name` 全为 `p013_behavior_lora`；进入 reviewer 的 pydicom 两题，review 的 model_name 均为 12B base。没有复发 P012“只挂协调器”的问题。

`model-requests.jsonl` 记录了 140 次 LoRA 的 `generate_content_async` 调用，以及 adapter 两题 reviewer 的16次 base调用；base臂全走base。此计数是生成方法调用，不等于成功完成的 HTTP 请求数或最终 `total_llm_calls`；trace 的角色/model_name 也独立支持路由结论。见[路由摘要与首次编辑](evidence/model-routing-summary.json)。

MarkupSafe adapter 在41.24秒已经补好 `import re`，45.55秒运行导入成功。之后同一 `import markupsafe; print('Success')` 命令却重复 **60次**。28个实际工具预算耗尽后，继续收到 **52次 BudgetExceeded**，最终跑满80 turns，没有交接给 reviewer。这是“修好后不会停止、把验证重复当进展”的真实行为缺陷，不能因为判据PASS而忽略。

其他限制：base MarkupSafe 在上下文达到24577输入、仍请求8192输出时超过32768上限；pydicom 1938 的 adapter 虽减少工具数并交接，但末尾出现工具调用格式文本，reviewer也没有正常完成。工具数下降不能直接解释为行为学习收益。pydicom 1867 adapter 比base多4个实际工具、5个LLM calls，且残留脚本更大。

## GPU 使用：推理阶段高利用率，加载拉低整体采样均值

四卡共享服务，`--max-num-seqs 4`。整受计时会话 **1321.28秒（22.02分钟）**，包含 bootstrap 安装、模型加载与归档；按历史2×倍率约 **0.734 配额小时**，是估算，不把共享账号账单差额归于本作业。

| 采样区间（从会话起计） | 时间点数 × GPU数 | 四卡 device utilization 均值 |
|---|---:|---:|
| 含加载的已采样区间：232.25–1312.90秒 | 214 × 4 | **76.26%** |
| 已采样的模型就绪前区间 | 47 × 4 | 6.52% |
| 模型就绪后：469.98–1312.90秒 | 167 × 4 | **95.89%** |

模型就绪时为468.48秒。首个生成调用北京时间20:21:54.505，最后一次调用20:35:39.576，调用窗口约825.07秒。模型就绪至会话结束约852.80秒；二者定义不同。前232.25秒没有GPU采样，76.26%不能称为全会话平均；95.89%也不是SM效率或有效修复吞吐。实际推理批处理已经保持高device utilization，下一步更应减少无效重复生成和错误命令，而不是只提高并发。

原始采样经一次性计算，摘要见[GPU采样口径](evidence/gpu-sampling-summary.json)。推理窗口平均每卡已用显存约20532 MiB。

## 对下一轮后训练的决定

暂不继续同一34条的 epochs、旧P012辅助层或未经修改的扩量配方。本轮没有新增修复，且具体弱点正是训练未覆盖的“定位搜索、错误命令纠正、修后失败恢复与停止条件”。已有34条全部固定为一次读取→修前失败→成功编辑→修后通过，不能把它们的 `behavior_recovery` 标签当作错误恢复训练证据。

下一次 SFT 若继续，应从训练划分补入每处修改前的源码观察、自然发生的错误及纠正、已验证后的明确停止交接，并保留未完成/验证未完成分支；不能把本轮4题或新8题的题目、补丁或专用判据转为训练。先明确新数据增加了什么有效行为，再投入训练。当前提交主线继续选31B实跑候选。

## 补充：结束文本参与监督，模板结束 token 被 mask

只读检查真实冻结编码文件（SHA `f0354c13…`）：34/34条的最终handover正文均在目标assistant steps和content spans中，不能说“没有监督结束交接文本”。但模板的 `<turn|>`（token ID106）全部标为 `-100`，没有参与loss；数据中也没有 `<eos>` 监督。编码器仅选 content/tool action 的差分字段，模板结束符位于字段外，因此被当前mask策略排除。

这说明“生成交接正文后如何结束一轮”的显式训练信号缺失，值得在下一版编码设计中研究；**尚不能证明它导致本次60次重复导入**。本次没有修改编码、标签或权重，也不据此立即重训。统计见[真实结束token标签观察](evidence/training-end-token-observation.json)。
