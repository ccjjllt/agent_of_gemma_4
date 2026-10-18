# 2026-10-18 Agent 与公共 12 题配对复核

复核 HEAD `be9b27b`，相对研究基线 `5a979ef`。只读检查已有 24 份 `result.json`、`context_metrics.json`、harness 日志与 patch，重点回读相关 trace、评测输出及两份实际 ZIP；没有运行模型/裁判、提交或读取新的任务正文/参考 patch/heldout 内容。以下结论不涉及另一分支负责的 12B 训练效果。

**建议先提交原样 `public022_v10`，不必等训练。** 两份候选均已有可复用 ZIP、同版本真实 compile 和完整模型运行证据；公开来源 0.22 为原样候选提供外部锚点，本地此前归档的正式分数是 0.13，最新账号状态由主代理另外核实。`behavior_v1` 是值得后续正式探索的候选，但本轮仅一对独胜且环境失真明显，不足以确认稳定优于原样版本。若原样候选尚未正式提交，一次新 submission 的目的应是获得官方环境下自己的成绩，不能承诺 0.22 或把 4/12 外推成公榜 0.33。

## 原件与可提交候选身份

运行根目录：`research/007/public12-pair-20261017/gpu-output-v2/results/007/run-007-public12-pair-20261017T063527Z-5d0f2c69/`，下文简称 `RUN/`。

`RUN/joint_summary.json` 与全部 24 份单题 result 一致：12 题、每臂一次，A=`public022_v10` 3/12，B=`behavior_v1` 4/12；唯一 B 独胜 `rich_3180`，无 A 独胜。这是**本次结果**，不是确定性、可重复或泛化优越性。1 个 discordant 的双侧 exact paired sign test p=1，样本不足以排序。

对实际 ZIP 逐成员核对：

| 候选 | ZIP SHA256 | 与当前候选源码 |
|---|---|---|
| 原样 public022_v10 | `640b8abc9c99e36aba8163bba84778e148698505da161e9fbe906386fb98d809` | 四文件全部字节一致 |
| behavior_v1 | `c12870bf49305f8b7d9cede7ed2da4df25a8cd2f3c5c19950ae097e518502990` | 四文件全部字节一致 |

ZIP 位于各臂 `RUN/<arm>/r1/agent-bundle-007-public12-<arm>.zip`，根目录含 `agent.yaml`，其余为 `configs/sampling.yaml`、`eval_config.yaml`、`prompts/system.md`，没有训练权重或任务材料。两臂仅 prompt 不同：替换 **Locate 的一个行为复现 bullet 与 Verify 的三个 bullet**；Edit 节没有变动。原报告所述“Locate、Read、Verify 三小节”不准确。

两臂均保留 31B CT v2、temperature .2/top-p .95、8192 输出/4096 thinking、原 8 分钟/28 calls/80 turns/240 秒命令超时。实际 worker 用 v31 harness，另固定 verification 300 秒、subprocess sandbox；一个服务、四 worker 滚动并发。因此这是 prompt 因子的对照，但不是正式 scorer 串行时延复现。

## 重新计算的事实

| 指标 | A 原样 | B 行为版 | 解释 |
|---|---:|---:|---|
| result.resolved | 3/12 | 4/12 | 唯一独胜 rich_3180 |
| resolved 且非空生产文件修改 | 2/12 | 4/12 | A fastapi_14851 空 patch 也过；该题不能证明 B 的新增修复收益 |
| 非空 patch | 7/12 | 9/12 | 包括只有 scratch 脚本的输出 |
| 实际含 `rich/`、`fastapi/` 或 `src/requests/` 生产修改 | **5/12** | **8/12** | 不是原报告的“生产补丁 7/9” |
| harness 明确记录 agent timed out | **6/12** | **5/12** | B 不能按 result.error 计成 3/12 |
| timeout 或明确 tool-budget error | **6/12** | **6/12** | B 另有 rich_3472，双方均 50% |
| context.agent_elapsed_seconds 平均 | **364.04s** | **380.25s** | B +16.21s；与含 grading 的 duration 分开 |
| result.duration_seconds 平均 | 381.64s | 422.99s | 包含 setup/verification；B rich_4006 含 300 秒裁判超时 |
| 至少一次主动 submit_patch | 7/12 | 6/12 | 不等于非空提交或成功完成 |

`context_metrics.json` 和 harness 原件揭示漏计原因：B `06-rich_3180`、`11-fastapi_14851` 均 agent 运行约 480 秒并明确记载 `agent timed out`，`patch_submitted=false`；随后恢复工作树补丁并通过评测，导致 `result.error=null`。按 error 字段统计会漏掉这两次超时。因此“收尾改善 50%→33%”没有原件支持。

总 supervisor 3397.60 秒，模型启动 333.41 秒。`coordinator.py:454` 的 `batch_start` 在 `server.start()` 之前，故 `batch_wall_seconds=3291.30` **已经包括服务启动**，不能再相加，也不宜称为纯 24 次 agent 批处理耗时。四并发下 agent 耗时包含共享模型排队，不能直接用于正式 sequential 12 小时预算预测。

## 实际机制：有收益迹象，也有明显失败

### rich_3180：正确生产编辑成功，行为闭环并未完成

证据：`RUN/behavior_v1/r1/attempts/06-rich_3180-r1-behavior_v1/traces/trace_rich_3180.json` 的 step 7–11、同目录 `harness.log:2`、`context_metrics.json`、`patches/rich_3180.patch`。

- step 7 在 174.94 秒写 `repro.py`；step 8 运行后只打印关键 mixed-width 输入当前返回 `[1]`，该关键例没有正确结果 assert。
- step 10 在 **415.41 秒**才首次修改生产 `rich/cells.py`；step 11 在 **474.80 秒**又写 `repro_2.py`，里面是含 `pass` 的占位函数。
- 没有修复后运行验证，没有 `submit_patch`；agent 最终超时，恢复 patch 后官方测试 **114 passed**。
- 最终 patch 含 `repro.py`、`repro_2.py` 和生产改动。

因此可以说“B 在本次及时落地了正确修改，A 只有 repro.py”；不能说“正反合同断言与独立验证已形成确定性闭环”。成功的 production edit 是真证据，其解释应收窄。

### requests_7205：后半段的导入定位是真机制证据

证据：`RUN/behavior_v1/r1/attempts/18-requests_7205-r1-behavior_v1/traces/trace_requests_7205.json`。

step 10 曾用自造 `get_netrc_auth_fixed` 替身验证，不能代表真实源码；step 11 修改 `src/requests/utils.py`；step 12 运行真实 API 仍显示旧行为；step 14 自查 `requests.__file__`，输出系统 `/usr/local/lib/python3.12/dist-packages/requests/__init__.py`；step 15 改用 `PYTHONPATH=/workspace/src` 后观察到空凭据返回 `None`、正常凭据保持原值；step 17 非空提交。这个“查清证据执行的是哪份代码”比抽象地称作验证增强更有依据。

但独立 verify 的输出依然是 `1 failed, 205 passed, 13 skipped`，关键断言拿到 `('', '')` 而不是 `None`。这支持当前 subprocess 评测存在 import 路径失真的诊断，不支持未经修正复评就把该题手工改记成功；本复核没有读取 gold patch，也不沿用“100% 正确补丁”措辞。

### 其他需要保留的负面证据

- **rich_4006**：双方无 shell 短超时的死循环复现各卡住 240 秒；B 首次编辑在 464.65 秒，只够修部分分支，最终裁判超时。应限制故障复现自身耗时，而不是简单增加总预算。
- **requests_7502**：A patch 修改 `src/requests/models.py`；B patch 只有 `repro.py`。二者裁判都失败且有 196 个 fixture 错误，但“零回退”最多指本次 resolved 表，不能扩展为生产编辑层没有回退。
- **fastapi_14482 B、fastapi_14459 A** 的 `test_outputs/*.log` 明确是 `ModuleNotFoundError: inline_snapshot` 收集失败；不能由此推断 agent 修复正确，也不适合计入可靠质量排序。
- **requests_6757**：最终两臂 patch 都为 0。A/B 分别调用 submit_patch 11/10 次，反复收到 `patch_size=0, files_changed=0`。即使因保护配置文件导致任务不可交付，重复提交仍浪费时间，不能当成有效完成。
- **补丁污染**：B 成功的 rich_3180 与 fastapi_14463 均携带 scratch；B requests7427 亦有 repro，requests7502 只有 repro。提示中“never create a file”没有可靠落实。

## 对提交/训练排序的判断

1. **原样 0.22 候选已可用于下一次正式提交。** 已有真实运行与冻结 ZIP，不需要先训练才能提交；官方提交后的分数属于我们，外部 0.22 仍只是来源记录。
2. **B 可作为随后的探索提交，但本地证据不足以替代 A 成为已证实更优方案。** 本轮只有一对独胜、一个空补丁通过、多个环境不可靠题，且更慢、收尾超时优势消失。不能以“确定性净增零回退”为理由直接推广。
3. **先修正解释，再修正有效评测环境，避免为错误反馈训练。** 针对已暴露开发题的 `src/` import、缺依赖和不可修改配置路径问题应单独处理；未校准的 false negative/false positive 不宜进入奖励或监督筛选。没有必要在此再堆新的测试/门控体系，也不需要用一轮大训练来回答这次候选提交问题。

本报告仅写入 review 文档；没有修改候选、原结果或计划。
