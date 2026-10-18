# P007 公共 12 题配对评测：public022_v10 vs behavior_v1

本报告于 **2026-10-18** 按原始 result、context_metrics、harness 日志、trace 和 patch 复核更正。原始产物保持不变，官方 resolved 仍是 **3/12 对 4/12**。这是一次开发集配对观测；不能称为确定性增益，也不能据此预测正式公榜分数。详细复核见 [agent-review.md](../../008/review-20261018/agent-review.md)。

- GPU 执行：2026-10-17 UTC 06:35:27–07:32:05，`ccjjllt/p007-public12-pair` V2，4×L4，24/24 attempts 完成。
- run_id：`run-007-public12-pair-20261017T063527Z-5d0f2c69`。
- 原件根目录：`gpu-output-v2/results/007/run-007-public12-pair-20261017T063527Z-5d0f2c69/`，下称 `RUN/`。
- CPU prepare V1 完成，两臂 compile 成功，151.03 秒。GPU V1 误分配 2×T4 后在 16 秒退出，未启动模型；V2 使用显式 NvidiaL4 完成实验。

## 1. 实际控制条件与耗时口径

A 为外部公开 0.22 的原样 `public022_v10`，B 为 `behavior_v1`。逐成员核对，`agent.yaml`、`configs/sampling.yaml`、`eval_config.yaml` 完全一致，只有 prompt 不同。B 替换 **Locate 的一个行为复现 bullet 与 Verify 的三个 bullet**；Edit 节未变，没有单独的 Read 节。改动同时涉及行为证据、证据失败诊断和修复时机，不能把单一 prompt 因子进一步解释成某一句话的独立收益。

两臂均为 31B CT v2、temperature=0.2、top_p=0.95、8192 output、4096 thinking、8 分钟、28 counted calls、80 turns、240 秒命令超时。没有额外 skills；原九工具声明保持不变。v31 harness 的 verification 另固定为 300 秒，sandbox 为 subprocess。12 个已选择开发题每臂仅一次，AB/BA 交替，一个模型服务、最多四 worker 滚动并发。

- prompt SHA：A `7f8dde62578a247ffbfc1a519b712ae65849fcfae1ee70c49e0628cab74cbef6`；B `079af84132a2931adf2aa84d48bf15407dce92aa07c4dfa6cf330e0ce146bdca`。
- 模型启动 333.41 秒；`batch_wall_seconds=3291.30` **已包含模型启动**，因为 coordinator 在 server.start 之前设置 batch_start。不能再把两者相加。
- Supervisor 总墙钟 3397.60 秒（56.63 分钟）。四并发下的批次耗时、单题排队时间均不等于正式 scorer 串行耗时。
- `context_metrics.agent_elapsed_seconds` 是 agent 阶段；`result.duration_seconds` 还包含 setup/verification。正式预算分析应区分两者，不能将含 300 秒裁判超时的 duration 直接外推。

## 2. 更正后的配对汇总

| 指标 | A public022_v10 | B behavior_v1 |
|---|---:|---:|
| 完成 / 计划 | 12/12 | 12/12 |
| 官方 resolved | **3/12** | **4/12** |
| resolved 且有非空生产修改 | 2/12 | 4/12 |
| 非空 patch（含 scratch-only） | 7/12 | 9/12 |
| 实际含生产文件修改的 patch | **5/12** | **8/12** |
| harness 记录的 agent 会话超时 | **6/12** | **5/12** |
| 明确 tool-budget error | 0/12 | 1/12 |
| 会话超时或 tool-budget error | **6/12（50%）** | **6/12（50%）** |
| agent 阶段平均耗时 | **364.04s** | **380.25s** |
| 含 setup/verification 的平均 duration | 381.64s | 422.99s |
| 含 setup/verification 的中位 / P95 duration | 465.85s / 488.61s | 454.62s / 786.03s |
| 平均工具调用 / LLM 调用 | 22.75 / 25.17 | 21.25 / 24.58 |

生产文件口径为最终 patch 中存在 `rich/`、`fastapi/` 或 `src/requests/` 修改；repro.py 等不算生产修改。P95 使用 nearest-rank，因此 n=12 时等于最大值。

配对四格仍为：双方成功 3、B 独胜 1、A 独胜 0、双方失败 8。唯一 B 独胜是 `rich_3180`。只能说“本次未观察到 resolved 回退”；只有一对 discordant，双侧 exact paired sign test p=1，不足以确认稳定优势。

B 的 `rich_3180` 和 `fastapi_14851` 都明确超时，但恢复工作树 patch 后通过，故 result.error=null。此前仅按 error 字段统计漏掉这两次，将 B 超时误记为 3/12。更正后的 JSON 保留 error 原值，为全部尝试独立增加 `agent_timed_out` 等派生字段。

## 3. 逐题结果与证据边界

| 题目 | A resolved / patch / duration | B resolved / patch / duration | 核实后的解释 |
|---|---|---|---|
| rich_4006 | False / 0B / 484.51s | False / 709B / 786.03s | 两臂无短超时复现各阻塞 240 秒。B 464.65s 才编辑一处分支，裁判又超时；不是完整修复 |
| rich_3067 | True / 1338B / 244.51s | True / 655B / 224.49s | 双方通过；B 13 tool calls，A 28。A patch 携带 repro.py，B 为较干净的生产修改 |
| rich_3180 | False / 499B / 485.75s | **True / 2761B / 485.70s** | B 正确修改生产文件后超时恢复成功；无修复后验证、无 submit_patch，含两个 scratch 文件 |
| rich_3472 | False / 0B / 482.41s | False / 0B / 307.26s | A 会话超时，B 工具预算耗尽，均没有最终生产修改 |
| fastapi_14482 | False / 0B / 485.27s | False / 1541B / 338.27s | B 有生产修改；裁判收集失败缺 inline_snapshot，不能据此判断修改是否正确 |
| fastapi_14851 | **True / 0B / 226.84s** | True / 526B / 489.21s | A 空 patch 通过，说明当前环境不能充分区分该题；B 超时恢复后通过，不能算新增修复胜场 |
| fastapi_14463 | True / 745B / 171.77s | True / 1900B / 302.84s | 双方修改生产文件并通过；B 更慢且包含 reproduction.py |
| fastapi_14459 | False / 1133B / 488.61s | False / 0B / 485.17s | A 只有 repro.py、裁判缺 inline_snapshot；B 无 patch。二者都未落地生产修改 |
| requests_7205 | False / 503B / 266.21s | False / 503B / 214.83s | 两 patch 相同；B 诊断系统 requests 导入遮蔽并用本地 src 验证。独立裁判仍失败，保留 False，不改记成功 |
| requests_6757 | False / 0B / 449.28s | False / 0B / 424.96s | 涉及受保护 tox.ini；最终 patch 均为 0，A/B 反复 submit_patch 11/10 次仍未交付 |
| requests_7427 | False / 1089B / 484.34s | False / 2442B / 484.28s | 均有生产修改并超时；B 另带 repro.py。src 导入问题使本地质量结论受限 |
| requests_7502 | False / 1715B / 310.24s | False / 1194B / 532.88s | A 有 src/requests/models.py 修改，B **只有 repro.py**；双方评测有 196 个 fixture 错误，存在生产编辑层回退 |

各数来自 `RUN/<arm>/r1/attempts/<attempt_id>/result.json`；生产修改来自该目录 `patches/`。两臂各四个 Rich、FastAPI、Requests 开发题，不能覆盖其他仓库或正式隐藏任务分布。

## 4. 可以支持的机制结论

**rich_3180 的正面证据是正确编辑落地，不是完整验证闭环。** B trace step 7 在 174.94 秒写 repro.py；step 8 对关键 mixed-width 输入打印当前 [1]，该例没有正确结果 assert。step 10 在 415.41 秒首次编辑 rich/cells.py；step 11 在 474.80 秒写包含 pass 的 repro_2.py。之后没有修复后检查或 submit_patch，harness 记录超时，恢复 patch 后裁判 114 passed。A 仅写 repro.py。这个独胜值得跟进，但不能证明提示要求被完整执行。

**requests_7205 支持“核实运行的是哪份代码”。** B trace step 10 先验证自造 helper，不能代表真实源码；step 11 编辑生产代码；step 12 实际 API 仍显示旧结果。step 14 查出 requests.__file__ 指向系统目录，step 15 使用 PYTHONPATH=/workspace/src 后观察到空凭据 None、正常凭据保持原值，step 17 提交非空 patch。然而独立 verify 仍报告 1 failed、205 passed、13 skipped，关键断言得到 ('', '')。它支持 subprocess 环境路径问题的诊断；未经修正环境复评，不称“100% 正确补丁”，不手工改写官方 resolved。

**短超时可以限制复现损耗，但不是成功保证。** rich_4006 的两个 run_command 均真实阻塞 240 秒，占 agent 预算一半。应为可能死循环的复现限定短 shell 超时，再依据源码证据编辑；仅增加总预算或提示篇幅没有得到本轮支持。

**环境与污染问题仍制约结论。** 缺 inline_snapshot、src 导入遮蔽、配置文件保护及 httpbin fixture 错误，使若干失败不能等价于模型修复错误；空 patch 通过又说明存在假阳性。新增 scratch 文件和反复空提交同时显示行为版仍有执行问题。这些结果不适合未经甄别直接作为训练奖励。

## 5. 实验结论

原样公开参考已完成本地真实运行，B 在一次同条件配对中观察到 1 个 resolved 净胜、更多生产修改，但 agent 平均耗时增加 16.21 秒，超时或工具耗尽比例没有降低。应将 B 视作待确认候选，保留原样参考，不宣称确定性升级或金牌水平。

后续正式提交与训练进展另记，不追加为本次 P007 原始结果。原 result/trace/patch 均未更改；[paired-analysis.json](paired-analysis.json) 记录更正口径和独立 timeout 信息。
