# Submit 后 terminal 的实际范围

2026-10-28，固定 100 train 题，只读源码与已有记录；没有模型、工具或 oracle 重放，也没有生成新训练视图。

**“99 条 terminal 在官方部署不可达”不成立。** pinned swegemma 0.2.11 的外层 `patch_submitted` 检查位于 `runner.run_async` 事件迭代返回之后；迭代内部的停止条件明确要求已经 submit 且收到 assistant final text。`submit_patch` 本身只设置 `ctx.patch_submitted` 并返回成功响应，没有设置 ADK 的 `skip_summarization` 或 `end_invocation`。

在 pinned ADK 1.36.1，普通 function-response 不是 final event。`BaseLlmFlow.run_async` 因而继续下一次 `_run_one_step_async`，后者调用 LLM；生成 final text 后，swegemma 才结束。这是预算允许时的具体源码可达路径，不是新比赛 HTTP 或模型实跑证据。相关 wheel/member SHA 与精确行号保存在 JSON 的 `pinned_sources`。

当前 `run.py` 只在 `collection_protocol=live-original-pair` 分支中于 submit 后立即 break；`historical-three-branch` 不共享该立即停止条件。当前注释 “Official execution stops immediately on submission” 与上述 pinned 官方调用链不一致。本次没有修改它。

逐实际 messages 核对发现：全部 99 个 terminal 都紧接真实 `submit_patch` 成功工具响应，响应为 `status=ok` 且 patch 非空；call ID 与实际 tool-events 一致，文本逐字等于 recipe.final_response。它们是 Codex off-policy recipe 的实际记录，不是 Gemma 已采样的 final response。只有 index 31 Tenacity425 原来源没有 terminal。

| 逐行源数据量 | 完整视图 | 99 个 terminal | 占完整视图 | 保留 706 个工具动作 |
| --- | ---: | ---: | ---: | ---: |
| 行数 | 805 | 99 | 12.30% | 706 |
| 输入 tokens | 2,054,401 | 341,627 | 16.63% | 1,712,774 |
| 监督 tokens | 78,148 | 3,209 | 4.11% | 74,939 |
| masked prefix tokens | 1,976,253 | 338,418 | 17.12% | 1,637,835 |

输入 token 数包含每行前缀与 completion，且行间重复计算历史；这不是唯一 episode token 数或实测计算成本。单纯保留全部工具行仍覆盖 100 个相同 train ID，706 行原字节及 prompt 前缀可以全部保留。原字节过滤的预期 SHA 为 `5ecadc1fa1d12c59b24ae6ba7dd401ad5670b6189edc58506a68f33a2dbd124a`，本次没有写出该数据集。

root 已决定保留全部 99 条 terminal 与原 805 行完整视图，不导出或推荐 706 行作为当前视图。上述过滤哈希只保留为只读统计，不代表数据准入决定。

修改前的 live `run.py` 全文已原字节冻结在 `source-pins/live-run.before-submit-fix.py.txt`，SHA `25c9fb4f2614aff712d50337fb38d5a9b0665e37b56db8cbb0e08209bb4368a5`，与 census 一致。官方/ADK 成员原件、精确片段和最小修正评估见 `source-pins/provenance-and-minimal-change.json`。没有更早的 submitted guard 在正常路径上阻止 sampling；删除 live 的立即 break 可在剩余预算内恢复 final-response 机会。文本结束后现有 no-call 分支不会再 nudge，late-tool guard 仍保留。此处只评估机会恢复，不声称整套 official 等价；root 拥有源码修正，CPU facade/oracle 验证等待具体指派。

完整范围绑定、统计定义和源码证据：[source-and-census.json](source-and-census.json)，SHA `884a2a9bdfbd05b39958d242d9f2856da9001d29fa4e22ef1187528f8798c631`。
