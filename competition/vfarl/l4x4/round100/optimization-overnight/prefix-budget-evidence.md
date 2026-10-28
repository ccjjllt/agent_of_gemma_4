# 原始前缀、续行反馈与逻辑预算审阅

2026-10-28，北京时间。只读分析 R3 的固定 train 轨迹，并用一次性临时 Python 命令核对 collector 机制；未启动 GPU、远程作业或训练，未读取隐藏题及预留四题正文，没有新增测试文件。

## R3 真实事实

来源为 `round100/cancelled-r3/vfarl-train/resident/batches/batch-001/worker-*/*/draw-*/*/actions.jsonl`，按现有 curriculum 的固定 train 身份过滤，并排除预留任务 ID。共 41 份动作文件、501 个动作，与 `cancelled-r3/log-summary.json` 的真实 `policy_action=501` 一致；kernel 原日志 SHA256 为 `5b5f77b9e163d80c5ddfb2dc46fde4c15307ed7b94bcaffeda4baf39ffd13daa`。

- 496 个动作为 `tool_calls`，5 个为 `stop`；末 token 分别为 tool-close 49、turn-close 106。没有 `length/context_limit` 动作，所以不能把本轮合成截断实验说成 R3 实际发生过的故障。
- 64 个动作各有一个强制控制 token，全部为 thought-close 101，loss mask 均为 0，behavior logprob 均为 0；存在 reference 数组时，对应位置也全为 0。没有用强制控制 token 提供动作监督。
- 每份文件内的相邻动作，下一 prompt 均以“上一 prompt + 上一完整 completion”开头；没有丢弃或重新编码原始 sampled token 的实例。
- 29 份已完整结束的 episode，其最终 turns 都等于 restore 起点 turns 加本段动作数，output_tokens 都等于 restore 起点 output_tokens 加本段全部 completion token 数。未见恢复后归零、只累计 mask=1 token 或漏算强制控制 token 的情形。
- R3 完整 episode 未见 `termination=submitted` 但 `patch_submitted=false`。下面修复的是当前代码可复现的自然 turn 耗尽分支，不把它归因为 R3 已发生问题。

## 真实模板的反馈布局

R3 worker 的 policy 身份记录 template SHA256 为 `94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413`。本地 `research/006/data-replay-20261013/tokenizer/chat_template.jinja` 与该指纹匹配，使用已有 Jinja wheel 直接渲染后，当前 `continuation_suffix` 实际得到：

```text
<turn|>
<|turn>user
feedback<turn|>
<|turn>model
```

如果原生成已经以 turn-close 106 结束，只去掉后缀中第一个重复的 `<turn|>`，保留换行、独立 user 反馈与新的 model turn。原 completion 的所有 token 均保持原位置。未闭合 thought 或工具字符串不会被补成一条可执行工具调用；反馈使用显式 turn 边界开始新的对话机会，没有伪造工具执行结果。未确认“未闭合局部语法使 user 消息落入 thought channel”这一推断；真实模型能否借此恢复仍需后续短验证，模板文本布局核对不是模型行为证据。

最初试用的本地 12B template 指纹为 `36e3a42e…bead3f0`，与 R3 不同，因此未用它冒充 R3 模板；随后取得上述完全匹配版本，确认相关反馈布局一致。

## 一次性续行与预算机制实验

调用真实 `episode` 与缓存 Jinja 模板，policy/tool/verifier 使用明确的合成对象，普通文本编码也使用合成编码器；临时结果 reward 保持 null、verified=false，不产生研究成功率。

序列为“未闭合 thought 且 length 截断 → malformed 工具调用 → 合法 get_status”。全过程保留旧 prompt 与全部 completion，末尾计数从 turns=4/output=123 累计为 turns=7/output=207；context 的 LLM 次数也从 4 累计为 7。反馈消息只增加下一次真实 context 长度，不伪装成模型 output token。get_status 不计工具预算，工具计数保持初值 2。

另核对：显式 submit 对照保持 `patch_submitted=true`；剩余可用输出不足 16 token 时不再调用 policy；连续四次普通文本结束只收到三次 nudge，不重置累计 turns/output。真实采样强制控制 mask 则使用上节 R3 的实际 64 个动作核对，不从合成 policy 推断。

`state._context/_restore_context` 的一次性组件实验使用合成 budget/harness dataclass，真实函数恢复后 tool_calls=7、llm_calls=9、剩余工具数=25、剩余 turns=31，逻辑 agent elapsed 约 20.0 秒。这只是 context 组件验证，不冒充完整 sandbox snapshot 新实验；完整 R3 restore 的本段累计关系已由真实记录核对。

## 已修复：自然耗尽 turns 被误记为 submitted

原 `episode` 的 while 条件自然结束时没有给 stop 赋值，summary 的 `stop or 'submitted'` 会把没有调用 submit 的自然 turn 耗尽误记为 submitted。上面的合成序列实际复现了 `termination=submitted / patch_submitted=false`；reward 仍为 null，故这是终止原因错误，没有伪造验收奖励。

现在自然耗尽且未提交明确记为 `turn_budget_exhausted`；默认 submitted 也要求真实 `ctx.patch_submitted`。仍允许官方 working-tree fallback 提取实际补丁并交给原 verifier，不根据终止标签改写奖励。

修复后分别用 historical/live 两个协议、末动作 get_status/submit_patch 进行一次性核对：两协议的未提交均得到 `turn_budget_exhausted / false`，两协议的显式提交均得到 `submitted / true`；turns/output/LLM 次数保持累计。AST 解析和 `git diff --check` 通过。没有新 GPU 修复收益证据。
