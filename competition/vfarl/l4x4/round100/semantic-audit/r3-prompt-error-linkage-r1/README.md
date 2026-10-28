# 题面修订与 R3 真实采样失败的联系

2026-10-28。只读分析固定 100 train 中的索引 25、26、6，三题均属于现有 38 篇题面修订。没有重放模型、运行新的 CPU oracle、创建 teacher／pair／SFT 或修改原奖励。本目录的 `evidence.json` 绑定源文件 SHA、实际 action/event 行 SHA、三个版本的 prompt 身份及记录的 token 计数。

这三例直接说明已有修订针对真实发生过的复现障碍；尚不能说明新版题面会节省多少 token、减少多少动作或提高修复率。R3 原始采样均为 thinking_budget=256、original_action_replay=false，与本轮 Codex 人工前缀和 thinking=0 teacher 不同。

| 固定索引 | 真实观察 | 对应修订 | 原 episode |
| --- | --- | --- | --- |
| 25，Tenacity wait_combine，`jd__tenacity.0d40e76f.func_basic__pvngsy7s` | 动作 0 照用旧题面的 `RetryCallState(retry_object=r, outcome=None, next_action=None)`，实际抛 TypeError。动作 7 修好构造参数后仍保留旧题面的 `seconds_since_start = 5`，实际抛只读属性 AttributeError。动作 8 才得到真实 double-delay 输出。 | v2 使用该 snapshot 的四参数构造 API，移除无关的只读属性赋值，以 fixed(2)+fixed(3) 有界展示 sum 契约，同时保留随机组合的契约说明。 | 11 个采样动作、10 个工具事件，2238 completion token；末尾 malformed edit，未 submit，官方 working-tree fallback 得 0。 |
| 26，Tenacity wait_fixed，`jd__tenacity.0d40e76f.func_basic__rlm5c89z` | 动作 0 直接使用旧题面的 `RetryCallState(1, 0)`，抛缺少 args/kwargs 的 TypeError，尚未执行到 wait。随后 grep 和读取类定义；动作 3 malformed read_file 后结束，未形成有效行为复现。 | v2 补齐同一四参数 API，保留 fixed(5)、timedelta(10) 的原预期与实际值。 | 4 个采样动作、3 个工具事件，375 completion token；未 submit，官方 fallback 得 0。 |
| 6，Bottle overlay，`bottlepy__bottle.a8dfef30.func_basic__r3tamcap` | 动作 6/7 沿用“先赋值、后创建 overlay”顺序，三个读取均正常；动作 10 重跑仍正常。动作 11 改为先创建子 overlay、后写 source 新 key，动作 12 才抛 KeyError；生产修复后动作 14 得到三个正常值。 | v2 明确最初复制值可正常读取，真正缺陷是子 overlay 已存在后的 source 更新不能传播。v2 改已有值为 source2；R3 后续触发是新增 key，两者是相关传播证据，并非字节相同的 v2 执行。 | 18 个采样动作、17 个工具事件，2354 completion token；实际 submit，官方得 1。 |

## 实际开销的边界

计数来自各行记录的 `completion_tokens`，总和与 episode 的 `ledger.output_tokens` 一致。`thought_tokens` 单列保留，不能再加到 completion_tokens 上。以下是已经发生的窗口，不能当作新版题面将省下的数量；不把重叠 prompt 长度相加为预填充成本。

- 索引 25：两次无效 MRE 执行动作 0、7 合计 **997 completion token**（记录 thought counter 131）；查找构造定义动作 1–6 合计 **368**；有效重写动作 8 为 **497**。完整复现准备窗口 0–8 为 9 动作、1862 token，其中包含正常探索。
- 索引 26：无效 MRE 动作 0 为 **217 token**；查定义动作 1、2 合计 **105**；最后 malformed 动作 3 为 53。不把后续 malformed 归因于题面。
- 索引 6：首次写入和运行非触发复现的动作 6、7 合计 **349 token**；动作 10 重跑为 **278**（其中记录 thought counter 256）；换顺序写入和运行的动作 11、12 合计 **362**。包含中间源码探索的完整窗口 6–12 共 7 动作、**1245 token**。

## wait_combine 既有分支说明

只读复核了原动作 1 与既有 branch-1 首动作的 **2316 个 prompt token 完全相同**，并与 restore.json 一致；没有创建新 pair。branch-1 的 16 个新后缀动作合计 2069 completion token；ledger 2455 还包含继承的原动作 0 的 386 token，不能重复统计。

该分支使用 MockState 得到有效复现、修改生产代码并观察到正常等待时间，随后以 malformed submit 结束；它没有实际 submit，官方 working-tree fallback 得 1。因此“original 0／branch 1”不能描述成干净提交成功或某一个动作的因果收益。历史分支曾写新测试文件，这是归档中的真实模型行为；本次分析没有创建测试文件。

本地有 state-001 manifest、restore 元数据及原有 `tenacity-live-original-review.json`；完整 snapshot root forest 未下载，原 focal action 的 reference forward 仍缺失。本次没有重建可续跑的完整状态、补 logprob 或制造训练行。

## 身份与来源

每题的 v4 索引、ID、机制、family、snapshot、train role 都保持一致；旧／新 task JSON 除 problem_statement 外逐字段相同。实际 R3 prompt 使用当时实际运行预算和显式 token limits，所以其 SHA 与旧 canonical initial_prompt SHA 不同；三条实际 prompt 都包含旧 problem_statement 的完整字节。v2 canonical initial_prompt SHA 单独记录，不冒充当时的采样 prompt。

主要来源为 `cancelled-r3/vfarl-train/resident/batches/batch-001/worker-{0,1}/<exact-ID>/draw-0` 中的 actions、tool-events、episode、actual-prompt、prompt-provenance、官方 result，以及 v4 和 question-revision-r2 的 selection/task/revision 文件。精确路径、SHA、行位置与实际输出见 `evidence.json`。所有原始来源、当前 fixed100 主 SFT、题面、算法和运行策略保持原样。
