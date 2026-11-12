# 007 训练吞吐优化：Checkpoint、KL 前向复用与 Prefix Tree

状态：实现完成，纯 CPU 核对通过；A800 实测待授权 / 实例当前不可达。
日期：2026-11-12。
范围：只改训练吞吐路径，不改 SPO 目标、Rank 128、奖励公式、训练数据和官方任务预算。

> A800 状态：2026-11-12 本地只读连通性探测显示既定 AutoDL SSH 端口 `Connection refused`；备用 22 端口返回已变更的主机密钥，未绕过校验、未连接、未启动训练。首次 A800 指标仍待实例与访问方式确认后采集。

## 0. 边界

- 不依赖跨训练轮次保留 GPU 或 CPU 内存。每轮结束仍把 FP32 LoRA、AdamW、scheduler、RNG 和当前 batch 持久化；下一轮重新加载。
- 不改变 `official_runtime.py` 的验收语义，不改变 `reward/` 的奖励计算，不改变 `target_action_tokens`、`max_policy_lag`、采样参数或课程。
- 没有启动 GPU、没有连接服务器。本文中的数字分为“已核对的 CPU 结果”和“待 A800 测量的目标”。

## 1. 启动阻断：`Policy._load_learner` 变量作用域

`__init__()` 里局部导入的 `FastModel` 和 `torch` 在 `_load_learner()` 中不可见。原代码会在真正加载 31B 时抛 `NameError`。已改为使用实例保存的：

- `self.FastModel.from_pretrained(...)`
- `self.torch.bfloat16`
- `self.FastModel.get_peft_model(...)`

纯 CPU 的 AST 核对确认 `_load_learner` 内没有裸 `torch`/`FastModel` 引用。

## 2. Checkpoint：两层保存与异步写入

### 2.1 优化前的保存次数

一次成功更新实际会完整序列化四次：

| 节点 | 触发位置 | 内容 |
| --- | --- | --- |
| `learning_started` | `optimize_batch()` | 全量 LoRA/Adam/scheduler/RNG/batch |
| `optimizer_step` | `_apply_optimizer_step()` | 全量，参数已更新 |
| `optimized` | `_run_active()` | 全量，仅多了 KL 统计 |
| 导出后 | `run_spo.finalize_update()` | 全量，参数未再变化 |

后两次与 `optimizer_step` 的张量完全相同，属于重复写盘。

### 2.2 优化后的两层

A. 完整训练 checkpoint：`spo_trainer_state.pt`

- FP32 LoRA、AdamW 一阶/二阶、scheduler、Python/NumPy/Torch/CUDA RNG。
- 当前 `UpdateBatch`、optimizer cursor、`pre_update_logprobs`、训练组和 KL probe 索引。
- 只在 `optimizer_step` 写一次；这是唯一发生参数变化的边界。
  - `learning_started` 不重复写 LoRA/Adam；它只写事务记录，指向最后一次已提交的 full checkpoint（首次更新没有 full 时为 `null`）。
  - `optimized` 也不重复写；它只写 KL/阶段统计。

B. 轻量事务记录：`spo_trainer_tx.json`

- `update_id`、phase、optimizer cursor、completed optimizer steps。
- KL 统计（`kl_value`、`token_count`、`estimator`、`post_update_kl`）。
- `kl_ready`、`policy_before`、`checkpoint` 文件引用（或首次更新时的 `null`）。
- `EXPORTED/REGISTERED` 阶段追加 `adapter`：`version`、`parent_version`、`optimizer_step`、`adapter_sha256`。
- 由 `training/checkpoint_io.py:write_transaction` 写临时文件、`fsync`、原子替换；读取时校验 checkpoint 引用和 adapter SHA256。

训练器状态的 schema 版本从 1 升到 2。旧 SPO checkpoint 不会被静默当作新格式恢复；`_verify_resume_consistency` 会给出明确的 schema 不匹配错误。

### 2.3 写入与等待策略

`training/checkpoint_io.py:CheckpointWriter` 使用单后台线程、单队列：

- 调用方先完成 CPU 侧克隆（`export_training_state`），后台线程只做 `torch.save`、文件 `fsync`、`os.replace` 和目录 `fsync`。CPU 克隆仍占用训练线程。
- 本轮唯一 full 快照是 `optimizer_step`，用 `sync=True` 等待；它是唯一真正的恢复边界。`learning_started` 与 `optimized` 只写事务记录，不再排队大权重。
- `sync=False` 接口保留给后续可能的非边界快照；本轮没有 full 调用使用它。
- 任一次写入失败会被保存，下一次同步提交或关闭时抛出，不会把失败静默当成可恢复边界。
- `learning_started` 期间 ledger 仍保持 `RESERVED`；只有 `optimizer_step` 全量落盘后才推进到 `LEARNING/1`。这样即使事务记录已写而 full 尚未完成就崩溃，恢复时仍可从头重放该 batch，而不会出现“ledger 已 LEARNING、checkpoint 还是上一条”的死路。
- 若崩溃发生在 `optimizer_step` 全量落盘之后、ledger 推进之前，恢复会看到 `RESERVED` + checkpoint cursor 1，直接恢复该 post-step 状态并继续 KL/导出，不重复 AdamW。

### 2.4 恢复语义

| 崩溃点 | 全量文件 | 事务文件 | 恢复行为 |
| --- | --- | --- | --- |
| forward 前/中，未 step | 上一条 committed（首次更新无 full） | `LEARNING` | ledger 仍 RESERVED；从头重放该 batch，AdamW 未执行 |
| step 后、checkpoint 前 | 上一条 committed | `LEARNING` | ledger 仍 RESERVED；从已提交 base 重放该 batch，post-step 结果从未持久化 |
| checkpoint 后、ledger 前 | `optimizer_step`（cursor 1） | `LEARNING`（旧） | 看到 RESERVED + cursor 1，直接恢复 post-step，不重复 AdamW |
| KL 前 | `optimizer_step`（cursor 1） | `LEARNING`（旧） | 恢复全量，重算 KL，不重复 AdamW |
| KL 后、导出前 | `optimizer_step` | `OPTIMIZED` | 合并 KL 统计，直接进入导出/发布 |
| 导出/注册后 | `optimizer_step` | `EXPORTED/REGISTERED` + `adapter` | 合并统计与 adapter SHA，重放幂等发布阶段 |
| 已 CONSUMED 后重启 | `optimizer_step`（cursor 1） | 旧记录 | 允许从 cursor 1 恢复已消费 base，不再要求 OPTIMIZED 全量文件 |

`_reconcile_ledger()` 对 `CONSUMED` 的检查从“必须是 OPTIMIZED 全量快照”放宽为“optimizer cursor 必须为 1”。已消费更新的 KL 不再影响后续训练；cursor 1 已足以证明 AdamW 执行过。

`ensure_kl_ready()` 用于 `EXPORTED/REGISTERED/ACTIVATED` 恢复：若事务记录没有有效 KL，则只做一次 KL probe，不把 ledger phase 回退到 OPTIMIZED。

## 3. KL：复用训练前向

### 3.1 原路径

1. `_new_active()` 调 `_capture_pre_update_logprobs()`，对最多 4 个 KL probe call 单独做一次 no-grad 前向。
2. `_apply_optimizer_step()` 再对同一批 call 做可微训练前向。
3. AdamW。
4. `_measure_post_update_kl()` 再做更新后前向。

第 1 步和第 2 步对相同 token、相同参数重复计算。

### 3.2 新路径

1. `_apply_optimizer_step()` 的正常训练前向产生 `current` logprobs。
2. 对 `kl_call_indices` 中的 call，立即 `detach().to(cpu, float64)` 保存到 `_active["pre_update_logprobs"]`。
3. 完成全部组的 backward。
4. 校验 KL probe 索引全部覆盖后，才执行 `optimizer.step()`。
5. `_measure_post_update_kl()` 只做必要的更新后前向，仍使用原 K3 公式。

不变量：

- token IDs、action mask、behavior logprobs 与训练损失使用的是同一份 `ActionCall` 证据。
- 更新前概率确实来自 `optimizer.step()` 之前的参数；保存的是 detached CPU 副本。
- 若训练前向没有覆盖某个 KL probe，直接失败，不回退到“更新后参数当作更新前”的伪证据。
- `pre_update_logprobs` 进入全量 checkpoint；中断恢复后不需要重新做更新前前向。

预期收益是每次更新少最多 `min(4, calls)` 次完整 31B 前向；实际秒数由 A800 的 `spo_update_timing` 记录验证。

## 4. Prefix Tree、Logit Chunk 与 Gradient Checkpointing

### 4.1 分组：`lcp_greedy`

`training/prefix_tree.py` 原有实现先按完整 token 序列字典序排序，再取连续最多 4 个调用建树。字典序本身会把相同前缀放在一起，但当紧随其后的调用因 tree token 上限放不进时，原实现会直接退化为单例，不会尝试后面共享前缀稍短但能放进的调用。

新增 `grouping="lcp_greedy"`：

- 仍以字典序为全局顺序。
- 每组只看后面 `lookahead=8` 个候选，按与 seed 的 LCP 从长到短尝试。
- 每次尝试都重新检查 tree token、mask 预算、dense attention 和最小共享 token。
- 放不进的高 LCP 候选不会阻断后续可放进的候选。
- 仍保留每个原始 call 的 `score_indices`、action mask、behavior logprob 和 advantage occurrence；不是把调用合并成一条轨迹。

配置：

```toml
training_prefix_grouping = "lcp_greedy"
training_prefix_lookahead = 8
training_prefix_max_calls = 4
training_prefix_min_tokens = 256
training_prefix_query_chunk_size = 256
```

CPU 合成用例（300 token 公共前缀，3 个调用，`max_tree_tokens=350`）：

- 字典序：3 个单例，`saved_tokens=0`。
- `lcp_greedy`：`[0,2]` 成树，`saved_tokens=301`；`[1]` 单例。
- 所有调用仍恰好出现一次。

这只是分组正确性与收益上界的 CPU 证据，不代表 31B 的实际加速。

### 4.2 Logit Chunk

`policy.logprob_chunk_size` 仍控制大词表投影分块（当前 64）。本轮不盲目改默认值：

- `_public_stats()` 记录实际 `logprob_chunk_size`。
- 首次 A800 运行用 `spo_update_timing.forward_seconds` / `backward_seconds` 对比 64、128、256。
- 只有显存峰值仍有余量时才提高；否则保留 64。

### 4.3 Gradient Checkpointing

`native_tree_forward()` 新增 `checkpoint_policy`，由 `training_tree_checkpoint_policy` 控制：

- `"all"`：默认，保持现有逐层非重入 checkpoint。
- `"none"`：只用于“已确认显存有余量”的对照测量，不设为默认。

代码不为了省显存无条件增加重计算；默认行为不变。首次 A800 需要记录 `checkpoint_policy`、激活峰值和 step 时间，再决定是否切换。

## 5. 首次 A800 需要记录的四个指标

`metrics.jsonl` 新增：

- `spo_model_loaded`：`phase=genesis/update/resume_*`、`seconds`。
- `spo_update_timing`：`forward_seconds`、`backward_seconds`、`optimizer_seconds`、`kl_seconds`、`checkpoint_learning_started_export_seconds`、`checkpoint_learning_started_submit_seconds`、`checkpoint_optimizer_step_export_seconds`、`checkpoint_optimizer_step_submit_seconds`、`checkpoint_optimized_seconds`、`checkpoint_writer.last`（实际磁盘写入秒数和字节数）。
  - `learning_started` 的 `submit_seconds` 只是入队时间；实际磁盘写入看 `checkpoint_writer.history`。
  - `optimizer_step` 的 `submit_seconds` 包含同步等待，因此就是“写盘+fsync”时间。
- `spo_checkpoint_queued`：`phase`、`sync`、completed steps。
- `spo_transaction_persisted`：update id、phase、`kl_ready`、`adapter_sha256`（导出/注册后）。

这四项分别回答：模型加载、单次前向、反向传播、checkpoint 写盘各占多少时间。

## 6. 本轮已执行的纯 CPU 核对

环境没有安装 torch，无法在本地执行 31B 或真实 `SPOTrainer` 前向；以下是不依赖 GPU/torch 的核对：

```text
python -m compileall -q training
compile_exit=0

import training.run_spo, training.spo.trainer, training.prefix_tree,
       training.tree_backend, training.checkpoint_io
imports_ok
```

- Prefix 分组：`lexicographic` 3 个单例 vs `lcp_greedy` 成树保存 301 tokens；索引覆盖完整；小 mask 预算下两种分组都退回单例（chunk floor）。
- Checkpoint 去重：`learning_started` → transaction(`LEARNING`)，`optimizer_step` → full(sync)，`optimized` → transaction(`OPTIMIZED`)；本轮 full 只写 1 次，timing 保留 `export/submit/seconds` 三类键。
- 轻量事务：首次更新没有 full 时 `checkpoint=null` 可读；`REGISTERED` 记录 `adapter_sha256` 和 full checkpoint 字节数；缺少 full 的 post-optimizer 记录被拒绝。
- 训练器事件：`learning_started` → `tx(LEARNING)`，`optimizer_step` → `full(sync)`，`optimized` → `tx(OPTIMIZED)`；单次更新 full 保存 1 次。
- 事务合并：KL 统计可从 `spo_trainer_tx.json` 叠加到恢复状态。
- `CheckpointWriter`：异步提交 1、同步提交 2，最终文件为 2，写入计数 2，history 2 条；启动时只清理 PID 已死的遗留 `*.tmp-*`，保留当前 PID 的临时文件。
- schema：`TRAINER_STATE_VERSION=2`，旧 v1 checkpoint 被明确拒绝。
- `Policy._load_learner`：AST 检查无裸 `torch`/`FastModel`。
- 恢复顺序：`checkpoint("optimizer_step")` 在 ledger cursor 推进之前；`_checkpoint_phase` 在 `_merge_transaction` 之前读取。
- `RESERVED` 更新如果看到 cursor-1 full checkpoint，走 `restore_training_state + resume_batch`；只有 cursor-0/已消费 base 才走 `restore_base_state`，并用 `_active_update_id` 做身份校验。
- 旧/新 checkpoint 的 active update id 都能从 `active_batch.update_id` 或 `active_batch.update.update_id` 读取；`_install_active` 会交叉校验两者，不匹配直接失败。
- KL 捕获顺序：AST 检查 `_freeze_logprobs` 行号小于 `optimizer.step()`，且 `pre_update_logprobs` 在 step 前赋值。

未执行：真实模型加载、真实前向/反向、真实 checkpoint 写盘、SGLang 握手、A800 显存和吞吐。

## 7. 首次 A800 运行协议

1. 用 `--check-only` 只做配置/模型目录/SGLang health 预检。
2. 受控 pilot：加载 → Genesis adapter → 一条完整轨迹 → 一次真实 optimizer step → 导出 adapter。
3. 记录 `spo_model_loaded`、`spo_update_timing`、`spo_transaction_persisted`、显存峰值和 adapter hash 变化。
4. 对比 `training_prefix_grouping=lexicographic` 与 `lcp_greedy` 的 tree tokens / saved tokens / step 时间；只改一个因素。
5. 只有显存有余量时才试 `training_tree_checkpoint_policy=none` 和更大的 `logprob_chunk_size`。

## 8. 风险与回滚

- `learning_started` 事务写失败会在进入 forward 前抛出；不会静默跳过。`optimizer_step` 的 full 写失败会在 KL 前抛出。
- `optimizer_step` 的 CPU 克隆/`torch.save` 准备仍在训练线程；只有文件写入与 `fsync` 进入后台线程。首次 profile 必须分开看 `export_seconds` 和 `submit_seconds` / `checkpoint_writer.history`。
- 事务记录损坏或指向错误 checkpoint 时，`load_transaction_state()` 直接失败，不回退到猜测。
- 全量 checkpoint 仍是唯一模型状态来源；事务文件只携带统计和引用，删除它最多导致一次 KL 重算。
- 训练前向使用 tree/batched masked-SDPA，更新后 KL probe 使用独立 causal 路径。两者数学等价，但 bf16/kernel 差异会给区间 K3 一个数值底噪。首次 A800 应对少量 call 同时计算 tree 前向与独立前向的更新前 logprob，记录最大差值；若明显大于预期，KL 数值只能作诊断，不能当作精确的版本间 KL。
- `lcp_greedy` 在 mask 预算只允许小于 16 行的 query chunk 时主动退回单例，避免“每 token 一次 attention launch”的悬崖。
- 回滚：恢复 `_checkpoint("optimized")` 的全量保存、恢复独立 `_capture_pre_update_logprobs()`、把 `training_prefix_grouping` 设为 `"lexicographic"` 即可回到旧行为；schema 已升到 2，旧 checkpoint 不会被误读。

## 9. 本轮不改

- 奖励公式与 `official_differential_spo_v1`。
- `training/official_runtime.py` 的逐节点 fallback 接线（已知的 P0 奖励信号问题，单独处理）。
- Rank、alpha、dropout、学习率、batch token 预算、官方评测预算。
- 不新增测试文件；验证均为一次性手动命令。
