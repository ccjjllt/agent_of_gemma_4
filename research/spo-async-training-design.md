# Single-stream SPO 异步训练系统设计

状态：实施基线，2026-11-10，分支 `codex/training-plan-refactor`。

本文是代码重构的直接依据。固定路线为 Gemma 4 31B IT QAT BF16、LoRA Rank 128 / Alpha 256 / Dropout 0、Single-stream Policy Optimization、异步 Agent RL、SGLang 优先、Unsloth + Fused AdamW、官方验收差分奖励、SFT + 在线 RL + 教师蒸馏/OPD、2×A800 80GB。不安排 GRPO/GSPO/SAPO 或不同 Rank 的对比实验。

设计和代码完成不等于已有训练收益。真实更新、吞吐改善、修复率提升和 CT 兼容必须由后续运行证据分别证明。

## 1. 不变量

1. 官方公开任务原目录只读；题面、`base_commit`、参考补丁和 `test_patch` 不改写。参考补丁与逐节点验收资产只在裁判侧使用，不读取 heldout。
2. 每条轨迹独立调度、独立验收、独立计算奖励和优势；不存在同题 8 条、至少 2 条、同题组均值或组序提交游标。
3. 只有完整官方 verdict、完整必需 JUnit 覆盖和完整模型消费 ACK 的轨迹能产生有效 RL 奖励。基础设施失败、验证不完整和报告缺失不补零入批。
4. 每个参与损失的 token 都能追溯到原始输入/输出 token ID、行为 logprob、action mask、adapter 版本及哈希。不得用文本重 tokenization 冒充 TITO。
5. 一条轨迹创建时绑定一个 rollout adapter 版本，不在轨迹中切换。更新批次仍逐 call 校验版本和 policy lag。
6. 优势先使用更新前价值计算，随后才写入新价值。批次优势冻结后，恢复不得重算。
7. 两张 A800 承载同一个 31B 分片模型，不划分独立推理卡和训练卡。CPU 环境执行与 GPU 工作流水并行；GPU 推理和反向传播分时。
8. 未实测显存前不得启用双 Adapter 常驻，不得宣称 32K、双引擎或双 Adapter 已稳定。
9. 旧 GRPO run、collection checkpoint、reward ledger 和 optimizer state 不混入新 SPO run。新 schema 从独立 run 目录开始。
10. 不新增测试套件。实现核对采用一次性手动 Shell/Python 命令；真实研究进展以有效轨迹、更新和官方修复证据为准。

## 2. 现有资产与必须拆除的同步假设

可复用资产：

- `official_runtime.py`：UID 隔离、官方 harness、私有裁判资产、交付计时、结果归因。
- `endpoint.py` / `request_store.py`：OpenAI 协议适配、幂等、ACK、取消和持久化事件。
- `generation.py` / `kv_cache.py` / `cache_budget.py`：连续批处理、KV/prefix cache 和显存预算思想。
- `prefix_tree.py` / `tree_backend.py`：算法无关的原始 token 树和前反向计算；每次 occurrence 必须继续携带自己的 mask 和 advantage。
- `policy.py`：双卡 Unsloth、Rank 128 目标层、完整前缀 logprob、BF16 adapter 导出。
- `recovery.py`、LR/RNG 工具：原子 checkpoint、预算、optimizer/scheduler/RNG 状态。

必须拆除：

- `collection.py` 的 group/slot、min-two、`commit_cursor`、组内奖励和整批 collection 生命周期。
- `session.py` 的“采集返回→停 worker→训练→下一批”相位。
- `grpo_trainer.py` 的 TRL GRPO、Dr.GRPO 固定长度分母、首次前向冻结 old logprob、组内 cohort 语义和额外 TIS 修正。
- `rewards.py` 的组内中心化、缺失区间优势和时间惩罚主导。
- `request_store.py` 以 `(rollout_id, policy_version)` 限定整轨迹读取的假设；新轨迹虽固定版本，训练读取必须按不可变 trajectory ID 和 call ID。

## 3. 总体架构

```text
TaskCatalog + SPO Sampler
          │ independent TaskKey
          ▼
RolloutScheduler ──► ExecutionManager ──► OfficialEnvironment/Worker
          │                  │                  │ tools / repo / tests
          │                  ▼                  ▼
          │             TITO RequestManager   VerificationBundle
          │                  │                  │
          │                  └────► TrajectoryStore
          │                                      │
          │                            DifferentialRewardEngine
          │                                      │
          │                            ValueTracker.observe
          │                                      │ raw advantage
          │                                      ▼
          │                               TrajectoryBuffer
          │                                      │ action-token budget
          ▼                                      ▼
SGLang/vLLM Backend ◄── GpuLease/AdapterManager ── SPOTrainer
   TP=2, sleep/wake          versions N/N+1       Unsloth Rank128
```

一个监督者拥有 run lock、预算和数据库写事务。Rollout worker、验收、奖励和 buffer 入队持续运行；trainer 达到 token 预算后申请 GPU lease。训练期间只暂停需要 GPU 的请求，不停止正在执行 CPU 工具、验收或写盘的 worker。

## 4. 目标目录与接口

### 4.1 `rollout/`

- `scheduler.py`：`RolloutScheduler.submit_next()` 从短/长任务队列选择独立任务，维护仓库覆盖、并发和公平配额。
- `engine.py`：`RolloutEngine.run(assignment)` 驱动单轨迹状态机；完成即触发验收，不等待其他轨迹。
- `request_manager.py`：迁移 ACK/幂等/取消，提供 `create_call`、`commit_generated`、`acknowledge`、`seal_trajectory`。
- `policy_version.py`：`PolicyVersion{version,adapter_sha256,parent,optimizer_step,config_hash}`；提供 lag 校验和不可变 manifest。

### 4.2 `environment/`

- `worker.py`：迁移官方 agent/judge UID、进程和结果校验。
- `sandbox.py`：私有可写 overlay、安全只读依赖层、清理和权限边界。
- `workspace_cache.py`：按 `(repo,base_commit,source_hash)` 的只读仓库 snapshot，按 `(lock_hash,python_abi,wheel_hash)` 的只读 venv；每 attempt 使用独立 overlay。
- `execution_manager.py`：独立 worker 提交、完成即验收、取消、长短队列和缓存预热。

### 4.3 `trajectory/`

- `schema.py`：严格校验下述不可变 schema。
- `buffer.py`：SQLite 事务化 `offer/reserve/mark_consumed/mark_stale`；按完整轨迹 FIFO，达到有效 action-token 预算即预留更新。
- `ledger.py`：轨迹、奖励、优势、更新和审计事件的可恢复账本，不再保存同题组统计。
- `token_alignment.py`：保存原始 ID、等长 action mask、稳定工具调用 ID 和逐 token 行为版本；校验 `len(completion_ids)==len(behavior_logprobs)==len(action_mask)`。

### 4.4 `reward/`

- `baseline.py`：裁判侧 `EmptyPatchBaselineStore.get(task_key)`，计算并缓存逐节点空补丁结果。
- `differential.py`：候选/基线逐节点配对，输出 P、G、覆盖率和缺失原因。
- `progress.py`：resolved、progress、regression、validity、效率及奖励资格计算。
- `critic.py`：从工具事件、编辑、测试和裁判私有证据生成过程评价；不读取模型自述作为证据。
- `engine.py`：组合 `VerificationBundle → RewardRecord`，只将 eligible 记录交给 SPO。

### 4.5 `spo/`

- `value_tracker.py`：每 `TaskKey` 持久化连续 V、独立 `success_p`、有效计数和 KL 游标。
- `advantage.py`：raw advantage 和 SPO++ action-token-measure 全局冻结归一化。
- `sampler.py`：不确定性优先、epsilon 全覆盖、仓库/类型配额和健康退避。
- `loss.py`：masked PPO-Clip、可选 reference KL、成功轨迹效率辅助项。
- `trainer.py`：行为概率校验、token-budget 梯度累积、global norm 1.0、Fused AdamW、prefix tree、KL 观测和恢复。

### 4.6 `adapter/`、`data/`

- `adapter/manager.py`：adapter manifest、版本引用计数、发布/回滚和 hash。
- `adapter/double_buffer.py`：training/rollout 所有权、GPU lease、resident/non-resident 两种安全模式。
- `adapter/exporter.py`：Rank 128 BF16 PEFT 导出和 CT 提交文件生成。
- `data/curriculum.py`：公开任务元数据、价值采样和在线能力记录。
- `data/sft.py`：从验收通过的轨迹生成 masked-CE 数据集。
- `data/distillation.py`：教师轨迹、OPD top-k 分布、tokenizer 对齐和官方验收筛选。

## 5. 核心数据 schema

```text
TaskKey {
  task_id, repo, base_commit, test_patch_sha256, judge_environment_hash
}

ActionCall {
  trajectory_id, call_index, request_id,
  prompt_ids, completion_ids, behavior_logprobs, action_mask,
  behavior_adapter: {version, sha256},
  sampling: {temperature, top_p, top_k, seed},
  parser_valid, finish_reason, raw_tool_call_ids,
  delivery_state, consumed_at, token_timing
}

Trajectory {
  schema_version, trajectory_id, task_key, stream_sequence,
  behavior_adapter, calls, action_token_count, status,
  verification_ref, reward_ref, raw_advantage, provenance
}

VerificationBundle {
  candidate_verdict, candidate_summary_ref, baseline_summary_ref,
  repair_counts, regression_counts, coverage, eligibility,
  process_evidence, patch_sha256, artifact_hashes
}

RewardRecord {
  resolved, P, G, progress, regression, validity,
  base_reward, efficiency, efficiency_weight, total_reward,
  eligible, exclusion_reason, objective_version
}

ValueRecord {
  task_key, V, success_p, effective_count,
  last_observation_sequence, last_policy_version, kl_cursor
}

UpdateBatch {
  update_id, policy_before, trajectory_ids,
  raw_advantages, frozen_advantages,
  token_mean, token_std, total_action_tokens,
  max_behavior_lag, optimizer_cursor, phase
}

AdapterManifest {
  version, parent_version, adapter_sha256, base_revision,
  rank=128, alpha=256, dropout=0, optimizer_step, config_hash
}
```

`action_mask=1` 只用于模型实际生成的 thinking、普通回答、工具调用和生成定界符；prompt、用户/环境消息、工具观察、padding 和宿主注入 token 为 0。原始工具调用 ID 必须保留，内部 request ID 不得替换它。

## 6. 独立轨迹与异步状态机

单轨迹：

```text
ASSIGNED → RUNNING ⇄ WAITING_GPU → VERIFYING → REWARDED → BUFFERED
                                      └───────→ EXCLUDED（仅明确原因）
BUFFERED → RESERVED → CONSUMED | STALE_EXCLUDED | NOOP_CONSUMED
```

全局 GPU/发布状态：

```text
SERVING → QUIESCING_GPU → LEARNING → OPTIMIZED → PUBLISHING → SERVING
```

规则：

1. buffer 达到 `target_action_tokens`，或最老样本接近 lag 上限时，按完整轨迹 FIFO 预留 batch。
2. `QUIESCING_GPU` 只停止新 GPU prefill/decode；CPU 工具、测试、验收、ACK 和数据库写入继续。
3. SGLang 完成安全排水并释放 KV/权重占用；learner 获得 lease 后加载/恢复训练状态。
4. 更新提交后记录 action-token KL、导出不可变候选 adapter，再释放 learner GPU 占用并恢复推理。
5. 新轨迹使用已发布新版本；旧轨迹继续使用原版本直至结束。不能安全并存时，推迟发布而不是切换旧轨迹。
6. `max_policy_lag` 在 batch 预留和每个 optimizer 边界逐 call 校验；超限只记 `stale_excluded`，不记模型失败或零奖励。

长短队列按任务历史执行时间 EWMA 分类，未知任务进入短队列并设探索配额；两个队列均有老化提升，防止长任务饿死或占满 worker。

## 7. SPO 数学实现

### 7.1 价值、优势与遗忘

连续价值 V 跟踪最终训练奖励，`success_p` 只跟踪官方 resolved，二者不得混用。

初始 `V=prior_value`、`success_p=prior_success`、`N_eff=prior_strength`。默认先验均为 0.5、强度 1；所有任务初始采样权重相同。

每次策略更新在保存的 action tokens 上估计相邻 adapter 的 KL，并追加到 KL registry。任务下次观察前懒惰应用尚未消费的 KL 段：

```text
D = sum(kl_segment since task.kl_cursor)
ρ = 2^(-D / D_half)
V ← prior_value + ρ (V - prior_value)
success_p ← prior_success + ρ (success_p - prior_success)
N_eff ← ρ N_eff
```

跨多版本的 KL 段和是本项目异步采样的散度代理，不声称是非相邻策略的精确全词表 KL。估计器、token 数和版本链必须落盘。

随后严格按顺序：

```text
V_pre = 当前已遗忘、但未观察本样本奖励的 V
A = R - V_pre
η = 1 / (N_eff + 1)
V ← V_pre + η (R - V_pre)
success_p ← success_p + η (resolved - success_p)
N_eff ← N_eff + 1
```

raw A、V_pre、更新后 V 和 η 在同一数据库事务中提交。价值更新绝不在 batch 归一化或恢复时重复执行。

### 7.2 SPO++ action-token-measure 归一化

对 batch 中轨迹 `i`，`L_i=Σcalls Σtokens action_mask`，raw advantage 为 `A_i`：

```text
μ_tok = Σ_i L_i A_i / Σ_i L_i
σ_tok = sqrt(Σ_i L_i (A_i-μ_tok)^2 / Σ_i L_i)
Â_i = (A_i - μ_tok) / (σ_tok + ε)
```

冻结 μ、σ、L 和 Â。最终 token loss 仍除以实际参与该次全局更新的 `Σ action_mask`；梯度累积 microbatch 以各自 action-token 数占全局总量的比例加权，不能先对各 microbatch 独立归一化后平均。

### 7.3 PPO-Clip

行为 logprob 直接来自采样引擎，old policy 就是 behavior adapter；不再叠加旧实现的 `old/behavior` TIS cap。

```text
r_j = exp(logp_current_j - logp_behavior_j)
L_actor = -Σ_j m_j min(r_j Â_i, clip(r_j,1-ε,1+ε) Â_i) / Σ_j m_j
```

默认一次 PPO epoch、global grad norm 1.0。reference KL 正则默认关闭，避免不必要的前向；若启用，其 reference namespace、缓存和系数必须独立记录。只有真实非跳过 optimizer 更新且 adapter hash 变化才发布新版本。

### 7.4 任务采样

```text
w(x) = sqrt(success_p(x) (1-success_p(x))) + ε_sampling
```

权重乘仓库/类型覆盖和近期暴露惩罚，再归一化采样。健康退避只表示基础设施状态，不写入 success_p。已掌握题和长期失败题保留 epsilon 概率，不硬删除。

## 8. 奖励与过程证据

### 8.1 空补丁差分

空补丁基线按 TaskKey 和裁判环境 hash 缓存。候选和基线必须覆盖同一组官方要求节点；逐节点 map 只留在私有裁判存储，learner 只获得聚合值和 artifact hash。

- P：官方修复要求中，从基线非通过变为候选通过的节点数 / 基线非通过修复节点数。
- G：官方回归要求中，从基线通过变为候选非通过的节点数 / 基线通过回归节点数。
- 必需节点缺失、JUnit 缺失、基础设施异常或验证不完整：`eligible=false`。
- 官方声明某类节点为空：该分项 `not_applicable`，数值为 0，并保留状态；这与报告缺失不同。

### 8.2 有界复合奖励

用户给定分项的原始效用为：

```text
progress = max(0, P-G)
U = 0.60 resolved + 0.25 progress - 0.10 G + 0.05 validity
```

`U∈[-0.1,0.9]`。为同时保留给定权重/负向回归和 `[0,1]` 总奖励，统一使用常数平移：

```text
R_base = U + 0.10 ∈ [0,1]
```

平移是奖励 schema 的一部分，不在组内或批内动态缩放。`validity` 只来自补丁可应用性、协议和官方验证证据；不能由资格代码虚构为 1。若官方模型侧补丁拒绝没有完整测试报告，则按硬约束排除，而不是补零。

效率仅在 `resolved=1` 时启用：

```text
E = bounded_mean(exp(-gpu_seconds/τ_gpu),
                 exp(-action_tokens/τ_token),
                 exp(-tool_calls/τ_tool))
R_train = R_base + λ resolved (E-R_base)
```

默认 λ=0.05，失败轨迹严格使用 `R_base`，不存在“更快失败”奖励。效率证据缺失时仅停用该辅助项并令 `R_train=R_base`，不得因此丢弃已通过官方验收的轨迹。V 跟踪 `R_train`，同时保存 `R_base` 和 E 供审计。

### 8.3 过程评价

`ProcessEvidence` 至少包含：

- 目标文件定位：工具实际读取/编辑路径与裁判私有目标集合的聚合匹配，不向 agent 泄露参考路径。
- 相关函数/调用链：真实 LSP、搜索、读取或工具事件证据。
- 失败测试识别：实际测试命令、返回码和结构化输出。
- 修改后改善：候选与基线/修改前测试的差分。
- 回归保持：完整回归节点覆盖及通过证据。

过程结果用于奖励审计、失败分类、SFT/教师轨迹筛选和课程记录，不额外发明未定义的标量权重。任何“模型声称已定位/已测试”但没有执行证据的字段保持 unknown。

## 9. SGLang、双 A800 与 Adapter 双缓冲

### 9.1 推理后端

统一 `InferenceBackend`：`start/submit/cancel/drain/pause/resume/release_memory/load_adapter/unload_adapter/health/metrics`。

- 默认 `sglang`，TP=2，启用 continuous batching、radix/prefix cache、分层 KV 和内存预算。
- 保留 `vllm` 同接口实现；两者都必须返回原始 input/output IDs、实际采样 logprob、finish reason 和 adapter hash。
- 后端不支持 Gemma 4、指定 LoRA 或所需 token/logprob 字段时明确失败，不静默换后端或重算行为概率冒充采样证据。
- OpenAI 兼容层仅负责官方 ADK 协议，不作为丢失 token 信息的抽象边界。

### 9.2 Colocated GPU 协议

默认采用分时 colocated 模式：推理阶段 learner 状态在 CPU/NVMe；SGLang 占用双卡。更新时 SGLang 停止接纳、排水并 `release_memory_occupation`，确认权重/KV占用释放后 learner 上卡。更新结束释放 learner，再 `resume_memory_occupation` 并加载正确 adapter。

该协议不承诺双引擎同时常驻。显存记录必须包括基座、两份 adapter、梯度、Adam、KV、prefix tree workspace 和激活；adapter 文件大小不能替代峰值测量。

### 9.3 双缓冲所有权

- `training`：唯一可写 N+1，由 learner 持有。
- `rollout`：冻结 N，由推理持有；每条 trajectory 固定绑定。
- 候选 adapter 完成 hash、manifest 和恢复 checkpoint 后才可发布。
- resident 模式只有实测预算允许时启用，并按 adapter hash 隔离 KV/prefix cache。
- non-resident 模式在旧轨迹仍有 GPU 请求时继续服务 N，暂缓发布 N+1；不得给旧轨迹换权重。CPU 工作仍可并行。

旧 adapter 仅在其轨迹引用和训练 buffer 引用均释放后删除；删除另行授权，默认只解除注册。

## 10. Rank 128 更新路径

固定 BF16 基座、FP32 LoRA 参数、Rank 128、Alpha 256、Dropout 0、Fused AdamW、global norm 1.0、BF16 导出。

1. Policy 仅对 training adapter 参数开启梯度，rollout adapter 冻结。
2. 使用原始 prompt/completion IDs 和 action mask；行为概率不再通过第一次训练前向重建。
3. 按 action-token 预算组织完整轨迹 batch，按 token/tree workspace 切 microbatch 累积。
4. prefix tree 可跨任务共享完全相同 token 路径；每个原始 call 的 mask、行为和 advantage 保留独立 occurrence。
5. gradient checkpointing 覆盖 decoder/tree attention；不在反向时切换推理 adapter。
6. 优化器、scheduler、Python/NumPy/Torch/CUDA RNG、batch cursor、冻结优势和 KL registry 全部进入恢复状态。
7. 记录梯度范数、实际 LR、非零更新、权重 hash、policy lag、KL 估计和各卡峰值；这些仍不等于修复收益。

## 11. SFT、教师蒸馏与 OPD

所有数据源复用 `Trajectory/ActionCall` 和官方 VerificationBundle，来源字段为 `student_rl`、`verified_sft`、`teacher_offline` 或 `teacher_opd`。

### SFT 冷启动

- 只接受独立官方验收确认的成功修复、正确工具轨迹、可验证错误恢复和多轮测试后有效修改。
- 使用 masked CE；prompt、工具输出和环境消息不计损失。
- SFT 更新同样产生新 adapter 版本和 KL 段，在线 RL 的价值遗忘不能忽略这一阶段。

### 教师轨迹蒸馏

- 教师在相同官方工具/沙箱中执行，补丁通过完整官方验收后才进入数据集。
- 教师模型只提供轨迹，最终评估始终运行 Gemma 4。
- 外部教师目录、模型/endpoint、来源和配额显式配置；不维护自动下载脚本，不擅自调用付费服务。

### OPD

- 在学生 action 位置保存教师 top-k token/logprob、教师 tokenizer/template hash 和学生 tokenizer hash。
- 同 tokenizer 且逐 token 对齐时计算 masked forward-KL 蒸馏损失，与 SPO actor loss 显式分账加权。
- 不同 tokenizer 只有通过可验证的原始 span 对齐才允许接入；否则拒绝，禁止文本重 tokenize 后冒充 OPD。
- 蒸馏样本不伪造 SPO reward、V 或 A；需要进入 RL 的轨迹必须独立通过奖励资格。

## 12. 原子提交与恢复

更新事务顺序：

```text
预留完整轨迹 batch并冻结优势
→ 写更新前checkpoint
→ 每个optimizer边界checkpoint
→ 写不可变候选adapter和最终learner checkpoint
→ 原子移动run pointer
→ 标记buffer消费并提交KL registry
→ 发布serving adapter
```

恢复顺序：

1. 获取 run lock，核对 source/config/schema/hash，清理仅属于本 run 的孤儿进程，丢弃未提交 staging。
2. 恢复 StreamDB 中的轨迹、ACK、奖励、buffer、V/P、冻结 batch 和 KL 游标。
3. 恢复 training adapter、Adam、scheduler、optimizer counter 和全部 RNG。
4. `LEARNING` 从最近 optimizer 边界继续原 batch；`OPTIMIZED` 只重做消费/KL提交；`PUBLISHING` 只重做已提交 adapter 激活。
5. `queued/generating` 请求标记中断并以同 call identity 重排队；已 generated/delivered 可重放，已 consumed 不撤销。未完成官方 attempt 使用新 workspace，已完成独立轨迹保留。
6. 一致性检查完成后才获取 GPU、恢复引擎并解除 worker 的 GPU 等待。

任何阶段不得重复价值观察、重复 KL 遗忘、重算冻结优势或把旧 adapter 的 KV 用于新版本。

## 13. 配置基线

新配置使用 `algorithm="single_stream_spo"`，删除 `num_generations`、`groups_per_batch`、`max_candidate_groups_per_batch`、GRPO `num_iterations/loss_type/scale_rewards` 和旧奖励权重。

核心字段：

```toml
[policy]
lora_rank = 128
lora_alpha = 256
lora_dropout = 0.0
adapter_export_dtype = "bfloat16"
max_context = 32768

[inference]
engine = "sglang"
tensor_parallel_size = 2
colocated = true
enable_prefix_cache = true
return_token_ids = true
return_logprobs = true

[spo]
target_action_tokens = 32768
max_policy_lag = 2
ppo_epochs = 1
clip_epsilon = 0.2
clip_epsilon_high = 0.28
advantage_epsilon = 1.0e-6
value_prior = 0.5
success_prior = 0.5
prior_strength = 1.0
kl_half_life = 0.05
sampling_epsilon = 0.05

[optimizer]
type = "fused_adamw"
learning_rate = 5.0e-6
max_grad_norm = 1.0
gradient_checkpointing = true

[rewards]
objective_version = "official_differential_spo_v1"
success_efficiency_weight = 0.05

[adapter]
double_buffer = true
allow_resident_pair = false # 实测显存后才允许改为 true
```

`allow_resident_pair=false` 是安全默认，不是永久禁用双缓冲；版本所有权和磁盘双缓冲始终存在。

## 14. 迁移与实施顺序

1. 建立新 schema、配置和 StreamDB；旧 GRPO 状态只读留作历史，不提供兼容续跑。
2. 改 TITO/action mask/稳定工具 ID/request store，并完成独立 trajectory seal。
3. 拆官方 worker/验收，加入逐节点空补丁基线、差分奖励和过程证据。
4. 实现 V/P tracker、SPO sampler、token-measure advantage、buffer 和 PPO trainer。
5. 用异步 orchestrator 替代 collection/session；实现长短队列、完成即入队和 lag 控制。
6. 接 SGLang/vLLM backend、GPU lease、双 adapter、colocated sleep/wake 和完整恢复。
7. 接 Fused AdamW、梯度累积、prefix tree 和 BF16 exporter。
8. 接 SFT、教师蒸馏、OPD，复用同一轨迹和 adapter 版本链。
9. 更新 paired evaluation、CT 导出、README 和研究现状；删除不再使用的 GRPO 入口及配置。
10. 只做手动静态/CPU 核对和完成度审计；全部实现完成前不启动 A800。

## 15. 完成度与真实运行验收

代码完成审计必须逐项证明：独立轨迹、TITO、action mask、差分奖励、V/P 顺序、KL 遗忘、token-measure 归一化、PPO-Clip、lag、Rank 128、Fused AdamW、gradient accumulation/checkpointing、prefix tree、双 adapter、SGLang/vLLM、缓存、恢复、SFT、教师蒸馏、OPD、评估和导出均有实际实现与调用入口。

后续真实运行另行取得以下证据，不能用静态核对代替：

| 目标 | 必需真实证据 |
| --- | --- |
| 采样效率 | 同口径有效轨迹数/GPU小时及排除原因 |
| 环境效率 | 仓库/依赖缓存命中、工具和沙箱耗时分解 |
| 推理效率 | SGLang tokens/s、GPU利用率、batch和cache指标 |
| SPO正确性 | 具体样本V_pre/R/A/后值、冻结归一化和恢复一致性 |
| 奖励安全性 | 失败轨迹不随速度增加奖励，缺失证据无有效奖励 |
| Adapter更新 | Rank 128非零权重diff、实际LR/梯度和独立导出 |
| 稳定性 | KL、lag、梯度、版本链和中断恢复记录 |
| 模型能力 | 同题同seed基座/adapter完整官方验收，报告修复与退化 |
| CT部署 | 官方CT加载非零BF16 adapter及端到端结果 |

训练题成绩不称独立泛化；BF16 结果不替代 CT；SPO 在其他模型/数学任务的结果不当作本项目收益。

## 16. 参考资料

- [Single-stream Policy Optimization](https://arxiv.org/abs/2509.13232)：持久 KL-adaptive value tracker、全局优势归一化、不确定性采样和 PPO-Clip。
- [SPO++](https://arxiv.org/abs/2608.24870)：action-token-measure 均值/方差归一化及长轨迹权重问题。
- [GLM-5 技术报告](https://arxiv.org/abs/2602.15763)：异步 Agent RL 系统参考，不作为本项目独立 GPU 池的证据。
- [verl 文档](https://verl.readthedocs.io/en/latest/) 与 [Miles](https://github.com/radixark/miles)：colocated 推理/训练与异步系统参考。
- [SGLang issue #22243](https://github.com/sgl-project/sglang/issues/22243)：`release_memory_occupation` / `resume_memory_occupation` 与 KV 恢复边界；实际 Gemma 4/LoRA 支持仍以部署版本核对为准。
