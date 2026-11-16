# 009 CPU 验收环境与奖励反馈修正

状态：2026-11-15，无卡实例（2 GiB cgroup），CPU-only。官方题面、`base_commit`、`test_patch`、参考补丁和官方验收语义未改；参考补丁仍只用于裁判侧正对照。

## 0. 结论

本轮只做三件事，目的是让后续短 GPU 采样更容易产生“可观测的修复进展”，不是宣称已有修复收益。

1. **依赖覆盖**：修复 subprocess 任务环境剥掉版本约束、总选最高 wheel 的问题；补充官方 wheel 缓存缺少的测试依赖。老 FastAPI 提交不再因过新 Starlette 产生 `Router.__init__()` 收集失败。
2. **失败反馈**：候选补丁导致测试收集/导入失败时，保留完整、合法的官方 verdict 和训练资格，但把奖励的 `validity` 置 0，总奖励 0.15 → 0.10；普通未修复候选仍为 0.15。目标版本升为 `official_differential_spo_v2`。
3. **回归观测与优化门控**：公开数据没有 `PASS_TO_PASS`，官方 `G` 仍为 `not_applicable`；新增仅裁判侧可见的 exact 节点 `shadow_regression` 与 `shadow_repair`，分母分别是空补丁基线 passed / nonpassed 节点，缺失/重复/歧义不补零，只进 reward audit，不改官方 `G`、`eligible` 或损失。正式训练路径把聚合 shadow 写入 ledger audit；没有 `P>0` 也没有 shadow 正信号（audit 缺失时按 `RewardRecord.P` 判定）的前缀标为 `NOOP_CONSUMED`，不执行优化步。参数化节点名中的沙箱路径规范为 `<workspace>`。

复核后另修正 6 个实现问题：3 份候选的 baseline/candidate 改为同一奖励复核身份 `f242e208…` 重跑；正式训练路径保留 shadow audit；`-r/-c` include 与约束文件不再静默忽略；精确 pin 保留可离线满足的 extras，并记录丢弃的 extras；按 wheel `Requires-Python` 排除不兼容版本；任何重复 exact 节点（含同状态）都判 unusable，`matching_nodes` 只数实际匹配的基线 passed 节点。第二次复核又修正 3 条恢复路径问题：丢弃无信号批次不再删除已训练 trainer checkpoint；旧式 `LEARNING`+cursor 0 恢复先过正信号门；`require_positive_signal=false` 在 resume 同样生效（见 3.1）。本次复核再修正恢复后的 learner 重绑定顺序：scheduler 必须在加载 AdamW 状态之前构造（见 3.2），并把“0.15/0.10 即无信号”“缺 audit 即无信号”两处文档条件改成按 `P` 与 `shadow_repair`/`RewardRecord.P` 判断。证据见第 3、4 节和 `results/cpu-improve-20261115/`。

## 1. 依赖修复

旧 `training/official_runtime.py` 在任务沙箱里调用官方 `sandbox/setup.py`；该脚本把 `pyproject.toml` / `requirements*.txt` 的版本说明符剥掉，再对每个发行版选择最高 wheel。FastAPI 老提交的 `starlette<0.51` 因此被忽略，装上 1.x 后 `Router.__init__` 与提交代码不兼容。

新实现：

- `training/task_dependency_setup.py`：解析 `project.dependencies`、非 docs/lint 的 optional extras、`requirements-tests*.txt` 和 `dependency-groups`；递归执行 `-r/--requirement` 与 `-c/--constraint` include，约束文件只收窄版本、不新增发行版，其他 directive 记录而不静默丢弃；选出的显式版本用精确 `==` pin，**原始约束同时以临时 `-c` 文件传给 pip**，因此只约束 transitive 依赖（例如 Starlette）也能生效；按 PEP 508 交集选最高满足版本的 wheel；保留可离线满足的 extras，缺少可选依赖的 extras 明确记录并丢弃；按 wheel 的 `Requires-Python` 排除当前解释器不兼容的版本；安装后核对 `importlib.metadata.version`，不一致写入诊断；只使用本地 wheel，不联网。
- `training/official_runtime.py`：在官方 `sandbox/setup.py` 的 `setup_git_exclude` / `install_editable_package` / `execute_fast_path` 之上执行该选择；官方 setup 源文件不改。额外 wheel 目录通过 `runtime.extra_wheels` **复制**到任务沙箱 `/wheels`；源目录和 `official/` 都不写回，但这不是只读挂载，沙箱内的副本由任务 UID 可读。源资产本轮复核未发现写回。
- 额外 wheel 目录只含官方缓存缺少的公共测试包：`inline_snapshot`、`dirty_equals`、`attrs`、`asttokens`、`executing`、`markdown-it-py`、`pytest-timeout`、`python-multipart`、`email-validator`、`orjson`、`ujson`、`pydantic-settings`、`passlib`、`pyjwt`、`coverage`、`flask`、`uvicorn`、`pytest-asyncio`、`trio`、`pytest-mock`、`argon2-cffi` 等；具体清单和指纹记录在 `results/cpu-improve-20261115/`。

单任务验证：

| 任务 | 旧症状 | 新结果 |
| --- | --- | --- |
| `fastapi_11355` | `ModuleNotFoundError: inline_snapshot`，exit 2 | 空补丁正常执行，`test_stringified_annotations` failure，基线健康 |
| `fastapi_11194` | `TypeError: Router.__init__() got an unexpected keyword argument 'on_startup'` | 选择 Starlette 0.48.0，2 failed / 2 passed，基线健康 |
| `fastapi_14986` | 60 秒任务超时导致 SIGKILL，无结果 | 300 秒/8 分钟覆盖后正常完成 |
| `fastapi_14964` | 快照提取失败 | 正常完成，7 passed / 2 failed，基线健康 |
| `requests_7427` | 参数化节点路径导致 exact shadow 错位 | 基线健康；节点路径规范后 exact 配对可测 |

### 1.1 覆盖复测（目标集）

目标集是 51 道旧不健康题加 7 道旧健康对照，共 58 道；不是 58 道全部旧不健康，也不是另加 11 道。2 GiB cgroup 加单题约 40–70 秒，58 道全部完成，但这些结果跨 3 个 judge environment hash，不能把 58 份都当作同一环境测量。

下表的“最终身份”只指**覆盖复测阶段**的最终身份 `690819fb…`；它不是本轮奖励/同环境复核使用的 `f242e208…`。46/27/26 是覆盖阶段在 `690819fb…` 下直接测到的数字，不能称为 `f242e208…` 的覆盖结果，也不能把覆盖阶段与奖励复核阶段的数字相加或互相替代。这里的“健康”指空补丁测试可完整收集并产出合法 verdict（允许 failure），不要求全部通过。

| 指标 | 数值 |
| --- | ---: |
| 目标题数 | 58（51 旧不健康 + 7 旧健康对照） |
| 已完成 | 58 |
| **覆盖阶段最终身份 `690819fb…` 直接验证** | **46** |
| 上述身份新官方 verdict 有效 | 27 |
| 上述身份新空补丁健康 | 27 |
| 上述身份旧不健康 → 新健康 | 26 |
| 上述身份旧健康对照仍健康 | 1 |
| 上述身份仍未健康 | 19 |
| 仍使用更早身份的结果 | 12（12 健康；`1296aafc…` 11 题 = 5 恢复 + 6 对照；`13dd119d…` 1 题 = 1 恢复） |
| 新健康目标题合计 | 39（27 `690819fb…` + 12 更早身份） |
| 旧不健康 → 新健康合计 | 32（26 `690819fb…` + 6 更早身份） |
| 旧健康 → 新不健康 | 0 |
| 仍未健康 | 19（全部在 `690819fb…`） |
| 旧全集健康基线 | 63 |
| 全集健康估计 | 95（跨阶段估计） |

32 道 recovered 覆盖 `both_arms_shared_compatibility_candidate` 16 道和 `both_arms_same_missing_module` 16 道。`estimated_total_healthy=95` 的算法是：目标集外旧健康 56 + 目标集内新健康 39；其中 39 里的 27 来自 `690819fb…`、12 来自更早身份，因此 95 是跨阶段估计，不是完整 129 题在任一单一身份下逐题重跑后的计数。`results/cpu-improve-20261115/coverage_recovery.json` 现在逐题记录 `environment_hash` 与 `identity_stage`，并在 `summary.identity_stages.final_environment_hash` 固定为 `690819fb…`。

仍未健康的题并非都能靠依赖修复解决。已确认至少 3 题是 `test_patch` 导入了参考修复才新增的名字：

- `fastapi_15030`：`cannot import name 'EventSourceResponse' from 'fastapi.responses'`
- `fastapi_14978`：`APIRouter.__init__() got an unexpected keyword argument 'strict_content_type'`
- `fastapi_14609`：`cannot import name 'PydanticV1NotSupportedError' from 'fastapi.exceptions'`

这些题的空补丁在定义上无法健康；不应伪造基线，也不应把参考补丁当空补丁。其中 `fastapi_14485`、`14512`、`14962`、`15023` 仍有 collection/error，需下一轮逐个定位，本轮不归因于模型。另有 `fastapi_14186`、`14262`、`14605`、`15745`、`rich_3472` 在本轮仍为 0 passed，同样不能归因于模型。

## 2. 失败反馈

`training/rollout/engine.py` 的 `_validity()` 新增：

```text
agent_pytest_collection_failure / agent_pytest_import_failure
+ verdict_valid=True + verification_complete=True  ->  validity=0.0
```

官方 verdict 仍由 `training/verification_outcome.py` 生成，`verdict_valid`、`verification_complete`、`infrastructure_error`、`success`、`eligible` 不变；只是奖励的 validity 分量归零。目标版本从 `official_differential_spo_v1` 升为 `official_differential_spo_v2`，旧账本不重贴。

10 份历史候选的离线复算（只使用已保存的官方 verdict 和旧 P/G，不重新执行补丁）：

| verdict_kind | 数量 | 旧总奖励 | 新总奖励 | eligible |
| --- | ---: | ---: | ---: | --- |
| `agent_pytest_collection_failure` | 3 | 0.15 | 0.10 | true |
| `tests_completed` | 7 | 0.15 | 0.15 | true |

这证明 3 个“验收破坏”不再与普通未修复候选得到相同标量；仍然不是负奖励，也不改变官方成功判定。

## 3. 回归观测

官方 `G` 使用 `PASS_TO_PASS` 要求；公开 129 题没有该字段，因此 `G=0, status=not_applicable` 只表示未测。新增：

- `training/reward/junit_bridge.py:parse_junit_node_statuses()`：保留 exact `(classname, name)`，参数化后缀不丢；**任何重复 exact 键都返回 unusable，即使两行状态相同**，避免去重后虚增分母。
- `build_node_report(..., include_observed_nodes=True)`：在裁判侧报告中保存 `observed_nodes`、`observed_nodes_usable`。
- `training/reward/baseline.py`：空补丁缓存保留 `observed_nodes` 与 `observed_node_duplicates`；旧缓存缺重复计数或计数非 0 时仍可读，但 shadow 强制 `unavailable`，防止旧解析器去重后的缓存继续配对。
- `training/reward/differential.py:shadow_regression()`：分母 = 基线 passed 节点；候选缺任一基线 passed 节点即 `unavailable`；`matching_nodes` 只数实际匹配的基线 passed 节点，不再把基线 failed 节点算进覆盖；候选新增节点单独计数；不进入官方 `G`。
- `training/reward/differential.py:shadow_repair()`：分母 = 基线 nonpassed 节点，分子 = 候选中转为 passed 的节点；缺失仍 `unavailable`，不补零；这是质量门控的正修复信号来源。只有 `status=measured` 且分子 > 0 才计入 `positive_repair_signal`；`unavailable` 与 `not_applicable` 都不算。
- `training/reward/engine.py`：`spo_record_with_audit()` 把学习侧 `RewardRecord` 与裁判侧 audit 分开；`RewardRecord` 字段不变。
- `training/rollout/engine.py`：正式训练路径把 `shadow_regression`、`shadow_repair`、`positive_repair_signal` 和 cache/patch 指纹写入 ledger 的 `reward_audit` 事件，不再在 `spo_reward_record()` 处静默丢弃；测试节点名不进入该事件。
- `training/trajectory/ledger.py` 与 `buffer.py`：`require_positive_signal` 默认开启。没有 `P>0` 也没有 exact shadow repair 的 FIFO 前缀被标为 `NOOP_CONSUMED`，不产生 optimizer step、LR 推进或版本号增长。判定先看 `reward_audit.positive_repair_signal`；该字段不是 true（缺失或 false）时回退读取 `RewardRecord.P>0`；缺少 audit 时 `P>0` 仍是正信号。已持久化的无信号 `RESERVED` 批次在同 ID 重试或 `--resume` 时会被丢弃；恢复路径**不删除** trainer checkpoint，而是把仅属于该批次的 pre-step 快照改写为 committed base，保留 adapter/AdamW/scheduler/RNG，并在需要时先装回 trainer。旧式 `LEARNING`+cursor 0 恢复同样先过信号门，只有已运行过 optimizer 的 cursor=1 路径继续完成发布。

候选节点名中的沙箱绝对路径在 `official_runtime.py` 中规范为 `<workspace>`，使 Requests 三对参数化节点可用同一身份比较。

3 份真实历史补丁在同一奖励复核身份 `f242e208...` 下重跑（baseline 与 candidate 的完整 TaskKey 相等；旧表 baseline 与 `f242e208…` 不是同一身份，且旧表未记录 `environment_hash`，已移到 `reward_shadow_validation.previous-cross-environment.json`，不再作为当前证据）：

| 任务 | verdict_kind | P | validity | 总奖励 | shadow regression | shadow repair | 正信号 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `rich_4070` | `agent_pytest_collection_failure` | 0.0 | 0.0 | 0.10 | unavailable：候选只有 2 个节点，基线 117 个 passed 节点均未匹配 | unavailable：基线 7 个 nonpassed 节点未匹配 | false |
| `fastapi_14258` | `tests_completed` | 0.0 | 1.0 | 0.15 | not_applicable：基线 passed 分母为 0 | measured 0/1 | false |
| `rich_4076` | `tests_completed` | 0.0 | 1.0 | 0.15 | measured 0/20（不是旧表的 0/24；`matching_nodes` 已只数基线 passed） | measured 0/4 | false |

这同时验证了：官方 verdict 和 eligibility 不变；collection failure 的标量确实与普通未修复候选分开；shadow 只在 exact 覆盖完整时给数值，缺失时保持 unavailable；三份候选都没有 `P>0` 或 shadow repair，`require_positive_signal` 会把该批标为 NOOP，不会仅因 `max_batches=1` 自动更新。若批次只有单条轨迹，`zero_frozen_advantage` 也会独立产生 NOOP；信号门的独立证据应看 `gate_validation.json` / `gate_fallback_validation.json` 的多轨迹批次。正式 `RolloutEngine` 路径另用真实 `RewardEngine` 和 fake request manager 跑通：轨迹进入 `BUFFERED`，`P=1.0`，`reward_audit` 事件含 `positive_repair_signal=true` 且不含测试节点名；结果在 `fix_verification.json` 的 `formal_rollout`。这里的 `P=1.0` 是 CPU 合成奖励的接线证据（fake request manager），不是模型修复收益，也不能当作训练效果。

### 3.1 恢复路径修正（复核后）

复核指出三条恢复路径问题，已按“任何路径都不得用无信号批次推进 optimizer，也不得删除已训练状态”修正：

1. **丢弃批次不再删除训练状态**：旧代码在 `--resume` 丢弃无信号 `RESERVED` 批次时，只要完整 checkpoint 的 active batch 指向该批次就 unlink `spo_trainer_state.pt`；该文件同时保存此前已训练的 adapter、AdamW moments、scheduler 和 RNG。现在改为把 pre-step 快照改写为 committed base（`active_batch=None`，`completed_optimizer_steps` 不变），原子写入成功后才删除 ledger 行；删除 ledger 行后，若 base 含已训练状态，在 training lease 内 `restore_base_state()` 装回 fresh trainer。checkpoint 属于更早 committed update 时保持字节不变。
2. **旧式 `LEARNING`+cursor 0 恢复先过信号门**：恢复不再直接 `resume_batch()` 跑完整个 optimizer step；`recovery_discard_reason()` 要求 ledger phase 为 RESERVED/LEARNING、ledger cursor 0、无 post-step 事务、无 `P>0`/shadow repair；对 LEARNING 还要求 checkpoint cursor 0。事务声称 cursor 1 但 checkpoint 不能证明时 fail closed；checkpoint 声称拥有该 update 但 cursor 不可读时也拒绝回滚。只有 cursor-1 checkpoint 佐证的 post-step 状态才继续走完成发布路径。事务的 `completed_optimizer_steps == offset + optimizer_cursor` 和 batch offset 一致性由非回滚的 `SPOTrainer._merge_transaction()` 在合并前校验，不满足直接拒绝。
3. **显式关闭选项在 resume 也生效**：`spo.require_positive_signal=false` 现在同时关闭 reserve 与 resume 的回滚；该丢弃在开关为 true（默认，当前配置显式设为 true）时执行，显式设为 false 时 reserve 与 resume 都不回滚。注意全零 frozen advantage 的 NOOP 路径不依赖该开关。`require_positive_signal` 不参与 trainer checkpoint 的配置等价检查（它是数据门控，不改模型、优化器或调度），因此可以在已有 run 上显式切换，resume 不会报“trainer configuration differs”。

另封闭两个崩溃窗口：① rebase 已落盘但 ledger 行未删时，checkpoint 的 `checkpoint_phase=discarded_unoptimized_batch` 标记会在下一次 resume 完成丢弃，`LEARNING` 行不再被 “belongs to None” 卡死；② 若 checkpoint 仍指向已删除 update 的 cursor-0 快照（例如 `reserve_update` 重试路径），`restore_committed_base()` 把它当作 committed base 用 `restore_base_state()` 恢复，而不是静默从 step 0 重新开始；未知 owner 的 cursor-1 快照直接拒绝。

一次性核对（`results/cpu-improve-20261115/recovery_path_validation.json`）：pre-step checkpoint 被改写后 adapter/AdamW 逐张量相等，fresh trainer 恢复 `completed_optimizer_steps=3`、`last_epoch=3` 且 `exp_avg` 非零；`LEARNING`+cursor 0 经真实 ledger 回滚成功；cursor 1 checkpoint 被拒绝且当时 ledger 行保留（`cursor1_row_preserved=true`，随后清理）；已 rebase 的未完成丢弃可在不重写 base 的情况下完成；orphaned pre-step owner 可作为 committed base 恢复；`require_positive_signal` 切换不破坏配置等价；属于更早 committed update 的 checkpoint SHA256 不变。该核对使用 CPU fake policy 和真实 ledger/trainer 代码，不调用模型，也不构成训练效果证据。

### 3.2 恢复后的 learner 重绑定（复核后）

`suspend_learner_binding()` 保存的 AdamW 状态带有恢复时的当前 LR（例如第 3 步的 `0.0008333…`，不是 base `0.001`），而 `make_scheduler()` 要求 fresh optimizer 仍在 base LR。旧 `bind_parameters()` 先 `optimizer.load_state_dict()` 再 `make_scheduler()`，所以恢复后的下一次 learner 重绑定会报 `Construct the LR scheduler on a fresh optimizer at the declared base_lr`；generation-change 分支还会把旧 scheduler 留在旧 optimizer 上，后续 `scheduler.step()` 不再驱动新 optimizer 的 LR。

现在两条路径统一走 `_rebuild_optimizer_binding()`：fresh AdamW → `make_scheduler()`（base LR）→ `optimizer.load_state_dict()`（恢复当前 LR 与动量）→ `scheduler.load_state_dict()`（恢复 `last_epoch`/`base_lrs`）。若 AdamW 有动量但缺少 scheduler 状态，在替换 optimizer 之前直接拒绝，不静默丢失调度。一次性核对（`results/cpu-improve-20261115/rebind_order_validation.json`）在真实 `SPOTrainer` 上跑 3 步后分别走 `restore_base_state → suspend → bind`、直接 suspended-state 和 generation-change 三条重绑定路径：LR 都恢复为 `0.0008333…`、`last_epoch=3`、scheduler 绑定新 optimizer、AdamW 动量保留；重绑定后的下一步 LR 与未 suspend 的对照同为 `0.0006666…`。修复前同一脚本在 suspended 路径报上述 `ValueError`，generation 路径的 scheduler 仍绑定旧 optimizer。

## 4. 证据与边界

结果目录：`results/cpu-improve-20261115/`。包含 `summary.json`、覆盖复测 `coverage_recovery.json`、3 份真实候选的同环境重跑 `e2e_same_environment.json`、`reward_shadow_validation.json`（旧跨环境表保留为 `reward_shadow_validation.previous-cross-environment.json`）、`dependency_selection_evidence.json`（真实 FastAPI extras/constraints 选择）、`gate_validation.json`（NOOP/正信号门控）、`fix_verification.json`（约束/Requires-Python/重复节点/matching/正式 audit 的一次性核对）、`recovery_path_validation.json`（恢复路径的 checkpoint/ledger/trainer 一次性核对）、`rebind_order_validation.json`（恢复后 suspended/generation 重绑定的 LR、scheduler 与 AdamW 核对）、`gate_fallback_validation.json`（总奖励 0.15 下按 shadow 正修复/P>0 判定门控的一次性核对）和额外 wheel 清单/指纹 `wheels-extra-manifest.json`。

边界：

- 2 GiB cgroup 内存限制导致不能无限并行；58/58 道目标题完成，但只有覆盖阶段最终身份 `690819fb…` 的 46 道是同环境测量：27 健康、26 恢复、1 道旧健康对照、19 仍不健康。另 12 道结果来自更早身份（`1296aafc…` 11 题 = 5 恢复 + 6 对照；`13dd119d…` 1 题 = 1 恢复），全部健康。这些数字不是 `f242e208…` 的覆盖结果；`f242e208…` 另有这 3 道候选的空补丁 baseline，但不是目标集覆盖扫测。
- `estimated_total_healthy=95` 是跨阶段估计（目标集外旧健康 56 + 目标集内新健康 39；39 中 27 来自 `690819fb…`、12 来自更早身份，并假设目标集外 56 道旧健康题未回归），不是任一单一身份下完整 129 题重跑；不能称最终环境覆盖率。
- 新环境只证明这些公开题在本机 CPU 配置下可执行；不证明官方 GPU 主机或 heldout 的覆盖率。
- `shadow_regression`/`shadow_repair` 不是官方 `PASS_TO_PASS`，不能替换官方 `G`，也不能单独当作模型修复收益。
- 10 份历史候选的 0.15→0.10 是 verdict 级复算；其中 3 份真实补丁已在同一奖励复核身份 `f242e208…` 下重跑，其余 7 份未在本次重跑。
- 本轮仍没有正修复候选；下一步仍是短 GPU 采样，寻找真实局部修复或最终成功，再谈扩大训练。`require_positive_signal` 已把“无信号不优化”从文档计划变成代码路径。

## 5. 短 GPU 采样方案（2026-11-16 已按此执行，结果见第 6 节）

首轮条件：

- 代码：`7783b44e`（依赖约束、shadow audit、正信号门控修正、恢复路径保留已训练状态）；新 run 目录，沿用同一 v0 和同一公开题集。
- 预算：单次 45–60 分钟，`max_batches=1`，`target_action_tokens=16384`，其余 Rank 128、奖励权重、8 分钟/28 工具调用不变；不再用“缩小 token 阈值制造梯度”。
- 通过条件：至少一条轨迹出现可观测的 `P>0` 或 `shadow_repair` 正修复。总奖励仍为 0.15/0.10 并不代表没有 shadow 正修复；只有没有任何轨迹 `P>0`（`P=0` 或不可用），且 audit 没有正信号（`positive_repair_signal` 为 false 或缺失）时，`require_positive_signal` 才把该前缀标为 `NOOP_CONSUMED`，不启动优化步，只记录该批没有质量排序信号；门控不读取 `total_reward`。`gate_fallback_validation.json` 在总奖励保持 0.15 时分别验证 shadow 正修复和无 audit 的 `P>0` 都会 reserve，`P=0` 且无 audit 正信号才 NOOP。
- 门控实现：`TrajectoryBuffer`/`TrajectoryLedger.reserve_update(require_positive_signal=True)` 在冻结 advantage 前检查 `reward_audit` 的 `positive_repair_signal`，该字段不是 true（缺失或 false）时回退读取 `RewardRecord.P>0`；`max_batches=1` 不再足以单独触发更新。旧 run 没有 audit 时 shadow 证据不可补算，只能按 `RewardRecord.P` 判定：`P>0` 仍是正信号，`P=0` 按无信号处理（不是断言 shadow 一定为 0），不静默放行。
- 失败条件：collection failure 仍可训练但为 0.10；基础设施失败、验证不完整、baseline 不可用继续排除，不补零。
- 产出：轨迹数、有效轨迹数、P/shadow 分布、collection failure 数、耗时和显存；不把训练题成绩称独立泛化。

## 6. 2026-11-16 短 GPU 采样实测（60 分钟，exit 124）

环境：2×A800 80GB；vLLM 0.32.0，TP=2、`max-model-len 32768`、LoRA rank 128、2 slot、sleep level 2、prefix cache。新配置 `configs/spo_gpu_short_60m.toml`（`max_batches=1`、`target_action_tokens=16384`、`walltime_seconds=2400`、`update_reserve_seconds=840`，其余与主配置一致），新 run `spo-short-20261116`，从 v0 开始，直接读官方公开题。代码内容为 `7783b44e`。

时间线（服务器时间）：13:47:28 首次启动 vLLM，因 venv 的 `ninja` 不在 PATH 于 13:54:09 失败；13:55:08 修正 PATH 后重启，13:59:23 就绪。14:00:18 启动训练；14:08:31 预留 `upd-0000`；14:33:13 才进入训练租约；14:47:29 被外部 60 分钟截止（`timeout --signal=TERM 2850`，exit 124）终止，当时仍在优化步内。

轨迹与奖励（17 条）：

- 状态：3 `RESERVED` + 3 `BUFFERED` + 11 `EXCLUDED`；有效（eligible）6 条，无效 11 条，collection failure 0 条。
- 信号：`P>0` 1 条（`requests_7205`，P=1.0、G=0.0、total 0.9926，shadow repair measured 1/14，`positive_repair_signal=true`）；`P=0` 5 条；`P` 缺失 11 条。另有 `requests_7315` 最终成功（P=0、total 0.7548），仍停在 `BUFFERED`，未进入该批。
- 门控：`upd-0000` 以 19,093 action tokens 预留，成员 advantage 为 `requests_7205 +1.6157`、`rich_4076 -0.6189`、`rich_4070 -0.6189`。这是首个带真实正修复信号的 GPU 批次，门控没有把 0.15 的总奖励当成无信号。
- 更新：训练确实启动（learner 载入、vLLM sleep、`spo_training_memory_started`），但 876 秒后被杀；无 `spo_trainer_state.pt`、无 EXPORTED/REGISTERED/ACTIVATED、registry `active_version=0`。即本轮没有更新成功保存或发布 adapter。

失败分类（11 条无效）：

- 6 条沙箱依赖安装失败：5 道 fastapi（`typer==0.16.0` 与 `fastapi-cli 0.0.34` 要求 `typer>=0.26.1` 冲突）、1 道 requests（`pytest==6.2.5` 需要 `py>=1.8.2`，wheelhouse 无该 wheel）。链路为 `ContainerSetupError` → agent 未产生 model call → `trajectory_seal_RequestUnavailable` → 奖励路径 `task_key` 异常覆盖了真实原因。
- 3 条 baseline 不可用（`fastapi_15023`、`requests_7433`、`rich_3480`）。
- 2 条候选验证不完整（`rich_3472`、`fastapi_15745`）。

本轮同时修复了一个归因缺陷：官方 worker 在验证前失败时结果不含 `task_key`，`training/rollout/engine.py` 的 `_bundle` 直接 `JudgeTaskKey.from_dict(None)` 抛 `TypeError`，把基础设施失败记成奖励错误；现改为缺字段时回退比对 `task_id`，字段存在时仍做严格 TaskKey 相等校验。

资源与耗时：rollout 阶段显存峰值 76,811/76,767 MiB（94%），训练阶段峰值 62,483/65,923 MiB；预留到训练开始之间排空 1,482 秒（24.7 分钟），是长尾 rollout 与一次 `rich_3480` 空补丁 probe 造成的。上一轮 35,964 token 的更新实测 forward 431 秒 + backward 1,243 秒，本轮 19,093 token 在 876 秒内未完成，和该量级一致。

结论与边界：

- 学习信号存在但不是普遍现象：6 条有效轨迹中 1 条 `P>0`、1 条最终成功；`P=0` 的 4 条总奖励仍为 0.15。不能把单条正修复称为泛化收益。
- 60 分钟预算下，排空 24.7 分钟 + 更新约 15 分钟使“采样—更新—保存”无法同时完成；下一轮应先修依赖选择、压缩排空（预留前先让在途轨迹收敛或限制 drain），再给更新留足时间，而不是扩大训练。
- `budget.json` 的 `consumed_seconds` 在 SIGTERM 后仍为 0，因为持久化只在正常收尾执行；续跑不能据此重新授予整份额度，需先修周期持久化或显式扣除。
- 证据：`results/gpu-short-20261116/summary.json`（逐轨迹 P/G/validity、audit、updates、buffer、adapter、耗时与显存），原始 run 目录在服务器 `/root/autodl-tmp/gemma4/runs/spo-short-20261116`。
