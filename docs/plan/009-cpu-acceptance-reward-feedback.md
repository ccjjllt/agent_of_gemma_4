# 009 CPU 验收环境与奖励反馈修正

状态：2026-11-15，无卡实例（2 GiB cgroup），CPU-only。官方题面、`base_commit`、`test_patch`、参考补丁和官方验收语义未改；参考补丁仍只用于裁判侧正对照。

## 当前策略：移除研究前置门控（2026-11-16）

当前代码已删除 `require_positive_signal`：没有 P>0/shadow repair 的有效轨迹也能进入更新，恢复时不再因缺少正修复而丢弃已预留批次。shadow 保留为诊断；全零冻结优势仍不执行无效优化步。

新 run 的奖励默认 v3。完整、有效的候选官方结果不再因 baseline/P/G 不可用而整条排除；完整差分维持原公式，否则只用真实 resolved/validity（`0.10 + 0.60 × resolved + 0.05 × validity`），缺失 P/G 保持 N/A，audit 明确稀疏分支。v2 兼容旧实验，服务器冻结批次及其奖励、优势、身份均未修改。

已移除 Rank=128/Alpha=256、max_grad_norm=1 的固定值限制，以及双 adapter 驻留必须关闭的历史限制；默认参数不变，配置仍须满足真实服务容量和当前实现。删除对配置全文的旧关键字扫描及正式启动的重复部署探针。未接入的 SFT/蒸馏模式、当前 prefix-sharing 的零 dropout 要求仍如实说明，不冒称已实现。

继续保留授权预算/硬截止、公开数据与裁判资产隔离、来源身份、完整验收与模型调用、非有限数值拒绝、checkpoint/事务一致性。后续工作以冻结批次完成更新和 v0/v1 同口径修复对照为主，不把重复基础设施复核当作研究阶段。

下方第 0–8 节保留历史实测及当时的策略；其中“无正信号即 NOOP”“baseline 不可用即整条排除”不再描述新 v3 run 的行为。旧 run 续跑仍使用已部署 v2 奖励配置；新的本地主配置只用于新实验。2026-11-17 冻结批次完成记录、计时口径与下一阶段设计见第 9 节。

## 0. 历史结论

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
- 截至 CPU 复核轮没有正修复候选；2026-11-16 短 GPU 采样首次出现真实 `P=1.0`/shadow repair（见第 6 节）。`require_positive_signal` 已把“无信号不优化”从文档计划变成代码路径。

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

时间线（服务器时间，按 `metrics.jsonl` 的 `unix` 差）：13:47:28 首次启动 vLLM，因 venv 的 `ninja` 不在 PATH 于 13:54:09 失败；13:55:08 修正 PATH 后重启，13:59:23 就绪。14:00:18 启动训练；14:08:10 预留 `upd-0000`（`version_barrier_hold`，t+493）；**14:32:53 才进入训练租约**（`spo_training_memory_started`，t+1975），`spo_transaction_persisted` 在 14:33:21；14:47:29 被外部 60 分钟截止（`timeout --signal=TERM 2850`，exit 124）终止，训练租约内约 **876 秒**。

轨迹与奖励（17 条）：

- 状态：3 `RESERVED` + 3 `BUFFERED` + 11 `EXCLUDED`；有效（eligible）6 条，无效 11 条，collection failure（eligible 且 validity=0）0 条。
- 信号：`P>0` 1 条（`requests_7205`，P=1.0、total 0.9926，shadow repair measured **1/14**，`positive_repair_signal=true`）；`P=0` 5 条；`P` 缺失 11 条。
- `requests_7205` 的正式 `G=0` **不是“无回归”**：该任务 `expected_universes.regression=0`，分母为零、不可适用；`shadow_regression` 也因 `candidate_missing_baseline_passed_nodes`（204/205 匹配、candidate_only 1）为 `unavailable`。因此只能声明“修复”，不能声明“没有回归”。
- 三条候选通过要分开看（3/17 是候选通过比例，不是修复率）：

| 任务 | 原始对照 | 可得出的结论 |
| --- | --- | --- |
| `requests_7205` | 基线 1 个失败节点，候选转为通过 | 明确新增修复（P=1.0，shadow repair 1/14） |
| `requests_7315` | 空补丁本来就通过（`baseline_nonpassed_nodes=0`） | 保持通过，没有新增修复证据 |
| `rich_3480` | 候选通过（`success=True`、`test_exit_code=0`，109 个用例 0.33 秒）；空补丁在原始代码上因任务自身的无限循环耗尽 2 GiB cgroup 被 OOM（原批次为 300 秒包装超时）、无 JUnit | 候选验收成功，修复增量未知；空补丁不可执行，按规则不计奖励 |

- 门控：`upd-0000` 以 19,093 action tokens 预留，成员 advantage 为 `requests_7205 +1.6157`、`rich_4076 -0.6189`、`rich_4070 -0.6189`。这是首个带真实正修复信号的 GPU 批次，门控没有把 0.15 的总奖励当成无信号。
- 更新：训练确实启动（learner 载入、vLLM sleep），但租约内 876 秒后被杀；无 `spo_trainer_state.pt`、无 EXPORTED/REGISTERED/ACTIVATED、registry `active_version=0`。即本轮没有更新成功保存或发布 adapter。

失败分类（11 条无效，不能统一算作环境故障）：

- 6 条沙箱依赖安装失败（明确证据）：5 道 fastapi（`typer==0.16.0` 与 `fastapi-cli 0.0.34` 要求 `typer>=0.26.1` 冲突）、1 道 requests（`pytest==6.2.5` 需要 `py>=1.8.2`，wheelhouse 无该 wheel）。链路为 `ContainerSetupError` → agent 未产生 model call → `trajectory_seal_RequestUnavailable` → 奖励路径 `task_key` 异常覆盖了真实原因。
- 2 条候选验证不完整：`rich_3472` 的 pytest 退出码 2、无可归因 verdict；`fastapi_15745` agent 超过 8 分钟会话上限后 pytest 退出码 2。属于测试/收集错误与会话超时，不是依赖安装问题。
- 3 条 baseline 不可用：`fastapi_15023`、`requests_7433` 的候选测试已完成（exit 1），但空补丁 baseline 报告不完整；`rich_3480` 在 **同身份 `f242e208…`、单独无并行负载**下重跑空补丁，pytest 打印 108/109 个通过点后约 **29.7 秒被 SIGKILL（exit -9）**、无 JUnit。根因已查明：任务自带的 `test_append_loop_regression` 在原始代码上触发 `text._spans.extend(...)` 自引用无限循环，持续分配内存直至 2 GiB cgroup OOM；原批次那次是 300 秒包装超时（exit 124）。两种死法都说明**空补丁在这道题的原始代码上不可执行**，不是暂态超时。因此“统一增加 baseline 重试”没有充分依据；应按稳定测试错误/不可执行与暂态超时分类，再决定是否重试。

本轮同时修复了一个归因缺陷：官方 worker 在验证前失败时结果不含 `task_key`，`training/rollout/engine.py` 的 `_bundle` 直接 `JudgeTaskKey.from_dict(None)` 抛 `TypeError`，把基础设施失败记成奖励错误。修复后的回退只允许验证前失败记录缺 `task_key`；完整可训练报告仍必须携带完整身份（见第 7 节）。

资源与耗时：rollout 阶段显存峰值 76,811/76,767 MiB（94%），训练阶段峰值 62,483/65,923 MiB。预留到训练开始之间等待 **1,482 秒（24.7 分钟）**，包含长尾 rollout、重试、验收和最后一次约 5 分钟的 baseline probe；训练前等待（`run_spo.py` 的 `wait_for_workers()`）当时**没有超时约束**，现有 600 秒 barrier 限制不覆盖它。2026-11-15 上一轮 35,964 token 的更新实测 forward 431.27 秒 + backward 1,242.96 秒（`docs/plan/008-vllm-backend-migration.md` 第 10.2 节）；本轮 19,093 token 在 876 秒内未完成，和该量级一致。这两个数字都不是 11-17 单步耗时。

结论与边界：

- 学习信号存在但不是普遍现象：6 条有效轨迹中 1 条明确修复、1 条保持通过；`P=0` 的 4 条总奖励仍为 0.15。不能把单条正修复称为泛化收益。
- 60 分钟预算下，24.7 分钟等待 + 未完成的更新使“采样—更新—保存”无法同时完成。更新应按 2026-11-15 实测的训练窗口约 29.26 分钟（含 learner 加载，属该 run，不是 11-17 单步）加保存余量规划，而不是只留 15 分钟。
- `budget.json` 的 `consumed_seconds` 在 SIGTERM 后仍为 0，因为持久化只在正常收尾执行；续跑不能据此重新授予整份额度。
- 证据：`results/gpu-short-20261116/summary.json`（逐轨迹 P/G/validity、audit、updates、buffer、adapter、耗时与显存）和 `rich3480-probe/`（同身份空补丁重跑结果），原始 run 目录在服务器 `/root/autodl-tmp/gemma4/runs/spo-short-20261116`。

以上是 2026-11-16 会话的当时记录：该轮在 60 分钟硬截止时批次仍为 `RESERVED`，没有完成更新。`upd-0000` 于 2026-11-17 续跑完成并发布 v1；单步计时以第 9.1 节为准，本节的 876 秒是未完成的训练租约，不是一步的耗时。

## 7. 2026-11-16 无卡修链与下次 GPU 顺序

### 7.1 依赖联合选择

`training/task_dependency_setup.py` 在原有“逐包选最高版本”之后增加联合兼容 pass：读取已选 wheel 的 `Requires-Dist`，只允许在仓库 specifier 内调整被依赖包的版本，或降级/丢弃依赖方；每次动作写入 `joint_resolution`，无法由离线 wheel 满足的传递依赖写入 `missing_transitive_distributions`。仓库约束始终优先，不会为了消除冲突而放宽官方 pin。

真实任务验证（沙箱解释器 Python 3.13，真实快照与 wheelhouse）：

- `fastapi_14512`：原选择 `fastapi-cli 0.0.34 + typer 0.16.0` 冲突；把 `fastapi-cli 0.0.32`（要求 `typer>=0.16.0`）放入 staging 后，联合选择为 `fastapi-cli[standard-no-fastapi-cloud-cli]==0.0.32` + `typer==0.16.0`，无 residual/missing；`pip install --dry-run --no-index` 在 Python 3.13 下通过。
- `requests_6589`：`pytest==6.2.5` 需要 `py>=1.8.2` 与 `toml`；两者都不在原 wheelhouse。放入 staging 的 `py-1.11.0`、`toml-0.10.2` 后，dry-run 通过（Would install attrs/py/pytest 6.2.5/toml）。
- staging 目录：`/root/autodl-tmp/gemma4/wheels-staging-20261116`（`py-1.11.0`、`toml-0.10.2`、`fastapi_cli-0.0.32`）。**尚未并入 `wheels-extra`**，因为 extra_wheels 清单进入 judge 身份；并入会改变新 run 的环境身份。

`fastapi-cli` 0.0.33 起要求 `typer>=0.26.1`，0.0.32 是与 `typer==0.16.0` 兼容的最高版本；`rich-toolkit 0.20.6`、`rich 15.0.0` 已在 wheelhouse。若某题仍无法兼容，联合 pass 会记录丢弃原因而不是静默安装冲突组合。

全量回归（129 道公开训练题，沙箱 Python 3.13，逐题解包、串行、0 错误）：

- 旧 helper + 旧 wheel 目录 vs 新 helper + 新 wheel 目录：86 题选择完全一致，43 道 fastapi 题只发生 `fastapi-cli 0.0.34 -> 0.0.32` 一个 pin 变化，没有其它包或 extra 变化。
- 同一新 helper 但**不加 staging**：那 43 题会直接丢掉 `fastapi-cli`，4 道 requests 题报告 `py`/`toml` 缺失。因此 staging wheel 是“保留 fastapi-cli 并降级”的必要条件，不是可选优化。
- 新选择在 129 题上 `residual_conflicts`、`missing_transitive_distributions`、`unreadable_metadata` 均为空；`unsatisfied_constraints` 的 12 题与旧 helper 完全相同（既有 wheelhouse 缺口：`pytest-timeout>=2.4.0` 10 题、`pyjwt==2.8.0`、`httpx_3672` 的 `pytest==8.3.5`），不是本次引入。
- 结论：联合选择没有改变任何无关题目的选择，可以进入续跑阶段。

### 7.2 身份回退收紧

`training/rollout/engine.py` 的 `task_key` 回退现在只允许**验证前失败**的记录：完整 verdict（`verification_complete=True`、`verdict_valid=True`、无 `infrastructure_error`、`success` 为 bool）缺 TaskKey 直接失败；`candidate_report`/`baseline_report`/`baseline` 缺 TaskKey 一律失败。一次性核对覆盖 6 种情况（完整缺键、完整缺键+匹配 id、报告缺键、验证前失败+匹配 id、验证前失败+错误 id、匹配/不匹配 TaskKey），并确认生产者（`official_runtime.py` 的 `spo_bundle` 与空补丁 probe）都带 TaskKey。

### 7.3 预算与排空控制

- 新增 `[run] update_min_seconds`（默认 1800，两个配置均显式写入）。`_drain_update_window()` 只允许在 `deadline - now - update_min_seconds` 内等待在途 worker；超时则把批次留在 `RESERVED`、发 `spo_update_deferred`、不启动优化步。`do_update`、resume 的 `resume_post_step`/`resume_reserved`/`resume_learning` 和 final 更新都走同一约束；resume 路径预算不足时明确报错，提示用 `--additional-walltime-seconds` 续跑。
- `budget.json` 改为每 30 秒心跳持久化，并在 SIGTERM 处理器中先写盘；优化步进行中（`in_update`）只记录停止请求、不打断耐久事务，空闲时抛 `KeyboardInterrupt` 正常收尾。一次性核对：SIGTERM 在 +1200 秒时记录 `consumed≈1200`，resume 读到同一剩余量；缺失/损坏记录仍拒绝。
- 正常路径语义不变：等待在预算内、批次正常 settle 时仍照常训练。

### 7.4 rich_3480 空补丁的根因

同身份 `f242e208…`、单独无并行负载重跑已确认：任务自带的 `tests/test_text.py::test_append_loop_regression` 在原始代码上触发 `rich/text.py` 的 `self._spans.extend(...)` 自引用无限循环，持续分配直至 2 GiB cgroup OOM（exit -9，29.7 秒，108/109 通过，无 JUnit）；原批次那次是 300 秒包装超时（exit 124）。候选补丁把 `_spans` 复制后 109 个用例 0.33 秒全过。结论：**空补丁在该题原始代码上不可执行**，这是任务性质，不是暂态超时；不做统一 baseline 重试，也不为此改代码（提高内存只会延长循环）。

### 7.5 下一次 GPU 的顺序

1. 先用**原身份**完成冻结批次：保持服务器上的旧 `task_dependency_setup.py` 与旧 `extra_wheels` 不变，`--resume` 运行 `spo-short-20261116`。已用真实账本验证 `recovery_discard_reason(upd-0000, checkpoint=None, tx, require_positive_signal=True) == None`，即会从 v0 重做整步而不是丢弃。预算处理：`budget.json` 已在补账中记为 `consumed=2851`（一次性操作，有 `budget.json.pre-20261116` 备份），**不要再手动重写**——续跑只用 `--additional-walltime-seconds` 显式追加额度，让单调写入者按实际消耗继续累加；此时原 allowance 2400 已不足，需再用 `--additional-walltime-seconds 3000` 之类显式追加。按 2026-11-15 的 29.26 分钟训练窗口估算，allowance 5400、remaining 2549 可覆盖更新加保存余量。11-17 实际执行时 allowance 最终为 11400、结束时消费 5901；实际单步为 23.43 分钟优化（含加载 24.23 分钟），264 秒只是发布回放，见 9.1。
2. 旧批次保存/发布后，再同步新 helper，并把 `py`、`toml`、`fastapi-cli 0.0.32` 并入 `extra_wheels`。这会产生新的 judge 身份，旧 run 的 3 条 `BUFFERED` 轨迹不能混入新环境。
3. 新身份下做同口径 v0/v1 对照（同一批题、同一验收身份），有稳定更新与修复改善证据后再扩大训练；实际改为固定 6 题池的 5 步连续训练，见 9.2。

续跑命令（旧身份；2026-11-17 实际执行时 config 为 `configs/spo_gpu_short_60m.v2.toml`，见 9.1）：

```bash
cd /root/gemma4/project
export CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       TMPDIR=/root/autodl-tmp/gemma4/tmp
/root/gemma4/envs/learner/bin/python -m training.run_spo \
  --config configs/spo_gpu_short_60m.v2.toml \
  --data-root /root/gemma4/official/competition \
  --model-path /autodl-fs/data \
  --run-dir /root/autodl-tmp/gemma4/runs/spo-short-20261116 \
  --backend-url http://127.0.0.1:8000 \
  --resume --additional-walltime-seconds 3000
```

`--additional-walltime-seconds 3000` 只是示例：allowance 变为 5400 秒、扣除 2851 秒后 remaining 2549 秒；29 分钟只是 2026-11-15 训练窗口（29.26 分钟，含加载）的规划参照，不属 11-17 单步。实际续跑把 allowance 追加到 11400、结束时消费 5901，单步 23.43 分钟优化（含加载 24.23 分钟），264 秒只是发布回放（见 9.1）；实际额度由用户授权决定。续跑前 vLLM 必须已按主配置部署并处于 v0（run 的 genesis adapter 会重新载入）。**整个会话必须在 `scripts/session_supervisor.py` 下运行（见 8.4），预算从 vLLM 启动前开始计时。**

### 7.6 边界

以上均为 CPU 侧代码与选择验证，尚未在 GPU 上重跑，也没有新的能力提升结论。依赖修复的有效性目前只到“选择与 pip dry-run 通过”，真实沙箱安装与验收仍需下一次 GPU 会话确认。

## 8. 2026-11-16 硬截止与记账修复（第二轮 CPU）

### 8.1 SIGTERM 现在有界

原实现（`_make_sigterm_handler`）在 `in_update=True` 时对每个 SIGTERM 都直接返回，重复 TERM 也被吞掉，因此更新期间收到停止请求后仍可能跑完整步。现已改为：

- 第一次 TERM：持久化预算；空闲路径抛 `KeyboardInterrupt`；更新路径保留事务并设置看门狗。
- 看门狗在 `run.sigterm_grace_seconds`（默认 60，已写入两个配置）后仍未结束则 `exit 75`，预算先落盘。
- 第二次 TERM：立即落盘并以 75 退出，即使优化步仍在运行。
- 步在宽限内正常结束时原子地取消看门狗（`_finish_update_window`）。

独立复核：更新中两次 TERM → 立即 75；更新中一次 TERM、grace=1 → 约 1 秒后 75；空闲 TERM → `KeyboardInterrupt` 正常收尾；预算均已持久化。

### 8.2 预算写入单调

原 `_write_budget` 只保证单文件原子，不能保证写入顺序：旧的 10 可以在新的 20 之后落盘。现由单一 `_BudgetWriter` 线程拥有全部写入，`submit()` 取 `max(last_written, submitted)`，`flush()` 有界等待，`close()` 提交最终值并刷盘。独立复核：racy 提交 10/20 → 20；随后 15 → 20；`close(final=15)` → 20；5 线程压力测试无回退。

### 8.3 依赖解析不再“删必需包并宣称解决”

`_JointResolution` 原先在无兼容版本时会删除一个被仓库直接要求（`source in constraints`）或被某个已选包的基础 `Requires-Dist` 要求的包，然后报告 `resolved=true`。现已改为：

- 必需包永不删除；无解时保留当前选择，写入 `action="unresolved"` 的 `residual_conflicts`，`resolved=false`，原因包含冲突 requirement、已选版本和仓库 pin。
- 只有仅由 optional extra 拉入的包才可删除，且先移除该 extra 并记录。
- `run()` 在无法推进时停止并按最终选择重建 residual，不再空转 200 次。

合成用例复核：`req-a→dep-b>=2` 与 `req-c→dep-b<2` 同时必需时，旧代码删除 `req-c` 并 `resolved=true`；新代码保留 `req-c`、`resolved=false`、residual 指明 `dep-b<2`。可行 extra 场景仍只删 extra 并 `resolved=true`；`changed_version` 场景仍正常降级。真实 wheelhouse 复测已在实例恢复后完成：129/129 题、0 错误。fixed helper + 旧 wheels 与旧 helper 的 pin 完全一致（0 差异），但 43 道 fastapi 题各带一条 `action=unresolved` 的 residual（冲突被点名，不再静默）；fixed helper + staging wheels 则把这 43 题降到 `fastapi-cli 0.0.32`、`resolved=true`、0 residual。12 条 unsatisfied 与旧 helper 完全相同（既有缺口）。`fastapi_14512` → 0.0.32 + `typer 0.16.0`；`requests_6589` → `pytest 6.2.5`（`py`/`toml` 走 staging）。

### 8.4 外部会话监督器

新增 `scripts/session_supervisor.py`：从**子进程启动前**计时（因此 vLLM 加载计入预算），在 `budget-grace` 向进程组发 SIGTERM，在 `budget` 发 SIGKILL；宽限期在预算内；会话 JSON 记录计划/实际软停与强杀时间、退出码和是否升级。

独立复核：合作子进程 budget=3/grace=1 → 约 2.0 秒软停、2.06 秒退出 0、无强杀；忽略 TERM 的子进程 → 约 2.0 秒软停、3.001 秒 SIGKILL、退出 137；后台孙进程同组被杀，无残留。

第三轮补齐（收尾条件与逃逸覆盖）：直属进程退出不再等于收尾——监督器在子进程被回收后继续监督整棵树（进程组 + cgroup 成员 + 快照后代），组内有存活者时硬截止前不返回，硬截止时走强杀路径并记 `hard_killed`（137），不再把 leader 的退出码当成功。Linux 下建独立 cgroup `spo-sup-<pid>`（v2 `cgroup.kill`，v1 逐成员杀），`setsid` 逃逸也一并清扫；无可写 cgroup 的平台（macOS、只读挂载的本机）降级为组杀 + ppid 链快照清扫并如实记录。独立复核：shell 在 TERM 后退出但后台 `sleep 60` 存活 → 监督器等到 4.0 秒强杀，exit 137，无残留；`setsid` + 忽略 TERM 的孙进程（leader 只活 1 秒）→ `tree_pids_at_child_exit` 非空而组为空，照样清扫，exit 137，无残留。已知残留：无 cgroup 平台上，在单个 50 ms 轮询内完成 fork + setsid + leader 退出的后代无法被快照关联（生产负载远超该窗口）。

下次 GPU 会话必须在监督器内运行（示例；2026-11-17 实际续跑按此形状执行，config 为 `configs/spo_gpu_short_60m.v2.toml`，见 9.1）：

```bash
cat > /root/autodl-tmp/gemma4/logs/session-20261117.sh <<'SH'
#!/bin/bash
# vLLM 在后台启动，健康后运行续跑；退出时停止 vLLM。
/root/autodl-tmp/gemma4/logs/start-vllm-20261116.sh &
VLLM=$!
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8000/health >/dev/null && break; sleep 5; done
/root/gemma4/envs/learner/bin/python -m training.run_spo \
  --config configs/spo_gpu_short_60m.v2.toml \
  --data-root /root/gemma4/official/competition --model-path /autodl-fs/data \
  --run-dir /root/autodl-tmp/gemma4/runs/spo-short-20261116 \
  --backend-url http://127.0.0.1:8000 \
  --resume --additional-walltime-seconds 3000
code=$?
kill $VLLM 2>/dev/null
exit $code
SH
chmod +x /root/autodl-tmp/gemma4/logs/session-20261117.sh
python scripts/session_supervisor.py \
  --budget-seconds 3600 --grace-seconds 180 \
  --session-file /root/autodl-tmp/gemma4/logs/session-20261117.json \
  --log /root/autodl-tmp/gemma4/logs/session-20261117.log \
  -- bash /root/autodl-tmp/gemma4/logs/session-20261117.sh
```

以上是 11-16 的计划示例：supervisor 3600 秒（含 vLLM 加载与 180 秒宽限），run allowance 5400、已消费 2851，remaining 2549。11-17 实际最后一次续跑的 supervisor 预算为 2400 秒，264.85 秒后 exit 0；该 264 秒只回放 `EXPORTED` → `REGISTERED` → `ACTIVATED` → `CONSUMED`，没有 optimizer step，不能当作一步的训练耗时。注意软停按进程组发送，vLLM 也会收到 TERM；若此时更新尚未完成，可能停在可续跑的 EXPORTED/REGISTERED 状态而不是发布，因此正常运行应让 run 自身先于软停结束，supervisor 只作硬截止兜底。

## 9. 2026-11-17 冻结批次完成与下一阶段设计

### 9.1 2026-11-17 冻结批次完成

2026-11-16 会话在 60 分钟硬截止时把 19,093 action tokens 的 `upd-0000` 留在 `RESERVED`（第 6 节），2026-11-17 用原 run、原 judge 身份续跑完成。账本最终状态：

| 项 | 结果 |
| --- | --- |
| update | `upd-0000`：`RESERVED` → `LEARNING` → `OPTIMIZED` → `EXPORTED` → `REGISTERED` → `ACTIVATED` → `CONSUMED`，`optimizer_cursor=1` |
| checkpoint | `spo_trainer_state.pt`，11.6 GB（11,640,386,877 bytes） |
| adapter | v1 位于 `adapters/versions/00000001-ba11bdc9827bc24c`：parent v0、rank 128、`optimizer_step=1` |
| 预算 | 结束时 consumed 5901 / allowance 11400 秒；11-16 采样会话已消费 2851 秒 |
| 运行方式 | `configs/spo_gpu_short_60m.v2.toml`，`--resume --additional-walltime-seconds 3000` |

计时口径（`results/gpu-short-20261116/resume_summary.json`）：

| 阶段 | 时长 | 说明 |
| --- | ---: | --- |
| 优化步（训练租约内） | 1,406 秒 = **23.43 分钟** | forward + backward + optimizer + checkpoint + 导出；其中 forward 336.5 秒、backward 1,029.7 秒、optimizer 0.09 秒、checkpoint 22.1 秒（含 state 导出 7.0 秒、落盘 15.1 秒） |
| 含模型加载 | 1,454 秒 = **24.23 分钟** | resume 启动到 `OPTIMIZED` |
| 最后一次续跑 | 264.85 秒 = **4.4 分钟** | **只回放发布**：`EXPORTED` → `REGISTERED` → `ACTIVATED` → `CONSUMED`，无 optimizer step |
| 2026-11-16 采样会话 | 2,851 秒 | 60 分钟硬截止被杀，批次仍 `RESERVED`（第 6 节） |
| 2026-11-15 pilot（不同 run） | 训练窗口 29.26 分钟；冻结到租约 12.84 分钟 | `docs/plan/008-vllm-backend-migration.md` 第 10.2 节；不能套用到 11-17 步 |

264 秒不是单步耗时，4.4 分钟不能代表“更新完成”。11-17 一步的真实成本是 23.43 分钟优化（含加载 24.23 分钟）；29.26/12.84 只属 2026-11-15 的 pilot run，不能用来描述 11-17 的续跑。

三个失败与处理：

1. **checkpoint 写入 ENOSPC**：写到 9.74 GiB 时磁盘只剩 9.8 G，异步 checkpoint 写失败（`unexpected pos ...`），该次 attempt 未留下 durable step。处理：把已被取代的 `vllm-pilot-20261113g` run 的 900 行 metrics 备份到本地后删除该 run，并清理 11-15 之前的 sandbox，再重试。
2. **发布拷贝 ENOSPC**：优化与 checkpoint 已完成后，发布阶段复制 adapter 时再次空间不足。处理：继续释放同一批 run/sandbox 空间后重试发布。
3. **恢复一致性检查拒绝**：checkpoint 已完成一步、serving 仍是 genesis 时，`_verify_resume_consistency` 报 “trainer checkpoint claims completed steps but serving is still genesis”。`39cae48b` 改为只允许**恰好一个** `OPTIMIZED`/`EXPORTED`/`REGISTERED` 在途 update，且其 id 与 checkpoint 的 active update 相同、`completed == CONSUMED + 1`；其余情况仍 fail closed。

这只证明真实优化步与 v1 发布成立；v0/v1 的配对修复比较尚未执行，能力提升仍未测量。证据：`results/gpu-short-20261116/resume_summary.json`（由服务器 ledger、registry、metrics、trainer 事务、budget 与会话 JSON 只读重建）；服务器 run 目录为 `/root/autodl-tmp/gemma4/runs/spo-short-20261116`。

### 9.2 下一阶段：多步连续训练（rank 128）

目标是把单步快照推进到连续多步，观察同一固定题池在后期步的官方 verdict 是否优于早期步。**容量必须沿用 rank 128 / alpha 256**：rank 64 的 checkpoint 与 adapter 虽然省盘，但容量减半，与既有 v1 不可比，不作为正式实验。

前置条件只有一个：数据盘扩容。理由见 9.3（rank 128 单次 checkpoint 写入峰值 21.7 GiB，当前 50 GB 盘最大只能腾出约 18.5 GB）。扩容后按原有配置族新建 run（新 judge 身份），`max_batches` 即步数上限与学习率计划长度，一经启动不可改。

比较方法（与工具一致）：

- 同一固定题池、同一 judge 身份，按 `scripts/report_repair_trend.py --run-dir <副本> --compare A,B` 生成逐题 × 逐 step 表；step 取 `trajectories.behavior_version`，update 归属另列；缺证据显示 `-` 并计入 missing，不当作 0，也不做零差推断。
- 若相邻两步同题覆盖不足，按“早期步对后期步”池化比较，并列出缺失的题与步。
- 题池同时参与训练，结论是 **in-sample trend**，不是泛化结果；单步 run 的 v1（旧 judge 身份）只作独立参照。

### 9.3 rank 128 的磁盘约束（实测）

checkpoint 只在固定路径上原子替换，**不会累积**；累积的是每步 1.94 GB（rank 128）的 adapter。rank 128 的 checkpoint 为 11,640,386,877 B（10.84 GiB），写入需要同尺寸临时文件，**单次峰值 21.7 GiB**。该实例数据盘 50 GB，已用 43.4 GB（官方数据 21 GB + 运行时 9 GB + 两个 run 12 GB），最大可回收到约 18.5 GB，仍小于 21.7 GiB，因此 rank 128 在当前磁盘上无法完成一次 checkpoint 写入（11-17 已真实出现过一次 ENOSPC 失败）。回到 rank 128 有两条路：扩容数据盘，或把 checkpoint 外置。2026-11-17 采用后者并经用户授权：`[run] checkpoint_dir` 把 `spo_trainer_state.pt` 放到 `/root/autodl-fs`（200 GB，实测写 147 MB/s，10.84 GiB 约 75 秒），run 目录只留 adapter 与账本；本地盘因此不再承担 21.7 GiB 的写入峰值。同时加入 `training/adapter/retention.py` 与 `[run] prune_finished_artifacts`：run 计划跑完（全部 update `CONSUMED`）后自动保留最终 adapter、删除中间版本与 checkpoint，删除前用 registry 的 SHA256 校验保留版本。

### 9.4 产物保留策略（2026-11-17 起）

已完成 run 只保留**最终 adapter + 账本/指标/身份文件**；中间 adapter 版本、trainer checkpoint、staging 与 workspaces/requests/endpoint 属于过程产物，收尾时删除。用 `scripts/prune_run_artifacts.py` 执行：

```bash
python scripts/prune_run_artifacts.py --run-dir <run> --keep <最终版本> --dry-run
python scripts/prune_run_artifacts.py --run-dir <run> --keep <最终版本>
```

脚本的约束：只处理全部 update 均为 `CONSUMED` 的 run；永不删 active 版本或 `--keep` 指定的版本；删除前用 registry 里的 SHA256 校验待保留版本的权重；默认删除 checkpoint 与瞬态目录（`--keep-checkpoint` 可保留）；把删除清单写入 `pruned_artifacts.json`。registry 行**有意保留**，因此被清理的版本仍有 manifest 与血缘，只是没有权重文件——finished run 的 serving 与发布校验只读 active 版本。

首次应用：`spo-short-20261116` 保留 v1（`ba11bdc9…`，已校验 SHA256 一致），删除 v0 genesis 与瞬态目录，回收 2.009 GB；数据盘可用从 16 GB 升到 18 GB。rank 128 仍需 ≥21.7 GiB 的 checkpoint 写入峰值，扩容后才可启动多步 run（见 9.3）。

## 10. 2026-11-17 rank-128 五步 run 与修复趋势结论

### 10.1 运行与产物

`configs/spo_v3_5step_rank128_20261117.toml`：rank 128 / alpha 256、v3 奖励、`max_batches=5`、`walltime_seconds=20400`、固定 6 题公开池（`fastapi_14258`、`requests_7205`、`requests_7315`、`rich_3782`、`rich_4070`、`rich_4076`）。checkpoint 外置到 `/root/autodl-fs`（本地盘只有 50 GB，rank-128 的原子重写峰值 21.7 GiB 放不下），run 目录只留 adapter 与账本。19:28 在 `scripts/session_supervisor.py`（21600 s / grace 180 s）下启动，22:08 全部完成，run 退出码 0，supervisor `exited`、无升级。

五步实测（账本 + `spo_update_timing`）：

| update | tokens | 起止 | 墙钟 | forward | backward | checkpoint 写 | 优势值 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| `upd-0000` | 22,913 | 19:43:14 → 20:16:46 | 2,012 s | 471.8 s | 1,247.3 s | 89.7 s | +0.846 / −0.807 / −0.807 / +1.496 |
| `upd-0001` | 19,770 | 20:16:58 → 20:40:34 | 1,415 s | 233.9 s | 693.6 s | 77.3 s | −1.014 / +0.986 |
| `upd-0002` | 18,763 | 20:40:45 → 21:08:21 | 1,656 s | 242.6 s | 704.2 s | 75.6 s | +1.026 / −1.311 / −0.465 / +0.846 |
| `upd-0003` | 21,815 | 21:08:32 → 21:37:18 | 1,726 s | 286.0 s | 869.3 s | 75.5 s | −0.409 / +2.145 / −0.498 |
| `upd-0004` | 19,614 | 21:37:29 → 22:07:54 | 1,824 s | 279.2 s | 810.2 s | 97.5 s | −0.840 / −0.857 / +1.595 / +0.315 |

五步合计 2 小时 40 分。checkpoint 写共享盘 75–98 s（本地盘同尺寸约 15 s，但本地盘装不下）。run 结束后 `prune_finished_run` 保留 v3 与 v5、删除 v0/v1/v2/v4、checkpoint 与瞬态目录，回收 19.517 GB（本地盘 7 GB → 14 GB 可用，共享盘 checkpoint 归零）。

### 10.2 同口径修复对照

采样分布（`behavior_version`）：v0 9 条、v1 3 条、v2 1 条、v3 4 条、v4 4 条；21 条轨迹全部有完整官方 verdict（P/resolved/validity 覆盖 21/21）。

**v3（第 3 步后策略）对 v4（第 4 步后策略）**，共同任务 3 题：

| task | N_L | N_R | resolved_L | resolved_R | Δresolved | P_L | P_R | ΔP |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `fastapi_14258` | 1 | 1 | 0/1 | 0/1 | 0 | 0.000 | 0.000 | 0 |
| `requests_7205` | 1 | 1 | 1/1 | 1/1 | 0 | 1.000 | 1.000 | 0 |
| `requests_7315` | 1 | 1 | 1/1 | 1/1 | 0 | 0.000 | 0.000 | 0 |

**v0（基座）对 v4**，共同任务 4 题：

| task | N_L | N_R | resolved_L | resolved_R | Δresolved | P_L | P_R | ΔP |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `fastapi_14258` | 2 | 1 | 0/2 | 0/1 | 0 | 0.000 | 0.000 | 0 |
| `requests_7205` | 2 | 1 | 1/2 | 1/1 | +0.5（分母效应） | 1.000 | 1.000 | 0 |
| `requests_7315` | 2 | 1 | 2/2 | 1/1 | 0 | 0.000 | 0.000 | 0 |
| `rich_4076` | 1 | 1 | 0/1 | 0/1 | 0 | 0.000 | 0.000 | 0 |

结论：**五步训练在这 6 题上没有产生任何官方 verdict 变化**——没有一道题从 fail 转 pass 或从 pass 转 fail；`requests_7205` 的 +0.5 只是 1/2 变成 1/1 的分母变化，P 始终为 1.000。`requests_7315` 在所有采样里都是 `resolved=True` 但 `P=0.0`，属于该题既有的判定特征，不是训练带来的改变。

### 10.3 边界与未测项

- **v5 从未被采样**：采样发生在每步训练之前，run 在完成第 5 步后退出，因此不存在 `behavior_version=5` 的轨迹。严格的"第 3 步 vs 第 5 步"在本 run 结构上不可测，最接近的同口径对照是 v3 vs v4（3 题）与 v0 vs v4（4 题）。
- 每步每题只有 1–2 条轨迹，所有 Δ 都是单样本；`report_repair_trend.py` 对每条比较都打印了 N<2 警告。这些结果不能当作趋势强度，只能说明"没有观察到通过/失败翻转"。
- 题池同时参与训练，属 in-sample；不构成泛化结论。
- 若要测 v5 自身，需要额外的只做 rollout+裁判的评测入口（把 v5 adapter 加载进 vLLM 跑同一批题），本轮未实现。

证据：`results/gpu-5step-r128-20261117/`（`steps.json`、`report-compare-3v4.txt`、`report-compare-0v4.txt`、`report-at-step3.txt`），服务器 run 目录与账本保留。代码指纹：`training/run_spo.py` `04c90557…`、`training/adapter/retention.py` `0994e75d…`、`task_dependency_setup.py` `9a05542a…`、`configs/spo_v3_5step_rank128_20261117.toml` `f18d9721…`、`scripts/session_supervisor.py` `1a9988cf…`。
