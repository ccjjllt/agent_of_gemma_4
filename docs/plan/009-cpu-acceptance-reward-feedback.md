# 009 CPU 验收环境与奖励反馈修正

状态：2026-11-15，无卡实例（2 GiB cgroup），CPU-only。官方题面、`base_commit`、`test_patch`、参考补丁和官方验收语义未改；参考补丁仍只用于裁判侧正对照。

## 0. 结论

本轮只做三件事，目的是让后续短 GPU 采样更容易产生“可观测的修复进展”，不是宣称已有修复收益。

1. **依赖覆盖**：修复 subprocess 任务环境剥掉版本约束、总选最高 wheel 的问题；补充官方 wheel 缓存缺少的测试依赖。老 FastAPI 提交不再因过新 Starlette 产生 `Router.__init__()` 收集失败。
2. **失败反馈**：候选补丁导致测试收集/导入失败时，保留完整、合法的官方 verdict 和训练资格，但把奖励的 `validity` 置 0，总奖励 0.15 → 0.10；普通未修复候选仍为 0.15。目标版本升为 `official_differential_spo_v2`。
3. **回归观测与优化门控**：公开数据没有 `PASS_TO_PASS`，官方 `G` 仍为 `not_applicable`；新增仅裁判侧可见的 exact 节点 `shadow_regression` 与 `shadow_repair`，分母分别是空补丁基线 passed / nonpassed 节点，缺失/重复/歧义不补零，只进 reward audit，不改官方 `G`、`eligible` 或损失。正式训练路径把聚合 shadow 写入 ledger audit；没有 `P>0` 也没有 shadow repair 的前缀标为 `NOOP_CONSUMED`，不执行优化步。参数化节点名中的沙箱路径规范为 `<workspace>`。

复核后另修正 6 个实现问题：3 份候选的 baseline/candidate 改为同一最终环境身份重跑；正式训练路径保留 shadow audit；`-r/-c` include 与约束文件不再静默忽略；精确 pin 保留可离线满足的 extras，并记录丢弃的 extras；按 wheel `Requires-Python` 排除不兼容版本；任何重复 exact 节点（含同状态）都判 unusable，`matching_nodes` 只数实际匹配的基线 passed 节点。证据见第 3、4 节和 `results/cpu-improve-20261115/`。

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

| 指标 | 数值 |
| --- | ---: |
| 目标题数 | 58（51 旧不健康 + 7 旧健康对照） |
| 已完成 | 58 |
| **最终环境身份直接验证** | **46** |
| 最终身份新官方 verdict 有效 | 27 |
| 最终身份新空补丁健康 | 27 |
| 最终身份旧不健康 → 新健康 | 26 |
| 最终身份旧健康对照仍健康 | 1 |
| 最终身份仍未健康 | 19 |
| 仍使用旧环境身份的结果 | 12（12 健康；其中 6 恢复、6 旧健康对照） |
| 新健康目标题合计 | 39（27 最终 + 12 旧身份） |
| 旧不健康 → 新健康合计 | 32（26 最终 + 6 旧身份） |
| 旧健康 → 新不健康 | 0 |
| 仍未健康 | 19（全部在最终身份） |
| 旧全集健康基线 | 63 |
| 全集健康估计 | 95（跨阶段估计） |

32 道 recovered 覆盖 `both_arms_shared_compatibility_candidate` 16 道和 `both_arms_same_missing_module` 16 道。`estimated_total_healthy=95` 的算法是：目标集外旧健康 56 + 目标集内新健康 39；其中 12 来自旧环境身份，因此它是跨阶段估计，不是完整 129 题在最终身份下逐题重跑后的计数。`results/cpu-improve-20261115/coverage_recovery.json` 现在逐题记录 `environment_hash` 与 `identity_stage`。

仍未健康的题并非都能靠依赖修复解决。已确认至少 3 题是 `test_patch` 导入了参考修复才新增的名字：

- `fastapi_15030`：`cannot import name 'EventSourceResponse' from 'fastapi.responses'`
- `fastapi_14978`：`APIRouter.__init__() got an unexpected keyword argument 'strict_content_type'`
- `fastapi_14609`：`cannot import name 'PydanticV1NotSupportedError' from 'fastapi.exceptions'`

这些题的空补丁在定义上无法健康；不应伪造基线，也不应把参考补丁当空补丁。剩余 `fastapi_14485`、`14512`、`14962`、`15023` 仍有 collection/error，需下一轮逐个定位，本轮不归因于模型。

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
- `training/reward/differential.py:shadow_repair()`：分母 = 基线 nonpassed 节点，分子 = 候选中转为 passed 的节点；缺失仍 `unavailable`，不补零；这是质量门控的正修复信号来源。
- `training/reward/engine.py`：`spo_record_with_audit()` 把学习侧 `RewardRecord` 与裁判侧 audit 分开；`RewardRecord` 字段不变。
- `training/rollout/engine.py`：正式训练路径把 `shadow_regression`、`shadow_repair`、`positive_repair_signal` 和 cache/patch 指纹写入 ledger 的 `reward_audit` 事件，不再在 `spo_reward_record()` 处静默丢弃；测试节点名不进入该事件。
- `training/trajectory/ledger.py` 与 `buffer.py`：`require_positive_signal` 默认开启。没有 `P>0` 也没有 exact shadow repair 的 FIFO 前缀被标为 `NOOP_CONSUMED`，不产生 optimizer step、LR 推进或版本号增长。已持久化的无信号 `RESERVED` 批次在同 ID 重试或 `--resume` 时会被丢弃并清理未执行的 trainer checkpoint；只有已运行过 optimizer 的 cursor=1 路径继续完成发布。缺失 audit 时回退读取 `RewardRecord.P`，崩溃窗口不会丢掉已知正信号。

候选节点名中的沙箱绝对路径在 `official_runtime.py` 中规范为 `<workspace>`，使 Requests 三对参数化节点可用同一身份比较。

3 份真实历史补丁在同一最终环境身份 `f242e208...` 下重跑（baseline 与 candidate 的完整 TaskKey 相等；旧表使用的 baseline 属于 `1296aafc...` 环境，已移到 `reward_shadow_validation.previous-cross-environment.json`，不再作为当前证据）：

| 任务 | verdict_kind | P | validity | 总奖励 | shadow regression | shadow repair | 正信号 |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `rich_4070` | `agent_pytest_collection_failure` | 0.0 | 0.0 | 0.10 | unavailable：候选只有 2 个节点，基线 117 个 passed 节点均未匹配 | unavailable：基线 7 个 nonpassed 节点未匹配 | false |
| `fastapi_14258` | `tests_completed` | 0.0 | 1.0 | 0.15 | not_applicable：基线 passed 分母为 0 | measured 0/1 | false |
| `rich_4076` | `tests_completed` | 0.0 | 1.0 | 0.15 | measured 0/20（不是旧表的 0/24；`matching_nodes` 已只数基线 passed） | measured 0/4 | false |

这同时验证了：官方 verdict 和 eligibility 不变；collection failure 的标量确实与普通未修复候选分开；shadow 只在 exact 覆盖完整时给数值，缺失时保持 unavailable；三份候选都没有 `P>0` 或 shadow repair，`require_positive_signal` 会把该批标为 NOOP，不会仅因 `max_batches=1` 自动更新。正式 `RolloutEngine` 路径另用真实 `RewardEngine` 和 fake request manager 跑通：轨迹进入 `BUFFERED`，`P=1.0`，`reward_audit` 事件含 `positive_repair_signal=true` 且不含测试节点名；结果在 `fix_verification.json` 的 `formal_rollout`。

## 4. 证据与边界

结果目录：`results/cpu-improve-20261115/`。包含 `summary.json`、覆盖复测 `coverage_recovery.json`、3 份真实候选的同环境重跑 `e2e_same_environment.json`、`reward_shadow_validation.json`（旧跨环境表保留为 `reward_shadow_validation.previous-cross-environment.json`）、`dependency_selection_evidence.json`（真实 FastAPI extras/constraints 选择）、`gate_validation.json`（NOOP/正信号门控）、`fix_verification.json`（约束/Requires-Python/重复节点/matching/正式 audit 的一次性核对）和额外 wheel 清单/指纹 `wheels-extra-manifest.json`。

边界：

- 2 GiB cgroup 内存限制导致不能无限并行；58/58 道目标题完成，但只有最终环境身份的 46 道是同环境测量：27 健康、26 恢复、1 道旧健康对照、19 仍不健康。另 12 道结果来自旧身份，全部健康（6 恢复、6 旧健康对照）。
- `estimated_total_healthy=95` 是跨阶段估计（目标集外旧健康 56 + 目标集内新健康 39），不是最终身份下完整 129 题重跑；不能称最终环境覆盖率。
- 新环境只证明这些公开题在本机 CPU 配置下可执行；不证明官方 GPU 主机或 heldout 的覆盖率。
- `shadow_regression`/`shadow_repair` 不是官方 `PASS_TO_PASS`，不能替换官方 `G`，也不能单独当作模型修复收益。
- 10 份历史候选的 0.15→0.10 是 verdict 级复算；其中 3 份真实补丁已在同一最终环境下重跑，其余 7 份未在本次重跑。
- 本轮仍没有正修复候选；下一步仍是短 GPU 采样，寻找真实局部修复或最终成功，再谈扩大训练。`require_positive_signal` 已把“无信号不优化”从文档计划变成代码路径。

## 5. 下一步短 GPU 采样（本轮未执行）

当前实例为无卡模式，`nvidia-smi -L` 无设备；短 GPU 采样需要用户切回 GPU 模式并另行授权预算。建议的首轮条件：

- 代码：`2162c6d5`（依赖约束、shadow audit、正信号门控修正）；新 run 目录，沿用同一 v0 和同一公开题集。
- 预算：单次 45–60 分钟，`max_batches=1`，`target_action_tokens=16384`，其余 Rank 128、奖励权重、8 分钟/28 工具调用不变；不再用“缩小 token 阈值制造梯度”。
- 通过条件：至少一条轨迹出现可观测的 `P>0` 或 `shadow_repair` 正修复；若仍全部为 0.15 或 0.10，`require_positive_signal` 会把该前缀标为 `NOOP_CONSUMED`，不启动优化步，只记录该批没有质量排序信号。
- 门控实现：`TrajectoryBuffer`/`TrajectoryLedger.reserve_update(require_positive_signal=True)` 在冻结 advantage 前检查 `reward_audit` 事件；`max_batches=1` 不再足以单独触发更新。旧 run 中没有 audit 的 buffered 轨迹按无信号处理，不静默放行。
- 失败条件：collection failure 仍可训练但为 0.10；基础设施失败、验证不完整、baseline 不可用继续排除，不补零。
- 产出：轨迹数、有效轨迹数、P/shadow 分布、collection failure 数、耗时和显存；不把训练题成绩称独立泛化。
