# 009 CPU 验收环境与奖励反馈修正

状态：2026-11-15，无卡实例（2 GiB cgroup），CPU-only。官方题面、`base_commit`、`test_patch`、参考补丁和官方验收语义未改；参考补丁仍只用于裁判侧正对照。

## 0. 结论

本轮只做三件事，目的是让后续短 GPU 采样更容易产生“可观测的修复进展”，不是宣称已有修复收益。

1. **依赖覆盖**：修复 subprocess 任务环境剥掉版本约束、总选最高 wheel 的问题；补充官方 wheel 缓存缺少的测试依赖。老 FastAPI 提交不再因过新 Starlette 产生 `Router.__init__()` 收集失败。
2. **失败反馈**：候选补丁导致测试收集/导入失败时，保留完整、合法的官方 verdict 和训练资格，但把奖励的 `validity` 置 0，总奖励 0.15 → 0.10；普通未修复候选仍为 0.15。目标版本升为 `official_differential_spo_v2`。
3. **回归观测**：公开数据没有 `PASS_TO_PASS`，官方 `G` 仍为 `not_applicable`；新增仅裁判侧可见的 exact 节点 `shadow_regression`，分母是空补丁基线 passed 节点，缺失/重复/歧义不补零，只进 reward audit，不改官方 `G`、`eligible` 或损失。参数化节点名中的沙箱路径规范为 `<workspace>`。

## 1. 依赖修复

旧 `training/official_runtime.py` 在任务沙箱里调用官方 `sandbox/setup.py`；该脚本把 `pyproject.toml` / `requirements*.txt` 的版本说明符剥掉，再对每个发行版选择最高 wheel。FastAPI 老提交的 `starlette<0.51` 因此被忽略，装上 1.x 后 `Router.__init__` 与提交代码不兼容。

新实现：

- `training/task_dependency_setup.py`：解析 `project.dependencies`、非 docs/lint 的 optional extras、`requirements-tests*.txt` 和 `dependency-groups`；按 PEP 508 交集选最高满足版本的 wheel，传给 pip 时使用精确 `==` pin；安装后核对 `importlib.metadata.version`，不一致写入诊断；只使用本地 wheel，不联网。
- `training/official_runtime.py`：在官方 `sandbox/setup.py` 的 `setup_git_exclude` / `install_editable_package` / `execute_fast_path` 之上执行该选择；官方 setup 源文件不改。额外 wheel 目录通过 `runtime.extra_wheels` 只读 staging 到任务 `/wheels`，不写 `official/`。
- 额外 wheel 目录只含官方缓存缺少的公共测试包：`inline_snapshot`、`dirty_equals`、`attrs`、`asttokens`、`executing`、`markdown-it-py`、`pytest-timeout`、`python-multipart`、`email-validator`、`orjson`、`ujson`、`pydantic-settings`、`passlib`、`pyjwt`、`coverage`、`flask`、`uvicorn`、`pytest-asyncio`、`trio`、`pytest-mock`、`argon2-cffi` 等；具体清单和指纹记录在 `results/cpu-improve-20261115/`。

单任务验证：

| 任务 | 旧症状 | 新结果 |
| --- | --- | --- |
| `fastapi_11355` | `ModuleNotFoundError: inline_snapshot`，exit 2 | 空补丁正常执行，`test_stringified_annotations` failure，基线健康 |
| `fastapi_11194` | `TypeError: Router.__init__() got an unexpected keyword argument 'on_startup'` | 选择 Starlette 0.48.0，2 failed / 2 passed，基线健康 |
| `fastapi_14986` | 60 秒任务超时导致 SIGKILL，无结果 | 300 秒/8 分钟覆盖后正常完成 |
| `fastapi_14964` | 快照提取失败 | 正常完成，7 passed / 2 failed，基线健康 |
| `requests_7427` | 参数化节点路径导致 exact shadow 错位 | 基线健康；节点路径规范后 exact 配对可测 |

### 1.1 覆盖复测（部分）

优先复测旧分类中 `both_arms_same_missing_module` 与 `both_arms_shared_compatibility_candidate` 的 58 道题，另加 3 道 Requests 参数化题和 4 道历史候选题。2 GiB cgroup 加单题约 40–70 秒，本轮在时限内完成 23 道；未完成题不推断结果。

| 指标 | 数值 |
| --- | ---: |
| 目标题数 | 58 |
| 已完成 | 23 |
| 新官方 verdict 有效 | 16 |
| 新空补丁健康 | 16 |
| 旧不健康 → 新健康 | 10 |
| 旧健康 → 新不健康 | 0 |
| 仍未健康 | 7 |
| 旧全集健康基线 | 63 |
| 保守估计新健康下限 | 72 |

10 道 recovered：`fastapi_11355`、`12942`、`14099`、`14349`、`14463`、`14482`、`14487`、`14492`、`14786`、`14964`。`estimated_total_healthy` 只把已完成目标题的新健康数加到旧健康全集上，未完成目标题仍按不健康计，因此是下限，不是完整 129 题复测结果。

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

- `training/reward/junit_bridge.py:parse_junit_node_statuses()`：保留 exact `(classname, name)`，参数化后缀不丢；重复键或状态歧义返回 unusable。
- `build_node_report(..., include_observed_nodes=True)`：在裁判侧报告中保存 `observed_nodes`、`observed_nodes_usable`。
- `training/reward/baseline.py`：空补丁缓存保留 `observed_nodes`；旧缓存缺该字段仍可读，shadow 为 `unavailable`。
- `training/reward/differential.py:shadow_regression()`：分母 = 基线 passed 节点；候选缺任一基线 passed 节点即 `unavailable`；候选新增节点单独计数；不进入官方 `G`。
- `training/reward/engine.py`：`shadow_regression` 只写入 reward audit；`RewardRecord` 不变。

候选节点名中的沙箱绝对路径在 `official_runtime.py` 中规范为 `<workspace>`，使 Requests 三对参数化节点可用同一身份比较。

3 份真实历史补丁的新代码重验（不是仅合成数据）：

| 任务 | verdict_kind | P | G | validity | 总奖励 | shadow |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `rich_4070` | `agent_pytest_collection_failure` | 0.0 | 0.0 | 0.0 | 0.10 | unavailable：候选缺 124 个基线 passed 节点中的大部分，不补零 |
| `fastapi_14258` | `tests_completed` | 0.0 | 0.0 | 1.0 | 0.15 | not_applicable：基线 passed 分母为 0 |
| `rich_4076` | `tests_completed` | 0.0 | 0.0 | 1.0 | 0.15 | measured 0/20，无回归 |

这同时验证了：官方 verdict 和 eligibility 不变；collection failure 的标量确实与普通未修复候选分开；shadow 只在 exact 覆盖完整时给数值，缺失时保持 unavailable。

## 4. 证据与边界

结果目录：`results/cpu-improve-20261115/`。包含 `summary.json`、覆盖复测 `coverage_recovery.json`、3 份真实候选的 `reward_shadow_validation.json` 和额外 wheel 清单/指纹 `wheels-extra-manifest.json`。

边界：

- 2 GiB cgroup 内存限制导致不能无限并行；本轮复测 23/58 道目标题，未完成题不推断结果，`estimated_total_healthy=72` 是下限。
- 新环境只证明这些公开题在本机 CPU 配置下可执行；不证明官方 GPU 主机或 heldout 的覆盖率。
- `shadow_regression` 不是官方 `PASS_TO_PASS`，不能替换官方 `G`，也不能单独当作模型修复收益。
- 10 份历史候选的 0.15→0.10 是 verdict 级复算；其中 3 份真实补丁已用新代码重验，其余 7 份未在本次重跑。
- 本轮仍没有正修复候选；下一步仍是短 GPU 采样，寻找真实局部修复或最终成功，再谈扩大训练。

## 5. 下一步短 GPU 采样（本轮未执行）

当前实例为无卡模式，`nvidia-smi -L` 无设备；短 GPU 采样需要用户切回 GPU 模式并另行授权预算。建议的首轮条件：

- 代码：本文件对应的 `91030e44` 及之后的修正；新 run 目录，沿用同一 v0 和同一公开题集。
- 预算：单次 45–60 分钟，`max_batches=1`，`target_action_tokens=16384`，其余 Rank 128、奖励权重、8 分钟/28 工具调用不变；不再用“缩小 token 阈值制造梯度”。
- 通过条件：至少一条轨迹出现可观测的 `P>0` 或 exact shadow 改善；若仍全部为 0.15 或 0.10，不启动优化步，只记录该批没有质量排序信号。
- 失败条件：collection failure 仍可训练但为 0.10；基础设施失败、验证不完整、baseline 不可用继续排除，不补零。
- 产出：轨迹数、有效轨迹数、P/shadow 分布、collection failure 数、耗时和显存；不把训练题成绩称独立泛化。
