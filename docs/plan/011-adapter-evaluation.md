# 011 已发布 adapter 的独立评测入口

状态：2026-11-18，CPU-only 实现与 preflight 验证；尚未在 GPU/vLLM 上执行真实 54 条采样。官方 `tasks.jsonl`、`base_commit`、`test_patch`、参考补丁与官方验收语义均未修改；本入口不读 heldout，也不运行 optimizer。

## 目的

训练 run 在每步更新前采样，run 结束时最后一版 adapter 往往没有被同口径采样。本入口把“已发布 adapter”当作固定策略，只做 rollout + 官方裁判，逐题报告 `resolved`、`P` 与 `validity`。一次调用只评测一个策略：

| 策略 | 服务对象 | 行为身份 |
| --- | --- | --- |
| `base` | 已部署的 base model（无 LoRA） | version 0 + 固定 64-hex sentinel digest |
| `vN` | 任意已发布版本目录（例如 v3/v5/v6/v10） | manifest version N + 权重 SHA256 |
| `adapter` | 路径由 `--adapter-dir` 给出，版本从 manifest 读取 | manifest version + 权重 SHA256 |

`base(v0)` 是**未加 LoRA 的已部署基座**，不是训练里被删除的零初始化 v0 genesis adapter，也不是旧 run 的 v0 轨迹。slots 固定为 6 道公开题 × 3 次重复 = 18 条/策略；评估几个版本就调用几次（v3/v5/base 三次即 54 条，v6/v10 可直接追加）。题池同时用于训练，结论是 **in-sample 诊断**；3 次重复只是比单次样本多一点信息，不构成显著性。

## 入口

核心：`training/eval_adapters.py`；CLI：`scripts/evaluate_adapters.py`。配置：`configs/eval_adapters_20261118.toml`（沿用六题、rank 128/alpha 256、v3 奖励与官方 runtime 设置）。

### CPU-only 计划校验（不联系 vLLM）

```bash
python scripts/evaluate_adapters.py \
  --config configs/eval_adapters_20261118.toml \
  --data-root /root/gemma4/official/competition \
  --model-path /autodl-fs/data \
  --run-dir /root/autodl-tmp/gemma4/evals/20261118-v3 \
  --policy v3 \
  --adapter-dir /root/autodl-tmp/gemma4/runs/spo-v3-5step-r128-20261117/adapters/versions/00000003-10c1a80f3a4d71b9 \
  --reference-run-dir /root/autodl-tmp/gemma4/runs/spo-v3-5step-r128-20261117 \
  --plan-only
```

`--plan-only` 会校验配置、`tasks.jsonl`、模型目录、adapter `manifest.json`/`adapter_config.json`/权重 SHA256、judge identity、共享 baseline 缓存和 `--reference-run-dir` 的 identity/task key，然后打印 6×3 的逐题重复矩阵（含固定 seed）；**不会构造 vLLM backend，也不会跑 empty-patch probe**。

### 真实评测（需要 GPU + 已部署 vLLM）

```bash
python scripts/evaluate_adapters.py \
  --config configs/eval_adapters_20261118.toml \
  --data-root /root/gemma4/official/competition \
  --model-path /autodl-fs/data \
  --run-dir /root/autodl-tmp/gemma4/evals/20261118-v3 \
  --policy v3 \
  --adapter-dir /root/autodl-tmp/gemma4/runs/spo-v3-5step-r128-20261117/adapters/versions/00000003-10c1a80f3a4d71b9 \
  --baseline-dir /root/autodl-tmp/gemma4/evals/baselines \
  --max-workers 6 --walltime-seconds 14400
```

`base` 不传 `--adapter-dir`；`v5` 换成 v5 目录；任意版本可写 `--policy v10`，或写 `--policy adapter` 让入口从 manifest 读取版本。`--policy vN` 与目录版本不一致会直接报错。服务器上应使用 learner 环境运行（`/root/gemma4/envs/learner/bin/python`）；vLLM 部署与训练相同，必须启用 dynamic LoRA、Level-2 sleep 和 prefix cache。本入口不会调用 `/sleep`、不会暂停/恢复服务，也不写入源 run 目录。被 SIGTERM/预算中断后，用同一 `--run-dir` 加 `--resume` 继续；已完整记录的 `(task, repeat)` 不会重复采样，配置、seed、baseline 目录或 identity 变化会被拒绝。

### 报告

```bash
python scripts/evaluate_adapters.py --report \
  --arm base=/root/autodl-tmp/gemma4/evals/20261118-base \
  --arm v3=/root/autodl-tmp/gemma4/evals/20261118-v3 \
  --arm v5=/root/autodl-tmp/gemma4/evals/20261118-v5 \
  --arm v10=/root/autodl-tmp/gemma4/evals/20261118-v10 \
  --reference-run-dir /root/autodl-tmp/gemma4/runs/spo-v3-5step-r128-20261117
```

报告以完整官方 verdict 为样本单位，打印逐题 N、resolved 率、P 均值和策略间差值。缺失的 P 显示 `-`，不按 0 处理；`N<2`、未达到 3 条、identity 不一致都会显式警告。identity 不一致时抑制跨策略差值。报告只读 `spo_stream.db` 的 `audit_events`（必要时回退到带 eval provenance 的轨迹和 `rewards` 表），不修改 training `run_spo.py` 或 `scripts/report_repair_trend.py`。

## 复用与隔离

- **同一 judge/运行时**：调用 `training.judge_identity.read_runtime_identity` 与 `bind_run_identity`，传入与训练相同的 `data_root`、`submission_dir`、baseline python、sandbox、evaluation overrides 和 extra wheels；TaskKey 由同一 `task_key_for` 生成。`--reference-run-dir` 只读比较 `judge_identity.json` 和 task key，不做跨 judge 的合并。
- **同一裁判链**：`OfficialWorker` 跑官方 runtime，`EmptyPatchBaselineStore` 缓存空补丁基线，`RewardEngine` 生成 `P/resolved/validity`，`RolloutEngine` 封存轨迹并写 ledger（入口已显式导入 `RolloutEngine`；此前 `--plan-only` 不经过该行，漏掉导入未被发现）。共享 `--baseline-dir` 让多个版本复用同六道题的空补丁结果；缓存键包含 TaskKey/judge identity，identity 变化不会误复用。共享 baseline 根目录在构造 store 前会按 store 要求创建/校正为当前用户所有、mode 0700；若存在符号链接或属于其他用户则拒绝。
- **同一 dispatch 协议**：eval 不伪造 0.5 基线。每次派发调用 `ValueTracker.freeze_dispatch(task, version)` 取真实快照，并调用 `note_dispatch` 写 marker；派发失败 unmark，尝试结束（无论完整、被排除还是等待重试）也 unmark，避免残留 marker 阻塞后续序列。完整样本若 reward 可用，`RolloutEngine` 会按原逻辑 observe；无 optimizer 读取这些值。
- **无训练**：不构造 trainer，不写 `spo_trainer_state.pt`，不调用 `buffer.reserve`/`ledger.reserve_update`。`RolloutEngine` 需要一个 eval-only 的 `TrajectoryBuffer` 和 `ValueTracker`；轨迹最多进入本地 `BUFFERED`，不会被任何 update 消费。运行结束会检查并记录 `updates=0`。
- **源 run 只读**：任意版本直接从原版本目录加载 1.94 GB 权重，不复制、不调用 `AdapterManager`（源 run 的 registry 可能仍列出已裁剪的中间版本，打开它会失败）。manifest、权重 SHA、`adapter_config.json` 在本地校验，vLLM 加载时再校验一次。
- **资源边界**：`--max-workers`（默认 6）、`--walltime-seconds`、`--shutdown-grace-seconds`、`--max-attempts`、`--max-sample-attempts` 都有上限；SIGTERM 先停止派发，在宽限期内收尾，然后取消 worker、`ExecutionManager.shutdown(wait=True)` 并关闭 endpoint。requests/workspaces/sandboxes/probe workspaces 都在 eval run dir 下，可直接整体清理。

## 采样计数

一个重复只有满足以下条件才算“完整样本”：

```text
verification_complete is True
and verdict_valid is True
and official success is bool
and reward.resolved is not None
and reward.resolved == official success
and exclusion_reason does not contain task_key_invalid
```

`reward.eligible` **不是**完成条件。候选让测试不可收集时，官方 verdict 仍完整、`validity=0`、`P` 可能缺失，但它是真实模型样本，必须占用 3 个重复之一；它只作为训练批次不合格，不影响评测计数。基础设施失败、worker error、超时、verdict 缺失、identity 不一致才会重试：`ExecutionManager` 内部最多 `--max-attempts`（默认 2），外层每个 `(task, repeat)` 最多 `--max-sample-attempts`（默认 3）。重试耗尽的 `(task, repeat)` 写入 `eval_summary.json` 的 `unfilled`，报告按 `N<3` 提示；被排除的尝试单独计数，不冒充样本。

固定 seed 规则：同一 `(task, repeat)` 在所有被评估版本上使用同一 seed（由 `sha256(base_seed:task_id:repeat)` 派生），不同 repeat 不同；这样策略比较不会因 seed 改变而混淆。采样仍由 vLLM 执行，seed 只固定随机源，不保证三条轨迹彼此独立。

## 产物

每个 eval run dir 独立包含：

```text
judge_identity.json     # 与训练同一 canonical judge environment
model_identity.json     # config/tokenizer 廉价指纹，不扫描完整权重
eval_policy.json        # policy/behavior/adapter 身份、task keys、18 个 slot 与 seed
spo_stream.db           # 轨迹、reward、reward_audit、eval_sample 事件；无 update
metrics.jsonl           # preflight/backend/adapter/dispatch/eval_sample/shutdown
eval_summary.json       # 完成数、unfilled、逐题 n/resolved/P/validity、耗时
requests/ workspaces/ sandboxes/ probe_workspaces/ endpoint/   # 可整体清理
```

`eval_sample` 事件逐轨迹记录 `verification_complete`、`verdict_valid`、`official_success`、`resolved`、`P`、`G`、`validity`、`total_reward`、`reward_eligible`、`exclusion_reason`、`infrastructure_error`、`status`、`action_tokens`、seed 与 sample attempt；不含测试节点名或官方题面。

## 已知限制

- 六道题同时参与训练；本入口的结果是 in-sample 诊断，不是独立泛化，也不能替代 heldout。
- 每策略每题 3 条；报告明确 `N<2` 警告。3 条不足以做显著性检验，不应把单题 1/3 与 2/3 的差别称为稳定提升或退化。
- `base(v0)` 是未加 LoRA 的已部署基座，不是训练 genesis adapter；不能把它与训练账本里的 v0 轨迹混为一谈。
- v5 在 2026-11-17 run 中没有被采样；本入口第一次为 v5 产生独立评测证据，但这次评测本身尚未在 GPU 上执行。六步 continuation 的 v6…v10 可直接用同一入口评估。
- P 依赖空补丁 baseline。baseline 不可用时 `P=None`，报告显示 `-`；resolved 仍可单独比较。`P=0` 是实测值，不是缺失。
- 真实评测需要 vLLM 已按训练配置部署且 base model id 能在 `/v1/models` 中被 `VLLMBackend.start()` 发现；`--plan-only` 不验证服务可达性。
- 信号处理会停止新派发并取消官方 worker；正在执行的 empty-patch probe 由 `EmptyPatchProbeRunner` 自己的超时约束，不能在同一瞬间中断。
- 本入口没有把训练 run 的成绩合并进报告；跨训练/独立评测的比较需要另做，且必须先确认 judge identity 与 task key 一致。

## 2026-11-18 CPU 验证

已做：`py_compile`；在服务器 CPU 模式用真实 `/root/gemma4/official/competition`、`/autodl-fs/data`、v3/v5 原目录和 base 路径分别执行 `--plan-only`（含 SHA256、manifest、judge identity 与 reference run 比较）；三次均 18 条/策略，`judge_identity_sha256=3b73b9c7…` 与训练 run **MATCH**，6/6 task key 精确一致。用训练 run 的 `baselines` 作为 `--baseline-dir` 时显示 `cached=6/6`、mode 0700、owner OK，因此真实评估可复用已有空补丁基线而不写源 run；若用新的 eval baseline 根，则首次会跑 6 次探针。用一次性内联 Python 验证 `VLLMBackend.submit_base` 的 `/v1/completions` payload 为 served base id + `cache_salt=base`；用临时 sqlite ledger 验证 `--report` 的缺失 P 显示 `-`、`P=0` 不被误判为缺失、N<2/未满 3 条警告和动态 `--arm vN` deltas；`--policy adapter` 与 `--policy 3` 均在服务器 CPU 模式通过。证据摘要：`results/eval-plan-20261118/summary.json`。未启动 vLLM、未跑官方 54 条样本、未产生 GPU 结果。
