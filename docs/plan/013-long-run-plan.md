# 013 长跑计划：rank 128 / v3 奖励 / 12 题公开池（2026-11-18）

状态：**配置与计划已同步服务器；base/v3/v5 三次 `--plan-only` 与 12 题 `load_config` 已通过；未启动训练、未启动 GPU。本轮代码与配置随本次提交入库。**
新配置为 `configs/spo_v3_long_rank128_20261118.toml`；`/root/gemma4/project`
已同步该配置与 in-run retention/评估入口，checkpoint 目录
`/root/autodl-fs/spo-checkpoints/spo-v3-long-r128-20261118` 已创建；尚未创建
长训 run 目录，也没有提交任何 optimizer step。另新增
`configs/spo_v3_continue_v5_rank128_20261118.toml`（从 v5 产出 v6…v10，未启动），
见第 10 节。

当前 5 步 run 的证据（`results/gpu-5step-r128-20261117/`、`docs/plan/009` 第 10 节）
**既不能证明稳定提升，也不能证明没有提升或退化**：v5 从未被采样，v3/v4 共同 3 题
的单次样本相同，v0/v4 共同 4 题的 P 相同，单题只有 1–2 条轨迹。因此本计划先补
独立评估入口，再用同题重复采样判断 v3/v5 是否值得进入更广题池的长跑；不把
“继续训练更久”当作默认结论。

## 0. 顺序与各阶段回答的问题

| 阶段 | 问题 | GPU | 预算 | 退出条件 |
| --- | --- | --- | ---: | --- |
| A. CPU 能力 | 能否在不训练的情况下评估已有 adapter？长跑期间能否把本地盘增长限制住？ | 无 | CPU 小时级 | 评估入口可用；in-run retention 接口与配置通过 `load_config`；两者均未在真实 run 上改动账本 |
| B. 同六题 GPU 评估 | v5 与 base、v3 在同一批公开题、同一预算、3 次重复下有没有可复现的方向差异？ | 1 个短会话 | ~2–2.5 h | 54/54 条官方 verdict 完整；评估轨迹不回流训练 |
| C. 广题池长跑 | 在 12 题公开池上连续 6 步能否稳定完成，并留下中间/最终 adapter 供后续评估？ | 1 个 6 h 会话 | 20,400 s run + 21,600 s supervisor | 6/6 步 `CONSUMED`、最终 adapter 与 v3 保留、无 ENOSPC、supervisor 无升级 |

顺序不可交换：B 的评估入口和 A 的磁盘回收都未落地前，C 不应启动。C 启动后
也不再把 B 的 6 题结果当作泛化证据。

## 1. 已核实的证据

### 1.1 5 步 run 的实测吞吐

来源：`results/gpu-5step-r128-20261117/steps.json`、`learning_signal_review.json`，
以及 `docs/plan/009` 第 10 节。

| update | action tokens | 墙钟（s） | forward（s） | backward（s） | checkpoint 写（s） |
| --- | ---: | ---: | ---: | ---: | ---: |
| `upd-0000` | 22,913 | 2,012 | 471.8 | 1,247.3 | 89.7 |
| `upd-0001` | 19,770 | 1,415 | 233.9 | 693.6 | 77.3 |
| `upd-0002` | 18,763 | 1,656 | 242.6 | 704.2 | 75.6 |
| `upd-0003` | 21,815 | 1,726 | 286.0 | 869.3 | 75.5 |
| `upd-0004` | 19,614 | 1,824 | 279.2 | 810.2 | 97.5 |

- 五步共 2 h 40 min；5 个 update 窗口合计 8,632.8 s，平均 1,726.6 s（28 min 47 s），
  最大 2,012 s（33 min 32 s）。
- 实际消费 17 条轨迹 / 102,875 action tokens；总采样 21 条，21/21 有完整官方
  verdict。按行为版本为 v0 9 / v1 3 / v2 1 / v3 4 / v4 4；v5 未采样。
- checkpoint 写共享盘 75–98 s（10.84 GiB，147 MB/s 量级）；本地盘装不下 rank-128
  的 21.7 GiB 原子重写峰值，所以本计划继续把 checkpoint 放 `/root/autodl-fs`。
- 单步 23–34 min 是 6 题池的实测，不是 12 题池的实测；第 5.2 节的 1.3 倍是规划
  余量，不是测量值。

### 1.2 官方公开题池（只读 SSH）

2026-11-17（本机日期）通过 `ssh seeta`（CPU 实例，`nvidia-smi -L` 为 0 张卡）
只读检查：

```bash
wc -l /root/gemma4/official/competition/tasks.jsonl
# 129

/opt/gemma-runtime/bin/python -c '
import json, hashlib
raw = open("/root/gemma4/official/competition/tasks.jsonl", "rb").read()
rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
print(hashlib.sha256(raw).hexdigest(), len(rows))
print(sorted(rows[0]))'
# e4b3fd60f69dbc2b9213e54eeb9636db78aefe92c1d06269d73d9f5f8f3c8ad6 129
# ['base_commit', 'created_at', 'hints_text', 'instance_id', 'patch',
#  'problem_statement', 'repo', 'test_patch']
```

仓库分布：`fastapi/fastapi` 67、`Textualize/rich` 48、`psf/requests` 13、
`encode/httpx` 1。该文件是公开池，不含 heldout；本次只读取了 `instance_id`、
`repo`、`created_at` 以及 `patch`/`test_patch` 的行数，没有把题面、参考补丁或
test_patch 内容复制到本地或用于训练。行数是选择“8 分钟会话内改动面有界”的
辅助证据，不构成难度或可解性保证。

### 1.3 磁盘实测

2026-11-17 只读 `df`：

| 卷 | 总量 | 已用 | 可用 | 用途 |
| --- | ---: | ---: | ---: | --- |
| `/root/autodl-tmp`（数据盘） | 50 GB | 37 GB | 14 GB | 运行目录、adapter、sandbox、临时文件 |
| `/root/autodl-fs`（共享盘） | 200 GB | 61 GB | 140 GB | 模型权重、官方题、checkpoint |

单个 rank-128 adapter 实测 1,940,039,826 B（1.94 GB）；trainer checkpoint
11,640,425,405 B（11.64 GB，10.84 GiB）。5 步 run 收尾时删除的 transient 目录
很小：`workspaces` 3.7 MB、`requests` 38 MB、`endpoint` 74 MB；真正占盘的是
adapter 版本和 staging 副本。收尾保留 v3/v5 后，run 目录 3.7 GB。

### 1.4 新增六题的空补丁 baseline 探针（2026-11-18）

用户在服务器上用**当前 runtime**（`/opt/gemma-runtime/bin/python`，
`extra_wheels=/root/autodl-tmp/gemma4/wheels-extra`）独立执行了 CPU-only 的
empty-patch baseline 探针：**无模型调用、无 GPU**。六个新增题
`fastapi_14349`、`fastapi_14360`、`fastapi_14430`、`fastapi_14487`、
`requests_7427`、`rich_4075` 全部健康：

- `top_verification_complete=true`、`top_verdict_valid=true`、
  `top_infrastructure_error=null`
- `baseline_verification_complete=true`、`baseline_verdict_valid=true`
- `junit_available=true`、`expected_universes_count=2`、
  `observed_nodes_usable=true`
- `all_healthy=true`

证据：`results/longpool-probe-20261118/summary.json`，schema
`gemma.longpool_baseline_probe.v1`；命令描述为 `EmptyPatchProbeRunner` 对
`/root/gemma4/official/competition` 运行，judge identity 和 extra wheels 沿用
`spo-v3-5step-r128-20261117`。这只证明**空补丁 baseline 在当前依赖环境下能产出
完整、有效的官方 verdict**，**不证明模型可解**，也不替代 Stage B 的真实
rollout。2026-11-15 关于 `inline_snapshot`/`dirty_equals` 缺失和 call-signature
不匹配的发现早于 extra_wheels 修复，不再作为这六题当前健康状态的证据。

## 2. 阶段 A：CPU 能力（无 GPU）

### 2.1 A1 独立 adapter 评估入口

现有训练循环只会在 rollout 时采样“当前正在服务的策略”，没有独立加载任意
adapter、只做 rollout + 官方验收的入口；v5 因此从未被采样。A1 的入口已落在
`training/eval_adapters.py`，并已同步到 `/root/gemma4/project`；本计划按它的
实际 CLI 在 3.2 给出命令。服务器 CPU 模式已实际执行 base/v3/v5 三次
`--plan-only`（各 18 条、judge identity 与训练 run 匹配、6/6 task key 精确一致、
共享 baseline 6/6 命中），未创建 run 目录；真实 54 条 GPU 采样仍未执行。A1 的
契约如下：

- 输入：公开数据根 `/root/gemma4/official/competition`、模型目录
  `/autodl-fs/data`、vLLM `http://127.0.0.1:8000`、一个 adapter 路径或“无
  adapter”（base）、固定题号列表、重复次数、固定 seed、输出目录。
- 行为：只做 agent rollout + 官方验收；不建 optimizer、不写训练账本、不发布
  adapter、不改变 `spo_stream.db` 的训练状态；评估轨迹不回流任何训练 run。
- 预算：与训练相同，`max_tool_calls=28`、`max_time_minutes=8.0`、
  `verification_timeout_seconds=300`、`max_turns=50`；使用与训练同一 judge
  identity 和同一 TaskKey（`base_commit` + `test_patch_sha256` + 环境哈希）。
- 并发：配置默认 `max_workers=6`；每 arm 18 条约 3 轮，三次调用串行。6 并发足以
  在 2–2.5 h 内完成 54 条；不要为了更快改成超过 vLLM 承载的并发。
- 输出：每个 eval run dir 的 `spo_stream.db`（`eval_sample` 事件）、`metrics.jsonl`、
  `eval_summary.json`、`eval_policy.json`；逐轨迹至少包含 `task_id`、`policy`
  （base/v3/v5）、`repeat`、`resolved`、`P`、`validity`、`verification_complete`、
  失败原因/异常、耗时；缺失值写 `null` 或 `-`，不写 0。
- 输出目录：`/root/autodl-tmp/gemma4/evals/20261118-base|v3|v5`，与训练 run
  目录分开。

A1 的验收只看事实：入口能在 `--plan-only` 下解析 6 题池、识别 base/v3/v5 的
adapter 并显示 18 条计划，且不写训练账本；真实 54 条的完整性属于阶段 B。不新增
测试文件，用一次手动运行和输出核对完成。

### 2.2 A2 运行中磁盘回收

本地实现已在本文第 9 节与 `training/adapter/retention.py` 固定
`[run]` 三个键，并已同步到 `/root/gemma4/project`：

```toml
in_run_prune = true
keep_adapter_versions = [3, 6]
keep_recent_adapters = 1
```

语义（已实现部分）：每次 update 提交并开始服务后，删除不再被 active、显式
keep 列表、最近 N 个版本、DoubleBuffer 轨迹引用或 pending ledger update 保护的
旧版本目录和已消费的 staging 副本；v0（genesis）按保守来源规则始终保留；删除
前写 `adapters/pruned_versions.json` 证据；失败只写
`spo_in_run_pruned`（`outcome=skipped/failed`），不让已成功的 update 失败。

对本配置（6 步，keep `[3,6]`）的预期驻留：

| active | 保留版本 | 本步删除 |
| --- | --- | --- |
| v1 | v0, v1 | – |
| v2 | v0, v1, v2 | – |
| v3 | v0, v2, v3 | v1 |
| v4 | v0, v3, v4 | v2 |
| v5 | v0, v3, v4, v5 | – |
| v6 | v0, v3, v5, v6 | v4 |

峰值 4 个版本（约 7.8 GB）+ 1 个 staging（约 1.9 GB）≈ 9.7 GB，比不裁剪的
7 个版本 + staging ≈ 15.5 GB 少约 5.8 GB。**2026-11-18 修正**：旧 `publish`
还会复制一份发布副本，六步真实峰值是 11.64 GB；现在同文件系统发布用硬链接，
不再增加整份权重，9.7 GB 才成立。20 步、keep `[3,6]`、recent 1 的峰值为
6 份 = 11.64 GB（不是 13.58 GB）；剩余约 2.96 GB，建议 recent 0（9.7 GB）
或启动前清理。硬链接跨卷会回退为复制，需按实际 `df` 复核。A2 的验收：在真实
或合成 run 上确认 `kept_versions` 包含 active/3/最近版本、`removed_bytes` 有
实际数值、`AdapterManager` 删除后仍能打开；不删除仍需评估的 v3/v6。

### 2.3 阶段 A 退出条件

1. A1 入口在服务器上可导入，且 3.2 的三条 `--plan-only` 全部通过（CPU-only，
   不连接 vLLM）；确认不会写训练账本。**2026-11-18 已实际通过：三次均 18 条、
   judge identity 与训练 run 匹配、6/6 task key 精确一致、baseline 6/6 命中，
   且未创建 run 目录。** 真实评估从 B 的 base arm 开始；若出现不完整样本，用
   `--resume` 补跑，不把失败样本静默计入。
2. A2 的代码与 `load_config` 的类型校验已同步到服务器；`[run]` 三键被接受。
3. 新配置在服务器上通过：

   ```bash
   cd /root/gemma4/project
   /root/gemma4/envs/learner/bin/python -c \
     'from pathlib import Path; from training.run_spo import load_config; \
      c=load_config(Path("configs/spo_v3_long_rank128_20261118.toml")); \
      print(c["run"], len(c["data"]["task_ids"]))'
   ```

   该命令要求 checkpoint 目录已存在（见第 6 节），只做配置解析，不启动 GPU。

## 3. 阶段 B：同六题 GPU 评估（base / v3 / v5 × 3 次）

### 3.1 设计

| arm | adapter | 证据 |
| --- | --- | --- |
| base | 无 adapter | 基座策略，不需要恢复已删除的零 adapter |
| v3 | `/root/autodl-tmp/gemma4/runs/spo-v3-5step-r128-20261117/adapters/versions/00000003-10c1a80f3a4d71b9` | 第 3 步后发布，`optimizer_step=3`，SHA256 `10c1a80f…`，权重仍在 |
| v5 | `.../adapters/versions/00000005-2ce46040af3eed69` | 第 5 步后发布，`optimizer_step=5`，SHA256 `2ce46040…`，权重仍在 |

题池固定为当前 6 题：`fastapi_14258`、`requests_7205`、`requests_7315`、
`rich_3782`、`rich_4070`、`rich_4076`。每 arm 每题 3 次，共 3 × 6 × 3 = **54 条**。
三次重复使用固定的 task 顺序和跨 arm 相同的 seed（例如 base_seed=3407，
repeat r 使用 3407+r）；不允许看结果后改 seed 或补样本。

### 3.2 启动命令

评估入口已由另一个子代理落地：核心 `training/eval_adapters.py`，CLI
`scripts/evaluate_adapters.py`，配置 `configs/eval_adapters_20261118.toml`
（六题、rank 128/alpha 256、v3 奖励、同一 runtime；一次调用只评估一个 policy）。
该配置的 `[eval]` 已固定 `repeats=3`、`seed=3407`、`max_workers=6`；下面命令
不再重复传 repeats/seed。改 repeats/seed 就视为改评估协议；`max_workers` 只影响
吞吐，不改变统计口径。先跑三次 `--plan-only`（CPU-only，不连接 vLLM），确认
题池、judge identity、adapter 权重和 baseline 缓存；三次都通过后再跑真实
rollout。不要用 12 题长跑配置，否则 `[data] task_ids` 会变成 12 题。

```bash
cd /root/gemma4/project
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=/root/autodl-tmp/gemma4/tmp

CFG=configs/eval_adapters_20261118.toml
REF=/root/autodl-tmp/gemma4/runs/spo-v3-5step-r128-20261117
EVAL=/root/autodl-tmp/gemma4/evals
V3_ADAPTER=$REF/adapters/versions/00000003-10c1a80f3a4d71b9
V5_ADAPTER=$REF/adapters/versions/00000005-2ce46040af3eed69

# CPU-only preflight：base
/root/gemma4/envs/learner/bin/python scripts/evaluate_adapters.py \
  --config "$CFG" \
  --data-root /root/gemma4/official/competition \
  --model-path /autodl-fs/data \
  --run-dir "$EVAL/20261118-base" \
  --policy base \
  --baseline-dir "$REF/baselines" \
  --reference-run-dir "$REF" \
  --plan-only

# CPU-only preflight：v3
/root/gemma4/envs/learner/bin/python scripts/evaluate_adapters.py \
  --config "$CFG" \
  --data-root /root/gemma4/official/competition \
  --model-path /autodl-fs/data \
  --run-dir "$EVAL/20261118-v3" \
  --policy v3 \
  --adapter-dir "$V3_ADAPTER" \
  --baseline-dir "$REF/baselines" \
  --reference-run-dir "$REF" \
  --plan-only

# CPU-only preflight：v5
/root/gemma4/envs/learner/bin/python scripts/evaluate_adapters.py \
  --config "$CFG" \
  --data-root /root/gemma4/official/competition \
  --model-path /autodl-fs/data \
  --run-dir "$EVAL/20261118-v5" \
  --policy v5 \
  --adapter-dir "$V5_ADAPTER" \
  --baseline-dir "$REF/baselines" \
  --reference-run-dir "$REF" \
  --plan-only
```

preflight 必须显示 `planned trajectories: 18 (6 tasks x 3 repeats)`、judge identity
与 reference run `MATCH`、reference task keys `6/6 exact`；baseline cache 应显示
`cached=6`（复用 5 步 run 的 `baselines/`，6 个 JSON 已在服务器上）。若显示
`missing>0`，把下面所有真实评估命令的 `--baseline-dir` 改为 `$EVAL/baselines`，
让入口生成空补丁基线，并把这一次生成计入评估时间。任何一项不符，先修配置/
路径，不要进入真实 rollout。

真实评估（vLLM 需按训练同款配置启动；三次调用串行，共用同一台服务，adapter
由入口自行 load/切换）：

```bash
# base：18 条
/root/gemma4/envs/learner/bin/python scripts/evaluate_adapters.py \
  --config "$CFG" \
  --data-root /root/gemma4/official/competition \
  --model-path /autodl-fs/data \
  --run-dir "$EVAL/20261118-base" \
  --policy base \
  --baseline-dir "$REF/baselines" \
  --reference-run-dir "$REF" \
  --backend-url http://127.0.0.1:8000 \
  --max-workers 6 --walltime-seconds 3600

# v3：18 条
/root/gemma4/envs/learner/bin/python scripts/evaluate_adapters.py \
  --config "$CFG" \
  --data-root /root/gemma4/official/competition \
  --model-path /autodl-fs/data \
  --run-dir "$EVAL/20261118-v3" \
  --policy v3 --adapter-dir "$V3_ADAPTER" \
  --baseline-dir "$REF/baselines" \
  --reference-run-dir "$REF" \
  --backend-url http://127.0.0.1:8000 \
  --max-workers 6 --walltime-seconds 3600

# v5：18 条
/root/gemma4/envs/learner/bin/python scripts/evaluate_adapters.py \
  --config "$CFG" \
  --data-root /root/gemma4/official/competition \
  --model-path /autodl-fs/data \
  --run-dir "$EVAL/20261118-v5" \
  --policy v5 --adapter-dir "$V5_ADAPTER" \
  --baseline-dir "$REF/baselines" \
  --reference-run-dir "$REF" \
  --backend-url http://127.0.0.1:8000 \
  --max-workers 6 --walltime-seconds 3600
```

一次调用只评估一个 policy；三次调用应串行，共用 vLLM，不要并发抢占同一组
adapter slot。vLLM 启动复用 C 的
`/root/autodl-tmp/gemma4/logs/start-vllm-20261116.sh`；评估入口只连接服务，不
负责拉起 vLLM。若要包在 supervisor 内，建议 `--budget-seconds 12600
--grace-seconds 180`，每个 arm 自己的 `--walltime-seconds 3600` 先停。评估结束后
用同一入口的 `--report` 汇总：

```bash
/root/gemma4/envs/learner/bin/python scripts/evaluate_adapters.py \
  --report \
  --arm base="$EVAL/20261118-base" \
  --arm v3="$EVAL/20261118-v3" \
  --arm v5="$EVAL/20261118-v5" \
  --reference-run-dir "$REF"
```

### 3.3 报告与判读

逐题 × arm 报告 `N`、`resolved` 数、`mean P`、`validity`、完整 verdict 数、
失败原因；同时给出 base-vs-v3、base-vs-v5、v3-vs-v5 的逐题差值。`P` 缺失写
`-`，不当作 0；`resolved` 与 `P` 分别判断可比较性。

**成功标准（操作性）**：54/54 条都有完整官方 verdict，0 条基础设施失败，评估
没有写训练账本，judge identity 与训练 run 一致。

**科学判读（不预设方向）**：
- v5 在多数题上优于 base 且不劣于 v3 → 可以记录“同题重复采样下的正向方向”，
  仍不是泛化结论。
- v5 与 base/v3 持平或互有胜负 → 记录为“未测出稳定差异”；3 次重复不足以
  证明等价，也不足以否决更广题池的长跑。
- v5 在多数题上明显差于 base → 先记录并检查失败类型；不要在同一轮同时改奖励、
  LR 和算法。是否仍启动 C 由用户决定，本计划不把“必须提升”设为 C 的前置门控。

**失败处理**：任何一条基础设施失败都先修环境再重跑该 arm；不得把失败样本
从分母里静默删掉。部分完成不进入判读。

### 3.4 预算

54 条 × 最多 8 min，配置默认 6 并发（18 条/arm 约 3 轮），9 轮 × 8 min ≈ 72 min；
加模型加载、adapter 切换、裁判和收尾，按 **2–2.5 h** 预留一个独立短会话
（supervisor 12,600 s 上限，3.5 h）。先跑 base arm，确认 18 条完整后再跑 v3/v5；
不完整时在同一 `--run-dir` 上加 `--resume` 补跑，失败样本仍留在记录里，不静默
删分母。若实际只能 4 并发，则拆成两个会话；不要为了赶时间降低
`max_time_minutes` 或 `max_tool_calls`，否则与训练时预算不可比。

## 4. 阶段 B 之后怎么决定 C

C 不是“B 有提升才跑”，也不是“B 没提升就不跑”。B 回答的是“现有 v5 在同六题上
有没有可复现的方向差异”；C 回答的是“更广公开题池上能否稳定完成并产出可评估
的中间/最终 adapter”。两者的证据不能互相替代。

启动 C 的条件：

1. A1/A2 完成，新配置在服务器通过解析；
2. B 的 54 条评估完成且没有未处理的基础设施失败；
3. 服务器本地盘可用 ≥ 12 GB（启动前手动 `df` 核对），checkpoint 目录可写；
4. 用户对本次 6 h GPU 会话明确授权。

若 B 显示 v5 在多数题上明显退化，先记录并以单独实验定位原因；不要在 C 中同时
改奖励、LR 和算法。C 本身只改题池、步数和预算，保留 rank 128 / alpha 256、
v3 奖励、基座 LR 5e-6、同一依赖环境、同一 16,384 action-token 采样目标。

## 5. 阶段 C：12 题公开池长跑配置

配置：`configs/spo_v3_long_rank128_20261118.toml`。

### 5.1 题池与选择证据

| task | repo | created_at | gold patch 行数 | test patch 行数 | 选择理由 |
| --- | --- | --- | ---: | ---: | --- |
| `fastapi_14258` | fastapi/fastapi | 2026-02-10 | 12 | 14 | 5 步池保留；可比；所有采样有完整 verdict |
| `requests_7205` | psf/requests | 2026-02-13 | 10 | 18 | 5 步池保留；v0 1/2 resolved，v3/v4 1/1 |
| `requests_7315` | psf/requests | 2026-05-10 | 10 | 12 | 5 步池保留；所有采样 resolved，P=0 |
| `rich_3782` | Textualize/rich | 2025-06-23 | 89 | 252 | 5 步池保留；未 resolved；一条 v0 样本 P=1/7 |
| `rich_4070` | Textualize/rich | 2026-04-11 | 401 | 30 | 5 步池保留；未 resolved；改动面较大但已实测可验收 |
| `rich_4076` | Textualize/rich | 2026-04-12 | 18 | 58 | 5 步池保留；未 resolved |
| `fastapi_14487` | fastapi/fastapi | 2025-12-10 | 15 | 22 | 新增；小改动/小测试；同 repo；11-18 空补丁 baseline 健康 |
| `fastapi_14349` | fastapi/fastapi | 2025-11-13 | 20 | 74 | 新增；小改动，中等测试；同 repo；11-18 空补丁 baseline 健康 |
| `fastapi_14430` | fastapi/fastapi | 2025-12-02 | 19 | 30 | 新增；小改动/小测试；同 repo；11-18 空补丁 baseline 健康 |
| `fastapi_14360` | fastapi/fastapi | 2025-12-02 | 12 | 78 | 新增；最小 gold patch 之一，测试中等；11-18 baseline 健康 |
| `requests_7427` | psf/requests | 2026-05-11 | 15 | 31 | 新增；与 `requests_7315` 同日邻期，repo/环境相近；11-18 baseline 健康 |
| `rich_4075` | Textualize/rich | 2026-04-11 | 44 | 14 | 新增；与 `rich_4070` 同日，测试小；11-18 baseline 健康 |

仓库分布：fastapi 5、requests 3、rich 4；官方公开池为 fastapi 67、rich 48、
requests 13、httpx 1。这个 12 题池不是按官方比例抽样：当前 6 题已含 2 个
requests，新增 1 个 requests 是为了保留同 repo 的近邻对照，因此 requests 占
3/12，高于其在公开池中的 10%。没有加入 `httpx_3672`（池中唯一 httpx，未在本
环境验证，且无法形成 repo 内对照）。

新增 6 题的当前证据是 1.4 的 2026-11-18 CPU-only 空补丁探针：六题全部 baseline
健康（`results/longpool-probe-20261118/summary.json`），但 baseline health
**不等于** solvability，也没有模型 rollout 证据。2026-11-15 的
`inline_snapshot`/`dirty_equals`/call-signature 发现早于 extra_wheels 修复，
不作为当前健康证据；本计划不再引用 2026-11-16 60 分钟筛选作为这六题的选择
依据。2026-11-17 只读核对：这 12 个 `task_id` 全部存在于 129 行公开池，且无
重复（同一 SSH 只读检查；没有读取题面/补丁内容）。

**重要限制**：新增 6 题只有空补丁 baseline 健康证据，没有任何模型 rollout 或
solvability 证据；C 的真实 rollout 仍可能失败。若 C 中某题持续 setup/verdict
失败，替换为同 repo、改动面相近的另一公开题，并记录原因；绝不使用 heldout。
C 的题池全部是公开题且同时参与训练，因此任何同题结果都是 **in-sample**。

### 5.2 步数与吞吐算术

实测 6 题池：单步 1,415–2,012 s，均值 1,726.6 s。6 题时调度器最多只有 6 个
不同任务可并发（`max_cpu_workers=8`，但 active/queued 任务不重复）；12 题时
8 个 worker 可以跑 8 个不同任务。并发从 6 提到 8、且不同题的最长 rollout
尾部可能更长，但**没有 12 题的实测单步时间**。规划取最坏实测 2,012 s × 1.3
= **2,615.6 s（43 min 36 s）** 作为单步上限；1.3 是保守余量，不是测量值。

| 项 | 秒 | 说明 |
| --- | ---: | --- |
| 启动（vLLM 加载、genesis、首批 rollout） | 900 | 5 步 run 实测 19:28→19:43 |
| 6 步 × 2,615.6 | 15,693.6 | 规划最坏 |
| 收尾 drain + prune | 600 | 10 min |
| **合计** | **17,193.6** | 约 4 h 46 min |
| run walltime | 20,400 | 5 h 40 min；余量 3,206.4 s（53 min 26 s） |
| supervisor | 21,600 | 6 h；比 run 多 1,200 s（20 min） |

- 若单步维持实测均值 1,726.6 s：900 + 10,359.6 + 600 = 11,859.6 s（3 h 18 min），
  run 会提前结束；`max_batches=6` 是上限，不会自动加步。
- 7 步按同样最坏值：900 + 18,309.2 + 600 = 19,809.2 s（5 h 30 min），只剩
  590.8 s（9 min 51 s）到 walltime；任何额外慢化都会让第 7 步无法在
  `update_min` 之前启动。因此选 **6 步**，不选 7。
- 8 步最坏 22,424.8 s，超过 walltime，不考虑。

`update_reserve_seconds=900`：剩余 15 min 停止派发；最大单题 8 min + 收尾，
15 min 足够。`update_min_seconds=3000`：新步至少留 50 min，覆盖 2,615.6 s
规划步时 + 约 6 min 收尾；旧配置的 1,800 s 小于实测最坏 2,012 s，不再沿用。

### 5.3 学习率日程

保留 `learning_rate=5e-6`、`warmup_ratio` 默认 0、`max_batches` 即
`total_optimizer_steps`。6 步的线性衰减为：

| step | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LR (×1e-6) | 5.000 | 4.167 | 3.333 | 2.500 | 1.667 | 0.833 |

5 步 run 是 5/4/3/2/1 ×1e-6；6 步的末尾更低是日程长度变化，不是改了基座 LR。
前四步的 LR 在历史事务文件中未逐步保留，不能冒称已逐步实测；这里是由配置和
`lr_schedule.py` 推导。

### 5.4 磁盘预算

本地 `/root/autodl-tmp` 启动时约 14 GB 可用：

| 项 | 估算 |
| --- | ---: |
| 4 个保留 adapter（v0、v3、最近、active） | 7.76 GB |
| 1 个 staging 副本（导出瞬间） | 1.94 GB |
| 发布副本（同文件系统硬链接） | 0 GB（回退复制时 +1.94 GB） |
| requests + endpoint + DB | ~0.15 GB |
| workspaces（5 步 run 实测 21 条仅 3.7 MB） | <0.01 GB |
| **峰值合计** | **~9.85 GB** |
| 剩余 | **~4.2 GB** |

不启用 in-run prune 时，v0…v6 共 7 个版本 + staging ≈ 15.5 GB，超过 14 GB
可用，会在运行中 ENOSPC；因此 `in_run_prune=true` 是启动前提，不是优化项。
共享盘 checkpoint 峰值 2 × 11.64 = 23.3 GB，140 GB 可用足够；正常收尾删除
checkpoint，不长期占用共享盘。按 2026-11-18 更新的约定，checkpoint 可以放
共享盘：本机数据盘只有约 14 GB 可用，rank-128 的原子重写峰值 21.7 GiB 放不下，
因此只把本 run 的 `spo_trainer_state.pt` 放 `/root/autodl-fs`；adapter、环境、
日志仍留在本地/数据盘。配置必须用 `[run] checkpoint_dir` 显式指定，并在实验
记录写明卷、可用空间和实测写入速度。

运行中手动核对（不是自动化门控）：每步后 `df -h /root/autodl-tmp`，并查看
`metrics.jsonl` 的 `spo_in_run_pruned` 事件；若可用空间低于 3 GB，或
`outcome=skipped/failed` 连续出现且 `removed_bytes=0`，停止 run 并人工处理，
不要删除 active/keep 版本。Stage B 的 `evals/20261118-*` 目录在评估结束后可整体
清理（requests/workspaces/sandboxes 都是 eval run dir 内的瞬态），启动 C 前先
确认它们不占用数据盘。旧 run（`spo-short-20261116` 约 1.9 GB）可按用户既有流程
清理后再启动，但不属于本计划自动执行。

### 5.5 C 回答与不回答什么

回答：
- 12 题公开池上能否连续完成 6 步，且每步都有完整官方验收；
- in-run retention 能否把 adapter 峰值压在约 10 GB；
- 中间 v3 与最终 v6 是否留下可评估的 adapter；
- 逐题 × 逐步的 in-sample 修复趋势，供后续与 base/v3 对照。

不回答：
- 不是 heldout 泛化结果；题池同时参与训练；
- 12 题 × 6 步、16K token 目标下每题只有约 2 条轨迹（6 步 × 约 4 条 ≈ 24 条），
  不能据此对单题做稳定结论；
- 不能证明“训练更久一定更好”，也不能仅凭 B 的平坦结果否决更广题池训练。

**训练量边界（必须保留）**：**6 h / 6 步**的 C 增加的是**题池 breadth**，不是
总 action-token 训练量。每步 `target_action_tokens=16,384`，6 步目标合计 98,304；
5 步 run 实测消费 102,875（整条轨迹越过目标会带来上偏），两者是同一量级。C 只是
把同样的 token 预算摊到 12 题、每题覆盖更少，因此**不能单独回答“训练量太小/
训练不足”的疑问**。10 h run 仍是后续的独立决定，不由 C 自动延长，也不能把
C 的 6 步当作 10 h 训练的替代或等价证据。

## 6. 阶段 C 启动命令（未执行）

以下命令只在用户授权的 GPU 会话执行。代码、评估入口和
`configs/spo_v3_long_rank128_20261118.toml` 已同步到 `/root/gemma4/project`，
checkpoint 目录已创建；服务器上的 `training/run_spo.py` 已包含 in-run retention
实现并通过 `load_config`。尚未创建长训 run 目录，也没有启动任何 optimizer step。

```bash
cd /root/gemma4/project
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=/root/autodl-tmp/gemma4/tmp

mkdir -p /root/autodl-fs/spo-checkpoints/spo-v3-long-r128-20261118
mkdir -p /root/autodl-tmp/gemma4/logs
df -h /root/autodl-tmp /root/autodl-fs

# 只解析配置，不启动 GPU；checkpoint 目录必须已存在。
/root/gemma4/envs/learner/bin/python -c \
  'from pathlib import Path; from training.run_spo import load_config; \
   c=load_config(Path("configs/spo_v3_long_rank128_20261118.toml")); \
   print(c["run"], len(c["data"]["task_ids"]))'
```

会话脚本（vLLM 后台启动，健康后跑 run，退出时停 vLLM）：

```bash
cat > /root/autodl-tmp/gemma4/logs/session-20261118.sh <<'SH'
#!/bin/bash
set -u
/root/autodl-tmp/gemma4/logs/start-vllm-20261116.sh &
VLLM=$!
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:8000/health >/dev/null && break
  sleep 5
done
/root/gemma4/envs/learner/bin/python -m training.run_spo \
  --config configs/spo_v3_long_rank128_20261118.toml \
  --data-root /root/gemma4/official/competition \
  --model-path /autodl-fs/data \
  --run-dir /root/autodl-tmp/gemma4/runs/spo-v3-long-r128-20261118 \
  --backend-url http://127.0.0.1:8000
code=$?
kill "$VLLM" 2>/dev/null
wait "$VLLM" 2>/dev/null
exit "$code"
SH
chmod +x /root/autodl-tmp/gemma4/logs/session-20261118.sh
```

supervisor（6 h = 21,600 s，软停宽限 180 s；run 自己的 walltime 20,400 s 先停）：

```bash
/root/gemma4/envs/learner/bin/python scripts/session_supervisor.py \
  --budget-seconds 21600 --grace-seconds 180 \
  --session-file /root/autodl-tmp/gemma4/logs/session-20261118.json \
  --log /root/autodl-tmp/gemma4/logs/session-20261118.log \
  -- bash /root/autodl-tmp/gemma4/logs/session-20261118.sh
```

运行中查看：

```bash
tail -f /root/autodl-tmp/gemma4/logs/session-20261118.log
df -h /root/autodl-tmp
grep spo_in_run_pruned /root/autodl-tmp/gemma4/runs/spo-v3-long-r128-20261118/metrics.jsonl | tail
```

收尾核对（自动 `prune_finished_artifacts` 之后，仍可手动 dry-run）：

```bash
/root/gemma4/envs/learner/bin/python scripts/prune_run_artifacts.py \
  --run-dir /root/autodl-tmp/gemma4/runs/spo-v3-long-r128-20261118 \
  --keep 3 --keep 6 --dry-run
```

若 run 以 75 退出（预算耗尽/软停）且还有未完成步，只能在用户明确追加额度的新
会话里 `--resume --additional-walltime-seconds N`；`max_batches=6` 不可改，
也不能用 resume 把本轮变成 10 小时。若出现 ENOSPC、active/keep 版本被删、
或 checkpoint 写失败，停止并按本文第 9 节的 marker/registry 规则排查，
不要手工删目录。

## 7. 风险、未解问题与边界

1. **评估入口已同步并随本次提交入库**：`training/eval_adapters.py`、
   `scripts/evaluate_adapters.py` 和 `configs/eval_adapters_20261118.toml` 已在
   服务器 `/root/gemma4/project`，三条 `--plan-only` 已实际通过。若实现方再改
   CLI，以实际的 `--help`/`--plan-only` 输出为准，不要凭本文件猜参数。
2. **新增 6 题只有 baseline 健康证据**：2026-11-18 的 CPU-only 空补丁探针显示
   六题都能在当前依赖环境下产出完整 JUnit/verdict（`results/longpool-probe-20261118/summary.json`），
   但**没有模型 rollout 证据，solvability 未测**；baseline health 不等于可解。
   C 的真实 rollout 仍可能失败；若某题持续 setup/verdict 失败，替换为同 repo
   的相近公开题并记录，绝不使用 heldout。
3. **1.3 倍单步余量是假设**：12 题没有实测步时；若实际单步超过 2,615.6 s，6 步
   可能只能完成 5 步并以 75 退出。`update_min=3000` 会阻止第 6 步在边缘启动，
   `max_batches=6` 不会自动增加；这是有意的安全行为。
4. **in-run prune 只覆盖 adapter/staging**：workspaces/requests 在 5 步 run 中
   很小（MB 级），但 12 题 × 6 步的 sandbox 数量更多；若实际增长，仍需在
   收尾前人工核对磁盘。A2 的 metrics 是事实来源，不靠猜测。
5. **本地盘余量约 4.2 GB**：若启动前有其他 run/缓存占用，先按用户流程清理；
   不要在剩余空间不足 12 GB 时启动。
6. **同题覆盖稀疏**：12 题、6 步、16K token 目标意味着每题约 2 条轨迹；C 的
   逐题曲线只能作为 in-sample 诊断，不能替代 B 的固定 6 题 × 3 次评估，也不能
   当作泛化或显著性结论。
7. **没有 heldout**：C 的题池是公开池；B 的 6 题也是公开训练题。任何“提升”
   都只能在公开题/训练集意义上说，不能称独立泛化。
8. **C 不回答“训练量太小”**：每步 16,384 token、6 步目标约 98K，与 5 步 run
   的 102,875 同一量级；C 扩的是题池 breadth，不是总训练量。10 h run 是后续
   独立决定，不能把 C 当作它的替代或等价证据。
9. **当前 5 步证据的边界**：v5 未采样；v3/v4 共同 3 题单次样本相同；v0/v4
   共同 4 题 P 相同；单题 1–2 条。它既不证明稳定提升，也不证明训练无效或退化。
   本计划的所有判读都必须保留这句话。

## 8. 证据路径

- 5 步 run 数值：`results/gpu-5step-r128-20261117/steps.json`、
  `learning_signal_review.json`、`report-compare-3v4.txt`、`report-compare-0v4.txt`。
- 5 步结论与边界：`docs/plan/009-cpu-acceptance-reward-feedback.md` 第 9–10 节。
- in-run retention 接口与验证：本文第 9 节（合并自 012）。
- 独立评估入口：`docs/plan/011-adapter-evaluation.md`、`training/eval_adapters.py`、
  `scripts/evaluate_adapters.py`、`configs/eval_adapters_20261118.toml`。
- 新增 6 题 baseline 健康：`results/longpool-probe-20261118/summary.json`
  （schema `gemma.longpool_baseline_probe.v1`，`all_healthy=true`；CPU-only、
  无模型/GPU；baseline health 不等于 solvability）。
- 新配置：`configs/spo_v3_long_rank128_20261118.toml`；
  continuation：`configs/spo_v3_continue_v5_rank128_20261118.toml`。
- 官方公开池：服务器 `/root/gemma4/official/competition/tasks.jsonl`，129 行，
  SHA256 `e4b3fd60f69dbc2b9213e54eeb9636db78aefe92c1d06269d73d9f5f8f3c8ad6`。
- 服务器磁盘：2026-11-17 只读 `df`，数据盘 14 GB 可用，共享盘 140 GB 可用。

## 9. 运行中 adapter 保留（合并自 012）

SPO 每步发布一个不可变 adapter（rank 128 约 1.94 GB），导出器还会写
`adapters/staging-vN`。`prune_finished_run` 只在整场结束后运行，旧实现删除版本
目录却不写“已删除”证据，导致 registry 行指向缺失目录、`AdapterManager` 重开
失败。当前的 in-run retention：

- 配置：`[run] in_run_prune`、`keep_adapter_versions`、`keep_recent_adapters`；
  `load_config` 校验类型/符号，bool 不当 int，keep 不得重复。
- hook：`run_spo.activate_and_commit()` 的 post-commit 一致性检查之后、离开训练
  租约；只删除 ledger update 已 `CONSUMED`、无 trajectory/registry 引用、非
  active/keep/recent、无 pending 更新的版本目录；staging 只删已发布且已
  `CONSUMED` 的副本；初始导入版本（无 update 行）保守保留。
- 证据：删除前原子写入 `adapters/pruned_versions.json`（version + full SHA256 +
  bytes + unix）；`AdapterManager` 只在 marker 精确匹配时容忍缺失目录，否则仍
  fail closed；active 目录缺失永远拒绝。旧 `pruned_artifacts.json` 可只读迁移，
  真实 `spo-v3-5step-r128-20261117` 已能重开（active v5、pruned {0,1,2,4}）。
- 发布副本：`AdapterManager.publish` 优先硬链接 `staging-vN` 到 `versions/`，
  同文件系统不再复制第二份 1.94 GB；跨卷回退复制。六步峰值从 11.64 GB 降到
  9.70 GB；20 步 keep `[3,6]`、recent 1 的峰值为 11.64 GB（建议 recent 0 或
  启动前清理）。
- 验证：一次性合成 run 删除 v2/v3 后 marker 写入、registry 重开 active=4、
  无 marker 的缺失目录被拒绝；真实旧 run 只读重开成功。没有新增测试文件。

## 10. 从已有 adapter 继续训练（v5 → v6…）

`[run] init_adapter_dir` 指向已发布版本目录时，新 run：

1. 读取该目录的 `manifest.json`，校验 version/rank/alpha/dropout/base_revision；
   `AdapterManager.import_adapter` 以硬链接安装到新 run 的 `versions/`，允许
   父版本行不存在（不导入整条历史），active 设为该版本（例如 5）。
2. 写 `initial_adapter.json`（source path、version、full SHA256、parent、
   optimizer_step、imported_unix）并把同一记录写进 `model_identity.json`；
   不修改源 run。
3. 第一次训练租约内 `policy.ensure_loaded()` 后调用
   `Policy.load_adapter_weights`，用 PEFT `load_peft_weights` +
   `set_peft_model_state_dict` 把 v5 权重装进 fresh LoRA；AdamW、scheduler、
   RNG、KL cursor 全部从头开始，第一次 optimizer step 产出 v6。
4. `_verify_resume_consistency` 用 `initial_version` 修正版本算术：
   `expected = initial + consumed (+1 pending)`，不再假设 genesis 0。
5. 保留策略必须包含初始版本和最终版本，例如 5 步 continuation 用
   `keep_adapter_versions = [5, 10]`。评估入口已支持任意版本（`--policy v10`
   或 `--policy adapter`；报告用 `--arm v10=RUN_DIR`）。

启动命令（未执行；需要 GPU 授权）：

```bash
# 首次启动前：run dir 必须不存在，checkpoint 目录必须存在且可写
mkdir -p /root/autodl-fs/spo-checkpoints/spo-v3-cont-v5-r128-20261118
# vLLM 部署与 5 步 run 相同（dynamic LoRA、max-lora-rank 128、Level-2 sleep、
# prefix cache、ninja 在 PATH），然后：
cd /root/gemma4/project
/root/gemma4/envs/learner/bin/python -m training.run_spo \
  --config configs/spo_v3_continue_v5_rank128_20261118.toml \
  --data-root /root/gemma4/official/competition \
  --model-path /autodl-fs/data \
  --run-dir /root/autodl-tmp/gemma4/runs/spo-v3-cont-v5-r128-20261118 \
  --backend-url http://127.0.0.1:8000
```

外层仍用 `scripts/session_supervisor.py --budget-seconds 21600 --grace-seconds 180`；
内部 `walltime_seconds=20400` 先停。若退出码 75，保留 checkpoint 并用同一配置
`--resume` 续跑；不要改 `max_batches`（即 LR 日程总长度）。本次 5 步预算约
3–4 h，6 h supervisor 留出 vLLM 加载与收尾余量。

评估 continuation 的最终版本时，在同一入口用
`--policy v10 --adapter-dir .../versions/00000010-<sha16>`，报告用
`--arm v10=<eval-run-dir>`；入口不再硬编码 base/v3/v5。

配置：`configs/spo_v3_continue_v5_rank128_20261118.toml`（未启动）。它沿用
12 题公开池、rank 128、v3 奖励、每步 16,384 token、5 步、6 小时预算，
checkpoint 放共享盘，首次更新从 v5 产出 v6。导入本身是一次性 CPU/磁盘工作；
首次训练租约内加载 1.94 GB 权重会增加启动时间，之后由 trainer checkpoint
恢复，不再重复导入。若源 adapter 或 `initial_adapter.json` 缺失，启动失败而
不是静默从零开始。
