# P005 T0 / single_state 四题配对载体

入口：`005-t0-single-state-pair.ipynb`，唯一代码 cell 与 `005-t0-single-state-pair-single-cell.py` 完全一致。执行会启动真实 GPU 推理；构建不会加载模型。本文件描述载体，不宣称已经取得修复收益。

```sh
PYTHONDONTWRITEBYTECODE=1 python3 notebooks/build_005_t0_single_state_pair.py
```

构建读取冻结 `research/002/variants/t0/agent/`、当前 `research/005/candidates/single_state/agent/` 和已有 P002 v30 / P005 single_state 构建来源。旧文件不被覆盖；新载体完整嵌入 agent、技能、逐文件 SHA、官方版本和模型指纹，无需远程克隆本仓库。T0 与 P002 v30 嵌入内容必须逐字一致。

2026-10-11 做 v30 官方默认命名兼容修正：single_state 的技能目录、frontmatter 和提示引用统一为 `source-edit`、`working-state`、`source-probe`，不启用 `SNAKE_CASE_SKILL_NAME`。脚本仍使用 `source_edit.py`、`working_state.py`、`source_probe.py` 等原文件名，8 个 Python 脚本字节及 CLI 行为均不改变。旧 single_state 载体与既有运行归档保留，不覆盖其历史内容。

## 运行条件

- 镜像固定为 `gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`，Linux x86_64 / CPython 3.12。Kaggle metadata 的镜像、GPU 和输入挂载由启动方设置。
- 挂载比赛 `gemma-4-developer-agent`、官方 `metric/gemma-4-developer-agent-wheelhouse` **v30**、`google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2`。本轮使用比赛 **4 × L4**（CLI `NvidiaL4`，官方 quota factor = 2），vLLM 使用四卡张量并行。两个 Evaluator 是请求并发数，不是 GPU 数。不要混挂另一个 wheelhouse 或 CT 版本。
- 禁用互联网。包安装来自挂载 wheel；运行前检查 CPython/ABI、全部 v30 wheel 的名称和大小，以及四个官方核心包 SHA。模型固定 CT v2，并核对已确认的 tokenizer SHA；不把 tokenizer 指纹当作全部权重的哈希。
- 仅研究运行，**不调用任何比赛提交接口**。导出的两份 agent ZIP 是可复查输出。

任务按以下四波执行，每波仅两臂并发，等该波全部退出后启动下一波。任务均为已暴露开发题，选择依据另存于本轮研究目录；不引入新题或预留评测题。

| 波次 | 开发题 | 子进程启动顺序 |
|---|---|---|
| 1 | rich_3942 | T0 → single_state |
| 2 | fastapi_14873 | single_state → T0 |
| 3 | fastapi_5077 | T0 → single_state |
| 4 | rich_3518 | single_state → T0 |

两臂相同：官方六工具声明，温度 0.2、top_p 0.95、输出上限 16384、thinking_budget 4096 / include_thoughts true；每题 5 分钟、50 工具调用、80 轮，单命令和验证超时 60 秒；context cache 为 2048 / 1800 秒 / 10，compaction 为 15 / overlap 2 / threshold 14336 / retention 5。single_state 额外获得官方 SkillToolset 提供的技能工具及三项技能，prompt 也有变化，因此比较的是**整个候选机制组合**，不能解读为状态卡单一变量消融。

服务通过官方 `VllmConfig.extra_args` 设置 `--max-num-seqs 2`，与本轮双并发匹配；实际完整启动命令归档到 run manifest。显式设置 `enable_lora=False`，但 v30 在没有挂载 adapter 时原本就不会发出 LoRA CLI 参数，不能把此项记为已实现的性能收益。相比旧载体，共同服务的并发上限属于本轮新研究条件。

## 服务、隔离与期限

唯一 coordinator 启动一次官方 VllmServer。两个 Evaluator 分别在独立 Python 子进程内运行，通过同一 HTTP 服务请求模型；各自创建新的官方 SubprocessManager，并拥有独立 submission 副本、TMPDIR、仓库/venv/验证目录、日志和结果。worker 的模型注册参数按官方 `create_model_registry()` 无 adapter 分支构造，不加载第二份模型，不向任一 agent 注入另一臂结果。

这保证评测进程及工作区不复用，沿用官方 subprocess sandbox 的边界，不额外宣称强安全隔离。两臂仍共享模型服务、prefix cache 与计算资源；AB/BA 只平衡启动顺序，不能保证缓存相同。并发 token 吞吐竞争会影响五分钟内能完成的工作，所以该研究条件不能与旧串行结果直接当作纯候选效应比较。四个暴露题、每题一对也不足以证明总体修复率提升。

外层监督进程给安装、加载、编译及全部评测合计 **2550 秒**。每个 attempt 另有 **480 秒**外层期限，包含 agent 运行、环境准备及裁判；官方 agent 五分钟预算不变。总期限到达即停止后代进程和服务，不重试，不启动第二个模型。另留至第 2680 秒以内完成日志/ZIP 输出；若打包超时，原始逐文件结果保留。启动方应设服务器总期限 **2700 秒**，把前两次安装前后失败也纳入本轮 2 小时账户预算，并记录真实配额扣减。载体不自行追加实验。

## 结果与观察

输出目录为 `/kaggle/working/results/005/run-005-t0-single-state-pair-<UTC>-<id>/`，同级生成结果 ZIP。

- `run_manifest.json`、`carrier-source.json`、`frozen_bundle/`：实际版本、GPU、预算、源码和配置指纹、唯一服务启动计数。
- `frozen_bundle/tasks-selected.jsonl` 仅保留这四题；完整源文件只按字节流计算 SHA/大小并识别行 ID，官方任务解析器也只读取四题子集。`task-source.json` 同时记录全源与子集指纹，不归档全题文件。
- `full_schedule.json`、`attempt_status.json`、`analysis_rows.json`、`joint_summary.json`：顺序、原始 resolved/错误和每臂数量；总完成要求 8 个 attempt 均正常归档，任何缺失都保留为部分结果并非零退出。
- 各臂 `r1/attempts/<id>/`：worker 配置、独立 stdout/harness 日志、官方结果、轨迹、补丁、验证输出、context 时间和文件清单。异常时保留已有 patch/trace、错误栈；不会伪造裁判成功。
- `gpu-telemetry.jsonl`：约每 10 秒记录每卡利用率/显存/功耗，以及 vLLM `/metrics` 的实际 running / waiting 请求数和 token 累积量。`phase.json` 标注 setup、server_start、evaluation、cleanup。
- `supervisor-report.json`：退出码、是否整体超时、真实完成数、按阶段区分的每卡采样平均利用率与最大采样请求并发数。`evaluation` 包含仓库准备、agent 和裁判，**不是纯解码阶段**；缺失指标记为不可用，采样平均也不是完整 GPU 忙碌率或 token 吞吐。
- `coordinator.log`、`vllm_server.archived.log`：安装/启动/故障原始证据；外层强停也尽量保留官方 server 的原日志。

本地仅做构建、嵌入内容/预算/官方 API 的手动内联核对，不建立或运行测试套件。GPU 容量、真实并发、技能使用、实际修复结果须以此次运行证据为准。

## 本轮接手位置

实际私有运行是 `ccjjllt/p005-t0-single-state-pair` **v3**；官方CPU构造准备已完成，两次早期GPU失败及CPU命名修正保留在[研究记录](../research/005/t0-single-state-pair-20261011/results.md)。当前研究结果优先回写该页和`docs/progress/005-medal-and-prize-research.md`，不新开检查清单。

读取状态及下载本轮结果的命令不启动新计算：

```sh
.venv/bin/kaggle kernels status ccjjllt/p005-t0-single-state-pair
.venv/bin/kaggle kernels output ccjjllt/p005-t0-single-state-pair/3 \
  -p research/005/t0-single-state-pair-20261011/output-v3 \
  --file-pattern '.*(\.zip|\.log|supervisor-report\.json)$' --quiet
```

先读取实际逐题`result.json`和原始trace，再解释候选收益；出现环境失败时保存已有证据，不把八次计划数写成已完成数。账户配额基线在`quota-before-v3.json`。训练支线的grouped SDPA仍默认关闭，外部来源也尚未转换为正式训练数据，均不随这次推理对照自动启动。
