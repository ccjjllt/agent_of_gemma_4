# 001：无微调基线与失败分析

## 研究目标

建立一个符合比赛接口的单 Agent、无 LoRA 基线，观察指定模型从问题描述到补丁提交的实际行为。首先区分运行环境故障、代码定位失败、错误修改、工具调用失败和预算耗尽，再决定后续改进方向。

本文件保留001的原始研究设计；实现和三题初步运行已完成，实际状态见[001进展](../progress/001-baseline.md)。后续已发现的接口、记录和运行载体问题，以[003修正交接](003-runtime-reliability.md)为准。原始资料依据为2026-09-28取得的官方说明、Getting Started v2和公开任务文件，指纹见[计划索引](README.md)。

## 已确认的实验条件

| 项目 | 已确认内容 |
| --- | --- |
| 参赛状态 | 当前账号已加入比赛，比赛元数据 `user_has_entered=true` |
| 开发设备 | M5、32 GB 统一内存；承担编辑、数据整理、轨迹与结果分析 |
| 账号计算额度 | 本次读取 GPU 每周额度 108,000 秒（30 小时），已用、预留均为 0；下次刷新 2026-10-03 00:00 UTC；这不证明某 GPU 型号可用 |
| 模型别名 | `gemma-4-31b-it-qat-w4a16-ct` |
| 官方示例模型资源 | `google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2` |
| 评分服务参考 | 4 × L4，vLLM，32,768 token 合计上下文，Gemma 4 工具与推理解析器 |
| 提交物 | `submission.zip` 根目录一个 `agent.yaml`，解压总大小小于 3 GiB |
| 主指标 | 补丁通过官方验证的任务比例 |
| 总预算 | 所有任务生成补丁共 12 小时，包含沙箱准备，不包含后续补丁验证 |
| 正式任务调度 | 主办方 9 月 24 日答复为顺序执行；本轮也逐题运行 |
| 正式提交次数 | 每日最多 1 次，可选择至多 2 个最终提交；日常研究首先使用公开开发任务 |

实际可分配 GPU 型号、数量、显存、当前 wheelhouse 包版本以及 Notebook 运行上限，仍应在实现后的实际环境中记录。不要用评分硬件或账号周额度推断训练能力。

## 输入数据与划分

官方数据包含 `tasks.jsonl`、仓库 `snapshots/`、`graphs/`、`embeddings/`、离线依赖 `wheels/`、`docker/`、`sandbox/setup.py` 和 `sample_submission/`。

实际读取任务分布如下：

| 仓库 | 任务数 |
| --- | ---: |
| `fastapi/fastapi` | 67 |
| `Textualize/rich` | 48 |
| `psf/requests` | 13 |
| `encode/httpx` | 1 |

任务字段为 `instance_id`、`repo`、`base_commit`、`problem_statement`、`hints_text`、`patch`、`test_patch`、`created_at`。Agent 输入只采用官方问题、可用提示和对应版本仓库；`patch`、`test_patch` 留在官方验证与研究分析侧，不拼入 Agent 提示或检索索引。

划分方案：

1. 按 `(repo, base_commit)` 分组，固定随机种子 `20260928`，同一组不得跨开发与保留集合；129 个任务只有 127 个不同仓库快照。
2. HTTPX 唯一任务作为探索案例单列，不用它宣称该仓库上的泛化效果。
3. 其余仓库按组分层，约 20% 任务留作保留集合；组大小导致数量略有偏离时，以不拆组为先，并记录最终数量。
4. 从非保留集合选每个可用仓库一个任务做初步运行观察，再选约 20～30 个任务作为固定开发面板；其余任务暂不进入提示词选择循环，可供后续训练研究使用。
5. 开发面板按仓库覆盖选取，不按某个候选 Agent 是否成功来挑题。保留任务一旦用于方案选择，即不再称为未使用保留集。

划分清单只存任务 ID、仓库、commit、集合标签、种子与原始数据指纹，不复制参考答案。

## 需要实现的文件

以下是本项目拟定结构，并非 Kaggle 要求的完整工程布局：

| 文件或目录 | 职责 |
| --- | --- |
| `agent/agent.yaml` | 一个 `LlmAgent`，指定模型别名，引用提示词与生成参数，注册本轮工具 |
| `agent/prompts/system.md` | 简洁的仓库修复流程：定位、阅读、形成修改、检查修改、最后提交补丁 |
| `agent/configs/sampling.yaml` | 固定采样和思考参数，保留版本标识 |
| `agent/eval_config.yaml` | 显式填写四个官方支持的单题预算字段 |
| `notebooks/001-baseline.ipynb` | 从官方 v2 示例改编，完成环境、数据清单、模型服务、逐题执行、结果读取和打包，各阶段独立单元格 |
| `research/task_splits.json` | 上述固定划分，不放入提交包 |
| `results/001/<run_id>/` | 官方原始产物，以及该次环境与配置快照 |

首轮不需要自建调度服务、图数据库、通用插件框架或替代官方评测器。`agent/` 内容才是打包范围，Notebook、数据、原始答案和运行记录位于包外。

## 官方接口与实现步骤

### 1. 复用官方 Notebook 的环境和路径

官方 v2 示例挂载：

- 数据：`/kaggle/input/competitions/gemma-4-developer-agent`。
- 运行依赖：`/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse`。
- 模型：`/kaggle/input/models/google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2`。
- 输出：`/kaggle/working`。

以实际挂载路径为准。官方安装单元格包含 CUDA wheel 文件名修复和依赖安装处理，应先理解并按当前版本复用。不要把该 Linux GPU 环境安装流程直接套在 Mac 上。

Notebook 记录 Python、CUDA、GPU 名称/数量/显存、`swegemma`、`adk-submission`、`adk-eval-core`、`google-adk`、`vllm` 实际版本。当前尚未安装这些库，不预先填写版本号。

### 2. 整理单 Agent 配置

从官方 `sample_submission/` 理解 YAML 结构，再建立独立 `agent/`。根配置使用 `name`、`model`、`instruction`、`tools`、`generate_content_config`；删除未使用的子 Agent 引用与文件，不设置 `adapter`。

本轮只注册 `run_command`、`read_file`、`edit_file`、`write_file`、`get_status`、`submit_patch`。图工具留给 002，避免基线同时改变多个因素。文本搜索通过 `run_command` 使用环境实际可用的命令，不能默认 `rg` 已安装。

提示词规定使用仓库证据、小范围读取和修改、处理工具错误，完成后调用 `submit_patch`。初始问题由官方 harness 注入，无需把参考答案写入系统提示。可选 `hints` 只有非空时才在 session state 中存在，避免无条件插值造成缺失变量错误。

采样初值沿用官方 v2：`temperature=0.2`、`top_p=0.95`、`max_output_tokens=16384`、`thinking_budget=4096`、`include_thoughts=true`。省略 `thinking_level`，避免它映射到服务端不兼容的参数。这里的输出上限不能额外加在 32,768 token 总上下文之外。

### 3. 显式设置预算

在 `eval_config.yaml` 的 `evaluation` 下填写 `timeout_seconds`、`max_tool_calls`、`max_time_minutes`、`max_turns`。

本轮小批开发的拟定起点：单命令 60 秒、50 次计费工具调用、每题 Agent 阶段 5 分钟、最多 80 个模型轮次。这是待观察的研究配置，不是官方默认，也未证明适合全量提交。模型服务启动时间单独记录。

后续源码复核补充：当前0.2.7的timeout_seconds同时限制Agent命令与最终pytest，不只作用于单命令；更改此值应重新固定所有对照组，详见003。

官方 README 与早期主办方答复关于省略字段时的默认值不完全一致，因此始终显式填写并核对实际运行记录。单题 Agent 计时不含准备，12 小时总预算含准备，不能用 `题数 × 5 分钟` 代替总耗时测量。

### 4. 复用模型服务与 Evaluator

官方 v2 使用以下真实接口，具体调用参照该版本 Notebook：

- `swegemma.models.load_tasks` 读取任务。
- `swegemma.models.discovery.validate_single_declared_model` 确认模型别名。
- `adk_submission.VllmConfig`、`VllmServer`、`discover_adapters` 管理服务与模型注册。
- `swegemma.config.EvalConfig`、`build_submission_limits` 配置运行。
- `swegemma.evaluate.Evaluator`，逐题调用 `evaluate_task(task=..., task_index=..., total_tasks=...)`。

官方示例使用 `sandbox='subprocess'`；模型服务使用 `tool_call_parser='gemma4'`、`reasoning_parser='gemma4'`、`max_model_len=32768`。其 tensor parallel 根据实际 GPU 数选择，不能因为示例可运行就假定任意 GPU 组合可用。

直接复用官方 Notebook 的同步/异步适配方式。不要把比赛的声明式提交改成自定义 `agent.py` 入口；Notebook 只是开发载体。

### 5. 保留原始结果并分析失败

官方结果目录包含 `summary.json`、`task_results.jsonl`、`patches/<instance_id>.patch`、`test_outputs/<instance_id>.log`、`traces/trace_<instance_id>.json`、`logs/<instance_id>.log`。

后续实现复核补充：手动逐题evaluate_task不会自动生成上述全部汇总，须显式调用append_task_result；JSONL中保存agent_patch_size而不是agent_patch正文。完整改动见003补充R08–R09，不能仅改成从trace猜测裁判结果。

官方 v2 实际读取的结果属性包括 `resolved`、`test_exit_code`、`agent_patch`、`tool_calls`、`duration_seconds`。原始返回与轨迹先保留，再生成分析视图；缺失字段记为空，不能用推测补值。

本项目分析表额外记录：`run_id`、任务 ID、仓库、commit、配置版本、执行环境、准备耗时、Agent 耗时、验证耗时、补丁是否非空、退出原因、失败类别、证据文件路径。上述额外字段为项目自定义，时间分段仅在日志确实提供时填写。

失败类别允许多标签，另记录主要原因：环境/依赖、定位、理解与修改、工具格式或执行、上下文截断、预算、提交失败、尚无法确定。不要将环境失败直接解释为模型修复能力不足。

### 6. 打包开发配置

复用官方打包单元格，以 `agent/` 为压缩根目录，得到 `submission.zip`；包内直接看到 `agent.yaml`。所有 `!include` 目标必须存在于包内，所有未使用的子 Agent 和 adapter 文件不进入包。

打包完成、Agent 能执行、补丁通过验证、正式比赛取得分数是不同结果，分别记录。公开任务生成的补丁表不是本赛要求上传的 Agent 包。

## 运行观察与环境问题

由使用者在 Notebook 中分阶段启动官方现有流程，先观察少量任务，再决定是否扩大。第一批重点看模型是否正确加载、工具调用是否完整、补丁能否提取、轨迹和验证输出能否对应到同一任务。

公开 subprocess 环境已有依赖缺失、导入已安装包而非工作区源码的社区报告。主办方 9 月 27 日表示这可能与 subprocess 沙箱有关，并将改进 Notebook；未证明所有任务、Docker 和正式评分都受相同问题影响。

遇到导入失败或参考修复也不能通过的任务，先记录实际模块路径、依赖版本和原始错误，并与当期官方环境说明比较。若使用官方现有功能观察未修改版本和参考修复的结果，这些结果属于环境诊断，不计为 Agent 成绩。不要通过改变测试预期、删除用例或修改评分器来消除差异。

报告时同时给出全部开发任务结果、环境错误数量、环境可解释子集的结果与选择依据。保留原始分母；不能静默剔除困难任务后宣称比赛解决率提高。

## 研究产出与下一轮

本轮预期得到单 Agent 提交目录、可打包的 zip、固定任务划分、原始轨迹和失败分类表。至少应能回答：失败主要发生在哪个阶段；准备和推理各花多少时间；哪些结果被公开运行环境影响；后续应该优先改定位还是改生成行为。

实际实现与运行结果写入 `docs/progress/001-baseline.md`，包括修改文件、数据与环境版本、实际任务 ID、未解决问题和下一轮依据。没有实际运行数据时不填写预期分数。

## 资料来源

- [官方 Getting Started](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent)：本次读取元数据 current_version_number=2、last_run_time=2026-09-25T06:14:17.557475700Z；已读取源码，未执行。
- [官方数据与 HARNESS_README.md](https://www.kaggle.com/competitions/gemma-4-developer-agent/data)：本轮接口和数据字段依据。
- [时间预算与调度说明](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743063)：主办方 2026-09-24 回复。
- [公开运行环境讨论](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/742882)：主办方 2026-09-27 回复。
- [比赛规则](https://www.kaggle.com/competitions/gemma-4-developer-agent/rules)：本次已取得正文。
