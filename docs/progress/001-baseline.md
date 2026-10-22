# 001 进展：无微调基线实现与三题初步运行

> 目录整理（2026-10-22）：本页历史载体路径保留当时记录；当前文件位置和重建入口见 [Notebook 目录导航](../../notebooks/README.md)。

日期：2026-09-28。状态：已实现基线，并完成用户在 Kaggle 的三题初步运行，记录到 1 题 resolved。原始运行轨迹尚未在本地归档，逐题观察来自用户提供的日志记录；结果保存、验证超时与计时口径的问题已在文末复核。

## 本次修改文件

- `agent/agent.yaml`：单 `LlmAgent`，模型 `gemma-4-31b-it-qat-w4a16-ct`，`!include prompts/system.md`，仅 6 工具（run_command/read_file/edit_file/write_file/get_status/submit_patch），无 `adapter`、无子 Agent。相对官方 sample（9 工具 + `agent_tool` + `adapter: main_lora`）做减法。
- `agent/prompts/system.md`：定位→阅读→修改→检查→提交；仓库证据、小范围 `read_file`（1 起含两端，`is_truncated` 分段）、`run_command` 先查可用命令（不默认 `rg`）、`get_status` 收尾、`submit_patch`；不改测试、不全量 pytest；`hints` 仅会话存在非空时参考，不做缺失变量插值。
- `agent/configs/sampling.yaml`：`temperature 0.2 / top_p 0.95 / max_output_tokens 16384 / thinking_budget 4096 / include_thoughts true`，省略 `thinking_level`。与官方 v2 采样一致，加版本注释。
- `agent/eval_config.yaml`：`evaluation: timeout_seconds 60 / max_tool_calls 50 / max_time_minutes 5 / max_turns 80`。官方 sample 为 `60 / 10 / 1 / 50`；本轮放宽调用/时间/轮次作小批观察起点，不是官方默认，全量前按实测重定。
- `notebooks/001-baseline.ipynb`：由官方 v2 改编；环境安装、清单校验、vLLM 服务、逐题顺序执行、结果读取、打包各独立单元格；显式用 `load_tasks / validate_single_declared_model / VllmConfig,VllmServer,discover_adapters / EvalConfig,build_submission_limits / Evaluator.evaluate_task(task,task_index,total_tasks)`，`tool_call_parser=gemma4, reasoning_parser=gemma4, max_model_len=32768, sandbox=subprocess`。
- `research/task_splits.json`：种子 `20260928`，按 `(repo, base_commit)` 分组不拆组；`heldout 28 / dev_panel 24（含 smoke 3）/ dev_future 76 / explore_singleton 1`；仅 ID/仓库/commit/标签，无答案。smoke：`fastapi_11194, rich_2725, requests_6592`；单例：`httpx_3672`。
- `results/001/`：空目录占位，运行后按 `<run_id>/` 存官方原始产物 + 环境配置快照。

## 数据与环境版本（本次 MCP 实测）

- 参赛：`user_has_entered=true`，截止 2026-12-02，每日 1 次，最多 2 最终提交。
- GPU 额度：周 108000s（30h），已用 0、预留 0，刷新 2026-10-03T00:00Z；不证明 L4 可用。
- 数据：`tasks.jsonl 1984455B sha256 e4b3fd60…f5f8f3c8ad6`，129 行，127 组（fastapi 67 / rich 48 / requests 13 / httpx 1），与计划指纹一致；`HARNESS_README.md 49356B` 一致；`.tgz129/.npz127/.json129/.whl124`。
- 官方 Notebook：`ryanholbrook/getting-started… current_version_number=2`，最新读到 `last_run_time=2026-09-25T07:06:36Z`（计划记录 06:14:17Z；中间曾观测 06:49:38Z，属重跑时间戳漂移）；模型资源 `google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2`。
- v1→v2 源码 diff（2026-09-28 实测，仅 3 处实质变更，其余为标题/时间戳）：
  - 新增顶层标题 cell `827a919e`；删除 §5 下预算说明 cell `7e3c6acc`（“sample 约束极低、可自设，仅 12h 强制”）。
  - `VllmConfig` 删除 `default_chat_template_kwargs={'enable_thinking': True}`；本轮 notebook 同 v2，不加该参数。
  - §5 预算由 v1 硬编码（`max_tool_calls=100`、`max_time_minutes=5.0`，忽略 eval 文件）改为 v2 全部从 `eval_config.yaml` 读取（默认 `300/100/60.0/None`）；本轮 `60/50/5/80` 在 v2 下真实生效，在 v1 下调用与时间会被强制覆盖。显式填写四字段的理由不变。
- 官方 sample：`agent.yaml` 9 工具 + 子 Agent + `main_lora`；`sampling` 与本轮一致；`eval_config 60/10/1/50`；`system.md 3888B` 快修约束已吸收；`code_analyzer.yaml` 确认 001 删除。
- 本地：Mac M5 只做校验，未安装 `swegemma/adk/vllm` GPU 栈；`agent` YAML 与 notebook JSON 已过本地解析。

## 本地验证

- `agent/*.yaml` YAML 解析通过；确认无 `adapter`、工具集合恰为 6、无 `thinking_level`、四预算字段齐全。
- `task_splits` 无跨组分裂（125×1 + 2×2 组均完整），无答案字段（`patch/test_patch/problem_statement/hints_text` 均未进文件）。
- notebook JSON 解析通过 8 markdown + 7 code；打包单元格以 `AGENT_DIR` 为根，校验 `agent.yaml` 在根、无 `sub_agents/adapters`、扩展名与 3GiB。
- 打包 dry-run（本地 `agent/` zip）：根见 `agent.yaml`，`!include` 两目标存在。

## 首次运行前的状态（历史记录，后续运行见下文）

- 未在 Kaggle 执行：模型加载、工具调用完整性、补丁提取、轨迹↔验证对应、准备/Agent/验证分段耗时均未知。
- 待 Kaggle 首跑 smoke 3 题后填：`run_id`、任务 ID、配置版本、执行环境（Python/CUDA/GPU/包版本）、分段耗时、补丁非空、退出原因、失败主因、证据路径。
- 版本漂移已核对：v1→v2 仅上述 3 处实质变更，环境安装/数据拷贝/图函数/打包均一致；重跑前仍须重取源码确认（若出 v3，重点看预算与 VllmConfig 是否再变）。
- 主办方 9-26/9-27 答复（embedding 已修 / async 缺失 / subprocess 沙箱）仍以重下载后实测为准；`EventsCompactionConfig(15/2/14336/5)` 为 Notebook 侧配置，不宣称为正式评分压缩。
- 12h 触顶、隐藏任务顺序/数量、token 统计口径待运行日志确认。

## MCP 代跑经过（2026-09-28，均失败，未产生有效结果）

- v1（`ccjjllt/001-baseline-smoke-private-run`）：提交时多传 `textNullable: null`，落盘为空 Notebook，papermill 报 `NotJSONError: ''`。教训：该 MCP 通道里 `has*=true` 表示取 `*Nullable` 值，`text` 与 `textNullable: null` 同传会清空内容（CPU 探针 `probe-text-field` 反证：只传 `text` 则内容正常）。
- v2（`ccjjllt/baseline-001-smoke-private-run`，内容已验证完整、GPU=L4）：cell1 写基线成功；cell2 `Installing 0 wheels` 后 pip 空参数报错。根因是比赛/wheelhouse/模型三个数据源都没挂载（`/kaggle/input` 为空）。
- 数据源挂载试过 6 种组合（CPU 探针验证，notebook 元数据始终无 sources，运行时 `/kaggle/input` 为空）：单数字段+Setter+has*+Nullable、仅Setter+has*、仅Setter、仅单数、id更新通道+Setter、单数+has*（v5，含模型）。结论：`kaggle_save_notebook` MCP 通道不支持挂数据源（比赛/dataset/模型），GPU 代跑走不通；`list_mcp_resources` 为空，无替代接口。slug 更新通道也拒收（`Invalid slug`），只能建新本。
- GPU 消耗共 30.19s（失败均在 10~16 秒内），剩余额度约 30h。残留 private 探针本 5 个（`probe-text-field/input-mounts/mounts-v2/v3/v4`）+ 2 个 smoke 本，无删除接口，保留不动。
- 注意：MCP 提交版内嵌的是精简 system.md（2238B），与仓库正式版（2555B）不一致；正式 smoke 以仓库版为准，即 `notebooks/001-smoke-single-cell.py`（已校验四段内嵌与仓库逐字节一致）。

## 浏览器首跑（2026-09-28，用户执行 smoke，MCP 只读）

- 实测环境（首跑日志）：`python 3.12.13 / Linux / torch 2.10.0+cu128 / cuda=True / gpus=4（4×L4，各 22GiB）`，`swegemma 0.2.7 / adk_submission 0.2.11 / vllm 0.19.1`，`tasks.jsonl` 指纹与计划一致，129 题，smoke 选取与预算校验通过。注意这是 4×L4 评分级环境，tp=4。
- 首个代码错误（已修，未重跑）：`adk_submission 0.2.11` 的 `discover_adapters` 返回 `AdapterManifest` 对象（无 `__len__`），`assert len(adapters) == 0` 报 `TypeError`。改为取 `adapters.adapters（或 manifest/__dict__）` 内部映射计数；`EvalConfig(adapter_manifest=…)` 传参保持官方类型不变。同步修了 `notebooks/001-smoke-single-cell.py` 与 `notebooks/001-baseline.ipynb`。
- MCP 读不到浏览器交互式运行的 cell 日志（session 输出接口返回空），报错仍需用户贴 traceback（只需最后的 `Error` 段）。

1. 官方 Getting Started → Copy and Edit（fork 继承比赛数据 + wheelhouse + 模型）。
2. Accelerator 选 GPU L4；新建空 cell，粘贴 `notebooks/001-smoke-single-cell.py` 全文运行（约 30~45 分钟）。
3. 把输出（`ROWS/RESOLVED`、`EVAL_WALL_MIN`、逐题行、环境版本行）贴回；产物（`results/`、`analysis_001_smoke.csv`、`submission.zip`）保留在 Kaggle working。
4. 用户告知 fork 的 slug 后，用 MCP 只读轮询运行状态与输出，不再经 MCP 提交运行。

## smoke 运行结果（2026-09-28，用户浏览器执行 `ccjjllt/001-baseline-smoke`，run_id 以 Kaggle 为准）

环境：4×L4（各22GiB，tp=4），vLLM 启动 346.4s；装包约1分钟；`EVAL_WALL_MIN=14.1`；`submission.zip` 打包校验通过。GPU 额度累计用 3858s（约1.07h）。

| 任务 | resolved | exit | patch | calls | 返回的整题时长 | 主因 |
| --- | --- | --- | --- | --- | --- | --- |
| fastapi_11194 | False | 1 | 2483（超时自动提取，未 submit） | 40/50 | 308s（超时） | 预算耗尽＋重复无效动作：同一 `reproduce_issue.py` 反复执行约25次、5 次 `read_file(params.py)` 无收敛、零源码修改；多次自判“已修复”却不收尾 |
| rich_2725 | **True** | 0 | 2936（超时自动提取） | 35/50 | 304.6s（返回的整题时长） | 本题修复通过，记录称约270s修改 `table.py:869 box_segments` 索引，随后自动提取；读取范围异常的归因仍需原始工具参数与响应 |
| requests_6592 | False | **124（验证超时）** | 579（`submit_patch` 正常提交，1 文件） | 33/50 | 162s（返回的整题时长） | 已生成并提交补丁，最终验证未完成；补丁正确性、验证预算与环境影响尚不能确定 |

环境观察：每题沙箱有`ensurepip unavailable`警告，未阻止运行启动；这不证明依赖和导入一致性已经验证。日志记录出现get_status调用，但状态的具体时间字段及提交过程仍需原始trace核对，不能把最后一次状态时间视为完整Agent时长。
工具观察：本次逐题调用后未生成task_results.jsonl/summary.json，汇总显示ROWS=0。后续源码已确认是载体遗漏显式持久化；先前“只改从trace提数”的建议已被替代，应调用官方append_task_result并读取正确字段。原始轨迹存于Kaggle `001-baseline-smoke`的working区（fastapi约159KB、rich约171KB、requests约94KB），MCP读取旧本子持续kernels.get denied；目前分类仍依据用户粘贴日志，未独立复核原始文件。

## 001 四问回答

1. 主要观察：3 题中 1 题验证成功；FastAPI 有重复动作与预算耗尽，Rich 虽超时仍通过自动回收获得成功，Requests 验证超时。退出事件与修复失败应分别统计。
2. 时间：模型启动约346s、批次墙钟约14.1分钟；返回的308/304.6/162秒包含准备、Agent和验证，不能直接称为推理时长，也不能据此预测12小时吞吐量。具体阶段仍需原始时间线。
3. 环境影响：未见明确致命初始化失败；验证时限、源码导入路径及读取范围异常的影响尚未排除。保留3题原始分母，不能把Requests计为已修复，也不静默剔题。
4. 下一步优先研究**固定预算下的无进展识别与收尾**。这是针对已观察行为的优先级选择；Rich单题成功不能证明定位已不构成瓶颈，扩大开发面板后再决定是否启动002。

## 下一步

1. （可选）MCP 对旧本子读取恢复后，拉 `traces/*.json` 存档核对；不阻塞 002/003。
2. 修结果持久化与汇总：逐题返回后调用官方保存函数，读取正确的补丁长度字段；traces用于补充行为时间线，不能替代裁判结果。具体见下文及003计划。
3. 002/003复用本轮划分、环境和日志；004目标候选为重复无效动作抑制，仍需多任务证据和完整轨迹归档。

## 2026-09-28 后续研究复核

本次只读检查仓库实现、官方最新讨论及官方 wheel 源码，未重跑模型或验证。旧 Notebook `ccjjllt/001-baseline-smoke` 的读取仍返回 `kernels.get` 权限错误，因此没有独立复核原始补丁和轨迹。

以下结论来自当前取得的 `swegemma 0.2.7` 源码，SHA-256 为 `2b74d402dc95a615f3b36c56fd3f9ba11ddbfb7db8a5a32ec259369a0f559d98`；后续运行应记录实际安装文件版本，避免同版本重打包造成差异。

- `evaluate.py` 的 `Evaluator.evaluate_task()` 返回 `TaskResult`，本身不写结果汇总；批量 `run()` 才调用 `swegemma.results.append_task_result`。当前逐题入口只打印结果，是 `ROWS=0` 的直接实现原因，不是没有运行结果。
- `append_task_result(result, output_dir, running_results)` 可保存 JSONL、补丁、裁判输出与累计汇总。官方 JSONL 用 `agent_patch_size` 表示补丁字符数，没有 `agent_patch` 正文字段。现有汇总读取 `agent_patch`，即使文件生成也会误报空补丁。
- `EvalConfig.timeout_seconds` 被传给 `harness.command_timeout_seconds`；`tools/execution.py` 的命令执行和 `harness/verification.py` 的最终 pytest 共用它。本轮60秒配置也限制了最终裁判，必须先对冻结补丁诊断，不能只加Agent时间。
- `TaskResult.duration_seconds` 从 `evaluate_task` 进入时开始，在验证后计算，包含准备、Agent与验证，不含外部模型服务启动。若表中数字来自其他日志，应保留原始出处，否则按整题时长解释。
- `tools/workspace.py` 的核心 `read_file` 及注册 wrapper 都传递并支持 `start_line/end_line`。目前无法认定是harness忽略正确行范围，应先查看模型实际传入的参数、工具schema与响应。
- FastAPI“无生产源码修改”与“补丁非空”可以同时成立；补丁可能包含复现脚本等文件，具体内容尚未取得。后续按实际diff区分生产代码、测试、临时文件，不把字符数当有效修复证据。

主办方在[9月28日澄清](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743964)中确认：12小时超限整次报错仍未修复，不支持并行模型调用，LoRA加载已修复，允许用Gemma-4-31B自身生成数据训练；其他模型输出仍待答复。这进一步支持先改善串行运行效率，暂不增加多Agent或训练复杂度。

下一轮按[003更新计划](../plan/003-budget-and-context.md)推进；本节是研究复核，不代表003已经实现或获得效果。

## 进一步的运行链复核

本次追加只读检查adk-submission 0.2.11、google-adk 1.36.1与两个运行载体，补充了安装前导入、旧wheel/缓存、缺失清单自动换题、scope误触发、输出覆盖、服务释放、异常恢复、模型身份、实际工具schema、字符截断和提交语义等问题。详见[003运行载体补充](../plan/003-runtime-reliability.md)的R01–R19。

已确认公有SwegemmaContext可用于记录完整Agent阶段时长；read_file字符截断时返回end_line可能超出实际显示内容；免费状态/提交工具仍受时限限制；成功提交后的新修改不会自动替换已提交补丁。上述源码事实尚不等于已定位001每个异常的唯一根因。

研究设计也相应修正：共同提示消除强制制造修改的要求，预算观察节奏在各组一致；检测非相邻短循环；平衡组别执行次序；保留28题不提前用尽。当前只更新交接文档，功能代码与远程运行状态未改变。
