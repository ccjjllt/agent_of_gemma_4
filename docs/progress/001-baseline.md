# 001 进展：无微调基线实现（未运行）

日期：2026-09-28。状态：已实现可打包基线与运行载体，尚未在 Kaggle GPU 运行，无模型结果，不填预期分数。

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

## 未运行与未解决问题

- 未在 Kaggle 执行：模型加载、工具调用完整性、补丁提取、轨迹↔验证对应、准备/Agent/验证分段耗时均未知。
- 待 Kaggle 首跑 smoke 3 题后填：`run_id`、任务 ID、配置版本、执行环境（Python/CUDA/GPU/包版本）、分段耗时、补丁非空、退出原因、失败主因、证据路径。
- 版本漂移已核对：v1→v2 仅上述 3 处实质变更，环境安装/数据拷贝/图函数/打包均一致；重跑前仍须重取源码确认（若出 v3，重点看预算与 VllmConfig 是否再变）。
- 主办方 9-26/9-27 答复（embedding 已修 / async 缺失 / subprocess 沙箱）仍以重下载后实测为准；`EventsCompactionConfig(15/2/14336/5)` 为 Notebook 侧配置，不宣称为正式评分压缩。
- 12h 触顶、隐藏任务顺序/数量、token 统计口径待运行日志确认。

## MCP 代跑经过（2026-09-28，均失败，未产生有效结果）

- v1（`ccjjllt/001-baseline-smoke-private-run`）：提交时多传 `textNullable: null`，落盘为空 Notebook，papermill 报 `NotJSONError: ''`。教训：该 MCP 通道里 `has*=true` 表示取 `*Nullable` 值，`text` 与 `textNullable: null` 同传会清空内容（CPU 探针 `probe-text-field` 反证：只传 `text` 则内容正常）。
- v2（`ccjjllt/baseline-001-smoke-private-run`，内容已验证完整、GPU=L4）：cell1 写基线成功；cell2 `Installing 0 wheels` 后 pip 空参数报错。根因是比赛/wheelhouse/模型三个数据源都没挂载（`/kaggle/input` 为空）。
- 数据源挂载试过 4 种组合（单数字段+Setter+has*、仅Setter、仅单数、id 更新通道）均失败，notebook 元数据始终无 sources。结论：`kaggle_save_notebook` MCP 通道不支持挂数据源，GPU 代跑走不通。slug 更新通道也拒收（`Invalid slug`），只能建新本。
- GPU 消耗共 30.19s（失败均在 10~16 秒内），剩余额度约 30h。残留 private 探针本 5 个（`probe-text-field/input-mounts/mounts-v2/v3/v4`）+ 2 个 smoke 本，无删除接口，保留不动。
- 注意：MCP 提交版内嵌的是精简 system.md（2238B），与仓库正式版（2555B）不一致；正式 smoke 以仓库版为准，即 `notebooks/001-smoke-single-cell.py`（已校验四段内嵌与仓库逐字节一致）。

## 浏览器跑法（待用户执行）

1. 官方 Getting Started → Copy and Edit（fork 继承比赛数据 + wheelhouse + 模型）。
2. Accelerator 选 GPU L4；新建空 cell，粘贴 `notebooks/001-smoke-single-cell.py` 全文运行（约 30~45 分钟）。
3. 把输出（`ROWS/RESOLVED`、`EVAL_WALL_MIN`、逐题行、环境版本行）贴回；产物（`results/`、`analysis_001_smoke.csv`、`submission.zip`）保留在 Kaggle working。
4. 用户告知 fork 的 slug 后，用 MCP 只读轮询运行状态与输出，不再经 MCP 提交运行。

## 下一步

1. 用户在浏览器跑通 smoke 3 题，贴回输出。
2. 按 001 计划填失败分类表（环境/定位/修改/工具/截断/预算/提交/待定），回答四问：主要失败阶段、准备与推理耗时、环境影响、优先改定位还是生成。
3. 002/003 复用本轮划分、环境和日志；004 目标等失败分布出来再定。
