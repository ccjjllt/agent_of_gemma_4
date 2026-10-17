# P007进展：公共12题配对评测完成、12B CPU NF4缓存就绪与12B GPU探针启动

2026-10-17，北京时间。按 [007计划](../plan/007-public022-and-transfer.md) 执行公共 12 题（24 次尝试）单变量配对评测与 12B QAT 真实样本训练准备。遵循最大化 GPU 利用率原则：所有依赖安装校验、任务快照与双臂编译检查、12B QAT 23.92 GB 权重下载与 NF4 量化均先在免费 CPU Kernel 完成（`0s` GPU 额度），随后再在 4×L4 GPU 上执行。未读取 heldout 或预留评测题内容，未新增测试文件。

## 1. 公共 12 题（24 次尝试）4×L4 配对评测完成：`public022_v10` vs `behavior_v1`

- **严格单变量设计**：基于冻结的 `public022_v10`（`flexonafft` V10 原样四文件，8 分钟 / 28 次工具调用 / 80 轮 / 240 秒单命令超时，31B CT v2，`temperature=0.2, top_p=0.95, max_output_tokens=8192, thinking_budget=4096`），构造单变量候选 [`behavior_v1`](../../research/007/candidates/behavior_v1/prompts/system.md)（`agent.yaml`、`configs/sampling.yaml`、`eval_config.yaml` SHA256 与基线完全一致，仅修改 `prompts/system.md` 中 `Locate / Read / Verify` 的行为验证准则，SHA256 `079af841...`）。
- **CPU 预检与 4×L4 滚动双配对并发**：
  - 免费 CPU 预检 `ccjjllt/p007-public12-pair-cpu-prepare` V1 `COMPLETE`（`151.03s`，`0s` GPU 额度），验证 41 个 v31 wheels、12 个 `dev_future` 任务与快照、双臂 `compile_submission` 通过。
  - 4×L4 GPU 评测 `ccjjllt/p007-public12-pair` V2 `COMPLETE`（`run-007-public12-pair-20261017T063527Z-5d0f2c69`）：单次启动共享 31B CT v2 vLLM 服务（`server_start_count = 1`，启动 `333.41s`，KV cache `57,664 tokens`），采用滚动双配对并发队列（`MAX_CONCURRENT = 4`，`--max-num-seqs 4`，峰值生成吞吐 `108.9 tokens/s`），**56.63 分钟**（墙钟 `3397.60s`，批处理 `3291.30s`）跑完全部 24 次尝试（注：V1 因 CLI 默认落到 T4 在第 16 秒被 4×L4 断言快速拦截退出）。
- **配对结果（详见 [完整评测报告](../../research/007/public12-pair-20261017/report.md) 与 [结构化数据](../../research/007/public12-pair-20261017/paired-analysis.json)）**：
  - 官方 `resolved = true`：`public022_v10` 为 **3 / 12 (25.0%)**，`behavior_v1` 为 **4 / 12 (33.3%)**（净增 **+1 题，0 回退**）。
  - **含非空源码修改**且 `resolved = true`：`public022_v10` 为 **2 / 12 (16.7%)**，`behavior_v1` 为 **4 / 12 (33.3%)**（翻倍，因 `fastapi_14851` 上基线耗尽 28 次调用未改代码 `patch_size=0`，仅靠当前环境 Starlette 尚未移除 `on_event` 空补丁通过，而 `behavior_v1` 实际产出 `526B` 生产补丁并通过）。
  - 配对四格表：双臂均通过 **3 题**（`rich_3067`, `fastapi_14851`, `fastapi_14463`）；`behavior_v1` 独胜 **1 题**（`rich_3180`：构造 `divide_line("A你好", 3)` 正反合同断言，9 次调用定位并修复 `rich/cells.py::chop_cells` 中 `reversed(characters)` 缺陷，114 passed；基线 16 次调用在 8 分钟超时前仅写出 `repro.py`）；`public022_v10` 独胜 **0 题**；双臂均未通过 **8 题**。
  - 预算与耗时分布：`behavior_v1` 将 8 分钟会话超时题数从基线的 **6/12 (50.0%)** 降至 **3/12 (25.0%)**，未正常收尾率（超时+工具预算耗尽）从 **50.0%** 降至 **33.3%**，平均工具调用从 `22.75` 降至 `21.25`（在 `rich_3067` 上从 28 次调用/5 次错误降至 **13 次调用/0 错误**）。
- **新发现的两项关键环境/复现机制**：
  1. **死循环复现吞噬半场预算（`rich_4006`）**：因单命令超时设为 `240s`，双臂直接运行 `python3 -c "from rich.cells import split_graphemes; print(split_graphemes('\ufe0f'))"` 复现死循环时单条命令卡满 `240s`（消耗半场 8 分钟预算）；`behavior_v1` 仍改出 `709B` 补丁修复了 `\ufe0f` 分支，但因剩余时间不足未及修复相邻 `\ufe0e` 分支。
  2. **离线 `SubprocessManager` 对 `src/` 布局仓库的导入遮蔽（`psf/requests` 四题）**：核查 `swegemma` 源码与 trace 证实，Notebook 离线 `SubprocessManager` 跳过 `pip install -e /workspace` 且仅设 `PYTHONPATH=/workspace`，导致 `psf/requests`（位于 `/workspace/src/requests`）在 `verify_task` 中实际导入系统 `/usr/local/lib/python3.12/dist-packages/requests`。在 `requests_7205` 上，双臂均写出了与官方 Gold 完全一致的补丁（`if _netrc and any(_netrc):`），且 `behavior_v1` 通过 `print(requests.__file__)` 诊断出遮蔽并用 `PYTHONPATH=/workspace/src` 验证通过，但离线评测容器因未挂 `/workspace/src` 误判失败（官方重跑使用 Docker `ContainerManager` 不受此限）。此外 `fastapi_14482` 与 `fastapi_14459` 因 wheelhouse 缺少 `inline_snapshot` 在 pytest 收集阶段报 `ModuleNotFoundError` (`rc=2`)，`requests_6757` 因修改 `tox.ini` 被保护文件过滤规则剥离。

## 2. 12B QAT CPU NF4 缓存完成与 12B GPU 训练/CT 探针启动

- **免费 CPU 12B NF4 缓存（`ccjjllt/p007-12b-cpu-nf4-cache` V1 `COMPLETE`，`721.02s`，`0s` GPU 额度）**：
  - 校验固定源 `google/gemma-4-12B-it-qat-q4_0-unquantized` @ `b6ed86275a6a5735884e208bfed95b445a684ca2`（`model.safetensors` 23.92 GB，SHA256 `26f2cee4...`）与官方部署模型 `google/gemma-4/Other/gemma-4-12b-it-qat-w4a16-ct/2`（`chat_template.jinja` SHA256 `36e3a42e...`），生成 7.71 GB `/kaggle/working/qat-12b-nf4`（331 个 `Params4bit` 张量、4 个 `<2GB` 分片及完整 manifest）与 6 个 Python 3.12 训练 wheels（清单归档于 [`cpu-cache-output/`](../../research/007/12b-qat-cache-and-train-20261017/cpu-cache-output/cache-manifest.json)）。
- **训练器升级与免费 CPU 预检（`ccjjllt/p007-12b-train-and-ct-probe-cpu-prepare` V2 `COMPLETE`，`92.69s`，`0s` GPU 额度）**：
  - 更新 `training/train_sft.py` 与 `training/configs/sft_12b_qat_nf4.yaml` 支持 12B NF4 缓存源哈希校验；在 CPU 预检中修复了 wheelhouse v31 `bitsandbytes-0.46.1` 与训练缓存 `bitsandbytes-0.50.2` 同名冲突，完整通过 43 个 wheels 安装、`--dry-run`、NF4 缓存溯源与 CT v2 协议校验（报告见 [`cpu-prepare-output-v2/`](../../research/007/12b-qat-cache-and-train-20261017/cpu-prepare-output-v2/p007-12b-probe/summary-report.json)）。
- **12B 4×L4 GPU 训练与官方 CT v2 对照探针（`ccjjllt/p007-12b-train-and-ct-probe` V1）**：
  - 已显式带 `--accelerator NvidiaL4 -t 2400` 推送（当前状态 `QUEUED`），将在单次会话的两个隔离子进程中依次执行：（1）`click` 单样本真实 NF4 QLoRA 前向/反向/Adam 更新、`export_peft.py` 独立导出、构造同结构零 delta 对照 `zero_r8_ctrl`，并在 `click`（train）与 `pydicom`（select，`no_grad`）上对比更新前后 NF4 loss；（2）启动官方 12B CT v2 `VllmServer`（保留 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`），在相同输入上对比 Base、`zero_r8_ctrl`、`click_r8_step1` 的全部监督 token 对数概率及现场工具调用执行结果。
- **外部 8 仓库候选源核验**：
  - 已通过 `git ls-remote` 与 GitHub 原始文件核实 `external-pr8-manifest.json` 全部 8 个外部 PR 实例在 `swesmith/<mirror>` 上的精确 buggy commit 与开源许可证（Apache-2.0 / BSD / MIT）。
