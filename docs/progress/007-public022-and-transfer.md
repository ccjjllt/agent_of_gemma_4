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

## 2. 12B QAT CPU NF4 缓存完成与 12B GPU 训练/CT 探针完成（V6 `COMPLETE`）

- **免费 CPU 12B NF4 缓存（`ccjjllt/p007-12b-cpu-nf4-cache` V1 `COMPLETE`，`721.02s`，`0s` GPU 额度）**：
  - 校验固定源 `google/gemma-4-12B-it-qat-q4_0-unquantized` @ `b6ed86275a6a5735884e208bfed95b445a684ca2`（`model.safetensors` 23.92 GB，SHA256 `26f2cee4...`）与官方部署模型 `google/gemma-4/Other/gemma-4-12b-it-qat-w4a16-ct/2`（`chat_template.jinja` SHA256 `36e3a42e...`），生成 7.71 GB `/kaggle/working/qat-12b-nf4`（331 个 `Params4bit` 张量、4 个 `<2GB` 分片及完整 manifest）与 6 个 Python 3.12 训练 wheels（清单归档于 [`cpu-cache-output/`](../../research/007/12b-qat-cache-and-train-20261017/cpu-cache-output/cache-manifest.json)）。
- **训练器升级与免费 CPU 预检（`ccjjllt/p007-12b-train-and-ct-probe-cpu-prepare` V2 `COMPLETE`，`92.69s`，`0s` GPU 额度）**：
  - 更新 `training/train_sft.py` 与 `training/configs/sft_12b_qat_nf4.yaml` 支持 12B NF4 缓存源哈希校验；在 CPU 预检中修复了 wheelhouse v31 `bitsandbytes-0.46.1` 与训练缓存 `bitsandbytes-0.50.2` 同名冲突，完整通过 43 个 wheels 安装、`--dry-run`、NF4 缓存溯源与 CT v2 协议校验（报告见 [`cpu-prepare-output-v2/`](../../research/007/12b-qat-cache-and-train-20261017/cpu-prepare-output-v2/p007-12b-probe/summary-report.json)）。
- **12B 4×L4 GPU 训练与官方 CT v2 对照探针（`ccjjllt/p007-12b-train-and-ct-probe` V6 `COMPLETE`，全程约 8 分钟墙钟 `493.7s`）**：
  - Phase 1（NF4 QLoRA 单步）：优化步 `14.827s`，峰值 `9.7362 GiB`，240/240 个 `lora_B` 非零；`click` loss `0.530146 → 0.515842`（Δ `-0.01430`），未参训 `pydicom` loss `1.654296 → 1.653286`（Δ `-0.00101`），三次运行稳定复现。
  - Phase 2（官方 12B CT v2，`--hf-overrides Gemma4ForCausalLM`，`max_model_len=8192` + `--max-num-batched-tokens 512` 解决长 prompt 全词表 `log_softmax` OOM）：启动 `195.2s`，3 模型注册，KV `182,848` token。Teacher-forced 覆盖率 100%（519 + 226 token，`missing=0`）；`zero_r8_ctrl` 与 base 差异精确 `0.0`（对照有效）。真实 adapter：`click` NLL `0.507977 → 0.503882`（Δ `-0.004095`，189 改善 / 158 退化 / 172 不变，最大单 token 变动 `1.248`），`pydicom` NLL `1.686483 → 1.684741`（Δ `-0.001742`，96 / 96 / 34）。Live 生成两案 adapter 均与 base 分叉，但尚未转化为成功修复（`edit_ok` 全 `False`）。
  - 结论边界：数值作用存在（含未参训样本），但幅度小且未形成修复增益；不报 SFT 收益。中间 V2–V5 的失败原因（`edit_file_content` 导入名、`--hf-overrides` 缺失致多模态后端 LoRA 断言、长 prompt `log_softmax` OOM、`supervised_spans` 字符偏移字段误用）已逐一定位修复。
- **外部 8 仓库候选源核验与 CPU 环境就绪检查**：
  - 已通过 `git ls-remote` 与 GitHub 原始文件核实 `external-pr8-manifest.json` 全部 8 个外部 PR 实例在 `swesmith/<mirror>` 上的精确 buggy commit 与开源许可证（Apache-2.0 / BSD / MIT）。
  - 免费 CPU 验证 `ccjjllt/p007-external-pr8-cpu-verify` V4 `COMPLETE`（`44.82s`，`0s` GPU 额度；V1/V2 因 dataset 管道未创建出数据源而失败，已改直连 codeload + SHA 校验）：`gpxpy`/`schema` 导入通过但根目录无 pytest 用例（待定测试命令）、`feedparser` 导入失败（需已移除的 `sgmllib` 且 conftest 需第三方 `responses`，记离线环境失败）、`cookiecutter` 导入通过、收集仅卡在缺 `pytest-cov` 插件、`conan` 导入通过、收集多文件 ERROR（待全量错误表）、`iniconfig` 当次导入命中了系统副本（src-layout 方法学漏洞，判无效待重测）。更新后：2 候选、1 待重测、2 待验依赖、3 环境失败，仍未达"至少 4 个准备就绪"门槛，不开 GPU 配对；方法学修复版 V5 已推送（`src/` 入 PYTHONPATH、addopts 置空重试、全量错误捕获、测试文件清单）。详见 [`external-env-readiness.json`](../../research/007/evaluation-design-20261017/external-env-readiness.json) 与 [`cpu-output-v4/`](../../research/007/external-pr8-20261018/cpu-output-v4/ext-pr8-verify/report.json)。
