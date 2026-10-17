# P007 公共12题（24次尝试）4×L4 配对评测报告：`public022_v10` vs `behavior_v1`

- **执行时间**：2026-10-17（UTC `06:35:27Z` – `07:32:05Z`）
- **CPU 预检 Kernel**：`ccjjllt/p007-public12-pair-cpu-prepare` Version 1（`COMPLETE`，`151.03s`，`0s` GPU 额度，验证 41 个 v31 wheels、12 个 `dev_future` 任务与快照、双臂 `compile_submission` 均通过，产物见 `cpu-output-v1/`）
- **GPU 评测 Kernel**：`ccjjllt/p007-public12-pair` Version 2（`COMPLETE`，`NvidiaL4` 4×L4，`run_id = run-007-public12-pair-20261017T063527Z-5d0f2c69`，完整产物与 24 条 trace/patch/test_output 归档于 `gpu-output-v2/results/007/run-007-public12-pair-20261017T063527Z-5d0f2c69/`）
- **注**：Version 1 因 CLI 未显式带 `--accelerator NvidiaL4` 默认落到 `NvidiaTeslaT4`（2×T4），在启动第 16 秒被 4×L4 断言拦截退出，未启动 vLLM；Version 2 显式指定 `--accelerator NvidiaL4 -t 5760` 一次跑完。

## 1. 实验设计与 GPU 利用率优化结果

- **严格单变量对照**：
  - `agent.yaml` SHA256：`680799291cc7d156d06fb7bfb5577f683ccc0d555b00e8ddb662d577f9bfa5e4`（双臂完全一致：8 min / 28 tool calls / 80 turns / 240s 命令超时，无额外 skills）
  - `configs/sampling.yaml` SHA256：`571a934e89911734927533f1d36cbcc7b8d493592febe63b217e911cf7ed4926`（双臂完全一致：`temperature=0.2, top_p=0.95, max_output_tokens=8192, thinking_budget=4096`）
  - `eval_config.yaml` SHA256：`916cf6cb34a6cc3de39837fa72cac681599b9ac2a3ee0b893adadfd36474c531`（双臂完全一致）
  - `prompts/system.md` SHA256：`public022_v10` 为 `7f8dde62578a247ffbfc1a519b712ae65849fcfae1ee70c49e0628cab74cbef6`；`behavior_v1` 为 `079af84132a2931adf2aa84d48bf15407dce92aa07c4dfa6cf330e0ce146bdca`（仅修改 `### 2. Locate`、`### 3. Read`、`### 5. Verify` 三个行为验证小节）。
- **单共享 31B CT v2 服务与滚动双配对并发（`MAX_CONCURRENT = 4`）**：
  - `server_start_count = 1`，vLLM 启动耗时 `333.41s`，批评测耗时 `3291.30s`，Supervisor 总墙钟 `3397.60s`（**56.63 分钟** 完成全部 24 次 8 分钟预算尝试）。
  - 4×L4 KV cache：`Available KV cache memory: 13.2 GiB`/卡，`GPU KV cache size: 57,664 tokens`（`32,768` 上下文下最大并发 `10.87x`）。
  - 4 路并发运行时 vLLM 生成吞吐最高达 `108.9 tokens/s`、预填充吞吐达 `985.9 tokens/s`，消除了双路 barrier 等待空闲。

## 2. 总体配对结果汇总

| 指标 | `public022_v10`（原样 0.22 基线） | `behavior_v1`（单变量行为验证升级） | 差值（`behavior_v1 - public022_v10`） |
|---|---:|---:|---:|
| 计划 / 完成任务数 | 12 / 12 | 12 / 12 | 0 |
| 官方 `resolved = true` 题数 | **3 / 12 (25.0%)** | **4 / 12 (33.3%)** | **+1 (+8.3 pct)** |
| **含非空源码修改**且 `resolved = true` 题数 | **2 / 12 (16.7%)** | **4 / 12 (33.3%)** | **+2 (+16.7 pct)** |
| 产出非空生产补丁（`agent_patch_size > 0`）题数 | 7 / 12 | 9 / 12 | +2 |
| 配对净胜题数（`behavior_v1` 独胜 vs `public022_v10` 独胜） | 0 独胜 | **1 独胜（`rich_3180`）** | **+1（0 回退）** |
| 单题平均 / 中位 / P95 耗时（含评测容器阶段，秒） | `381.64s` / `465.85s` / `488.61s` | `422.99s` / `454.62s` / `786.03s`（注：`rich_4006` 含 `300s` 测试容器超时） | 见注 |
| 平均工具调用数 / 平均 LLM 调用数 | `22.75` / `25.17` | **`21.25`** / **`24.58`** | `-1.50` / `-0.59` |
| 8 分钟会话超时题数 / 未正常收尾比例（超时+工具预算耗尽） | 6 题 / **50.0% (6/12)** | **3 题** / **33.3% (4/12)** | **-3 题 (-16.7 pct)** |
| 共享 vLLM 启动耗时 / 24 次尝试总批处理墙钟 | `333.41s` / `3291.30s`（单次启动共享） | 同左（双臂同服并发） | 0 |

- 配对四格表（`(public022_v10, behavior_v1)`）：
  - `(1, 1)` 双臂均通过：**3 题**（`rich_3067`, `fastapi_14851`, `fastapi_14463`）
  - `(0, 1)` `behavior_v1` 独胜：**1 题**（`rich_3180`）
  - `(1, 0)` `public022_v10` 独胜：**0 题**
  - `(0, 0)` 双臂均未通过：**8 题**（其中 5 题受 `SubprocessManager` `src/` 路径或 `dev_future` 缺失 `inline_snapshot` / 保护 `tox.ini` 限制，详见第 4 节）

## 3. 逐题配对明细（12 题 × 双臂 = 24 次尝试）

| Slot | 任务 ID | 仓库 | 配对顺序 | `public022_v10` 结果 / 补丁 / 调用 / 耗时 | `behavior_v1` 结果 / 补丁 / 调用 / 耗时 | 配对结论与关键机制观察 |
|---:|---|---|---|---|---|---|
| 1 | `rich_4006` | `Textualize/rich` | AB | `False` / `0B` / `5` calls / `484.5s` (timeout) | `False` / `709B` / `9` calls / `786.0s` (timeout) | 双臂复现 `split_graphemes('\ufe0f')` 死循环时未加短超时，单条命令卡满 `240s`；`behavior_v1` 仍改出 `rich/cells.py` 补丁修复了 `\ufe0f` 分支，但因耗时耗尽未及修 `\ufe0e` 分支 |
| 2 | `rich_3067` | `Textualize/rich` | BA | `True` / `1338B` / `28` calls / `244.5s` (5 tool errs) | `True` / `655B` / `13` calls / `224.5s` (0 tool errs) | 双臂均通过（82 passed）；`behavior_v1` 仅用 **13 次工具调用、0 错误**即完成最小干净补丁并提交，而基线用满 28 次调用且留下未清理 `repro.py` |
| 3 | `rich_3180` | `Textualize/rich` | AB | `False` / `499B` (`repro.py` only) / `16` calls / `485.7s` | **`True`** / **`2761B`** / **`9` calls** / `485.7s` | **`behavior_v1` 独胜（0→1）**：`behavior_v1` 构造 `divide_line("A你好", 3)` 正反合同断言，9 次调用定位并修复 `rich/cells.py::chop_cells` 中 `reversed(characters)` 顺序错误，114 passed；基线前 14 次调用反复读/搜，478.8s 才写 `repro.py`，未改生产代码 |
| 4 | `rich_3472` | `Textualize/rich` | BA | `False` / `0B` / `28` calls / `482.4s` | `False` / `0B` / `28` calls / `307.3s` | 双臂均在 `rich/repr.py` 与 `rich/pretty.py` 间耗尽 28 次调用未完成有效编辑 |
| 5 | `fastapi_14482` | `fastapi/fastapi` | AB | `False` / `0B` / `28` calls / `485.3s` | `False` / `1541B` / `28` calls / `338.3s` | `behavior_v1` 完成 `fastapi/encoders.py` 生产修改并 `submit_patch`（基线 `0B` 超时）；但官方评测容器收集 `tests/test_arbitrary_types.py` 时因 wheelhouse 缺少 `inline_snapshot` 报 `ModuleNotFoundError` (`rc=2`) |
| 6 | `fastapi_14851` | `fastapi/fastapi` | BA | `True` / **`0B`** / `28` calls / `226.8s` | `True` / **`526B`** / `28` calls / `489.2s` | 基线耗尽 28 次调用未改任何文件（`patch_size=0`），仅因当前环境 Starlette 尚未移除 `on_event` 而空补丁通过；`behavior_v1` 实际修改了 `fastapi/routing.py` 并通过全部 9 个测试 |
| 7 | `fastapi_14463` | `fastapi/fastapi` | AB | `True` / `745B` / `16` calls / `171.8s` | `True` / `1900B` / `28` calls / `302.8s` | 双臂均用 `copy.deepcopy(additional_response)` 修复 `fastapi/openapi/utils.py` 并通过评测 |
| 8 | `fastapi_14459` | `fastapi/fastapi` | BA | `False` / `1133B` (`repro.py` only) / `28` calls / `488.6s` | `False` / `0B` / `28` calls / `485.2s` | 双臂均未改出生产文件；且评测测试同样因缺少 `inline_snapshot` 在收集阶段报 `ModuleNotFoundError` (`rc=2`) |
| 9 | `requests_7205` | `psf/requests` | AB | `False` / `503B` / `22` calls / `266.2s` | `False` / `503B` / `14` calls / `214.8s` | **双臂均写出与官方 Gold 完全一致的源码补丁**（`src/requests/utils.py`: `if _netrc and any(_netrc):`），且 `behavior_v1` 主动发现 `requests.__file__` 指向系统目录并用 `PYTHONPATH=/workspace/src` 验证通过；但 `SubprocessManager` 在 `verify_task` 中未把 `/workspace/src` 加入 `PYTHONPATH`，导致评测导入了系统未打补丁的 `requests` 而误判失败 |
| 10 | `requests_6757` | `psf/requests` | BA | `False` / `0B` / `28` calls / `449.3s` | `False` / `0B` / `27` calls / `425.0s` | 该题要求修改 `tox.ini` 增加 `urllib3_1_26` 环境；双臂均正确修改了 `tox.ini`，但 `swegemma` 的 `_is_protected_test_or_config_path` 将 `tox.ini` 视为保护文件在 `submit_patch` 时剥离为 `0B` |
| 11 | `requests_7427` | `psf/requests` | AB | `False` / `1089B` / `18` calls / `484.3s` | `False` / `2442B` / `15` calls / `484.3s` | 双臂均修改了 `src/requests/utils.py::should_bypass_proxies` 域名边界逻辑，但因 `SubprocessManager` 默认导入系统 `requests`，复现实验与 `verify_task` 均实际运行了系统旧代码 |
| 12 | `requests_7502` | `psf/requests` | BA | `False` / `1715B` / `28` calls / `310.2s` | `False` / `1194B` / `28` calls / `532.9s` | `verify_task` 全量运行 `tests/test_requests.py` 触发 196 个 httpbin fixture 错误，且同样受 `SubprocessManager` `src/requests` 路径遮蔽影响 |

## 4. 新形成的三项核心机制证据

1. **正反行为验证指令（`behavior_v1`）带来确定性净增且零回退**：
   - 在完全相同的 31B CT v2 模型、采样参数与 8min/28-call 预算下，`behavior_v1` 将官方 `resolved` 从 `3/12` 提升到 `4/12`（非空有效补丁 `resolved` 从 `2/12` 提升到 `4/12`），在 `rich_3180` 实现 `0 -> 1` 突破，在 `rich_3067` 将工具调用从 28 次降为 13 次（0 错误），在 `fastapi_14851` 与 `fastapi_14482` 均完成了 `public022_v10` 未能完成的生产源码修改。
2. **死循环/阻塞复现耗尽预算机制（`rich_4006` 证据）**：
   - 当任务本身是修复死循环或卡死（如 `rich_4006` 的 `split_graphemes`）时，`eval_config.yaml` 的单命令超时为 `240s`（占 8 分钟总预算的一半）。模型直接执行 `python3 -c "..."` 复现 bug 会立刻挂死 `240s`，导致单次工具调用吞掉半场时间。在复现命令中显式使用短超时（如 `timeout 10 python3 -c ...`）可完全避免此类 240s 挂死。
3. **本地/Notebook `SubprocessManager` 对 `src/` 布局仓库的导入遮蔽机制（`psf/requests` 四题证据）**：
   - 核查 `swegemma/harness/container_setup.py` 与 `swegemma/sandbox/subprocess.py` 证实：官方比赛重跑使用 Docker `ContainerManager`（会执行 `pip install -e /workspace` 与 `sandbox/setup.py` 注入 `/workspace/src`），而 Notebook 离线评测使用的 `SubprocessManager` 跳过了 `install_editable_package` 且仅设置 `PYTHONPATH=/workspace`。因此对采用 `src/` 布局的 `psf/requests`，评测容器 `python3 -s -m pytest` 实际导入的是镜像自带的 `/usr/local/lib/python3.12/dist-packages/requests`，导致 `requests_7205` 双臂写出的 **100% 正确补丁**被评测容器忽略。同时，`behavior_v1` 在 `18-requests_7205-r1-behavior_v1` 的第 14–15 步自主通过 `print(requests.__file__)` 诊断出该遮蔽并用 `PYTHONPATH=/workspace/src` 完成了真实行为验证。
