# P005 下一轮：得到比赛环境中确实增加修复数的 adapter

日期：2026-10-10。执行计划，对应交接文档第 8 节「下一轮研究顺序」。
源规格：`docs/plan/005-implementation-handoff.md`。
原执行分支：`p005-next-round-sft`。本轮已完成项、未执行项和后续安排见文末「合并审查后的下一轮工作」。

目标：一个在官方 CT 基座 + 官方 v30 harness 上实际生效的 rank8 LoRA，并在 8–12 个未进入训练集的已暴露开发任务上相对 T0 base 比较修复数。

## Global Constraints

- 不覆盖 P002 冻结变体、历史结果、heldout 内容。
- 不正式提交比赛。
- 本轮 GPU 额度封顶 6 小时。启动前重读 `kaggle__get_accelerator_quota`。本轮开始时已用 7011s；4×L4 按两倍扣费。超过上限停止后续 GPU，不要再推。
- CPU 负责数据和依赖准备。不要把数据清洗放到 GPU。
- 不要原样重跑已证明的四步 31B 诊断训练。
- 不要在 Python 3.13 上强装 v30 的 cp312 wheel。3.13 时在安装前失败。
- 诊断 adapter（`ccjjllt/p005-training-ddp-probe` step4）只用于兼容性，不当修复收益，不提交。
- 不截断历史观察来伪造短轨迹。超长交易整段拒绝。
- 训练过的 issue 不能再当独立收益证据。配对集与训练集 issue 必须不相交。
- 推理对照固定 T0 提示、六工具、预算；不要同时换 prompt 和 adapter。
- MCP `save_notebook` 挂不上数据源。Kernel 用 CLI `kaggle kernels push`。
- 推理镜像优先钉官方 Starter 仍在使用的
  `gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`
  （2026-10-10 仍是 Starter v2 / 003-A0 的 docker_image，历史 Python 3.12）。
  当前默认 `gcr.io/kaggle-images/python@sha256:81b1e7c4b4f0f2a7e0a33943c95754b1b7af6b30d65221633a284b6b601cd84a` 是 3.13，不能用于 v30 推理安装。
- 训练路径可继续用 Python 3.13 + 已有 7-wheel/NF4 缓存，与推理 ABI 分开。
- 协议：`p005-harness-tools-v1`，官方六工具。
- 实现者自己提交 git commit；不要改 `main`。

## Task 1: CPU 钉住 v30 推理环境并解决 3.13/cp312 ABI

在 **CPU、零 GPU 额度** 上证明可以安装官方 v30 推理 wheelhouse，并记录可复用的 docker 钉。

### 要做的

1. 新建隔离目录 `research/005/inference-env-20261010/`（生成器 + kernel 源 + 证据）。不要改 P002 冻结 notebook。
2. 写一个短 CPU kernel：打印 `sys.version`、`sys.executable`、平台、`docker` 无法在容器内自证时至少记录 metadata 里声明的 `docker_image`。
3. `kernel-metadata.json` 必须：
   - `enable_gpu: false`，`enable_tpu: false`，`enable_internet: false`，`is_private: true`
   - `docker_image` 钉官方 Starter 的 3.12 sha（见 Global Constraints）
   - 挂载 `metric/gemma-4-developer-agent-wheelhouse` 与比赛数据（用于核对 v30 文件清单）
   - 不挂 CT 权重（本任务不加载模型）
4. 若 `sys.version_info[:2] != (3, 12)`：写出 `abi_status=pin_failed`，**不要 pip install 任何 cp312 wheel**，正常退出 0 并保存报告。这是有效负结果。
5. 若是 3.12：按 v30 载体同样方式 staging（跳过 cutlass、`cu128`→`+cu128`），安装全部可安装 wheel，记录每个 wheel 的 name/size/sha256 与 `importlib.metadata` 版本。至少成功 import `swegemma`、`adk_submission`、`google.adk`。CPU 上 `vllm`/`torch.cuda` 失败要记为环境限制，不假装 GPU 可用。
6. 用 CLI `kaggle kernels push -p <dir>` **不要** `--accelerator`。等待 COMPLETE。下载输出 JSON。把实际 python 版本、是否安装、失败 wheel 写入 `research/005/inference-env-20261010/result.json`。
7. 本地加一个不联网的小检查：metadata JSON 合法、docker_image 等于指定 sha、enable_gpu false。

### 完成标准

- 有一次真实 CPU kernel 运行证据（kernel slug、version、python 版本）。
- 明确二选一：`abi_status=ok_python312_v30_installed` 或 `abi_status=pin_failed`（附实际 python）。
- 后续 GPU 推理任务必须复用这里验证过的 `docker_image`；若 pin 失败，后续 GPU 推理不得安装 v30，并在报告里升级为阻塞。

### 不要做

- 不启动 GPU、不加载 CT、不跑 vLLM、不训练。
- 不修改 `notebooks/005-single-state-single-cell.py` 的 3.12 门闩。

## Task 2: 真实样本完整模板长度普查并决定训练窗口

CPU 整理真实局部修复/错误恢复样本，统计完整 chat 模板后的长度，按仓库划分 train/select。不截断。

### 要做的

1. 用现有 `training/data/prepare_local.py --include-failures`（若 CLI 无该开关则加一个默认关闭的开关，不要改变现默认 resolved-only）。只读 `results/` 与 `research/task_splits.json`、资格表。排除 heldout。不读 `patch`/`test_patch`/`problem_statement` 答案字段。
2. 保留官方六工具语义。坏动作不当正向监督。错误恢复交易必须包含导致恢复的观察。
3. 每 issue 最多 3 条；按 issue 与 repo+commit 分组；`--whole-repo`。训练 issue 集合写入 `research/005/length-census-20261010/train_issues.json`。
4. 用官方部署 tokenizer 做完整模板编码（`training/encoding/gemma_tool_chat.py`，`--on-oversize skip`）。本地没有 tokenizer 时：Kaggle **CPU** 下载/使用已缓存 tokenizer，或挂已有 CPU stage；不要为 tokenizer 开 GPU。核对 tokenizer.json sha `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`（32169626 字节）若使用 CT v2 tokenizer。
5. 对每个交易记录 raw token 长度（拒绝前）。输出直方图与分位数：2048 / 4096 / 8192 / 16384 各能完整编码多少条、多少独立 issue。写入 `research/005/length-census-20261010/length_report.json`。
6. 决策规则（写入同一报告的 `chosen_max_seq_len`）：
   - 若代表性训练交易多数能在 4096 完整编码，选 4096。
   - 若多数超过 4096 但 8192 可覆盖主要集合，选 8192，并标明 4K 容量未在本轮训练中证明。
   - 若主要样本超过所选窗口：记录缺口，**不要截断**，`trainable_ok=false`。
   - 2048 只作为对照统计，不能因为诊断合成 2K 成功就定为正式窗口（T0+六工具+官方说明在极短占位问题上已约 1435 tokens）。
7. 选出实际可训 JSONL（完整编码、无 JSON 截断）到 `training/artifacts/repair-sft-v1/`（train/select 分开）。样本数为 0 时仍提交长度报告，`trainable_ok=false`。
8. 本地 unittest：至少覆盖 include_failures 不会把失败动作改成 span_apply、heldout 仍排除、超长 skip 不截断。

### 完成标准

- `length_report.json` 含分位数、各窗口条数、chosen_max_seq_len、trainable_ok、train issue 列表。
- 可训数据与 24 题开发面板的关系写清楚：哪些 issue 进了 train，因此不能再当独立收益题。

## Task 3: 官方 CT 上加载诊断 adapter 并完成一次工具调用

仅当 Task 1 `abi_status=ok_python312_v30_installed`。否则本任务记录 BLOCKED，不推 GPU。

用已有诊断 adapter 验证官方 CT 基座加载、路由、工具调用。兼容性实验，不是修复收益。

### 要做的

1. 启动前重读 GPU 配额。本任务 wall timeout ≤ 1500s，accelerator NvidiaL4。若本轮已用额度 + 本任务预估 2×1500s 会超过 6h 封顶，停止。
2. 复用 Task 1 验证过的 `docker_image`。挂载：比赛数据、v30 wheelhouse、CT 模型 `google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2`、kernel 源 `ccjjllt/p005-training-ddp-probe`。
3. 将诊断 adapter 放进提交布局 `adapters/<name>/`（`adapter_config.json` + `adapter_model.safetensors`），`agent.yaml` 设 `adapter: <name>`。T0 六工具与 T0 sampling/eval 预算。不要用 single_state 技能，避免和训练协议混淆。
4. 使用官方 `discover_adapters` + `VllmServer(..., adapter_manifest=adapters)`，`enable_lora=True`。启动后做最小探针：
   - 记录 adapter 是否出现在 manifest 与 vLLM LoRA 列表/日志
   - 一次真实工具调用（`get_status` 或一次短 `read_file`），证明生成走了工具协议
   - 分类：`adapter_not_loaded` / `inference_error` / `tool_call_ok_no_repair_claim`
5. 不要跑完整题目评分当本任务成功标准。最多 1 个已暴露 smoke 题且仅当工具探针需要；该题结果不当修复证据。
6. 证据写入 `research/005/ct-adapter-probe-20261010/`：run_manifest、分类、adapter 文件指纹、配额前后快照。诊断权重不要下载回 git。

### 完成标准

- 有 GPU 运行证据和四分类之一。
- 明确写出：这不是修复 SFT 成果。

## Task 4: 一个小规模真实 SFT 候选

仅当 Task 2 `trainable_ok=true` 且有至少若干完整编码训练条。否则 BLOCKED，不推 GPU。

### 要做的

1. 启动前重读配额。沿用已有 NF4 缓存 `ccjjllt/p005-cpu-nf4-cache` 与 CPU 7-wheel `ccjjllt/p005-training-cpu-stage`。不要重新量化 62.5GB 基座，不要装 41 个推理 wheel。
2. 配方：rank8/alpha16/LR 5e-5、`kbit_prepare: norms_fp32`、`loss_backend: selected_chunked`、`sft_qat_nf4_ddp.yaml` 为底。`max_sequence_length` 必须等于 Task 2 的 `chosen_max_seq_len`。`epochs: 1`，`max_steps` ≤ 100，实际按样本量更小。
3. 按长度分桶，减少 DDP 等待。一次加载后连续训练。
4. 数据是 Task 2 的真实编码 JSONL，禁止诊断合成重复文本。
5. 四卡 `torchrun`。导出独立 PEFT 到 Kaggle working，记录 training_manifest（QAT 来源、NF4、模块、rank、协议、数据指纹、adapter sha）。
6. 本地：配置 dry-run + 现有 unittest；不要在本地加载 31B。
7. GPU timeout 按步数保守设置，且本轮累计不得超过 6h 额度。失败保留最近完整 adapter。

### 完成标准

- 至少 1 个真实优化步（期望远多于 1，但受样本限制）。
- 云端 adapter 文件清单与指纹。
- 明确：loss 下降 ≠ 修复增益。

## Task 5: 8–12 题 T0 base vs adapter 配对

仅当 Task 3 证明 CT 能加载 adapter，且 Task 4 产出真实数据 adapter。否则跳过配对并写明原因。

### 要做的

1. 启动前重读配额。多数剩余额度给本任务。4×L4，一次加载，同一服务上先 base 后 adapter 或交错同一批题；两臂共享 T0 提示、六工具、预算（5min/50/80/60s）、`max_output_tokens 16384`。
2. 选 8–12 个已暴露开发任务，**issue 不在 Task 2 train_issues 中**。优先 `research/task_splits.json` 的 `dev_panel_ids` 减去训练集。不足 8 个则用已暴露非 heldout 的其余题，并记录为什么小于 8。
3. Adapter 用 Task 4 导出，布局与 Task 3 相同。Base 臂 `n_adapters=0`。
4. 记录每题：task_id、resolved、exit、duration、patch 非空、tool_calls、adapter 指纹、失败类别。汇总新增修复、回退、耗时。
5. 载体从现有 joint 载体派生，另存 `notebooks/005-base-adapter-pair-single-cell.py`，不覆盖 P002/P003 冻结文件。
6. 结果表 `research/005/base-adapter-pair-20261010/results.md` + JSON。无增益也要保留负结果。
7. 不提交比赛 zip。

### 完成标准

- 配对表完整，train/eval issue 不相交。
- 结论只能来自 resolved 差异，不用 NLL 或更早编辑代替。

## 合并审查后的下一轮工作（2026-10-10）

本节接续上述五项任务；上文保留为本轮原始执行计划。Task 1、3 已完成环境和协议兼容验证；Task 2 完成长度普查；Task 4 未训练，Task 5 未评分。本次合并整理只运行本地 CPU 检查，不新增 Kaggle GPU 作业。审查修复还使 v30 成功状态要求库存和官方 wheel 哈希匹配，并严格核验探针的 `get_status` 参数。86 项本地检查通过（含真实 tokenizer）；自定义输出目录的 CPU 普查复现原报告。修复后的探针代码尚未重新推送；历史 v2 原始结果不改写，当次源代码可在提交 `673d357` 回溯。

### 已有证据与结论边界

- 官方 Starter 镜像实际为 Python 3.12.13，v30 安装成功，CPU 运行 171.155 秒。见 `research/005/inference-env-20261010/result.json`。
- CT 探针 v2 在官方 CT v2 上加载诊断 adapter，`/v1/models` 包含 adapter 名，以该名请求返回 `get_status` 工具调用，脚本 562.139 秒。它证明加载、路由与工具格式兼容；未执行真实修复评测，也未测量开关 LoRA 后的数值差异，不能据此宣称修复收益或单独证明 LoRA 数值作用。见 `research/005/ct-adapter-probe-20261010/`。
- 当前 train 为 15 条、5 个 issue：4K 覆盖 0 条，8K 覆盖 3 条，16K 覆盖 12 条，完整模板长度中位数 14,250。16K 尚未训练实测；`trainable_ok=false` 记录有效。
- 配额最近快照为 2026-10-10 09:47 UTC：已用 8,442.630 秒（约 2.345h）、剩约 27.655h，无预留。CT 两次运行较本轮起点净增 1,431.620 秒（约 0.398h）。这是历史账号快照，启动任何后续 GPU 作业前重新读取。

### 第一优先：CPU 修正数据抽样，再决定训练窗口

合并审查发现了既有数据管线的研究偏差：39 条 trace 产生 214 个交易，标签为 194 个 `local_edit`、20 个 `tool_recovery`；每 issue 最多 3 条采用先到先得，9/9 个 issue 均保留第一条 trace 的 tx01–tx03。最终 27 条全是 `local_edit`，20 个恢复候选全部被丢弃。这批长度统计是当前样本的真实测量，但不能代表恢复训练分布。

`local_edit` 是 `prepare_local.py` 的默认标签，不能直接等同于发生编辑或成功修复。`transactions.py` 按同工具名连接 error→ok，旧失败还可能跨交易重复计数；20 个候选中的 23 条关联仅有 10 个不同的 trace＋failed_call，仍需核对是否针对同一操作。

下一项实现限定在 `training/data/{prepare_local,transactions,split_registry}.py` 和普查入口：

1. 按实际工具动作标记编辑、错误恢复、验证收尾；失败调用只作上下文。恢复需关联同一文件/操作和失败观察，后续成功应结束该失败事件，避免重复监督同一恢复。
2. 每 issue 仍最多 3 条，先按恢复事件去重，再跨 trace、按真实行为选样，使用固定种子消除文件顺序偏差；有证据时覆盖编辑、恢复、验证，缺失类别直接报告，不能捏造。
3. 保留完整因果前缀、工具 JSON、来源 trace、调用 ID、失败/成功观察和最终 resolved 状态；工具调用成功与最终任务修复成功分别记录。按 issue 与 repo＋commit 去重，保持 train/select 分离。
4. 在新目录 `training/artifacts/repair-sft-v2/` 和独立普查报告中重算 4K/8K/16K 覆盖、独立 issue/事件数、监督 token 数及密度。旧 v1 数据和报告保留。随后扩充独立 issue 与仓库；当前五个训练 issue 只支持方法探索，不能说明广泛修复能力。

交付物是可复算的选样清单、真实行为分布和完整编码数据；无需新增门控平台。先完成这一项，再根据新分布选择 GPU 实验，避免为了有偏的 first3 样本直接投入 16K 改造。

### 第二优先：一次针对真实长度的容量实验，然后一个 SFT 候选

若重选样后主要数据仍超过已证明的 2K，在同一次模型加载中检查能代表新数据的完整长度，记录每卡峰值、有效监督 tokens/s 和步耗时。使用同源 NF4 与 7-wheel 缓存；输入用完整交易，避免再次只测稀疏标签的合成 2K。CPU 文本能编码不代表 GPU 能反传，8K/16K都需以实测为准。

容量实验使用独立输出，不把它的临时更新产物当正式候选。DDP 容量不足时，先根据峰值定位激活或权重占用，再决定更省内存的上下文策略或 FSDP-QLoRA 是否值得做；不自动连续重试。新的上下文构造若改变部署协议，应另设研究对照，不能只删训练历史使其变短。

容量与数据可用后，仅做一个 rank8/alpha16、LR 5e-5、一个 epoch 以内的真实 SFT 候选，继续使用 `norms_fp32`、`selected_chunked`、长度分桶。更新步数由独立样本数量决定，100 步只是上限，不是必须重复小数据达到的目标。导出 PEFT、数据指纹和监督 token 统计。

### 第三优先：固定 T0 的小样本配对，决定是否扩大投入

首轮建议 8 题，固定 T0 提示、官方六工具、模型及预算，base/adapter 共用一次 vLLM 加载。当前可预留的 8 题为：`fastapi_13713`、`requests_6592`、`requests_6644`、`requests_7309`、`requests_7505`、`rich_3130`、`rich_3296`、`rich_3469`。这些题不在当前 train 或 select 中；v2 扩充数据前重新排除它们及相同 repo＋commit 的训练样本。原 Task 5 的 12 题仅是未执行提案，其中 4 个 fastapi 题已作为 select，不把它们当未参与选择的确认集。

记录新增修复、回退、每题耗时和有效工具调用。八题结果只决定是否值得扩大研究，不能外推金牌或奖金。若出现稳定净收益，再扩大已暴露开发对照；尚无真实 SFT adapter 时不跑空的收益配对。`single_state`/交接机制继续保留为另一条研究线，避免与首个 adapter 同时更换。

### 配额安排与交接材料

继续沿用本轮 6 个额度小时上限。后续建议上限：容量研究 0.6h、首个真实 SFT 1.0h、八题配对 3.5h，合计 5.1h；加已花约 0.398h，仍留约 0.502h 余量。4×L4 按两倍 wall time 计费，预算不是运行时长保证。CPU 数据准备优先完成，GPU 时间主要用于有效训练和真实配对。

用户分支的 37 项非缓存忽略文件已保存到 `training/artifacts/worktree-archives/p005-next-round-sft-673d357-20261010/ignored-files.tar.gz`，对应 `manifest.json` 记录逐文件 SHA256 和符号链接。14 个训练数据文件和 3 个原始日志已恢复到主工作区原相对路径；`results` 符号链接未跟随打包，主目录实际结果未改动。这份本地归档不进 Git，删除旧 worktree 后仍可恢复过程资料。
