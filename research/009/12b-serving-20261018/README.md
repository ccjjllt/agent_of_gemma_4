# P009：12B 短 SFT → 官方 CT 路径 → 整题配对诊断

2026-10-18。本目录准备可执行研究载体；截至交付未启动 GPU、未得到新训练或修复结果。根代理统一决定并提交 GPU 运行。

## 架构结论与 P008 边界纠正

12B 使用 `--hf-overrides '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}'` **属于官方 per-model 服务配置**，不是本项目自选的不受支持替代后端。Ryan Holbrook（Kaggle STAFF/ADMIN）2026-10-11 14:39:27 UTC 的 comment 3536939 给出此配置，并确认 “The scorer has this update already.”；2026-10-12 19:56:46 UTC 的 comment 3537606 又确认新版 wheelhouse 修复 vision_embedder 权重加载问题。[官方讨论](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748287)

日期/作者/评论原件来自 `research/007/official-data-20261017/topic-748287.json`，2026-10-18 Web 重新检索确认正文。此前 `research/006/official-data-20261013/report.md` 已记录这条官方配置，P008 复核遗漏了它。因此不能把 V6 的 architecture override 本身当成非官方兼容边界。V6 依然只证明一步真实 click SFT、独立导出和 CT 数值作用；其短上下文、低并发、HTTP 动作诊断尚未证明官方 Agent 整题收益。

提交 `agent.yaml` 声明模型和 adapter；它不携带任意 vLLM server 参数。`SandboxedLlmAgentConfig` 没有 server 配置字段且禁止额外字段。官方 `VllmConfig.extra_args` 属于服务启动层，由 scorer 的 per-model 映射应用。本载体使用该映射，不修改 CT 预训练权重。

## 具体研究条件

- 固定三题 `rich_3067`、`fastapi_14463`、`requests_7205`，每题 base/new_adapter 各一次。三题在查看本轮新模型输出之前选定；均为此前已暴露开发诊断，不是随机样本或泛化评测。只嵌入这三行既有原件，不读取全量 competition `tasks.jsonl`。
- 两臂保留 public022 V10 提示、temperature 0.2/top_p 0.95、输出 8192/thinking 4096、8 分钟 Agent/28 工具调用/80 turns/命令 240 秒；评测 300 秒。两臂仅保留训练协议六工具，删除三个图工具声明。它是新的 **12B 六工具诊断配方**，不继承来源的 0.22 分数，也不能据此将与 31B 的差异归因于模型大小。
- NF4 冻结基座、四 rank DDP 的新短 SFT 由 `research/009/training-run-20261018/training_stage.py` 执行，成功后独立导出 adapter，再起服务。载体本身不重新训练 click；不挂载旧 V6 输出。训练/数据的具体收益与可用性取决于真实运行，保持 optimizer checkpoint。
- 服务：12B CT v2、v31 固定官方四包 SHA 与完整 wheel 库存；TP4、32768 context、GPU memory 0.90、官方 gemma4 tool/reasoning parser、enable_lora、官方 12B override。显式 `--max-num-seqs 4`，最多四个独立 Evaluator 同时使用同一服务，滚动两组 task pairs；两臂在偶数题交换启动顺序。不同 sandbox/venv/Agent 进程，服务和前缀缓存共享。
- 这是提高研究利用率的并发条件；争用、共享缓存和运行顺序不等同正式逐题资源条件，不声称 scorer 所有参数精确一致。不生成全词表 prompt logprobs，避免重复 V4 的额外 OOM 来源。
- LoRA 放在 `adapters/trained_adapter/`，官方 `discover_adapters` 发现该名称；worker 注册 `trained_adapter` 和 `adapter:trained_adapter`，对应 vLLM 的同名 LoRA 路由。base 始终指向基座模型。

## 源码导入修正

两臂统一修正官方 `swegemma/sandbox/subprocess.py` 的执行环境：存在 `workspace/src` 时 `PYTHONPATH=workspace/src:workspace`，否则仍为 workspace。这是 requests7205 CPU 重放发现的实际源包导入问题；Agent 和 verification 都调用同一 SubprocessManager，因此两条路径都覆盖。载体保存原文件、修改后文件与 unified diff 到 `frozen_bundle/`。没有改动题目断言、gold patch、Agent 实现或 CT 权重。

该修正是显式研究条件，不能伪装成未改动的官方 sandbox。正式提交不包含这项运行环境修改；若后续需要讨论正式分数，必须单独报告这层差异。

## 打包、挂载与预算

`python3 research/009/12b-serving-20261018/build_carrier.py` 只打包，不执行或提交。依赖根代理最终完成的训练源码与 `research/009/training-data-20261018/encoded-train-12b.jsonl`；脚本缺失时直接指出真实缺件。生成 `kernel/experiment.py` 与 `kernel/kernel-metadata.json`。随后若训练源码/数据变化，需要重新打包。清单保存每个嵌入文件大小和 SHA；同时嵌入 data-usage.json 和 source-manifest.json 保留训练来源，不打包下载的仓库源码或原始轨迹。

挂载清单：

- dataset：`metric/gemma-4-developer-agent-wheelhouse`（代码检查 v31 库存及官方包 SHA）。
- competition：`gemma-4-developer-agent`（只取已选诊断 snapshots、wheels 等运行资产）。
- kernel：`ccjjllt/p007-12b-cpu-nf4-cache`（`qat-12b-nf4/cache-manifest.json` 及旁边三个 training wheels）。
- model：`google/gemma-4/Other/gemma-4-12b-it-qat-w4a16-ct/2`。
- image：复用 P007 CPython 3.12/Linux x86_64 镜像，SHA 在 metadata 和 manifest 中固定。

监督器默认 2520 秒（42 分钟）截止，41 分钟停止评测，43 分钟前归档，根代理设置平台上限 **2640 秒（44 分钟）**。训练 torchrun 上限 900 秒，服务启动上限 600 秒；二者都包含在总预算中，不另加时。若准备/训练接近上限，六次整题可能不能全完成，保存真实 partial 结果且不自动重试。四 L4 当前计费估算最多约 1.47 配额小时，实际以账号前后读数为准。每 10 秒保留 GPU 利用率、显存、功率和 vLLM 指标，明确标记 training 阶段；平均采样利用率不是吞吐量。

可选运行参数（均通过环境变量；默认无需设置）：

| 参数 | 用途 |
|---|---|
| `P009_ADAPTER_DIR` | 跳过本轮训练，使用已导出的新 adapter；需提供独立 config+safetensors。可显式挂载 V6 作为后备，但默认 metadata 不挂它。 |
| `P009_ADAPTER_SHA256` | 核对外部 adapter 权重；新训练默认记录实际哈希。 |
| `P009_NF4_CACHE` | 显式缓存路径；默认在挂载中寻找唯一 `qat-12b-nf4/cache-manifest.json`。 |
| `P009_TRAIN_DATA` | 根代理提供的编码训练 artifact 路径；默认用嵌入新训练数据。 |
| `P009_DATA_DIR` | 含 snapshots/wheels/graphs/embeddings 的运行数据根目录。 |
| `P009_TASKS_PATH` | 三题同一固定字节内容的 JSONL 镜像；不允许悄悄换题。 |
| `P009_TRAIN_TIMEOUT_SECONDS` | 单次 torchrun 上限，默认 900；受总监督器预算限制。 |
| `P009_WALL_SECONDS` | 更短监督器总上限；本轮授权仍为平台不超过 44 分钟。 |

输出包括训练日志/实际数据摘要/独立 adapter/optimizer checkpoint、服务命令与日志、每次 Agent trace/patch/官方评测输出、配对结果及 GPU 分阶段采样。运行结果目录和 ZIP 均保留；为避免复制大 optimizer 状态，`.pt` checkpoint 留在原输出，不再重复进总 ZIP。

## 本地完成的核对

只做一次性 CPU 源码核对：执行归档官方 discovery/path helper，临时 adapters/trained_adapter 目录被识别为名称 trained_adapter、rank 8（文件为占位字节，仅验证目录/别名发现，不是权重有效性或服务加载）；生成的 Python 均可 AST 解析；配置与训练模块通过显式文件列表打包；固定三题 JSONL 来自旧公开诊断原件；12B CT config/tokenizer 身份固定；官方 adapter 路径和别名逻辑与归档源码对照。没有在本机运行 vLLM、GPU、真实训练或六次 Agent 任务。真实部署与整题结果以根代理后续保存的运行原件为准。
