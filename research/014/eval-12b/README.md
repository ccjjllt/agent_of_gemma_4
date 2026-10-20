# P014：已完成行为 adapter 的 12B 同条件对照

**2026-10-20 实跑已完成：`ccjjllt/p014-12b-behavior-pair-v1/1` COMPLETE，8/8 attempts返回；base4/4、adapter3/4，无新增、Click回退1题，暂停原34条继续训练。** 实际模型加载与LoRA角色路由均已证实，模型就绪后已采样GPU utilization均值95.89%。见[实际结果与失败机制](result-review.md)。以下为组件和运行前准备说明，不代表仍待启动。

两臂是同一 `SequentialAgent` repair→review：`agents/base/` 与 `agents/adapter/` 的唯一文件内容差异是 `adapter/repair.yaml` 的 `adapter: p013_behavior_lora`。repair 保持训练的五工具，review 是六工具 base，显式接收原题 `{problem_description}` 和 `{repair_handoff}`。预算 8 分钟/28 工具调用/80 turns，采样 0.2/0.95，12B CT v2。

`adapter-source.json` 固定已完成的 `ccjjllt/p013-12b-behavior-sft-34/1` 独立导出（rank8、alpha16）。本地权重 115,020,728 字节，SHA256 `54de0ca38cdcb284509f7678cd59077217b169e3ee84649e78791faa8b5ae2a3`；无需重训或重新上传。metadata 的 `kernel_sources` 挂载原 SFT kernel 后，`runtime.stage_adapter()` 只接受该 kernel 的最终 `training/adapter/`，排除其 18/36/54/72 checkpoint 与 train-run/adapter，再核对 config 和权重的完整指纹。

## 与共享 P014 载体的接口

根代理提供冻结 panel 的 `agent-tasks.json`、快照、依赖及 host-only criterion；在任何本轮模型输出出现前固定槽位 **t01、t02、t06、t08**，覆盖 Click、MarkupSafe、pydicom 三仓，共 **4 个不同开发问题**、8 attempts，见 `subset-freeze.json`。不能读取 31B 结果后挑题，不能将 criterion/参考修改传进解题 Agent。

共用根代理新 31B 的 `run_agent_sandbox` worker 和独立 host judge；12B 只换以下部件：

- Agent bundle：本目录 `agents/{base,adapter}` 的文本全部嵌入唯一上传 `code_file`。
- 模型：`model-pin.json` 固定 12B CT v2 的 config/tokenizer；沿官方 `--hf-overrides`，使用 gemma4 parser。
- LoRA：先展开两臂配置，再调用 `stage_adapter()` 和 `prove_role_routing()`；vLLM `enable_lora=True`，`adapter_manifest=discover_adapters(<adapter-arm>)`。worker 用 `register_worker_aliases()` 注册基座及两种 adapter 别名（普通名称与 `adapter:` 前缀）。
- 并发：用户本轮要求提高利用率，使用 **同一 vLLM 服务，最多 4 个隔离工作树和 Agent 上下文**；`--max-num-seqs 4`。这是共享推理资源的研究条件，不能称与正式单题墙钟完全等价。首批固定 AB/BA 顺序，调度与等待时间入记录。
- 不使用旧 P012 的五角色架构、旧题 ID、旧 adapter；不在 31B 上挂 12B LoRA。

`runtime.py` 只含可嵌入的运行组件，导入不会启动模型。真实模型加载后保留服务命令、实际 adapter 路由请求/模型别名及每题补丁，避免把“配置声明了 adapter”当成“实际调用了 adapter”。

## 挂载与预算

账号 `ccjjllt`（推送前重读配额/活动会话）。挂载：官方 wheelhouse、比赛资源（4×L4）、12B CT v2、原 SFT kernel，以及根代理打包的冻结 panel 资产。复用固定 CPython 3.12 image。Kaggle 只上传一个 `code_file`，邻接 Python/JSON/YAML 不是有效自包含载体；配置、辅助源码、任务元数据必须真嵌入后运行时展开。

预算 **4 配额小时**，按历史2×折算最多2小时墙钟，包含安装、模型加载、Agent、裁判及归档。建议整会话硬限 6900 秒，平台 7200 秒，deadline 前为归档预留120秒；剩余时间不足启动完整配对时停止新增配对，报告实际已完成数。不得自动重推或从其他账号补预算。

运行前本地完成了最终权重与 config 哈希核对、两臂唯一差异、review 原题注入及路由与工具协议的一次性手动核对；之后已完成真实CT加载与8次attempt，结果见上方报告。

已生成并真实推送 `kernel/run.py`（约50KB，唯一上传code_file）和 `kernel/kernel-metadata.json`，实际挂载同账号 `ccjjllt/p014-frozen-eight-material-cpu` 与既有SFT kernel。每个worker另存实际 `model-requests.jsonl`，记录真正发生的基座/LoRA请求别名，以区分配置声明与实际路由。
