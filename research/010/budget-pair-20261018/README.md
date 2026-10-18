# P010：联合收紧单次思考与总输出预算

准备并执行于2026-10-18。UTC 08:41已通过CLI成功push V1，账号`ccjjllt`、4×L4、平台上限1800秒/30分钟，随后回读为`RUNNING`。尚未取得本轮结果；它是GPU研究作业，不是正式competition submission。P009原目录保持冻结。见[job-request.json](job-request.json)。

P009 已证明 12B 真实短 SFT 的导出 adapter 能在官方 CT 服务路径上执行完整 Agent 任务，但三题中两臂均 1/3。长响应出现第一段思考到预算后，继续在正文或新思考通道循环的情况。本轮同时把 `thinking_budget` 从4096降到1024、`max_output_tokens` 从8192降到4096。这是**两个相配预算参数的联合处理**，不是单变量实验；与P009跨轮比较仍受生成随机性、并发、缓存和重复使用开发题影响。

## 固定条件

- 三个此前已暴露开发诊断：`rich_3067`、`fastapi_14463`、`requests_7205`；base/new_adapter 各一次，共六次。题目文件逐字节复用P009，只读取这三行。
- 双方同一 P009 提示、六工具、temperature0.2/top_p0.95、include_thoughts=true；任务8分钟/28次工具调用/80 turns/命令240秒，verification300秒。只有两个采样预算值变化。
- 一次12B CT v2服务启动：v31固定 wheel 库存与官方包SHA、TP4、32768 context、maxseq4、GPU memory0.90、gemma4解析器、官方12B `Gemma4ForCausalLM` override。最多四个隔离Evaluator共享服务，保留交替的配对启动顺序。
- 双臂继续使用P009的 `workspace/src:workspace` 导入修正（仅src目录存在时启用），同时覆盖Agent和verification。运行时保留官方源码前后原件与diff。这是显式研究环境条件，不代表正式scorer已应用该修正。
- Base使用CT模型路径；adapter声明 `trained_adapter`，通过官方 `adapter:trained_adapter` 别名路由。继续导出两个独立agent ZIP作为研究产物，不自动正式submission。

## 唯一 adapter 来源

只挂载 `ccjjllt/p009-12b-sft-agent-pair` **V1**，选取 run `run-009-12b-sft-pair-20261018T075746Z-5ad7782d` 下、目录末尾严格为 `training/adapter` 的独立导出。忽略 `agent_exec/.../adapters/trained_adapter`、`train-run/adapter` 和 checkpoint 中的副本。

- `adapter_model.safetensors` SHA256：`c7845c23852dbc701a267eaa58be87464af90e25c62bfffb0286ac1033945ba2`。
- `adapter_config.json` SHA256：`c7b7e91f822b4fb546199dfbe42fad553285d38d84e79c7474a55d7a4895a3ca`。
- 只复制这两个文件，保存实际身份记录。挂载选择使用kernel slug，代码再以精确run ID和上述SHA锁定V1，不会悄悄接受未来版本产生的另一adapter。

无NF4缓存、无训练数据、无训练代码或训练阶段，也不下载或修改预训练权重。

## 运行与预算

运行 `python3 research/010/budget-pair-20261018/build_carrier.py` 仅生成载体，不执行它。生成目录 `kernel/` 内为 `experiment.py` 与 `kernel-metadata.json`；根代理负责Kaggle CLI push和启动，kernel ID预设 `ccjjllt/p010-12b-budget-agent-pair`。

挂载：官方 wheelhouse dataset、competition运行数据、上述P009 kernel输出、`google/gemma-4/Other/gemma-4-12b-it-qat-w4a16-ct/2`。沿用P009成功的CPython3.12/Linux镜像SHA。

预算明确包含安装、加载和评测：平台 **1800秒/30分钟**；监督器1680秒；评测1620秒截止；归档1740秒截止。平台上限由根代理CLI设置，metadata不冒称能设置平台时长。超时保留真实partial、日志、trace和GPU记录，不重试、不延时。输出使用独立 `run-010-budget-pair-*` 和 `/kaggle/working/results/010/`；不写P009输出。

`P010_WALL_SECONDS` 仅能缩短1680秒监督器上限。保留可选 `P010_DATA_DIR` 和 `P010_TASKS_PATH` 同源镜像定位；后者必须仍为固定三题相同字节。adapter不提供任意路径替换，以保持本轮处理固定。

## 完成的必要CPU核对

一次性手动核对：Python源码和生成bootstrap可解析；嵌入ZIP SHA与文件清单一致；两臂相较P009只有sampling.yaml两个数值变化；prompt/agent/eval_config、worker、模型pin和三题字节保持一致。核对旧checkpoint/agent镜像不会满足adapter选择条件。没有新建测试或check/verify/gate脚本，没有运行GPU。

成功与否以之后真实run的六次结果、动作轨迹和耗时为准。预算截断能限制单次成本，但不保证模型选择正确编辑；总输出过小也可能截断工具参数。
