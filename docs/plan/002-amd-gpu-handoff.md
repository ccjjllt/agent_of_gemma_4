# P002/P004：AMD GPU 可行性与实施交接

状态更新：2026-10-01，用户决定等待Kaggle的4×L4额度恢复，当前不再开启AMD实验或实施迁移。TPU路线继续停止。本文保留可行性取证与未来可选交接，不是当前执行计划；重新启用需用户另行决定。

历史取证范围：仅远程只读检查、源码读取及扩展导入，没有下载模型、安装软件、修改远端项目代码或启动模型推理/训练。大量实现由用户完成。

## 1. 决策与优先级

以下是此前的技术可行性建议，已被上面的用户资源决策暂缓。当前P002主线为CPU前置核验→L4最小服务探针→冻结的12次T0/T1探索→有候选才做24题完整配对，不以AMD迁移为前置条件，不重跑已完成的0C。

**AMD可以成为后续开发实验平台。优先验证比赛指定QAT模型的推理，再做AMD环境内的T0/T1对照；LoRA/QLoRA训练另做兼容性门槛，不立即启动31B训练。**

理由已从“上游可能支持”推进到“实机有对应量化源码、原生算子和Gemma4解析器”。但完整模型、32K上下文和Agent轨迹尚未实跑，不写成已验证成功。

TPU资料保留为历史，停止申请和推进。Kaggle正式提交的HTTP 400是另一问题，切换硬件不能解释或解决它；保留无submission id/无分的记录，不自动重试。

## 2. 已核实的机器条件

通过amd-gpu MCP的list_profiles、inspect_remote、list_jobs与短时run_command取得：

| 项目 | 实测 |
|---|---|
| 可用profile | amd_a；GPU ID 0；amd_b未配置启用 |
| GPU | 单卡AMD Radeon Graphics，架构gfx1100，具体商业型号未确认 |
| 显存 | 51,522,830,336 bytes，约48 GiB |
| 解释器 | /opt/venv/bin/python；Python 3.12.3 |
| 推理栈 | ROCm 7.2.4；torch 2.10.0+rocm7.2.4；vLLM 0.27.0+rocm724 |
| 模型/适配器库 | Transformers 5.15.0、compressed-tensors 0.17.0、PEFT 0.20.0、Accelerate 1.14.0 |
| 未安装 | swegemma、adk-submission、google-adk、bitsandbytes、TRL |
| 容器资源 | memory.max约55 GiB；cpu.max折算16核，不能把宿主报告128核当可用配额 |
| 工作盘 | 约93 GiB空闲；未发现本项目模型或数据 |
| Docker | 当前未发现docker可执行文件 |
| 基础GPU能力 | torch.cuda.is_available=True、BF16支持；原生RDNA3算子可导入并注册 |

证据：[host-evidence.json](../../research/002/amd-review-20261001/host-evidence.json)、[source-evidence.md](../../research/002/amd-review-20261001/source-evidence.md)。

ROCm正常复用PyTorch的torch.cuda接口，因此旧载体的is_available检查本身不必删除，也不能因torch.version.cuda为None误判没有GPU。识别AMD应同时记录torch.version.hip及gfx架构。[PyTorch HIP语义](https://docs.pytorch.org/docs/stable/notes/hip.html)

## 3. 指定模型的推理支持证据

### 实机已证

- vLLM的RDNA3W4A16LinearKernel存在，ROCm候选优先列表包含它。
- vllm._C与vllm._rocm_C均导入成功；torch.ops._rocm_C.gptq_gemm_rdna3已注册。
- 该kernel要求gfx1100，接受FP16/BF16、对称INT4对应的uint4b8、正group size与维度整除等条件。
- Gemma4模型实现文件存在；gemma4工具与reasoning解析器均已注册。
- CT WNA16桥接会依据对称性和bits选择量化类型，再走平台kernel。

Google公开配置是对称INT4、group_size=32、pack-quantized、BF16，与上述路径相符；仍须核对比赛挂载v2权重/config/tokenizer/模板指纹，不能仅凭同名模型视为同一版本。

来源：[RDNA3上游实现](https://github.com/vllm-project/vllm/blob/v0.27.0/vllm/model_executor/kernels/linear/mixed_precision/rdna3_w4a16.py)、[原始PR](https://github.com/vllm-project/vllm/pull/41394)、[模型配置](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct/raw/main/config.json)。

### 仍需实跑

完整checkpoint加载、每层kernel选择、数值有限性、32K KV cache余量、真实工具往返、reasoning留存及五分钟内的有效动作数。算子已注册不等于算子已在本模型上运行通过；源码分派条件匹配也不等于所有层均可加载。

当前vLLM是源码树安装，版本字符串为0.27，但Git身份读取返回128，commit/dirty状态未知；已保存关键源码SHA-256。以后冻结完整依赖时不能宣称它必然等于无修改的官方tag，不修改该共享源码树。

48 GiB足以支持“值得尝试量化推理”的判断，不保证32K一定装得下。权重之外还有未量化层、KV cache、临时workspace和图捕获等消耗。若32K失败，不得悄悄缩到8K后作为原P002实验；短上下文只属于另一个诊断条件。

## 4. 迁移时保留什么、修改什么

### 保留

- 比赛固定模型v2及身份核对。
- T0/T1的冻结题目、提示、采样、16384最大输出、5分钟/50工具/80轮预算。
- 每次新沙箱、新会话、独立attempt、AB/BA顺序、完整原始结果与异常。
- 官方工具语义和验证入口，heldout继续冻结。
- 同一AMD环境内同时重跑T0与T1。

### 必须另建AMD载体处理

1. **安装**：禁止照搬002-joint-explore第174–199行的整包Kaggle CUDA wheelhouse强装，避免覆盖现成ROCm/PyTorch/vLLM。隔离推理服务与Harness客户端依赖，记录所有解析后的版本；先做依赖冲突检查。
   隔离后必须明确服务解释器：官方VllmServer.build_cmd使用sys.executable启动vLLM，若直接从Harness venv启动会选错环境。应明确以/opt/venv/bin/python启动现成ROCm服务、由客户端连接已验证endpoint，或另行证明包装器选用了正确解释器。
2. **路径**：把/kaggle/input、/kaggle/working改为明确配置的本项目路径；只搬运需要的已暴露任务资源，不整体索引heldout内容。特别检查T1脚本内部的/workspace：subprocess只重写shell命令，未证明会重写Python源码内的绝对路径；AMD宿主本身也有/workspace，必须验证技能访问的是当前题临时沙箱而非宿主项目。
3. **GPU记录**：保留torch.cuda接口，增加HIP/gfx/BF16信息；单卡TP=1，不把4×L4配置直接照抄。
4. **服务**：确认官方VllmConfig包装与0.27 CLI参数相容，使用gemma4双parser、实际生成字段、32768上限。存在同名parser并不证明输出行为与0.19.1完全相同。
5. **裁判**：无Docker时使用官方subprocess入口，但新宿主可能改变依赖/fixture；对拟跑任务重做空补丁/参考补丁资格检查。Kaggle的9/24不能直接当AMD环境资格。
6. **证据**：新增amd环境manifest，不改原GPU冻结文件；记录实际服务请求、selected kernel、显存峰值、prompt/completion、finish_reason、阶段耗时与原始错误。

建议新增路径（均为待实现）：

| 路径 | 职责 |
|---|---|
| research/002/amd/experiment_manifest.json | AMD环境身份、题目/变体/预算和对照冻结 |
| research/002/amd/verification_eligibility.json | 新宿主资格与全量分母 |
| scripts/amd/inspect_contract.py | 实际参数、工具协议、上下文与kernel证据 |
| scripts/amd/run_explore.py | 复用官方Evaluator的独立AMD载体 |
| results/002/amd/<run_id>/ | 所有attempt与运行记录 |

这里不建立可提交的自定义主机入口。AMD脚本是研究载体，比赛提交仍是Agent YAML/提示/技能/adapter。

## 5. 推理实验的最小验收顺序

1. 保存机器/软件/源码指纹，确认GPU未被其他managed job占用；准备同一模型与6题资源。
   在GPU模型启动之前，先做CPU真实技能工具链验收：在宿主与本题沙箱放不同合成符号，通过run_skill_script确认只返回本题符号，并确认编辑/检查不会越出该题工作区。当前源码表明存在路径映射风险，尚未实证发生；若失败则阻断批量探索。若需修改技能本身，另记T1新版本与哈希后重新冻结，不静默改动原清单。
2. 短请求加载指定QAT模型，记录真实kernel、无CPU回退/全模型解压、数值有限、服务响应。
3. 合成一次完整tool_call→tool_response→后续调用→提交形态，检查引号/换行/Unicode与reasoning字段。用合成标记检查实际渲染/编码后的下一次模型输入；字段存在、parser已注册或模型自述记得都不等于留存。无法取得渲染证据时标“转发已证、最终渲染未知”。
4. 验证逐级长输入以及32K上限，输出预留与输入相加不得越界；记录finish_reason和峰值显存。
5. 用已暴露且本环境eligible的单题验证完整轨迹和产物，T1要证实实际调用了技能工具且读写的是当前题工作区。
6. 完成新的AMD环境冻结后，运行6题×2组=12次探索；保持原题目，资格异常另列，不按成绩换题。
7. 至多选一个候选做24题完整配对。AMD内收益用于开发筛选，候选回4×L4确认后再判断正式效率与部署。

失败停止条件：指定权重/关键层无法加载；kernel不匹配；32K不可用；工具调用协议错误；关键参数没有生效；裁判失真未分开。这些情况下保留原件，不用换另一个模型或格式静默“修好”本实验。

AMD比4×L4快或慢都可能改变5分钟内的步骤数量。主指标报告平台内配对，同时单列吞吐；不能把GPU平台变化造成的成功率变化归给T1。固定调用数/短序列可另作机制诊断，但不能替代原五分钟主指标。

## 6. 训练的可行与不可承诺范围

| 方案 | 目前判断 |
|---|---|
| 31B全参数训练 | 不适合这台单卡；权重之外还需梯度、优化器、激活 |
| 31B BF16 LoRA，整模型驻留GPU | 按31B估算仅BF16权重约62 GB/57.7 GiB，已超过48 GiB |
| BF16 LoRA+CPU卸载 | 不是默认可行方案；55 GiB主存也不能保证整模型驻留，分片/流式/速度需另研究 |
| 通用4-bit QLoRA | 有上游ROCm/gfx1100支持路径，值得后续小验；当前bnb/TRL未安装 |
| 直接对当前CT-QAT用PEFT并保持4-bit训练 | 当前加载链不支持这种直接推论，见下文 |

### 本机源码确认的训练陷阱

实际Transformers 5.15的quantizer_compressed_tensors.py：

- 使用apply_quantization_config(..., run_compressed=False)。
- dequantize=True在加载后调用decompress_model。
- dequantize=False也会通过hook在首次forward解压。
- is_trainable只返回not use_fp8_kernel，不代表低显存4-bit反向路径已经实现。

因此“CT文件只有4-bit”或“is_trainable=True”都不能作为这台卡上31B低显存训练可行的证明；不要直接整模型加载后开训。

当前bitsandbytes官方文档列出ROCm7.2.4/gfx1100支持，通用QLoRA有组件基础。但NF4、BF16原版、QAT未量化权重、CT解压后重新量化均不是比赛CT运行检查点的同一量化条件，必须记录训练基座变化并回到指定基座验证收益。[bitsandbytes支持](https://huggingface.co/docs/bitsandbytes/v0.50.2/installation)、[PEFT量化训练](https://huggingface.co/docs/peft/developer_guides/quantization)。

### P004进入前的小验

在训练目标和数据划分明确后，先选一条记录清楚的基座/量化路线，用batch=1、序列128–256、rank=8、1–3个优化步验证：loss有限、adapter有非零梯度、只有adapter更新、GPU/主存峰值受控。随后保存并重载独立PEFT adapter，逐步增加序列长度。

标准PEFT adapter可以跨AMD/NVIDIA加载，但必须核对基座、模块名、形状、tokenizer/模板、量化误差与官方vLLM兼容性；最后在比赛指定QAT模型与官方服务上验证短工具调用和所需长上下文。训练在AMD不构成天然障碍，训练loss下降也不代表解决率提高。[PEFT格式](https://huggingface.co/docs/peft/developer_guides/checkpoint)

本轮仍优先P002推理/导航实验，不因为有新GPU就跳过P004六条件。不要同时在约93 GiB工作盘堆积多份BF16模型、重复缓存和全量快照；先制定下载体积和保留清单。

## 7. AMD MCP执行约定

- 唯一物理实例继续使用canonical workspace=/workspace；项目子目录建议agent_of_gemma_4，不为不同研究创建绕过锁的第二个workspace。
- 调用固定profile=amd_a。GPU长任务用submit_job，显式gpu_ids=[0]；纯CPU任务用[]，project可标p002-amd/p004-amd。
- cwd传workspace内相对目录（例如agent_of_gemma_4）；工具不接受绝对cwd。
- 明确使用/opt/venv/bin/python或经验证的隔离环境。不要无依据升级、删除或覆盖现成ROCm栈及依赖源码。
- 启动命令保持前台；服务与实验通过同一managed job管理，保留job_id，之后用job_status/job_logs。结果不确定时查询原job，不提交重复作业抢卡。
- 进程状态不代表云实例计费状态；本工具不负责关闭云实例。按平台管理方式处理实例生命周期，不通过删除项目或杀不明进程清理资源。

## 8. 当前未完成项

指定权重未下载/加载，32K与工具往返未实跑，AMD资格表未建立，AMD T0/T1未运行，训练兼容探针未做。当前结论为**具备值得优先推进的实机基础**，不是新的实验成绩或训练成功报告。

上游MI300系列的Gemma4配方不能单独证明gfx1100可用；本次判断主要依靠实机RDNA3 kernel/op证据。正式提交400及官方评分服务器部署状态另行取证，不与AMD可用性混为一谈。
