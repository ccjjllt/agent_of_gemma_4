# P002：TPU 可行性判断与有界验证交接

核对日期：2026-10-01（北京时间）。本次只读核对记录、代码、Kaggle MCP 与上游资料，未运行 TPU/GPU、未安装新推理栈。迁移代码由用户实现。

## 决策

**可以研究 TPU，指定模型已有新支持；不能把当前 Notebook 的 Accelerator 改成 TPU 后直接续跑 P002。**

建议保留冻结的 L4 T0/T1 主实验，另设 TPU 兼容性支线。首轮建议最多投入2小时TPU会话作可行性判断，包含下载、安装、编译和探针；这是研究投入上限建议，不是运行时间保证。若需要自研多芯片量化内核或长时间追依赖，停止支线，不因还有20小时额度而消耗全部时间。

TPU双方都重跑可以形成一个独立后端内的对照，但不能用GPU的T0对TPU的T1，也不能用TPU补齐GPU缺失的重复。5分钟预算在不同吞吐和编译条件下代表不同可用动作数。

## 1. 当前额度与设备边界

本次Kaggle MCP返回：

| 额度 | 已用 | 预留 | 总额度 | 算术余额 |
|---|---:|---:|---:|---:|
| GPU | 97752.0073023秒 | 0 | 108000秒 | 约2.85小时 |
| TPU | 0 | 0 | 72000秒 | 20小时 |

刷新时间为2026-10-03 08:00北京时间。见[额度快照](../../research/002/tpu-review-20261001/quota.json)。

用户页面显示GPU已用完，与API快照不一致。不能替用户推断某个值一定错误；也不能把API余额等同L4可申请、整批必能完成。运行前核对当前账号页面及实际可选机型，避免发起任务来试探分配。

Kaggle官方CLI机型列表含TpuV5E8/TpuV6E8，但部分型号可能限赛事/管理员；本次未建立TPU会话，因此本账号实际可分配型号未知。v5e每芯片16GB，v6e每芯片32GB；v5e-8合计128GB不等于模型可在单芯片自动使用全部内存。

来源：[Kaggle CLI机型](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)、[v5e](https://docs.cloud.google.com/tpu/docs/v5e)、[v6e](https://docs.cloud.google.com/tpu/docs/v6e)。

## 2. 指定模型已有TPU支持，但证据很新且范围有限

通过GitHub MCP实时API核实，[tpu-inference PR #3653](https://github.com/vllm-project/tpu-inference/pull/3653)于2026-09-27 06:33:30 UTC合并，merge SHA：

`1bd2d5a42af324706baff94f85cf6b8e8db8eea2`

它在JAX路径增加compressed-tensors的对称INT4、group/channel、pack-quantized、无activation quantization支持。作者明确测试了 `google/gemma-4-31B-it-qat-w4a16-ct`，不是仅测试同系列普通BF16模型。

作者报告的31B条件与限制：

- 单个v6e，max_model_len=2048，HBM约21.67GiB。
- v5e未测；TP>1未测。
- 冷编译约27分钟，保留其编译缓存后约9分钟。
- 测试吞吐来自16并发请求，不能用于预测我们串行Agent的速度或五分钟成绩。

原始PR正文与合并信息见[证据快照](../../research/002/tpu-review-20261001/pr-3653.json)。这些是上游贡献者测试，不是本项目Kaggle实测。

由此推断：若账号拿到v5e，则必须验证分片来满足显存，正落在未测试的多芯片路径；若能拿到v6e，可以先以单芯片短上下文接近上游条件。两者都仍需扩展到32K和Gemma4工具/推理协议。

版本陷阱：本次实时检查的最新发布v0.30.0中，JAX量化分派仍有WNa16未实现分支；main已含新实现。不能用“最新稳定版”代替具体源码确认。固定包含#3653的commit及兼容vLLM/JAX/libtpu组合，不盲目混装release与main。

来源：[v0.30.0分派源码](https://github.com/vllm-project/tpu-inference/blob/v0.30.0/tpu_inference/layers/jax/quantization/compressed_tensors.py)、[合并实现](https://github.com/vllm-project/tpu-inference/blob/1bd2d5a42af324706baff94f85cf6b8e8db8eea2/tpu_inference/layers/jax/quantization/wna16.py)。

## 3. 为什么当前载体不能直接运行

以 `notebooks/002-joint-explore-single-cell.py` 为例：

| 位置 | 当前行为 | 迁移要求 |
|---|---|---|
| 174–199行 | 安装冻结GPU wheelhouse，含cu128处理 | TPU服务采用单独依赖环境，不直接强装CUDA整包 |
| 222–225行 | 断言torch.cuda存在并读取GPU属性 | 记录真实TPU设备/拓扑/每芯片内存，不伪造CUDA通过 |
| 269–270行 | 根据GPU数量确定TP | 明确TPU mesh/sharding，验证量化scale分片 |
| 332–338行 | CUDA VllmServer、gemma4双parser、32768、20分钟启动超时 | 单独TPU服务适配，保留OpenAI兼容接口，逐项验证协议与长度 |
| 351行 | 创建模型registry并固定别名 | 别名后必须确为比赛固定v2权重，不能以BF16或其他checkpoint冒名 |

BF16运算本身不是TPU障碍。真实障碍是GPU安装链、量化算子、模型分片、接口版本和编译行为。原GPU载体的20分钟startup_timeout小于上游报告的约27分钟冷编译；TPU支线要单独设计有日志与总时限的启动等待，不能沿用后将正常编译误判卡死。

建议隔离推理服务与评测客户端依赖，复用官方Evaluator/工具/沙箱，不改变裁判或已有研究冻结文件。如果更换Python/依赖影响SubprocessManager验证，应重做同环境资格诊断，不能复用旧资格表而不检查。

## 4. 有界验证顺序（供实现，尚未执行）

### A. CPU准备

- 确认实际可申请机型；记录固定模型config、tokenizer、模板与权重清单指纹。
- 核对冻结v2的quantization_config符合新实现：对称INT4、group32、pack-quantized、无activation quantization。
- 冻结包含新实现的服务commit、vLLM/JAX/libtpu版本和安装方案；保留评测客户端版本。
- 新建独立TPU manifest和结果目录，不改 `explore_freeze.json`、T0/T1配置及原载体。
- 预先约定停止条件及编译缓存保存方式；不把模型回答“运行正常”当验收。

### B. 最小TPU会话

1. 识别真实设备与内存；验证权重确实由TPU执行，排除CPU回退。
2. 从2048上下文、短输出的基础生成开始；检查量化加载和多芯片输出数值有无NaN/异常。
3. 用合成消息检查一次真实tool_call、tool_response、后续调用和最终回复；包含引号、反斜杠、Unicode。使用真实gemma4工具/推理parser。
4. 逐级检查8K、16K、32K上下文及合理输入/输出额度，分开记录编译和稳态延迟。每次请求仍须满足输入+输出的总容量。
5. 在一个已暴露且eligible的任务上完成 `读取→技能→编辑→提交→验证`，保存trace/patch/result/request摘要；不得以打印Hello代替Agent验收。
6. 全部通过后才考虑同TPU环境的T0/T1小批配对；最终需要L4确认才能用于GPU部署决策。

**停止条件**：指定量化权重无法加载；TP>1数值/scale分片错误；工具协议无法往返；32K不可用；两小时投入上限到达仍无完整兼容性证据。短上下文能跑而32K未过，只算短上下文支线成功，不能混入P002主结果。

**可报告的结果**：TPU后端内的相对策略表现、工具协议兼容性、稳态吞吐。不能直接报告“官方五分钟GPU通过率提高”或以TPU证明私有L4评分器已修复reasoning。

## 5. 当前不用等GPU的工作

- 实际ADK/LiteLLM消息转换与HTTP参数捕获可以用CPU mock端点验证。
- 用同版本tokenizer/chat template和合成标记做渲染检查，不加载31B权重；这仍不证明私有评分服务部署了同样代码。
- compaction的事件选择、触发、工具调用/返回配对保留、长度控制可用CPU mock模型检查。实际Gemma摘要质量与时延需要真实模型。
- `compaction_interval`的语义应按当前ADK源码核实，它计user-initiated invocations，不应解释为每N次工具调用；token threshold另有触发路径。
- 技能脚本验收、图与源码覆盖、冻结补丁验证、分析补齐都可继续CPU。

当前preflight的 `RUN_MINIMAL_GPU_PROBE` 只有定义与末尾打印，没有真实推理分支。其94–125行的手写bridge与stdlib JSON往返不能代替真实ADK→服务合同验证。因此0A仍是部分完成；这是已披露边界，不使已经冻结运行的0C作废。

## 6. 本轮0C记录复核

- 原始12对：T0 7/12、C8192 6/12；按预先eligible四题×两重复：T0 5/8、C8192 4/8。
- `requests_6644`空补丁已通过，两组各两次成功不能归因于修复；作为不破坏行为的过程对照可保留。不要改名单抬分。
- 保留16384符合冻结规则；不等于证明8192无效。
- 已恢复23条trace实际包含738个Agent步骤的completion计数。C8192 12条/360步、最大2093；T0 11条/378步、最大2239。每条合计匹配final_metrics。
- 缺的是一个T0 trace以及全部trace的finish_reason字段；“没有单步计数”是文档错误，“没有length标记”也不能证明无截断。
- 图抽查是rich_3518单快照，snapshot HEAD偏移、仅一个async且属测试协程，不能把91.3%和async缺失推广整个开发集。

证据：[`analysis_0c`](../../research/002/analysis_0c_run-002-cap-20260930T082350Z-6dedcd65.json)、[`verification_eligibility`](../../research/002/verification_eligibility.json)、[`graph_audit_rich_3518`](../../research/002/graph_audit_rich_3518.json)、[`002进展`](../progress/002-code-localization.md)。

## 7. 最终资源安排建议

优先完成CPU合同取证与过程数据纠正；确认L4实际额度后，继续冻结的12次T0/T1探索。若L4确实不可申请，等待10月3日刷新期间，可按上面界限投入一次TPU兼容性支线。不要为了利用剩余额度提前启动P004，或把整个20小时投入未验证的迁移。

正式提交是另一条执行路径：已有Agent包的打包与上传准备不需要本地模型推理；正式评分由比赛服务执行。本次没有取得其是否扣个人Notebook GPU额度的明确证据，因此既不保证零配额可评分，也不应仅因本地CUDA不可用就断言正式提交必然不能进行。

比赛TPU讨论[744589](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744589)截至本次回读没有回复，不能当作主办方兼容性认证。最新[744566](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744566)已获主办方“将修补thinking_budget/seed”的答复；公告仍不能代替实际请求验证。

