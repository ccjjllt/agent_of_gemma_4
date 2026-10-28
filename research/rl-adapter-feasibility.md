# 官方 Gemma 4 31B CT 基座的 RL adapter：路线研究

研究日期：2026-10-28。目标是训练最终能够挂到比赛指定基座上的独立 LoRA，而不是训练一个只能合并权重后使用的模型。本次只查官方材料、开源代码和公开研究报告，另作显存与参数量计算；没有租卡、下载模型权重、读取题目正文或执行训练。以下把公开实证、源码推论和待实测方案分别标明。

补充说明：研究完成不等于训练问题已经解决。当前仍没有本项目真实跑通的RL→独立adapter→官方CT有效部署证据。文中的CT解量化方案是我们提出的工程候选，不是比赛官方给出的训练教程。下面附有本次通过Kaggle MCP直接读取官方材料的复核结果。

## 建议采用的主线

**原始官方 CT 用于采样与最终评测；同一 CT 解量化得到的 BF16 有效权重用于 learner；冻结基座，以标准组相对 RL 更新普通 PEFT LoRA；每次只同步 adapter。优先用一张 H200 做完整路径验证，成立后再扩展多卡。**

这是一条有源码依据、仍需完整模型实测的工程路线，不是已经跑通的比赛配方。对应 QAT-unquantized BF16 LoRA 是重要对照与备选；有大显存租卡条件后，不把再次量化成 NF4 作为默认主线。新的算法不默认沿用已删除的 VFARL 实现。

| 层面 | 推荐 | 原因 |
| --- | --- | --- |
| 基座 | 比赛实际挂载的 31B W4A16 CT，固定版本与指纹 | HF 同名仓库不能代替比赛实际权重身份 |
| learner | CT 原整数与 scale 解量化出的 BF16 基座，保持冻结 | 尽量对齐评测有效权重，不引入另一种量化表示 |
| adapter | 普通 LoRA，起点 r32、alpha64、dropout0；仅文本 decoder | 有同规模 agentic RL 经验，独立文件可提交 |
| 算法 | 无 reward 标准差归一的 GRPO；完整轨迹，每批一次更新 | 有公开 31B 实例，避免先引入 critic 和复杂分叉目标 |
| 系统 | veRL 训练组件＋官方 CT vLLM 服务＋独立 CPU 工具进程池 | 复用训练、批处理、adapter 同步接口 |
| 初期版本调度 | 同步策略版本迭代，版本内并发采样 | 避免同时引入参数滞后与数值错配 |
| 最终验证 | 比赛当前 wheelhouse、CT、TP4、L4×4、32K | 租卡机器上的训练成功不等于评分环境中的有效增益 |

## 官方允许什么，现成实证到哪里

官方工作人员已说明存在 rank≤64 的 LoRA 成功提交，并要求参考与 scorer 相同的 Getting Started 配置。这个证据支持 adapter serving/提交可行，不代表某种 RL 已证明有效。官方在小变体 LoRA 基座问题上指向未量化版本；未查到覆盖本任务全部条件的官方 RL 教程。[官方 LoRA 答复](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/749191)、[基座答复](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748287)

本地保留的 2026-10-24 官方说明要求独立 PEFT adapter、rank≤128、提交总解压大小<3 GiB，评测为 CT、TP4、32,768 tokens。这些是历史缓存；执行前以当前官方版本核对，不能自行修改评分后端。[原始说明](../official/HARNESS_README.md)

本次找到真正相关的 31B agentic RL 案例 **TermGrade**：发布方报告 plain-IT 基座上的 veRL GRPO++、r32/alpha64，五次训练的平均 Terminal-Bench 提升为2.1分；一次运行使用8×H200约7小时，含验证。它发布了 adapter，但不是比赛 QAT/CT 基座，结果也不是比赛成绩。[模型卡](https://huggingface.co/ai-and/termgrade-gemma4-31b-lora)、[发布方研究](https://www.aiand.com/newsroom/termgrade)

其配置摘要显示训练窗口49,152、rollout TP4，评测才是262,144/TP8；每步16题×16条轨迹，动态筛组还会额外采样。不能把“5次更新”理解为很少的模型调用，也不能把其耗时线性换算成我们的预算。配置未给完整框架 commit 与同步实现，属于可参考的一手报告，尚非我们独立复现的一键方案。[配置摘要](https://huggingface.co/ai-and/termgrade-gemma4-31b-lora/blob/main/training_config.yaml)

另一个31B数学RL发布方报告：其 adapter 在 vLLM0.28 的 runtime LoRA 下效果不完整，正式成绩采用 BF16 合并权重评测。这种交付方式不能直接满足本比赛；它提醒我们必须验证独立 adapter 的实际增量，而不只是能加载。[AE Studio 模型卡](https://huggingface.co/aestudio/gemma-4-31B-it-GRPO-math-lora)

## 为什么优先研究 CT 解量化 learner

设 QAT 未量化权重为 W，CT 保存的整数与 scale 对应有效权重为 W_CT，NF4 有效权重为 W_NF4，LoRA 增量为 ΔW。

| 路线 | learner 的有效基座 | 评测的有效基座 | 剩余问题 |
| --- | --- | --- | --- |
| QAT BF16 LoRA | W | W_CT | 原权重与量化值有差异，另有算子与 adapter 执行差异 |
| **CT 解量化 BF16 LoRA** | **W_CT 的浮点展开** | **CT kernel 使用的整数与 scale** | 浮点舍入、归约、attention、LoRA 加法顺序仍需比较 |
| QAT→NF4 QLoRA | W_NF4 | W_CT | 额外引入不同量化规则；主要优势是省显存 |

CT 解量化不会恢复 Google 量化前的 W，也不重新训练量化 scale。我们要训练的仍是普通浮点 A/B 矩阵，基座始终冻结。Google 模型卡将 QAT 未量化版本与 CT 推理版本明确区分。[Google 模型卡](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct)

源码依据：Transformers v5.20.2 支持加载时 dequantize=True；compressed-tensors0.17.0 的 packed decompressor 使用原始 packed integers、scale、shape及必要的zero point/g_idx进行恢复，不重新估计 scale。DECOMPRESSED 状态会跳过后续 weight QDQ。[HF quantizer](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/quantizers/quantizer_compressed_tensors.py#L124)、[packed 解码](https://github.com/vllm-project/compressed-tensors/blob/0.17.0/src/compressed_tensors/compressors/pack_quantized/base.py#L104)、[状态定义](https://github.com/vllm-project/compressed-tensors/blob/0.17.0/src/compressed_tensors/quantization/quant_config.py#L28)

这仍不等于逐位相同：审计的 vLLM 主线自己说明 Marlin tiling/K-split 会随 token 数变化；其 batch-invariant 分支采用原INT4＋scale展开后浮点计算。该主线功能不能当作比赛 patched v0.19.1 已具备的能力。[vLLM 源码，固定commit](https://github.com/vllm-project/vllm/blob/fd3469f4533d0517b0c8a6b72459d0fe7fd58d2f/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py#L62)

## 必须处理的 adapter 语义问题

31B 有50个 sliding-attention层和10个 full-attention层。HF 中 full层没有独立v_proj，K/V共享归一化前的投影输出；之后分别归一化，K还应用RoPE。因此给这些层的k_proj挂LoRA会同时影响K和V，而非表示最终送入attention的K/V完全相同。[HF定义](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/gemma4/modeling_gemma4.py#L1164)、[forward](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/gemma4/modeling_gemma4.py#L1231)

审计的vLLM主线基座加载器会复制K权重到融合QKV的K/V槽，但LoRA manager按q/k/v名字分别打包，缺v adapter时为None。由此存在“训练时K/V一起改变、部署时只有K改变”的静态风险。**尚未对比赛当前补丁运行复现，不能直接宣布scorer有此故障。** [基座映射](https://github.com/vllm-project/vllm/blob/fd3469f4533d0517b0c8a6b72459d0fe7fd58d2f/vllm/model_executor/models/gemma4.py#L89)、[LoRA打包](https://github.com/vllm-project/vllm/blob/fd3469f4533d0517b0c8a6b72459d0fe7fd58d2f/vllm/lora/model_manager.py#L849)

首版建议保留文本q/o、MLP和sliding层k/v，暂不训练10个full层的k_proj；等官方实际映射确认后再扩展。目标层必须是明确的文本decoder全路径，不能盲目对多模态整模型匹配七个叶子名字。部署端K→V复制可以研究，但不能未经验证改变提交adapter结构。

**零B adapter看不出这个风险。** 首次实测至少要比较无adapter、零B adapter和非零adapter；比较的是各端相对自身base的logprob增量，而不只是两个base的绝对误差。还需核对训练与服务的LoRA dtype：PEFT会默认将部分半精度adapter提升到FP32训练，不能假定导出及服务时精度不变。[PEFT文档](https://huggingface.co/docs/peft/package_reference/peft_model)

## 转换与软件版本不能混用

这是待完整checkpoint执行的转换契约，不是已运行脚本：

1. 固定比赛实际CT权重、原始config、tokenizer/template和generation配置；不以另一个同名仓库自动替换。
2. 独立转换环境使用已审计的TF5.20.2＋CT0.17.0加载并解量化，不加LoRA、不更新基座。
3. 该加载路径会自动删除根级hf_quantizer和quantization_config；但逐层weight_scale/weight_shape等附加状态可能保留。按目标普通模型的参数名与shape集合筛出完整dense state dict，仅删除已识别的量化附加项，再保存分片safetensors。
4. 在独立训练进程中重新加载普通BF16模型，确认没有CT hooks、packed权重或意外缺失参数，再冻结基座、注入LoRA。
5. 不调用prepare_model_for_kbit_training：这条路线已经是普通BF16 LoRA，该函数可能把冻结基座整体提升为FP32。也不在上述加载后再次调用model.dequantize()，它与加载时解压不是同一接口。

上述行为可追溯到[HF后处理](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/quantizers/base.py#L180)、[CT模块解压](https://github.com/vllm-project/compressed-tensors/blob/0.17.0/src/compressed_tensors/compressors/base.py#L115)、[HF保存state dict](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/modeling_utils.py#L3421)、[PEFT k-bit准备](https://github.com/huggingface/peft/blob/main/src/peft/utils/other.py#L160)。

特别注意：TF5.20.2会将旧global_head_dim、num_global_key_value_heads转成per_layer_config，而TF5.9不理解这个新结构。若选择旧训练栈，转换输出应保留官方原始配置的架构字段，仅去掉量化配置、声明BF16，不能直接拿新版重新序列化的配置给旧版加载。[新版配置转换](https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/gemma4/configuration_gemma4.py#L210)、[旧版attention构造](https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/gemma4/modeling_gemma4.py#L1161)

## 为什么选择 veRL 组件，仍需哪些适配

已核查正式tag veRL v0.9.1：

- 默认learner和rollout共用model.path。必须增加明确的独立rollout模型配置，指向原始官方CT；learner读取对应dense BF16 checkpoint。
- merge=False支持独立adapter。标准naive同步在base_sync_done之后可以只发A/B；非naive、分离GPU或异步同步不能直接假定同样安全，必须明确禁止向CT引擎覆盖基座。
- 每轮采集使用固定adapter版本，收齐该固定cohort后更新一次，再发布新版本。版本内可以连续批处理、并发工具调用，单任务结束立即补入该cohort的待办任务；跨版本异步训练留待后续。同版本、同token前缀才复用prefix/session KV；发布新adapter前排空旧请求，并通过受支持的adapter身份或缓存刷新机制避免跨版本复用。

依据：[模型路径](https://github.com/verl-project/verl/blob/v0.9.1/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L421)、[同步路径](https://github.com/verl-project/verl/blob/v0.9.1/verl/workers/engine_workers.py#L604)、[LoRA接口](https://github.com/verl-project/verl/blob/main/docs/advance/ppo_lora.rst)。

这些源码证明的是veRL支持版本栈内的能力，不能据此承诺它能直接连接比赛patched vLLM0.19.1。建议训练框架与官方采样服务使用分离Python环境/进程，通过token IDs、行为logprob、mask和adapter发布接口连接；该跨版本接口尚未实现验证。应避免为满足训练库依赖而静默替换官方采样端。

TRL v1.16.0默认vLLM同步对PEFT调用merge_adapter，再同步整套合并权重；server模式和use_vllm=True的自定义rollout也不能绕过这一点。因此不能直接把它接到固定CT基座服务。TRL仍可作为普通训练实现参考，但需要替换同步逻辑。[实际同步代码](https://github.com/huggingface/trl/blob/v1.16.0/trl/generation/vllm_generation.py#L396)

Unsloth有31B微调示例，但其Gemma4 RL指南仍采用fast_inference=False。SkyRL的Gemma4/QLoRA特定PR在本次查询时仍Open。两者都不能直接当作本任务已验证的高吞吐CT RL配方。[Unsloth指南](https://unsloth.ai/docs/models/gemma-4/train)、[SkyRL PR](https://github.com/NovaSky-AI/SkyRL/pull/1968)

veRL v0.9.1约束Transformers<5.11，CI用5.9；而新的CT加载接口与Gemma4修复来自更高版本。该tag是源码审计基线，**不是已确认可直接安装运行的最终锁文件**。可以独立转换、回移必要修复，或选定经过核查的较新训练栈；最终还要核对与比赛vLLM进程的接口和adapter同步。不能混抄不同版本文档形成貌似完整的启动命令。[依赖约束](https://github.com/verl-project/verl/blob/v0.9.1/requirements.txt)

## 初版 RL 的目标与数据流

建议初始配置：每题K=4条独立完整轨迹，组内同任务初态与预算；固定任务采样规则，奖励为完整官方验收的成功1/失败0。基础设施失败单独处理，不能编码成模型失败。K=8可在正负奖励对比过少时作为明确的新配置，不能用早停或缩历史制造差异。

优势采用 A_i=R_i−mean(R)，不做组内reward标准差归一。固定K的RLOO仅多一个K/(K−1)系数，也不需要critic。首版保留完整cohort与固定分母，不按得到的奖励动态丢弃同奖组。TermGrade的筛任务、动态筛组、零KL等属于可做的独立对照，不机械照搬。[veRL优势实现](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/core_algos.py)、[RLOO论文](https://arxiv.org/abs/2402.14740)、[Dr.GRPO论文](https://arxiv.org/abs/2503.20783)

若目标是等权任务成功率，轨迹内对模型生成token求和，再以固定cohort/固定常数归一；不要默默改成每条轨迹按自身长度平均。PPO clipping和rollout correction是代理目标与偏差/方差权衡，不能称为无偏保证。每批一次更新降低参数滞后，但不能消除跨端执行差异。[TRL归一选项](https://huggingface.co/docs/trl/grpo_trainer)、[veRL校正说明](https://github.com/verl-project/verl/blob/main/docs/algo/rollout_corr.md)

工程起点可用r32/alpha64/dropout0、microbatch1＋梯度累积、学习率5e−6、clip0.2；这些是待实测的起始值，不是本比赛已证最优值。从零LoRA开始时，reference若启用KL，可复用同一冻结learner基座关闭adapter计算，避免再常驻一份31B；这是需要明确接入的实现方案，不能假定veRL默认已这样配置。KL系数应显式记录，并以真实漂移与收益调整，不能用它掩盖结构映射错误。

每次模型调用保存真实prompt IDs、生成IDs、行为logprob、策略版本和mask。只有模型实际生成的reasoning、工具调用及回答参与策略损失；工具输出、prompt和框架插入token只作为条件。以采样当时的完整调用前缀重评分，避免重新渲染最终messages改变历史。必要时逐调用计算完整前缀，再按同一轨迹终局奖励聚合；不可通过detach训练前缀KV或截断历史换取一个未声明的新梯度目标。

训练数据继续只使用运行时挂载的官方公开任务，不引入TermGrade数据、旧自建课程或参考补丁监督。租卡机不会像Kaggle Notebook那样自动出现比赛挂载；实施时需通过有权限的官方来源在运行机器的数据卷准备原始资产，再只读挂载，不在本地项目另建训练包。独立工具工作树、并发数量、验收超时和CPU/RAM要与GPU分开管理。

## 32K训练的显存约束

官方QAT未量化checkpoint索引给出的存储大小为62,546,177,752字节，约58.25GiB。它是权重文件大小，不是峰值训练显存，也不是CT解量化输出实测值。[官方索引](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized/blob/main/model.safetensors.index.json)

按本地官方31B配置一次性计算（非实际加载结果）：

| 项目 | 估算 |
| --- | --- |
| r32全部实际文本投影 | 410模块，244,858,880个LoRA参数 |
| 排除10个full层k_proj | 400模块，242,483,200个参数 |
| 后者BF16 / FP32纯A/B文件 | 462.5MiB / 925MiB |
| 后者FP32参数＋梯度＋两个FP32 Adam状态 | 约3.61GiB，不含工作空间 |
| 60层各保存一个32K×5376 BF16 hidden | 约19.69GiB，仅说明激活量级 |
| 完整32K×262144 BF16 logits | 16GiB；全FP32为32GiB |

因此训练应使用activation checkpointing、必要的activation offload和经核对的分块logprob/loss，避免常驻全词表整段logits；这些优化必须保留正确的softcap和梯度。

另一个关键约束是全局attention的head_dim=512。FA2官方CUDA支持范围仍为≤256；SDPA可能退回二次方显存的MATH实现。TF5.9的GQA选择存在相关问题，v5.13.1及更高版本已有D512保护。用veRL旧环境必须回移修复或选择核对过的新环境，并确认实际forward/backward走高效后端。[FA2范围](https://github.com/Dao-AILab/flash-attention#nvidia-cuda-support)、[HF修复](https://github.com/huggingface/transformers/pull/46960/files)

可复用Axolotl的Gemma4混合attention接线：滑窗FA2、全局SDPA，并正确处理mask；它也有D512 Triton backward实现。其公开31B 32K FSDP配置是plain base、SFT、r256，不能整套用于比赛，但可作为底层实现参考。GQA保护只避免已知的错误后端选择，Axolotl路径也可能回退SDPA；实测应记录实际kernel，必要时临时禁用MATH以识别静默回退。单张H200能否承载我们的32K训练仍需实测，不用缩短目标上下文替代这个问题。[Axolotl指南](https://docs.axolotl.ai/docs/models/gemma4.html)、[混合mask实现](https://github.com/axolotl-ai-cloud/axolotl/blob/main/src/axolotl/monkeypatch/gemma4_hybrid_mask.py)、[D512实现](https://github.com/axolotl-ai-cloud/axolotl/blob/main/src/axolotl/monkeypatch/attention/flash_attn_d512.py)

## 租卡建议与预算

价格为2026-10-28约22:30北京时间Runpod MCP的实时目录读数：Secure Cloud、POD、CUDA最低12.8。单卡来自list_gpu_types；双卡来自对应count=2的get_gpu_type。这里只作为一家供应商的参考报价，未创建资源。LOW表示该次读数可用但供应低；不保证以后仍有同价同库存。

| 配置 | GPU费用/小时 | 查询时库存 | 适合的阶段 |
| --- | ---: | --- | --- |
| 1×H200 SXM 141GB | $5.29 | LOW，例如AP-JP-1、CA-MTL-4 | 首选：一个进程训练，采样与训练分阶段安排，减少多卡变量 |
| 2×A100 SXM 80GB | $3.58（2×$1.79） | LOW，US-MD-1/US-WA-1 | 更低时价，但需同机连接与FSDP；性能须实测 |
| 2×H100 SXM 80GB | $7.98（2×$3.99） | LOW，例如AP-JP-1、US-GA-2 | 单卡容量不够时的同机分片路线 |
| 2×H200 SXM 141GB | $10.58（2×$5.29） | LOW，EUR-IS-4 | 路径稳定后的吞吐扩展；可研究两份learner副本或采样/训练分工 |

GPU型号为目录返回的NVIDIA H200、NVIDIA A100-SXM4-80GB、NVIDIA H100 80GB HBM3。多卡还需核对实际同机NVLink拓扑，不能仅凭型号推断连接。H200官方规格为141GB、4.8TB/s。[NVIDIA规格](https://www.nvidia.com/en-us/data-center/h200/)

第一笔可提议的实验上限是1×H200四小时，GPU预算$21.16；这是费用上限示例，不是承诺四小时完成训练。存储、CPU配套、税费等另计。建议主机RAM≥256GB、可用盘≥250GB，属于权重转换、卸载、官方环境与临时文件的规划余量。整轮训练费用须用实测的完整轨迹吞吐与验收时间推算，不能由公开8×H200案例线性缩放。

## 进入付费实验时的最小证据链

1. **先在CPU完成资产和任务环境准备。** 固定版本、完成CT到dense转换的完整性核对，准备一个官方任务的原始环境。不要让GPU付费时间耗在首次安装与依赖排错。
2. **先比较同一前缀上的非零adapter。** 同时看base差异、adapter增量、不同batch/prefill/decode路径、真实行为概率与learner重评分；特别核对full K/V和导出dtype。零B相同不足以证明结构正确。
3. **完成实际32K forward/backward和峰值显存记录。** 只保留必要的一次性手动验证；若不满足，调整硬件或正确的attention/activation实现，不悄悄删历史。
4. **做少量完整官方任务的真实RL更新。** 固定cohort，保留所有成功/失败与同奖组，保存梯度、实际A/B变化和每次发布的adapter身份；不以loss下降替代任务收益。
5. **在官方部署方式下比较base与adapter。** 相同题、预算、采样和完整验收；训练题上的改善只算训练研究表现。独立公开开发评估的划分应在训练前另行确定，不读取隐藏或已有预留内容，不把全129题训练后的成绩叫独立泛化。

最重要的判别是：若普通HF learner改善而官方CT＋独立adapter没有对应改变，应先查映射、dtype、模板和同步，不继续盲目加训练步数；若两端一致却持续全同奖，再研究奖励产率和算法。租更多GPU不能替代这两个判断。

## 当前结论的边界

已经取得：官方adapter提交可行的答复；31B agentic RL的公开一手案例；CT解量化、LoRA映射、框架同步、版本配置和D512 attention的源码依据；参数量与显存量级计算；实时租卡报价。

尚未取得：本项目完整31B转换/加载、官方scorer非零adapter等价、32K反向容量、真实RL收益。它们属于下一阶段实施与GPU实验，不在本次研究中冒充完成。本次核查的开源项目中，未找到完整验证上述链路的即用方案；推荐路线已经能明确指出要复用哪些组件、修改哪些接口，以及先花钱验证什么。

## 补充：Kaggle MCP 官方训练信息复核

2026-10-28，按用户明确要求，通过本机已有配置连接 https://www.kaggle.com/mcp；服务返回Kaggle.Web 1.0.0.0、76个工具。只调用读取/检索工具，没有执行Notebook、启动GPU、提交作品或下载模型/训练数据。

实际读取范围：get_competition、list_competition_pages（10个页面）、get_notebook_info（官方Getting Started完整源码）、search_notebooks（主持人在本比赛下的公开Notebook）、list_forum_topics（train检索24项、LoRA检索49项并读取全部分页、reinforcement检索2项）、get_forum_topic（相关帖子与回复）、get_model/get_model_variation（Google官方31B QAT与CT）、list_competition_data_tree_files（仅sample_submission及adapter目录元数据）。检索命中数不是已读取全部讨论正文的数量，也不能推出全站没有其他教程。

| 官方来源 | 直接观察 | 支持的结论 |
| --- | --- | --- |
| [模型与adapter规则](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview/model-selection-budget-and-harness-rules) | 四个CT变体、同一提交共用一个基座，adapter可选，要求PEFT两份文件和YAML引用 | 明确规定如何交付和加载adapter；该页没有训练算法或训练循环 |
| [官方Getting Started](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent) | 六节为安装、数据/示例提交、图工具、vLLM服务、推理验收和打包；源码明确称示例为dummy LoRA adapters，调用discover_adapters后交给VllmServer | 这是部署/评测示例，不是训练教程；没有LoRA初始化、反向、优化器或训练循环 |
| [官方模型答复](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748287) | message3536992，Ryan Holbrook，author_type=ADMIN、tier=STAFF；答复小变体LoRA基座问题时指向unquantized模型页面 | 官方给出了训练来源方向；该答复不是31B RL的完整配方 |
| [官方欢迎帖](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743007) | Elan Markowitz，author_type=HOST，署名Google Research and The Gemma Team；提供starter submission、公开开发任务、图和embeddings，邀请探索PEFT/RL | 官方将post-training方法本身作为研究内容，没有在该帖附训练教程 |
| [Google QAT版本](https://www.kaggle.com/models/google/gemma-4/Transformers/gemma-4-31b-it-qat-q4_0-unquantized) / [CT版本](https://www.kaggle.com/models/google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct) | MCP确认两者当前版本均为2，version_id分别897279/897258；模型说明区分半精度QAT与vLLM CT推理格式 | 确认对应资产存在；元数据/模型说明没有构成RL训练教程 |

官方Notebook的MCP返回source为18,034 bytes，SHA256为8e7d84a326acae509fb596163ba085831961b588b86d51af31e037c5319b917c；与独立Kaggle SDK只读获取的源码一致。示例main_lora目录只有adapter_config.json（664 bytes）和adapter_model.safetensors（217,672 bytes），目录中没有训练脚本。没有读取或运行这些权重。

需要纠正的来源归属：

- [31B训练该用哪个checkpoint，750597](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/750597)：推荐未量化QAT的回复来自参赛者RISHAV KUMAR，不是主持人；其中对CT不可微调的绝对化说法也不作为技术结论。
- [SFT/RL基础设施，743186](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743186)：Unsloth文档链接来自参赛者，不是官方推荐。
- [Colab训练，745269](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/745269)与[adapter限制，745154](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/745154)：本次MCP返回中没有主持人训练配方。

MCP确实检索到了社区训练Notebook，但不能按标题判断已跑通：keykavus/gemma-4-31b-qat-lora-sft-trainer当前源码主要是LoRA配置，没有训练循环；manimaran2308/gemma4-31b-swe-lora-training有NF4/PEFT/Trainer训练与导出代码，但属于SFT，且模型选择、12B/31B文字和目标层存在需要审查的差异。两者不是官方RL配方，本次也未验证其输出或效果。

本轮可确认的是：官方提供了基座、开发环境、adapter交付接口和未量化版本的指引；在上述直接读取的官方材料中，未找到完整31B RL训练教程。“支持挂载adapter”不能单独证明官方已经提供训练它的实现。我们的训练方法仍需实际实现并在官方CT上验证，不能以候选设计或Notebook标题代替这个结果。
