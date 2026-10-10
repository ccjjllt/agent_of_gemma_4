# Gemma 4 coding agent：训练路线研究（2026-10-10）

研究范围：面向本项目比赛成绩与可发表贡献，重新评估 LoRA SFT、蒸馏、偏好训练与在线 RL。只做代码/文献/元数据研究；本轮未训练、未安装训练栈、未下载模型权重、未调用教师生成数据。本文区分作者报告、已读取实现与本项目建议，不把论文分数折算为比赛收益。

## 1. 结论与优先级变化

**把针对工具交互的 LoRA SFT 从“等待其他路线全部做完”上调为近期并行主线。** 首个学习目标是：在已有源码和实际工具反馈下，完成一次局部修复交易，或者在错误后作出有效恢复。训练应与确定性编辑工具互补：工具解决字节替换，模型学习何时修改、修改什么、如何构造有区分力的检查，以及何时提交。不能指望只学习格式就提高修复率。

建议顺序：

1. **现在准备同协议的训练数据和模块化 SFT 实现**，使用公开外部轨迹扩大 issue/repo 多样性；已有项目片段用于定义行为、检验转换器。并行保留 T2/T3 工具与提示研究，不以“仍需更多前置审查”为理由长期搁置训练。
2. **首轮训练一个 rank 8/16 的 LoRA**，以 QAT 上游未压缩权重为优先训练基座，正式推理仍为指定 CT 基座。先完成一个可归因的训练版本；不同时搜索多个数据配方和 rank。
3. **同前缀反事实偏好训练列第二轮**。它能利用失败，但真实正确/错误配对仍须执行产生；不能把不同历史的成功和失败简单配对后宣称学会错误恢复。
4. **长程在线 RL 是明确的扩展线，不是首个训练实现。** 当前已有可借鉴算法和公开框架，但 31B、多轨迹、长上下文、可执行仓库的成本远高于离线 SFT；用小规模 LoRA RL 验证特定恢复目标，比复刻大型 RL 系统更符合当前资源。

重要的新证据是：训练格式并非无路可走——Google 已发布 QAT 未压缩上游，Unsloth 提供 31B 的现成 QLoRA 配方；同时，比赛社区的教师对照显示“换更强模型”本身的增益有限，训练更应压缩有效行为成本。真正尚未知的是本项目 adapter 在当前评分服务上的有效加载和收益，不能将其说成已证明，也不能因此无限推迟数据与训练代码建设。[Google QAT 发布](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/)、[Unsloth 31B 配方](https://www.kaggle.com/code/danielhanchen/gemma4-31b-unsloth)、[同 harness 教师对照](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747168)

## 2. 项目输入与目前不能夸大的部分

历史参考是 `docs/plan/004-targeted-lora.md` 和 `research/002/literature-20261001/learning-and-compute.md`。其中“004 暂缓”是 9 月 30 日的研究排序，不是训练禁令。本轮重新排序；不修改冻结配置。

无 adapter 的实际研究对照应从 `research/002/variants/t0/agent/` 读取，不能把仍是 001 初版的根 `agent/` 当作当前冻结 T0。训练候选与 T0/T1 的每次比较都需明确 tool/prompt 版本；更换编辑协议会改变训练分布。

现有 `research/002/candidates/training_prep/fragments_index.json` 基于 27 个已暴露任务，定位片段涉及 21 题、小编辑 10 题、真实读写错误恢复 9 题、推翻假设的转向 3 题；类别重叠。**它是索引，不是现成训练语料。** 很多读取只保留路径/范围，文本摘录封顶，原模型 thinking 没有保存，三条转向为研究者标注。必须回查原始轨迹/相应仓库状态补全实际观察，不能从最终补丁倒推并伪造当时推理。

若将已暴露 27 题用于训练，它们只能是训练/诊断集，不能再承担独立训练收益评估。保留既有 28 题 heldout；从尚未暴露的开发池预先分出独立选择集，登记后固定。外部数据同时按 issue、repo+commit、补丁归一化和近重复分组去重。

## 3. 五项深读：能借什么，不能照抄什么

### 3.1 SWE-Gym：少量高质量交互示范足以构成训练理由

作者用 491 条专家成功轨迹微调 Qwen2.5-Coder-32B，OpenHands 的 SWE-bench Verified 从 7.0% 到 20.6%。但在该路径继续加入 868 条自产成功轨迹后，Lite 从 15.3% 降至 8.7%。这同时支持“训练值得做”和“成功筛选不等于数据质量”。迁移到本项目：先学习可验证的局部行为，按独立题控制权重，不照抄成功轨迹中的循环。作者环境/步数和本比赛五分钟不同。[论文 §4.2、附录 B](https://arxiv.org/html/2412.21139)

公开的 491 条 OpenHands 轨迹可作小规模外部起点，数据卡标 MIT；但其工具协议与本项目不同，且 README 正文为空，必须保留来源并检查重放语义。不能直接替换工具名就当作等价监督。[作者数据集](https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories)

### 3.2 SWE-smith：规模来自可执行任务与去重，不是重复切片

作者用 5,016 条专家轨迹训练 32B，单次 Verified 为 40.2%；每题最多保留 3 条轨迹，避免易题支配。论文任务池与如今数据仓库不同，不能把最新数据条数当作论文训练规模。对本项目最实用的迁移是：先选 500–1,500 个不同外部 issue 的可验证短片段，再增加规模；按仓库和失败类型抽样，降低对 fastapi/rich/requests 的过拟合。[论文 §4、附录 F](https://arxiv.org/html/2504.21798)

作者当前轨迹数据卡标 MIT，包含成功/失败及不同序列化 split；正文称原训练集 5,017 条，和论文 5,016 有一条差异，本报告论文效果采用论文口径。作者工具支持创建任务环境，但要求 Linux/Docker；不是本机 macOS 无成本可复刻的管线。[轨迹卡](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories)、[作者实现](https://github.com/SWE-bench/SWE-smith)

### 3.3 Teaching Agents to Code Reliably（10 月 2 日）：训练有区分力的验证

论文把成功质量、探索、效率用于 SFT 权重，并加入测试区分错误修复的 RL 奖励。5,061 个示例来自 230 题；270 个未参与 SFT/RL 的题上 pass@1 从 31.9% 经 SFT 到 35.2%，再经 RL 到 43.0%。其 500 题整体先用于失败分析，不能称完全未暴露测试；作者也是 Qwen3-Coder-30B-A3B、长 token 预算，非 Gemma 五分钟。最值得迁移的是在 buggy/fixed 两状态验证同一断言，保留“错误补丁也通过”的难负例。不要复制其“多读就加权”：本项目已有重读问题。当前阅读页面未找到明确作者训练代码入口，视作新预印本的算法证据。[论文 §2.2–2.3、§3.1、附录 B](https://arxiv.org/html/2610.03984v1)

### 3.4 FC-SWE（10 月 6 日）：失败应保留自己的负标签

FC-SWE 在失败后干净重置仓库，携带失败补丁与验证反馈重试；每条实际执行轨迹保留独立奖励，不能将“失败→成功”两条都标成功。Qwen3.5-4B 下，GRPO 的 Resolved@1/2 为 38.9/48.5，FC-SWE 为 41.7/52.8。实验用 4,518 题、每题 8 条初始链、最多两次尝试，1800 秒/100步；测试重试获得 benchmark 反馈，且 100 个 Verified 题参与 checkpoint 选择。它支持恢复训练思路，不提供本比赛可直接复用的成绩或算力结论。本项目只将**在线本来可见的工具错误和自己运行的检查**作为恢复输入。[论文 §3、§4、附录 A–B](https://arxiv.org/html/2610.07898v1)

### 3.5 EntroPO：偏好学习有用，但不能只压低失败文本概率

作者在成功轨迹 SFT 后，进行多轮 DPO/KTO 的熵增强版本，以减轻策略多样性下降；有公开代码和模型。训练用 rank8/alpha16、18K context、1e-5、各一 epoch，但资源为 8×H100 加辅助 A100，推理允许 200步/131K tokens。迁移：第二轮做相同决策状态的短动作偏好学习，监测是否只学到缩短输出、少搜索或提前提交；不要照搬多候选 verifier 和长步数选择器。LoRA 偏好训练可共用 SFT 的数据、编码和导出模块，新增 paired loss 即可。[论文 §3.3、附录 D](https://arxiv.org/html/2509.12434v1)

公开 RL 实现可参考 rLLM 的 DeepSWE 环境/rollout/奖励接口；其报告 42.2% pass@1 与 59.2% test-time scaling 是不同指标。本项目若扩展 RL，应接入现有工具环境适配器，不整体迁移 Qwen 的工具格式或 70K context 配方。[作者 DeepSWE 实现](https://github.com/agentica-project/rllm/blob/main/examples/agents/swe/README.md)

## 4. 训练基座、量化与推理的具体关系

### 4.1 已核实的模型对象

2026-10-10 通过 Kaggle MCP 读取 Google 模型列表：

| 对象 | Kaggle 类型/版本 | 总文件字节 | 用途 |
|---|---|---:|---|
| gemma-4-31b-it | Transformers / v1 | 62,578,670,545 | 普通 IT，已有 Unsloth 示例，但与比赛 QAT 存在权重差异 |
| gemma-4-31b-it-qat-q4_0-unquantized | Transformers / v2 | 62,578,682,779 | **首选训练权重来源**，QAT 流程输出的未压缩半精度权重 |
| gemma-4-31b-it-qat-w4a16-ct | Other / v2 | 23,297,590,856 | **正式指定推理基座** |

模型条目均标 Apache 2.0。以上是文件总量，不是训练峰值 VRAM。[Google QAT 模型卡及谱系](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct)、[未压缩 QAT 上游](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized)

指定 CT 配置是 compressed-tensors 的对称 4-bit、group_size=32、浮点激活；不是 bitsandbytes NF4。模型有 60 个文本层、hidden_size5376、局部/全局 attention 不同，且 attention_k_eq_v=true。**不能把 W4A16 文件直接喂普通 QLoRA 脚本并宣称可原生训练。** [官方 config.json](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct/raw/main/config.json)

### 4.2 推荐的两条训练路径

**路径 A，低部署差异优先：QAT-unquantized BF16 冻结基座 + LoRA。** 只训练语言线性层的低秩增量，导出独立 PEFT adapter。推理时官方 CT 权重保持原样，叠加 adapter：训练近似为 `W_QAT + ΔW`，服务为 `dequant(Q_CT(W_QAT)) + ΔW`。仍有量化差异，不能声称两者数值等价。需要较大显存或分片，适合可用 80GB 卡/稳定多卡时。

**路径 B，低成本优先：将同一个 QAT-unquantized 基座以 NF4 加载，再训练 LoRA。** PEFT 的 prepare_model_for_kbit_training、NF4、double quant 与 BF16 compute 是可用的通用机制；这里训练的是 `Q_NF4(W_QAT) + ΔW`，正式服务仍是 `Q_CT(W_QAT) + ΔW`。因此增加了 NF4→CT 的量化偏移，要把“在训练加载器上有效”与“在正式 CT 上有效”分开报告。初轮不使用 LoftQ 来拟合 NF4 误差，避免把特定 NF4 补偿当成可迁移能力。[PEFT 量化文档](https://huggingface.co/docs/peft/developer_guides/quantization)

普通 IT + NF4 是现成教程可行性的参考/备用路线，**不是首选比赛权重对齐方案**；也不能把 adapter_config 的 base_model_name_or_path 改名就消除权重差异。保持 adapter 独立，不合并后重新量化并替换官方基座。

### 4.3 模块选择与导出

第一版目标文本 `q_proj/o_proj/gate_proj/up_proj/down_proj`；先避开 k/v 的共享与打包复杂性，仍覆盖注意力输出和 MLP。第二版若第一版欠拟合，可扩到**模型实际存在**的 k_proj/v_proj，不要求每层都有 v_proj；不训练 vision tower、projector、embedding、lm_head、norm，不增加 token。

当前 vLLM Gemma4 源码把 gate/up 打包，q/k/v 按层类型映射；全局 K=V 层只有 k_proj，映射到 K/V 两片。必须用显式文本模块清单，而不是不加检查的 all-linear。当前 main 的支持也不等于评分 wheel 的支持，adapter 导出器应单独维护 inference-compatible key map。[vLLM Gemma4 源码](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/gemma4.py)

以官方结构推算，60 层的上述 q/o+MLP 在 rank16 约 1.06 亿 adapter 参数，BF16 约 212MB；rank8 约一半。计算为 rank×[50×(2×(5376+8192)+3×(5376+21504))+10×(2×(5376+16384)+3×(5376+21504))]；该估算忽略少量配置开销，最终以实际 key/shape 求和。远低于历史 3GiB 包上限不代表 KV/启动内存无问题。导出内容为 adapter_config.json、adapter_model.safetensors、基座/模板/数据/模块清单的 manifest；训练语料不进入提交包。

## 5. 数据路线与学习目标

### 5.1 首个目标：有证据的局部交易与真实恢复

以 2–6 个 assistant 动作构成一个样本，包含足够的真实前文：

- issue 相关片段、已读源码、当前文件哈希/仓库状态；
- 当前工具调用、实际错误或差异观察；
- 正确下一步：范围回读→精确修改→针对性检查，或补充一个缺失触发条件→检查→修改；
- 成功提交前的 diff 卫生与终止动作。

**学习重点是 evidence-to-action，不是最终 patch 文本。** 对“读过正确位置但未修改”“伪复现全绿”“重复失配后不恢复”分别打行为标签，首轮数据混合仍围绕同一局部交易目标。完整多轮成功示范用于保持流程；未知验证结果的样本不作为高权重正例。

建议首批数据规模是工程预算，不是文献保证的阈值：500–1,500 个去重外部 issue，1,000–3,000 条交易片段，总训练文本控制在约 4–12M tokens，先一 epoch；项目 27 题中的高质量片段不应靠多次切片占训练大头。每题最多 3 条、每 repo 设权重上限；初始可按外部多样交易 60%、真实恢复/反例 25%、完整紧凑解题 15% 配比。没有足够合格恢复样本时降低比例，不通过生成无证据“正确恢复”填满。

### 5.2 来源与许可事实

| 来源 | 本次确认 | 建议用途 |
|---|---|---|
| 项目自身 Gemma 轨迹 | 主办方已允许自生成轨迹训练 | 行为定义、同协议核心例；回查完整状态 |
| SWE-Gym/SWE-smith 公开轨迹 | 作者数据卡标 MIT；主办方明确允许既有非 Gemma 公共 SWE 轨迹 | 首批外部 SFT，逐项保存来源与原仓库许可 |
| 本地开源 teacher 的外部公开任务轨迹 | 主办方 746046 明确允许 | 补充学生反复失败而教师能稳定闭环的类型 |
| TFD 的 50 条 Gemma 轨迹 | 数据集描述写 Apache2.0，Kaggle licenses 字段却是 CC0-1.0 | 仅列候选，尚未逐条检查数据，不作为已验收高质量数据 |
| 向外部 API 发送比赛 released tasks | 746251 目前未见主办方回复 | 首轮不依赖该路径；公开外部数据/本地 teacher 可推进 |

主办方 742807 的通用回复附带模型与比赛许可条件；746046 随后的明确答复涵盖开源 teacher 和公开外部 SWE 轨迹。数据卡许可不自动覆盖所有内嵌仓库内容及教师供应商条款，来源字段与原许可须一起保留；这不阻止先选上述公开资源开展工程。 [742807](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/742807)、[746046](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/746046)、[746251](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/746251)、[TFD 数据](https://www.kaggle.com/datasets/rajrajak99/gemma-4-tfd-agentic-trajectories)

### 5.3 样本格式与损失

建议统一中间格式，避免绑定某训练框架：

```json
{
  "sample_id": "external-issue-transaction-01",
  "source": {"uri": "...", "license": "...", "trace_sha256": "..."},
  "group": {"issue_id": "...", "repo": "...", "base_commit": "...", "split": "train"},
  "protocol": {"tool_schema_sha256": "...", "chat_template_sha256": "...", "thinking_mode": "..."},
  "state": {"repo_state_sha256": "...", "remaining_budget_seconds": 180},
  "behavior_target": "evidence_to_local_edit",
  "messages": ["真实按序角色消息和tool_calls/tool结果"],
  "supervised_assistant_steps": [12, 14, 16],
  "verification": {"local_action_valid": true, "final_resolved": null, "basis": "..."},
  "offline_labels": {"gold_metadata": "只供筛选，编码时强制剔除"}
}
```

这只是说明结构，实际 messages 保存对象而非上述说明字符串。训练 encoder 用**计划部署版本**的 Gemma tokenizer、system/tool schema、thinking 与 function-call 模板；标签只覆盖选定 assistant 输出 token，用户/工具回包/复制的旧错误/离线 gold 标签全部 loss=-100。保留调用边界，不在参数 JSON 中截断；按整个交易的依赖切片，不伪装成从零开始的完整轨迹。

本项目原轨迹未保存 thinking：不补写假的隐藏思考，不把缺失当作空思考的监督。首轮主要监督可见工具动作；若保留推理能力需要完整 reasoning 示范，应使用许可证合适且确有记录的新样本，独立标识，避免大规模 action-only 训练隐式教会模型关闭 thinking。

### 5.4 第二轮：同前缀反事实 DPO

每对样本保存同一 issue、仓库状态和消息 prefix。rejected 为真实无效动作（如旧串失配后重复原调用）；chosen 必须是从同状态执行并证实有进展的动作或短后缀。可以由人工/teacher 提供替代动作，但明确其来源，不伪称 Gemma 历史恢复。

偏好排序以可执行正确性为首、局部进展为辅；只有都正确时才用延迟/调用数作效率偏好。不同环境错误、两个都失败但未知哪条更好、不同源文件状态的动作不强行配对。先用 300–1,000 对、DPO beta 0.05–0.2、lr 5e-6–2e-5、0.5–1 epoch；保留 SFT 混合损失作为项目候选设计，监测实际工具成功率和求解率，不把偏好 margin 当成研究结果。

## 6. 算力、初始超参及投入产出

### 6.1 可执行的 SFT 起点（本项目建议，未运行）

| 参数 | 初始值/窄范围 | 理由 |
|---|---|---|
| 基座 | QAT-unquantized v2；优先 BF16 LoRA，资源不足用同源 NF4 QLoRA | 减少普通 IT→QAT 的权重漂移 |
| 目标层 | 文本 q/o + gate/up/down，完整 60 层 | 首轮避开 K=V 导出复杂性 |
| rank / alpha | 首个 r=8、alpha=16；数据充足后单独比较 r=16、alpha=32 | 小 adapter、少量数据、便于归因 |
| lr | 2e-5–1e-4，首个 5e-5 | 比通用60步演示的2e-4保守 |
| 序列 | 4096 首批；完整交易确需时 8192 | 不截断决定下一步的观察；长上下文另分桶 |
| batch | micro=1，grad accumulation 8–16 | 按有效监督 token 平衡，不仅看样本条数 |
| epoch / warmup | 1 epoch；warmup 3%–5% | 小语料过拟合风险，先固定一轮 |
| 其他 | dropout0，bias none，weight decay0–0.01，clip1.0，gradient checkpointing | 冻结基座，激活占用可控 |
| checkpoint | epoch内少量固定点；独立选择集选一次 | 不用训练题反复挑最优 |

这不是官方推荐超参的逐字复制。作者 Kaggle v3 配方实际为普通 IT、NF4、8192、device_map=balanced、2×T4、rank8/alpha8、micro1/accum4、lr2e-4、60steps。本次 MCP 读到的 notebook 单元 outputs 为空：它提供可读实现，**没有可据此核实的训练时间/峰值显存**。Unsloth 文档的“31B QLoRA 22GB”也只是其条件下的资源宣称。[作者 notebook](https://www.kaggle.com/code/danielhanchen/gemma4-31b-unsloth)、[Unsloth 训练文档](https://unsloth.ai/docs/models/gemma-4/train)

### 6.2 资源选择

- **1×48GB CUDA GPU：近期 QLoRA 工程首选（推断）。** 相比24GB更能容纳4096/8192动作上下文、激活及框架开销。是否支持同源 QAT→NF4、实际吞吐和内存仍需运行测量，本文不编造小时数。
- **1×80GB：较低部署差异的 BF16-LoRA 候选，也可更从容做 QLoRA（推断）。** 62.6GB权重加激活并非必然适配所有长序列，不能说80GB就保证32K训练。
- **2×T4 或 4×L4：可作为低成本 QLoRA路线，但分片工程风险更高。** 独立卡显存不能简单相加；device_map 示例与分布式训练方案不能混用。社区 745364 报过2×T4 device mismatch/OOM，不能拿教程替代本机证据。
- **M5 32GB：CPU数据工作和静态检查合适，非本轮主训练平台。** 不把MLX产物导出到CUDA/CT的额外适配作为主线依赖。
- **TPU：存在作者公开的 Gemma4 31B LoRA 工程，但使用Google Cloud TPU及Orbax转换路径。** 它证明另一条研究路径存在，不能代表Kaggle TPU已有同样可用栈，首轮CUDA能做时不平行扩建。[TPU作者实现](https://github.com/h2loop/gemma-tpu)

训练时间用可复算预算表达：`训练秒≈总实际处理token/实测tokens_per_second`，分别记forward/backward、数据加载、checkpoint保存及CT验证时间。比如首批4M–12M tokens不等于4M–12M个有效监督token；长工具观察虽被mask，仍消耗前向算力。先跑完整首版数据的一小部分取得吞吐，再决定资源投入，而不是承诺未经测量的金钱/小时收益。

### 6.3 方法投入产出

| 方法 | 最可能贡献 | 主要成本/风险 | 当前顺序 |
|---|---|---|---|
| 同协议局部交易 SFT | 更早编辑、减少无效循环、学习真实恢复 | 数据转换和量化迁移；少题过拟合 | **立即并行实现** |
| 公开教师轨迹蒸馏 | 多样修复模式与更紧凑动作 | 工具协议错配；复制无效探索 | SFT数据来源，不单列巨型系统 |
| 同前缀 DPO/偏好学习 | 压低已识别的错误选择 | 真实对照生成/执行；错误偏好和模式塌缩 | SFT之后最有价值的升级 |
| 拒绝采样自训练 | 得到Gemma自身分布的数据 | 易题重复、只保留成功导致偏差 | 定向补充，不能全盘无权重回灌 |
| 在线 LoRA RL | 直接优化行为与最终成功 | 31B rollout、多容器、稀疏奖励、环境漂移 | 独立后续分支 |
| 全参数训练/超大教师蒸馏 | 更广能力变化 | 远超当前适配需求与计算资源 | 当前不选 |

比赛教师对照 747168 使用25分钟/50调用、修过本地环境、旧swegemma0.2.7。Gemma两次约54/124与53/123，teacher约58/124与59/123；两次Gemma有19/123题翻转。它不是本项目配对实验，但提醒不要把小增益当稳定教师优势；教师中位3.5–3.9分钟对Gemma5.2分钟，更支持学习高效闭环，而非堆大teacher API账单。[实验原帖](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747168)

## 7. 模块化实施交接

以下均为待实施模块，不在本轮伪装成已有代码；可并行开发，最终通过明确中间格式衔接。

| 模块 | 输入→输出 | 职责和可独立验收产物 |
|---|---|---|
| `training/data/catalog.py` | 来源元数据→source_manifest | 许可/模型/issue/repo/commit/内容hash；先元数据筛选，避免一次下载整个语料 |
| `training/data/trajectory_adapter.py` | ATIF/OpenHands/SWE-agent→canonical JSONL | 保留真实顺序、工具调用ID、观察与执行状态；工具语义不等价时明确拒绝转换 |
| `training/data/transactions.py` | canonical→交易片段及审查表 | 依赖闭包切片；原观察缺失明确标缺，不重建thinking |
| `training/data/replay_labels.py` | 片段+原快照→局部动作/反事实标签 | 同状态重放chosen/rejected；失败恢复保留前失败标签；服务训练，不改官方评分逻辑 |
| `training/data/split_registry.py` | issue+commit+近重复→固定split | 冻结28题不进入训练/挑样本，外部数据查重 |
| `training/encoding/gemma_tool_chat.py` | canonical+冻结模板→tokens/masks | 对齐部署tool/thinking协议，只对有效assistant目标算损失 |
| `training/train_sft.py` | tokens+配置→LoRA checkpoint | QAT-BF16/NF4两后端可选，记录可训练参数/显存/吞吐，训练循环不负责环境构建 |
| `training/train_preferences.py` | prefix/chosen/rejected→偏好LoRA | 共用编码/导出；后续模块，不阻塞首轮SFT |
| `training/export_peft.py` | checkpoint→PEFT adapter+manifest | 文本模块白名单、prefix/key映射、shape/rank/文件大小核查，不合并官方基座 |
| `training/eval_adapter.py` | adapter+CT服务→行为/解决率结果 | 同agent协议和预算，报告base与adapter；单独标加载失败/推理失败/修复失败 |

训练配置应固定 source revision、tokenizer/template hash、工具 schema hash、QAT base revision、量化后端、目标模块清单与损失定义。每个训练样本可回查到原数据与执行证据；每个结果可回查到adapter hash。这样实现模块可按资源逐步推进，不需要先建一个复杂的全自动训练平台。

## 8. 尚未知、但不应阻止准备工作的关键点

1. **评分服务 adapter 是否实际应用。** 9月28日官方称修过加载；10月社区仍报告极小零adapter提交Kaggle Error，无adapter可得分。两者不能互相替代，当前本项目没有adapter评分成功证据。这是训练结果解释的真实边界。[官方旧回复](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743964)、[当前社区报告](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/745364)
2. **QAT-BF16/NF4上的LoRA能否稳定迁移到固定CT。** 官方谱系支持选基座，但没有我们这种工具行为adapter的已公开、可直接照搬的结果。
3. **缺失thinking与工具序列化的影响。** 原索引缺文本/思考不能靠人工伪造补齐；新训练数据应记录实际模型输出及部署模板版本。
4. **训练收益是否跨任务类型。** 三个主要公开仓库不是隐藏分布；外部多仓库语料和独立选择集是增益可信度来源，训练loss不是替代品。
5. **资源和分片。** 现成2×T4教程与社区失败并存；本文提供两条可实施训练路径，未承诺本项目特定卡型吞吐。

本报告改变的是训练研究的行动优先级：已有依据足以建设首个可训练数据集和SFT模块，随后得到一个独立adapter。最终比赛收益仍须来自部署在指定CT基座、同预算的实际修复结果；本文没有新增成绩。
