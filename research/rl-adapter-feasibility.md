# GRPO 后训练方案

更新：2026-11-02。目标是用比赛129道公开任务提升修复能力，产出可挂载官方 W4A16 CT 基座的独立 adapter。**目前尚无完整有效题组、GRPO 参数更新或已训练 adapter，不能宣称能力提升。**

## 训练方案

| 项目 | 当前选择 |
| --- | --- |
| 基座与后端 | Gemma 4 31B QAT 对应未量化基座，BF16冻结；双A800 NVLink，Unsloth＋TRL GRPO |
| LoRA | r128 / alpha256 / dropout0；400个文本投影，暂排除10个共享K/V的 full层 k_proj |
| 采样 | 32K完整上下文；temperature=1、top_p=1、top_k=0；4路生成、8个任务worker |
| 题组 | 每题8条完整轨迹，每批目标4个有效组、最多尝试12组；同批最多更新2次 |
| 优化 | 学习率5e-6；比率裁剪0.8～1.28，基座reference，KL系数0.01，优势归一分母下限0.05 |

old概率在更新前冻结；仅生成token计损失，完整前缀参与反向，不detach前缀KV。同批旧策略漂移超过0.05时停止剩余复用，Adam与调度器状态跨批保留。

复合奖励：`完整修复 + 0.3×修复率 − 0.5×回归率 − 0.02×成功轨迹token成本`。奖励来自同一次完整官方验收；缺失分项保留未知及区间，基础设施故障不当作模型零分。采题按20%均匀探索、30%未访覆盖、50%近期反馈混合；完成整组后筛选学习信号，不足目标组数按实际有效组更新。

r128实测有969,932,800个FP32可训练参数；参数、梯度及两份Adam矩合计预计约14.45 GiB，另需基座、KV和激活空间。BF16 adapter纯权重约1.81 GiB；完整训练峰值及CT挂载效果仍待实际更新观察。

数据直接使用服务器上的官方公开集，参考补丁与验收资产仅供裁判侧；训练集成绩不称独立泛化结果。详细参数见[主配置](../configs/a800_2x80.toml)。

## 对后续训练有用的证据

| 采集轮次 | 完成模型调用 | 已消费生成tokens | 引擎忙时采样tokens/s |
| --- | ---: | ---: | ---: |
| [003](../results/grpo-20261102/003-generation.jsonl) | 116 | 15,822 | 8.97 |
| [004](../results/grpo-20261102/004-generation.jsonl) | 41 | 5,484 | 10.16 |

两轮均无完整有效题组或参数更新。吞吐包含取消请求的采样tokens，不是有效训练轨迹吞吐。按消费顺序对齐两轮前41次调用，输入/输出长度序列相同，平均生成耗时却为50.62→51.20秒，未见明确加速；长度一致仍不等于严格的token级对照。[逐调用数据003](../results/grpo-20261102/003-calls.jsonl)、[逐调用数据004](../results/grpo-20261102/004-calls.jsonl)用于区分工作负载与优化效果。[004实际配置](../results/grpo-20261102/004-config.toml)将每批有效组数和批次数设为1。

生成端已实现按轨迹前缀复用与动态显存预算；sliding KV改为镜像环形存储，成员退出用view选择连续行并裁掉共同padding，非连续行只复制survivors，以减少重复prefill和KV搬运。CPU固定轨迹对比保持生成概率一致；镜像存储的额外显存已计入预算，真实31B吞吐及完整训练耗时收益仍待后续实验。

## 当前未解缺陷

2026-11-02经Kaggle MCP复核：公开集129题来自4个仓库，约120道隐藏题来自私有仓库；最终目标是在**全部任务合计12小时**内交付修复，包含沙箱setup、不含补丁验收。公开训练成绩不能替代跨仓库能力证据。[官方数据](https://www.kaggle.com/competitions/gemma-4-developer-agent/data)、[评分说明](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview/evaluation)

| 优先级 | 已确认机制与尚未证实的影响 |
| --- | --- |
| P0 长任务状态丢失风险 | 官方wheelhouse31与本地均锁ADK1.36.1，默认摘要器只提取`part.text`，结构化tool调用/返回不进入摘要提示；不能把可能由此产生的回读或遗忘直接归因于模型能力。实际影响待轨迹证据，不能只改训练侧而假定官方侧同步解决。[版本源码](https://github.com/google/adk-python/blob/v1.36.1/src/google/adk/apps/llm_event_summarizer.py) |
| P0 采集尚未转化为学习 | 每题8轨迹，任一slot耗尽重试会使整组退出更新，已完成轨迹也失去本轮训练机会；采题优先级未计入真实成本。统计已补齐，仍须看固定预算内有效组与实际非零LR更新，不能只追求有效组比例。[采集逻辑](../training/collection.py)、[DAPO动态采样](https://arxiv.org/html/2503.14476v2) |
| P1 成本目标错位 | 训练扣除纯GPU排队时间；效率奖励仅惩罚成功轨迹生成token，65536后饱和，未直接表达setup、prefill、工具执行成本。应按官方墙钟下的修复数判断，不能把验收时间混入官方12小时成本。[奖励](../training/rewards.py) |
| P1 部署收益未闭合 | 学习与新配对入口均为BF16，提交用W4A16 CT；LoRA覆盖400个投影，另有10个full层k_proj暂不覆盖。尚无非零adapter的CT修复收益证据，不等于已证明量化导致退化。[模型限制](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview/model-selection-budget-and-harness-rules) |
| P1 概率校正口径不同 | old/behavior截断权重只乘advantage；TRL的KL校正用current/old，复用停止又直接平均behavior样本上的old漂移。生成/训练概率有差异时三者不对应同一分布；新增4调用KL探针只是有限诊断。[训练器](../training/grpo_trainer.py)、[锁定TRL源码](https://github.com/huggingface/trl/blob/v1.15.0/trl/trainer/grpo_trainer.py) |
| P1 奖励降级与题目加权 | 缺失奖励fallback会改成全组共同可见分项，可能暂失回归约束；组内归一与动态采题叠加题目加权。可在其他参数固定时单独比较`scale_rewards=group/none`；终局advantage广播本身是合法估计，不当作确定bug。[奖励实现](../training/rewards.py)、[Dr.GRPO](https://arxiv.org/abs/2503.20783) |
| P1 泛化证据缺位 | 默认129题全部可训练，内部评测默认in-sample；r128约9.7亿可训练参数的成本/收益未证实。需要在后续训练前按公开repo/base_commit元数据明确训练与评测边界，不读取官方隐藏题，也不能把分题成绩称跨私有仓库泛化。 |
| P1 短程复用对照可能无效 | 正warmup下首个optimizer的LR为0；单批reuse=1可能完全不改变adapter，reuse=2才有一次非零LR更新。固定调度已消除额外变量，但对照仍须有实际参数更新，不能据此断言两次复用更优。 |
| P2 训练前缀重复计算 | 当前逐调用重算完整前缀并反向；生成KV优化不消除此成本。先量精确token前缀重合率，再研究梯度复用；长上下文不保证高共享比例，不能detach推理KV冒充等价训练，Gemma4收益尚未实测。[PrefixFlow](https://arxiv.org/abs/2606.01143v4)、[Tree Training](https://arxiv.org/abs/2511.00413v5) |

## 下一步

实例关闭期间已补齐本地收益记录与单变量实验入口，**仍无更新后效果数据**。先保持当前算法配置产出首个adapter，再按下表一次只改一个因素；不自动调参。

| 待证实选择 | 单变量对照与判断依据 |
| --- | --- |
| r128 | 比较另一rank，保持alpha/r；看实际更新耗时、显存与相同任务/seed的修复增益，不能只看参数量 |
| 奖励权重 | 每次只改一项；看配对完整修复率、repair与regression及缺失覆盖率，离线奖励重算不算训练收益 |
| 动态采题 | 与纯均匀采题比较，保持题组信号筛选不变；看有效优势组/完整候选组、有效组/全部尝试组和有效组/小时 |
| 同批两次更新 | 与一次更新比较，固定同一LR调度尺度；看非零LR步数、同批更新前后KL与配对修复/退化，零LR的optimizer调用不算参数改善 |

训练摘要由已提交checkpoint生成，区分更新前采样成绩和更新后收益。更新后KL在固定、最多4个完整调用上测量，标为behavior样本上的K3近似；不是全分布KL。基座/adapter固定公开题与seed、相同墙钟和流程做完整官方验收，报告失败→成功与成功→失败。当前配对入口用BF16后端，训练题成绩不称独立泛化，官方CT迁移效果仍须另测。

实现入口：[训练器](../training/grpo_trainer.py)、[奖励](../training/rewards.py)、[采题器](../training/curriculum.py)。
