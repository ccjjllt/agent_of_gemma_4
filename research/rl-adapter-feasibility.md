# GRPO 后训练方案

更新：2026-11-04。目标是用官方公开任务提升修复能力，产出可挂载比赛CT模型的独立adapter。**目前尚无完整有效题组、GRPO参数更新或已训练adapter，没有训练收益证据。**

## 当前方案

| 项目 | 当前选择 |
| --- | --- |
| 基座与后端 | `google/gemma-4-31B-it-qat-q4_0-unquantized`，BF16冻结；双A800 NVLink，Unsloth＋TRL GRPO |
| LoRA | r128 / alpha256 / dropout0；400个文本投影，暂排除10个共享K/V的full层k_proj |
| 采样 | 32K完整上下文；temperature=1、top_p=1、top_k=0；4路生成、8个任务worker |
| 题组 | 每题目标8条；故障槽位剔除后至少2条完整验收轨迹可成组；每批目标4个有效组、最多尝试12组；同批最多更新2次 |
| 优化 | 学习率5e-6、首步正LR线性调度；比率裁剪0.8～1.28，基座reference，KL系数0.01 |
| 采题 | 在健康可选题中分配20%均匀探索、30%从未调度覆盖、50%近期反馈；运行故障独立退避 |

复合奖励为`完整修复 + 0.3×修复率 − 0.5×回归率 − 0.02×补丁交付墙钟秒数/3600`。修复分项来自同一次完整官方验收；时间项覆盖任务setup、prefill、生成、工具、补丁提取和排队，排除裁判setup/验收及归因对照，对成功和失败均线性扣费且不封顶。这是修复—时间权衡，不再保证任意慢的成功都优于失败。缺失分项保留奖励区间，只使用方向可确定的优势；时间也没有有限上界时整组保持未知、不给更新信号。group归一分母下限0.05，基础设施故障不当作模型零分。

仅生成token计损失，完整前缀参与反向。old概率在更新前冻结；策略损失、reference KL和复用漂移使用统一的截断概率校正（cap=2），加权漂移超过0.05时停止剩余复用。生成端使用动态前缀缓存与镜像sliding KV；训练端默认精确token树共享，保留所有原调用损失与前缀梯度。

当前r128有969,932,800个FP32可训练参数；参数、梯度和两份Adam矩预计约14.45 GiB，BF16 adapter纯权重约1.81 GiB，均不代表完整训练峰值。详细参数以[主配置](../configs/a800_2x80.toml)为准，启动与续跑见[运行说明](../training/README.md)。

## 官方依据与兼容边界

以下于2026-11-03核对；公开集仍为129题、4个仓库，官方wheelhouse为Version 31，31B CT为Version 2。

- **基座**：[Google CT模型页](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct)明确列出上述未量化QAT权重为量化前来源。[主办方答复](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748287)也指向官方未量化版本，其问答上下文为新增小型号。对应基座训练LoRA有依据，具体非零adapter的CT加载与修复收益仍待实测。
- **方法**：[比赛公告](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743007)鼓励参数高效微调和强化学习；[Google QLoRA＋SFT教程](https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora)与[HF GRPO＋LoRA示例](https://github.com/huggingface/huggingface-gemma-recipes/blob/main/scripts/carla_vlm_gemma.py)均为通用示例。尚未找到本赛31B专用的完整官方配方；GRPO、r128和奖励配置是本项目选择。
- **交付**：[正式规则](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview/model-selection-budget-and-harness-rules)支持CT模型加PEFT LoRA目录，通过YAML引用。[官方Starter](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent)中的adapter为dummy，只演示加载、评测和打包。完整接口查[官方原文](../official/HARNESS_README.md)。

## 真实结果

| 2026-11-02采集轮次 | 完成模型调用 | 已消费生成token | 引擎忙时采样token/s |
| --- | ---: | ---: | ---: |
| [003](../results/grpo-20261102/003-generation.jsonl) | 116 | 15,822 | 8.97 |
| [004](../results/grpo-20261102/004-generation.jsonl) | 41 | 5,484 | 10.16 |

两轮均未形成完整有效题组或参数更新。吞吐包含取消请求的采样token，不能作为有效训练吞吐。按消费顺序对齐前41次调用，输入/输出长度相同，平均生成耗时为50.62→51.20秒，未见明确加速；这也不是严格token级对照。来源：[003调用](../results/grpo-20261102/003-calls.jsonl)、[004调用](../results/grpo-20261102/004-calls.jsonl)、[004实际配置](../results/grpo-20261102/004-config.toml)。

## 关键未解问题

| 问题 | 当前影响与待取得证据 |
| --- | --- |
| 采集尚未转化为学习 | 已实现故障槽位剔除、任务健康退避和更新时间预留；仍需在固定GPU预算内完成有效组、真实参数更新和adapter导出 |
| CT部署与泛化 | 现有训练及配对入口都是BF16；r128覆盖、共享K/V映射及量化后的修复收益未验证。默认129题均可训练，成绩只能解释为训练集表现 |
| 成本目标与赛时迁移 | 奖励已采用所有有效轨迹的补丁交付墙钟；训练会话的超时预算仍扣排队，但奖励不扣。当地并发排队、BF16服务及逐轨迹线性成本仍只是比赛总墙钟修复目标的代理，需要CT端到端验证并校准时间权重 |
| 奖励与采样偏差 | cap=2仍有截断偏差，前缀来自behavior轨迹；缺失奖励只能保守识别部分优势，故障槽位剔除、组大小、归一和动态筛题也改变题目暴露，KL不能称全轨迹KL |
| 性能与训练稳定性 | 树注意力已改为分块祖先路径；9K/30K共同前缀可共享，CPU已核对attention前反向和小型Gemma4输出/梯度。512 MiB仅约束临时mask/index；31B双卡native兼容性、实际共享率、总峰值和耗时仍未知，同批两次更新及r128收益也未验证 |
| 官方摘要丢工具状态 | ADK默认摘要器只取文字，可能遗漏结构化工具调用与返回；官方提交接口未开放宿主摘要器替换。按用户决定保持`event_summarizer="official"`，暂不启用本地修复或宿主补丁 |

下一次经授权的GPU实验先产出首个真实更新后的adapter，再用相同公开题、seed、墙钟预算和完整官方验收比较基座与adapter，报告修复与退化；BF16结果不能代替CT结果。后续一次只比较一个因素，已准备[复用次数](../configs/experiments/reuse-one-batch/change.json)、[组内归一](../configs/experiments/reward-scale-none/change.json)和[前缀共享](../configs/experiments/prefix-no-sharing/change.json)对照，尚无GPU效果结论。

数据直接读取服务器官方公开集；不读heldout，不使用旧课程，参考补丁与验收资产仅供裁判。训练集成绩不称独立泛化结果。[官方数据](https://www.kaggle.com/competitions/gemma-4-developer-agent/data)、[官方评分](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview/evaluation)
