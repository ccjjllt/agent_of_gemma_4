# 005进展：官方/公开方案/前沿研究与实现交接

日期：2026-10-10，Asia/Shanghai。用户要求围绕金牌与奖金继续深度研究，优先机制研究；大量实现写成交接，不纠结门控和测试。

## 2026-10-10 合并审查：CT 兼容已验证，下一步先修数据抽样（最新）

`p005-next-round-sft` 的七个提交提供了 Python 3.12/v30 的 CPU 安装证据、官方 CT 上诊断 adapter 的加载/路由/工具格式证据，以及真实训练数据的完整长度普查。尚未训练真实修复 adapter，收益配对未运行。诊断工具调用不代表实际修复或 LoRA 数值作用的单独验证。

审查复算与原普查一致：15 条 train 中 4K/8K/16K 分别覆盖 0/3/12 条；但既有 first3 选样丢弃了全部 20 个标记恢复候选，39 条 trace 最终只保留 9 条 trace 的早期片段。下一步先在 CPU 修复恢复事件标注和跨 trace 行为抽样，再据新长度分布安排一次容量研究、一个 SFT 候选及八题 T0 配对。执行规格见[下一轮计划](../plan/005-next-round-sft.md#合并审查后的下一轮工作2026-10-10)。

合并审查修正 CPU 长度与 GPU 容量的混淆、自定义输出路径、v30 来源一致性及工具参数误报；86 项本地检查通过，真实 tokenizer 普查复现。修复后的探针未重新推送，历史运行报告保留，当次源代码可在 `673d357` 回溯。未纳入 Git 的数据/日志/过程材料已逐项校验归档，训练数据及日志已复制回主目录。本次整理未启动新 GPU 任务。最新已记录配额剩约 27.655h（09:47 UTC 快照），后续运行前重读。

## 2026-10-10 后续四卡31B训练成功（历史阶段）

修复后的私有`ccjjllt/p005-training-ddp-probe`已COMPLETE，wrapper success=true、worker退出码0，四个rank全部完成四个真实优化步骤：256、256、1024、2048长度，每步四个不同诊断样本。每步四rank梯度有限且有非零值，adapter均更新且参数指纹一致。云端保存step4的600-tensor adapter，权重212,261,592字节；本地只拉取报告和配置，没有下载这份合成诊断权重。

| 项目 | 实际结果 |
|---|---|
| 原FP32转换OOM | 已通过保留大矩阵BF16、只提升小norm修复 |
| CPU量化缓存 | 完整官方QAT v2→NF4，601模块、18,295,563,073字节；文件量减少70.75% |
| GPU加载与准备 | 四rank并行，每rank180.4–180.9秒；上次单卡原始加载839.3秒 |
| 整个GPU脚本 | 264.48秒；训练含诊断与checkpoint约36.57秒 |
| 2048长度更新 | 15.89秒，扣除已测诊断开销的估算为15.19秒 |
| 每卡峰值PyTorch allocated | 20.398 GiB；reserved20.666 GiB，物理可见22.03 GiB |
| 账号GPU扣额 | 1.80h→1.95h，净0.15h，剩28.05h；账号级取整读数 |

逐卡利用率需要分清阶段：全程27次采样均值约8.4%，受到约3分钟加载的影响；训练阶段只有3个10秒采样点，均值约67–68%，2048步骤内两次采样四卡均100%。这些是采样统计，不能声称全程满载。后续应一次加载后执行足够多的有效训练步骤，减少反复启动的固定成本。

已交付`training/memory.py`、`training/train_ddp.py`、同源NF4缓存加载、真实DDP全局监督token归一化及checkpoint恢复。两rank CPU与单进程全局批次数值对齐，尾部空rank不重复计权，恢复后参数逐位一致。CPU缓存私有Notebook`ccjjllt/p005-cpu-nf4-cache`已完成且可复用，不需要在GPU里重新安装torch或量化62.5GB原始基座。

本轮是明确标记的重复代码合成诊断，只证明所测2048及以下长度的加载、反传、更新和导出。loss下降不是修复能力或比赛提分证据；未验证4K/16K、CT基座adapter推理或正式评分。下一步优先收集能在2048内完整表达的真实修复/恢复样本，再做小规模SFT与固定CT上的base/adapter对照；不能截断已有超长轨迹来伪造短样本资格。

可复算证据：[结果摘要](../../research/005/kaggle-training-probe-20261010/ddp-result-summary.json)、[原始四卡报告](../../research/005/kaggle-training-probe-20261010/ddp-output/ddp-probe/wrapper-report.json)、[CPU缓存来源清单](../../research/005/kaggle-training-probe-20261010/cpu-cache-output/qat-nf4/cache-manifest.json)。以下保留首次失败与此前研究历史。

## 2026-10-10 首次Kaggle CLI训练探查结果（已由上述修复推进）

已完成一轮免费CPU准备与一次有硬时限的私有4×L4探针。Kaggle外层Notebook为COMPLETE，内部训练worker退出码2，**现有单卡训练路径失败，优化器更新0步，没有adapter产物**。原wrapper保存失败报告后正常退出，因此COMPLETE不能解读为训练成功；本地生成器已改为传播worker失败码，未重推或追加GPU运行。

- Python3.13.15、stock torch2.11.0+cu128、Transformers5.13.1、PEFT0.21.2、bitsandbytes0.50.2正常；离线安装7个CPU缓存wheel约12.19秒。小型NF4线性层前反向成功，证实当前训练依赖可用，推理wheelhouse的cp312问题并不阻断这条训练依赖路径。
- 官方QAT未量化v2→NF4真实31B加载完成，耗时839.30秒（约13分59秒），PyTorch已分配约17.03 GiB。机器确为4×L4，每卡实际可见22.03 GiB，当前加载器仅用GPU0。
- 随后`peft.prepare_model_for_kbit_training`将非4bit权重转FP32，试图额外分配5.25 GiB时OOM，当时只剩4.63 GiB。失败发生在训练准备，尚未到31B前向、反向或LoRA优化器更新；并非序列过长造成的本次失败。
- 整个脚本约885.34秒（14分45秒）；账号GPU配额从已用1.30h变为1.80h，剩余28.20h，账面净变化0.50h（接口按0.01h取整，属于账号级读数）。没有再启动第二次GPU任务。

后续先修复FP32转换峰值并设计有效利用4卡的训练路径，再运行下一次31B探针。加载本身占本轮脚本约95%，应研究可复用同源量化缓存与更高效权重加载，避免反复花14分钟冷启动。当前结果不能外推4K/16K训练或比赛提分。

证据：[结果摘要](../../research/005/kaggle-training-probe-20261010/result-summary.json)、[worker逐阶段报告](../../research/005/kaggle-training-probe-20261010/gpu-output/gpu-worker-report.json)、[逐卡采样](../../research/005/kaggle-training-probe-20261010/gpu-output/gpu-telemetry.jsonl)、[资源改造交接](../plan/005-implementation-handoff.md#8-每周30小时配额下的训练资源交接2026-10-10)。以下是此前实现与研究轮的历史状态。

## 2026-10-10 后续实现完成

用户随后授权补齐可执行路径。本节为当前实现状态，后文“本轮未做”为此前研究轮的历史记录。

- 新建完整 [single_state载体](../../notebooks/005-single-state.ipynb)及[生成器/使用说明](../../notebooks/005-single-state.README.md)：15文件全文与SHA打包，默认已暴露单题smoke，真实服务启动、逐题Evaluator、补丁/轨迹/错误归档与清理；没有启动远程GPU。Python3.13不兼容v30 cp312依赖时在安装前停止。
- 接通两个candidate的官方Docker/subprocess工作区解析、本题临时issue与handoff；修复patch_store恢复仅覆盖部分文件的问题，现可恢复快照后其他tracked变化及新文件。one_handoff真实ADK模型消息历史仍未实跑。
- [训练入口与说明](../../training/README.md)已完成：实际Gemma4加载、BF16 LoRA/同源NF4 QLoRA、监督token训练、checkpoint恢复、独立PEFT导出。缺依赖/缺权重不再返回伪成功。默认不下载权重。
- 修正真实数据链：保留官方工具语义和完整前文，参数错误及机械编辑错误只作上下文，坏动作不作正向监督；同issue/同快照不跨split。training/data源码已从.gitignore通用data规则中豁免。
- 历史真实轨迹导入15条、5题、89个片段，按整仓划分并限制每题3条后为6条train/9条select。这只是已暴露诊断语料。6条train均不适合4K/8K；16K完整编码4条、跳过2条，55,536输入tokens/1,559监督tokens，无JSON截断。未读heldout内容。
- 本地40项数据/真实tokenizer/训练/导出检查通过；随机微型Gemma4完成两步真实LoRA，基座不变、LoRA参数变化、checkpoint恢复后与连续训练参数相同。两个包的真实skill CLI及载体成功/异常/启动失败模拟检查通过。
- 验证栈位于/tmp隔离环境，torch2.14.1/transformers5.13.1/peft0.21.2，项目已有.venv未改。没有下载31B权重、没有31B训练、没有比赛提交或新增修复成绩。CUDA/NF4、真实31B资源及官方CT adapter加载仍未验证。

可复算验证记录：[implementation-verification.json](../../research/005/implementation-verification.json)。旧DRAFT保留为历史，不作为运行入口。

## 已完成

- Kaggle MCP只读取得主赛元信息、全部内容页、前50条排行榜、公开数据摘要、近期讨论、官方wheelhouse及论文赛资料；快照见`research/005/official-20261010/`。
- 下载官方adk_submission0.2.13与swegemma0.2.11两份小wheel（合计186,048字节），只解包读源码，确认数值thinking_budget/seed转发、0合法、工具错误回包、异常后补丁回收和图词法fallback。没有安装这些包。
- 获取八份公开Notebook源码，浏览器只读核实0.18 Budget-Fit V5、0.17 v19 V1；纠正标题“LB0.24”实际最佳0.13。没有取得榜首0.24方法与源码对应证据。
- 精读六项新增架构/行为研究；训练线精读五项（两项与历史重复，三项新增），加上模型卡、训练教程和作者实现。详细来源分别保存在frontier/training的sources.json。
- 将官方数据、社区负结果、本地93次结果的已有机制证据综合，形成[研究路线](../plan/005-medal-and-prize-research.md)与[开发交接](../plan/005-implementation-handoff.md)。

## 新决策

1. T2保留为精确动作基础；社区robust-edit三轮16题均6/16打平，不继续把机械编辑成功率当主要提分目标。
2. T3提升为行为约束、修复层次与两个竞争假设；加入按需实际调用路径。
3. 一次真实上下文交接上调；源码/diff等事实由工具保留，模型负责假设和下一动作。状态卡与真正更换历史明确区分。
4. 同协议局部修复/恢复LoRA SFT成为近期并行线；DPO随后，在线RL后置；训练基座和CT推理差异明确记录。
5. 将主赛金牌、主赛前三奖金、论文赛奖金分别规划。论文截止北京时间11月13日07:59；主赛12月3日07:59。没有建立自动日程或提交。

## 本轮未做

没有新GPU/TPU/AMD运行、教师调用、模型下载、训练、正式提交或公开发布；没有新增本项目修复率。未改变P002冻结变体、既有任务划分、原始结果或候选代码；未读取28题heldout内容。仅新建研究资料/交接并更新研究索引及004当前状态。

## 阅读入口

- `docs/plan/005-medal-and-prize-research.md`：主决策及冲牌/奖金节奏。
- `docs/plan/005-implementation-handoff.md`：推理组件、训练数据/编码/配方/导出接口。
- `research/005/official-20261010/report.md`：官方事实与历史结论修订。
- `research/005/public-solutions-20261010/report.md`：源码、真实分数、编辑与训练负结果。
- `research/005/frontier-20261010/report.md`：六项新增前沿论文与算法。
- `research/005/training-20261010/report.md`：基座关系、训练方法、参数、资源与模块。
