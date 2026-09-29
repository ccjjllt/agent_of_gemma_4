# 首批研究计划

资料核对日期：2026-09-28。

本项目研究指定 Gemma 4 模型在仓库级软件修复任务中的行为、代码定位效率和时间预算利用。001已实现，并完成三题初步运行，用户记录为1题resolved；原始轨迹尚未本地归档。下一轮优先003：先落实[运行载体修正](003-runtime-reliability.md)的R01–R19，再比较无进展抑制与软收尾。目前没有微调结果。

## 研究顺序

| 轮次 | 计划 | 核心产出 |
| --- | --- | --- |
| 001（已有实现与初步结果） | [无微调基线与失败分析](001-baseline.md) | 见[进展与证据复核](../progress/001-baseline.md)；结果保存与计时问题在003补齐 |
| 003（下一轮） | [行为研究](003-budget-and-context.md)＋[运行载体修正](003-runtime-reliability.md) | A0共同修订基线、B抑制重复无效动作、C增加20秒软收尾，先6题探索再24题配对 |
| 002（后续候选） | [代码定位与图检索](002-code-localization.md) | 面板结果显示定位瓶颈后再比较文本、符号相似检索与图邻域 |
| 004（后续候选） | [针对失败行为的 LoRA](004-targeted-lora.md) | 在多任务证据与可验证轨迹积累后确定训练目标 |

2026-09-29补充：L4排队已成为实际研究成本，下一批按[减少GPU排队的实验组织](003-queue-efficient-execution.md)实现一次模型加载、逐题轮换A0/B/C；CPU先完成资料、独立补丁诊断和产物分析。现有三个单变体载体尚未实现联合调度，不改动已排队任务。

研究顺序调整为001 → 003 → 依据结果选择002或004，保留原编号。003使用开发数据确认，28题保留集合（26个独立快照组）优先留到最终方案冻结后；原3题smoke已参与设计，不能作为独立推广证据。003两份计划的实现与结果统一记录在同轮进展文档，不将计划写成已完成实验。

## 001之后的关键判断

- FastAPI的重复操作支持研究无进展识别；Rich单题成功不足以证明所有定位问题已解决。
- Requests退出码124只说明本次验证超时，不能称为补丁正确或Agent行为满分。
- 已只读核对官方0.2.7源码：逐题evaluate_task不写汇总，应调用append_task_result；JSONL使用agent_patch_size；timeout_seconds同时限制命令与最终验证；duration_seconds包含准备、Agent和验证。
- 因此先修记录和诊断，再比较策略；暂不直接增加单题时间或开展LoRA。

## 已读取的资料

- Kaggle 官方 MCP：比赛元数据、全部内容页、数据文件清单第一页、公开讨论、官方 Getting Started Notebook v2 源码、当前账号加速器配额。
- 官方 `HARNESS_README.md`：49,356 字节，SHA-256 `3d6e57a13234cb4e783ba24eaab486459af76e0923ea4c6607cd41cc8961bbbb`。
- 官方 `tasks.jsonl`：1,984,455 字节，SHA-256 `e4b3fd60f69dbc2b9213e54eeb9636db78aefe92c1d06269d73d9f5f8f3c8ad6`。
- 实际任务记录：129 条，127 个唯一 `(repo, base_commit)`；FastAPI 67、Rich 48、Requests 13、HTTPX 1。

上述文件指纹来自首次只读资料获取，当时仅下载到临时目录；001后来已基于官方示例实现自己的运行载体。执行时从官方资源重新取得并核对，资料更新应形成明确的新实验版本，不能静默换题或混装依赖。

## 会影响实验的当前信息

- 比赛只支持 `gemma-4-31b-it-qat-w4a16-ct`。官方 Notebook v2 使用模型资源 `google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2`。
- 官方说明中的评分环境是 4 × NVIDIA L4、32,768 token 上下文。这不表示个人 Notebook 一定能分配相同资源，也不是训练显存承诺。
- 主办方 9 月 26 日表示空 embedding 文件已修复，缺失的 `async` 函数尚未补齐；重新下载后仍需检查实际版本。[图数据讨论](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/742911)
- 主办方 9 月 27 日表示公开 Notebook 的 subprocess 沙箱可能有问题，将改进可靠性；它与正式 Docker 评分环境不能直接视为等价。[运行环境讨论](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/742882)
- 闭源教师模型蒸馏的专项讨论中，当前读取到的主办方答复仍是等待进一步说明；一般外部数据条款不能代替该具体问题的结论。[蒸馏讨论](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/742807)
- 主办方9月28日进一步确认：12小时超限整次报错尚未修复、不支持并行模型调用、LoRA加载已修复、允许使用Gemma-4-31B自身生成的数据训练；其他模型输出仍待说明。[最新澄清](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743964)
- 当前wheelhouse依赖为正式环境依据，不支持ADK2.x；本地更换依赖不能改变正式评分服务。[版本说明](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743800)

## 官方入口

- [比赛概览](https://www.kaggle.com/competitions/gemma-4-developer-agent)
- [比赛数据与 HARNESS_README.md](https://www.kaggle.com/competitions/gemma-4-developer-agent/data)
- [官方 Getting Started](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent)
- [官方评估依赖包](https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse)
- [比赛规则](https://www.kaggle.com/competitions/gemma-4-developer-agent/rules)
