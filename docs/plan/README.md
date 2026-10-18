# 首批研究计划

**当前入口（2026-10-18）：[011批量SFT进展](../progress/011-batch-sft.md)，[010实际结果](../progress/010-budget-transfer.md)。** P010已完成，base 2/3、adapter 1/3，没有adapter净收益。P011首批139个独立问题/58repo已冻结；V1错误删除比赛关联，实际2×T4，训练0步失败。已修复并提交V2，平台RUNNING，运行时日志确认实际4×L4，计划两轮70步，优化进度仍待确认。此前[009](../progress/009-real-sft-and-agent-transfer.md)只有8题/7repo、4次更新，下一档扩至512。31B正式提交57340172截至UTC10:30仍待评分，已确认成绩仍是旧提交的0.13。下文为历史索引与当时官方限制，不能覆盖最新事实。

状态更新：2026-10-10；早期官方资料记录的日期另见各节。

后续实施更新：P005已接通[single_state完整载体](../../notebooks/005-single-state.README.md)与[真实训练/导出链](../../training/README.md)。本地40项检查及随机微型Gemma4 LoRA计算通过，已导入真实历史诊断语料；没有31B训练/GPU比赛结果。以下“交用户实现”的段落为此前研究交接，最新事实见[005进展](../progress/005-medal-and-prize-research.md)。

2026-10-10研究更新：[005冲牌与奖金路线](005-medal-and-prize-research.md)与[实现交接](005-implementation-handoff.md)已完成。通过Kaggle MCP核对最新官方数据、榜单、论文赛及wheelhouse，阅读八份公开Notebook与新增前沿研究。可核实公开源码参考为0.18/0.17；未取得榜首0.24的对应方案。相似精确编辑工具的三轮对照降低编辑错误但修复数打平，因此T2作为基础能力，研究重点转向修复层次/假设、状态保持与一次真实交接；同协议局部修复及错误恢复LoRA SFT上调为近期并行线。当前只有研究和原型，无新GPU成绩或训练权重。后文“004暂缓”等为10月1日及以前的历史排序，以005最新决策为准。

本项目研究指定Gemma 4模型在仓库级软件修复任务中的行为、代码定位效率和时间预算利用。001基线与003主实验已完成；003面板A0/B均3/24，C已回退。下一步按[9月30日研究交接](002-next-stage-handoff.md)推进002：先校准评估、上下文与工具执行，再研究文本/AST定位和按需图检索。004仍未开始训练。

2026-10-01最终资源决定：等待Kaggle的4×L4额度恢复；停止TPU路线，暂缓AMD迁移，不开启AMD实验。P002的0C容量对照已完成并保留16384，不重跑；冻结的12次T0/T1探索继续保留。[AMD交接](002-amd-gpu-handoff.md)仅作未来备选，不是当前前置任务。

2026-10-01晚间研究更新：已联合分析93份结果/92条轨迹、精读9个案例并核对官方信息与10项相关研究。当前首选研究“源码证据如何转成有效修改”：T2读取句柄与精确编辑，随后T3有区分力的最小反例，分层上下文另作增量。大量实现交用户，不继续以环境检查为等待期主线。见[研究结论与实现交接](002-research-opportunities-20261001.md)；这是候选设计，尚无新GPU成绩。

## 研究顺序

| 轮次 | 计划 | 核心产出 |
| --- | --- | --- |
| 005（研究与交接完成） | [冲牌与奖金研究](005-medal-and-prize-research.md)；[实现交接](005-implementation-handoff.md)；[进展](../progress/005-medal-and-prize-research.md) | 推理机制与同协议LoRA SFT并行，源码/文献/官方证据归档；大量实现交用户，尚无新成绩 |
| 001（已有实现与初步结果） | [无微调基线与失败分析](001-baseline.md) | 见[进展与证据复核](../progress/001-baseline.md)；结果保存与计时问题在003补齐 |
| 003（主实验完成） | [行为研究](003-budget-and-context.md)＋[进展与审计](../progress/003-budget-and-context.md) | 6题探索与24题配对完成；B面板无净增，未冻结采用 |
| 002（0C完成，探索冻结待L4） | [原阶段交接](002-next-stage-handoff.md)；[10月1日研究与实现交接](002-research-opportunities-20261001.md)；[实际进展](../progress/002-code-localization.md) | 保留16384；T0/T1探索后优先研究编辑转化，再做候选完整面板 |
| 004（暂缓） | [针对失败行为的 LoRA](004-targeted-lora.md) | 待执行环境稳定、目标行为与adapter兼容性明确 |

2026-09-30补充：联合载体已完成一次模型加载、逐题轮换与独立attempt归档。当前已恢复003的69条轨迹，66份产物清单中660项文件哈希核对一致；CPU复核发现上下文超限、编辑失败与工具参数循环不能都归为定位/时间问题。仍沿用[减少GPU排队的实验组织](003-queue-efficient-execution.md)，本次不运行新GPU实验，大量实现交由用户。

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
- 数据源挂载分三层验证（2026-09-29实测）：MCP的`save_notebook`通道挂不上数据源（6种组合元数据均无sources）；官方SDK SaveAndRunAll与CLI `kernels push`均可注册三项挂载（比赛/wheelhouse/模型v2，回读元数据确认）。元数据/页面显示挂载只证明“注册了挂什么”，不保证运行时布局。
- `/kaggle/input`布局随worker而变：A0-smoke worker为嵌套布局（`competitions/…`、`datasets/metric/…`、`models/…`），joint v2/v3 worker为扁平布局（`/kaggle/input/<slug>/`直接放文件，v3日志 input_files 清单570+文件证实）。载体不得硬编码任一布局，应候选+marker文件双重解析，解析结果记入run_manifest；缺失即fail-fast（约13秒），不要按固定路径假设继续。

## 官方入口

- [比赛概览](https://www.kaggle.com/competitions/gemma-4-developer-agent)
- [比赛数据与 HARNESS_README.md](https://www.kaggle.com/competitions/gemma-4-developer-agent/data)
- [官方 Getting Started](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent)
- [官方评估依赖包](https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse)
- [比赛规则](https://www.kaggle.com/competitions/gemma-4-developer-agent/rules)
