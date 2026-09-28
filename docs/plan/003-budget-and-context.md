# 003：固定预算下的无进展识别与收尾策略

更新日期：2026-09-28。承接[001进展](../progress/001-baseline.md)，替代原先“优先比较读取范围”的安排。本轮只确定研究与实现交接，尚未修改运行代码或获得新实验结果。

详细调用链复核见[003运行载体修正与证据补全](003-runtime-reliability.md)。先落实R01–R19的现有实现修改，并通过可取得的补丁和小批A0诊断固定共同环境，再做行为对照。原001补丁缺失不阻塞其他修复，具体归因保留未知，新A0诊断另行命名；不以文档补充代替修复结果。

## 研究依据

001记录中，FastAPI反复执行相同复现脚本且没有生产源码修改；Rich生成通过验证的补丁，但直到超时才自动回收；Requests主动提交了补丁，最终验证却以124超时。优先研究能否用相同5分钟预算减少无效动作并及时完成已有修改。

三题已参与研究设计，只能用于发现问题，不能推断全部任务的定位能力或解题率。Rich的自动回收已经获得成功，因此主动提交率不能替代比赛解决率。同时需先修复结果保存和计时口径，才能区分策略效果与验证问题。

| 事项 | 当前事实与边界 |
| --- | --- |
| 基线配置 | 指定QAT模型、无LoRA、6工具、温度0.2、50次调用、80轮、5分钟Agent预算、timeout_seconds=60 |
| 三题结果 | 用户记录为FastAPI未解决、Rich解决、Requests验证超时；原始trace与patch未在本地归档 |
| 旧Notebook读取 | 本轮再次访问仍收到kernels.get拒绝，没有独立核对原始补丁和轨迹 |
| ROWS=0 | evaluate_task只返回结果，当前逐题调用后未执行官方持久化函数 |
| 补丁长度 | 官方JSONL是agent_patch_size，不包含agent_patch正文；现有汇总会误报空补丁 |
| 60秒限制 | 当前官方0.2.7源码中，同一timeout_seconds限制Agent命令与最终验证pytest |
| 返回时长 | duration_seconds包含准备、Agent和验证，不含外部模型服务启动 |
| 读取范围 | read_file核心与wrapper均支持start_line/end_line；实际异常还需原始参数、schema与响应 |
| 正式运行 | 主办方9月28日确认12小时超限报错未修复，不支持并行模型调用 |

源码依据：官方`swegemma-0.2.7-py3-none-any.whl`，111548字节，SHA-256 `2b74d402dc95a615f3b36c56fd3f9ba11ddbfb7db8a5a32ec259369a0f559d98`。实际运行仍应记录包指纹，版本号相同不保证未重打包。

## 研究问题

1. 保持预算与模型不变，识别重复无效动作并切换假设，能否增加实际解决数？
2. 增加很短的软收尾阶段，能否减少拖延，同时保留最后时刻完成有效修改的机会？
3. Requests超时主要来自验证预算、环境还是补丁本身？与提示词实验分开诊断。

首轮固定图工具、模型权重、上下文压缩、读取窗口和采样参数，不引入多Agent或新的模型服务。

## 第一部分：结果持久化与验证诊断

### 1. 逐题立即保存

修改`notebooks/001-baseline.ipynb`的逐题执行和汇总单元格，以及`notebooks/001-smoke-single-cell.py`的对应部分。继续逐题调用`Evaluator.evaluate_task`；返回后先把结果加入当前批次`running_results`，再调用：

`swegemma.results.append_task_result(result, output_dir, running_results)`

该官方API已通过`results.py:39–88`核实，写入JSONL、非空patch、非空test_output和累计summary；官方批量入口`Evaluator.run()`也是显式调用它。无需替换评测器，也无需将逐题执行改成自动批量调度。

结果汇总应读取`agent_patch_size`，并与`patches/<instance_id>.patch`核对；保留`resolved/test_exit_code/duration_seconds/error/tool_calls/total_llm_calls`原值和完整`test_output`。traces补充行为时间线，不能替代裁判结果。

每次使用独立`results/003/<run_id>/<variant>/`。Notebook重跑时创建新run_id或明确跳过已完成attempt，避免重复追加和同名trace覆盖。一题异常也立即记录任务ID、错误和已有产物；缺失结果不能填成0或成功。3题执行却ROWS=0应报告结果采集失败。

### 2. 最小运行清单与分析字段

项目自定义`run_manifest.json`保存任务ID与顺序、variant、配置/提示词指纹、数据指纹、模型版本、wheel版本/指纹、Python/GPU、沙箱、外层起止时间和产物路径。运行结束归档完整run目录及配置，不只保留submission.zip。

官方格式保持不变，研究分析另存`analysis_003.csv`：

| 自定义字段 | 来源与含义 |
| --- | --- |
| agent_end_reason | 原始退出事件或trace；无法取得时留空，不从resolved倒推 |
| verification_status | passed/failed/timeout/execution_error/missing；另保留原始退出码 |
| production_source_changed | 依据实际diff；测试与临时文件另标记，不从补丁长度推断 |
| explicit_submit | 是否实际调用submit_patch，仅作为过程指标 |
| repeated_no_progress_actions | 按下文定义人工计数，不把全部重复命令当错 |
| first_source_edit_at | 首次生产源码修改时间，缺失不推算 |
| task_duration_seconds | 官方duration_seconds，包含准备、Agent和验证 |
| agent_duration_seconds | 每题新建公有SwegemmaContext并传给evaluate_task，从冻结的agent_elapsed_seconds读取；接口不可用时才退回有证据的日志，不能把最后一次get_status值当完整时长 |

任务清单中的每题应有返回结果或明确异常；JSONL里缺失正文不代表空补丁。源码中的补丁size是字符数，不自行改称文件字节数。

配置、日志、zip和结果均按run/variant/repeat隔离；每题重试另有attempt标识。官方append不是多文件原子事务，恢复必须核对patch/test_output等产物完整性，不能只凭JSONL有ID跳过。具体实现与分段时间接口见补充文档R04、R08–R10、R13。

### 3. 冻结Requests补丁诊断

优先从原working目录取得`requests_6592`原补丁和完整验证输出，记录指纹。取不到则标为缺失，新生成补丁不能冒充001原补丁。

人工使用官方已有`verify_task`在干净沙箱验证同一补丁，分别观察60秒与300秒验证时限，其他输入一致。只调用现有验证能力，不重新请求模型生成补丁，不更改测试内容、用例选择或评分判断。

真实接口位于`swegemma.harness.verification`：`verify_task(docker, config, task, snapshot_path, *, base_snapshot_path=None, patch_path=None, fast_path=True, agent_patch='', agent_error=None, trace=None, start_time=0.0)`。复用该版本的快照路径解析与沙箱管理，传原始补丁和真实计时起点；默认start_time=0不能产生有意义的时长。分层快照和完整快照的参数必须对照源码，不猜路径。

准确的配置、manager创建和快照解析顺序见补充文档R16。特别注意patch_path是仓库快照增量路径，Agent补丁通过agent_patch文本传入；skip_agent_patch=True只会生成空补丁，不能用来回放。直接verify_task的计时和调用统计不代表原Agent运行。

必要时仅对已用于开发的`requests_6592`使用未修改仓库与公开参考修复作环境对照，结果不计入Agent成绩。参考答案和诊断日志只供研究分析，不注入A0/B/C的提示、会话或保留任务输入。遇到证书、网络等待、导入或依赖问题，保留具体错误。社区已报告包括`requests_6592`在内的若干任务存在版本/证书问题，但来自不同环境，不能直接当作本次失败根因。

60秒超时而300秒通过，支持本地验证预算不足；两者都失败则继续区分补丁与环境；参考修复也失败时不能据此判模型能力。诊断结果单列，不覆盖001原始resolved。

环境对照还应检查未修改版本是否按预期失败、必需测试是否真正运行及官方JUnit判定。空补丁已经通过、零收集或关键用例跳过也会破坏修复收益判断。subprocess继承宿主环境，fast_path=False不等于完整重建依赖；参照R17–R19记录包导入路径、缓存和数据读取边界。

### 4. 固定A0条件

A0是本轮共同修订基线：统一环境、记录、输入和生命周期修复，并统一消除提示冲突。现有提示“每题必须产生源码修改”与没有证据时不制造修改矛盾，应改为依据实际证据；明确不将临时复现文件混入补丁、只使用已注册工具、处理字符截断，并在成功提交后停止编辑。这些共同修正会改变001行为，必须显式记录差异并重跑A0，不能仍称与001完全相同。

所有组共享相同的get_status观察节奏：阶段切换、长命令前和修改完成后查看状态，不在C中独自增加观测频率。C只增加余量不足时的收尾决策。默认仍为5分钟、50次调用、80轮和原采样；若验证时限需要改变，先统一配置再运行候选。

不能将主实验timeout_seconds从60改300称为“只改裁判”：它也改变Agent命令时限。调整后A0与全部候选必须一致；001只作历史参照，不作严格配对对照。不修改官方harness来制造只有本地可用的提交参数。

## 第二部分：固定5分钟的行为对照

| 组别 | 相对变化 | 比较 |
| --- | --- | --- |
| A0 | 共同修正后的基础提示词与可靠运行记录 | 本轮重新取得的基线，001仅作历史参照 |
| B | 仅增加重复无进展识别和换方向要求 | B对A0 |
| C | B再增加20秒软收尾要求 | C对B |

三组固定模型、6工具、5分钟Agent预算、50次调用、80轮、采样、沙箱和统一命令/验证时限；读取、压缩与生成上限不同时调整。

### B：无进展定义

自上次相关源码、输入或可检验假设有实质变化以来，同一动作反复得到等价结果且未增加可核查证据，就属于待识别的无进展模式。除相邻重复，也应识别“执行脚本→读同一文件→再执行脚本”等短循环；输出时间戳、打印顺序等噪声不算新证据。

以两次无新增证据的重复动作或短循环作为首轮触发候选。模型应简述已知事实、已排除假设和下一项可检验的不同方向，再换范围、检查调用方或修改已有明确问题；只换一种说法不算推进。触发后换方向，不自动放弃整题。改代码后的复跑、阅读真正未显示的下一段、等待确有变化的状态不属于无效重复。人工标注保留参数、相关源码状态和新增证据。

这是提示词策略，不是假定可向提交包注入Python控制回调。根据trace记录遵守程度，写入提示词不等于可靠拦截已实现。

### C：软收尾而非提前强停

模型已知剩余不超过20秒时，停止扩展新的调查方向，优先审阅现有生产源码diff并submit_patch。没有补丁时，只完成证据充分的局部修改，不为提高提交率制造无依据修改。

使用各组共同固定的get_status观察节奏，避免在剩余时间内启动明显做不完的工作。20秒是实验候选值，不是最优阈值，也不是可靠硬截止控制器。免费状态/提交工具仍可能因到期被拒绝；余量包含模型调用、diff、提交和最终纯文本回复，需按实测判断。

Rich记录中关键修改约在270秒，提前60秒硬停可能损失成功，因此不采用固定240秒终止。保留普通超时后的官方自动diff回收；主动提交率上升而resolved下降时判为退步。

## 实验规模与顺序

1. 原3个smoke先观察A0产物是否完整、诊断是否可解释；它们仍是开发数据。
2. 从`dev_future`按固定种子20260928、以(repo,base_commit)组为单位预先选FastAPI、Rich、Requests各1题，不看成功率或参考补丁挑题。当前future各组为单题；后续版本有多题组时整组登记已暴露。加原smoke组成6题探索集；保留原task_splits，另登记使用范围。按问题ID、commit关系等检查明显近重复，不向策略设计侧展示heldout答案。
3. 6题比较A0/B/C，共18次运行，配置的Agent预算合计90分钟；同步工具、收尾等可能超出名义时限，另加服务启动、准备、验证和重试，不能称为墙钟硬上界或额度承诺。模型服务可复用，仓库与会话必须重置，不共享其他组答案。六题预先分别采用ABC六种不同排列各一次，平衡组别先后，并记录冷/热缓存状态。
4. 探索后最多选一个候选，在固定24题dev_panel与A0配对比较。面板为FastAPI12、Rich8、Requests4；3个smoke已参与设计，与其余21题分开报告。
5. 24题配对顺序预先平衡：A0先和候选先各12题，尽量在仓库内均衡。首轮完整面板是主结果；仅对胜负变化重跑只能作为案例稳定性，不能选择性并入总体平均。若报告重复运行总体结果，必须在首轮结果未知前固定随机子集与次数，或统一复跑整面板；不能看结果后挑选再并入总体。温度0.2有随机性，记录seed但不宣称完全确定。
6. 本轮通常到开发面板确认即止，保留28题heldout供后续002/004及最终方案完成后一次冻结评估。其组成是FastAPI13、Rich11、Requests4，共26个独立快照组。若提前使用，其结果影响后续方向或参数时就必须登记为已暴露评估，不再复用“独立保留集”的称呼。HTTPX单例另作探索。

先不加长Agent时间。如果B/C仍在有效修改阶段稳定耗尽预算，再单独比较5分钟与更长时限，其他条件不变。长时限仅作诊断，不直接全量采用：官方数据说明约120个隐藏任务，120×6分钟已经占满12小时且尚未包含准备与启动。

## 指标与判断

- 主指标为相同任务上的resolved数和配对胜/负/同结果。验证超时保留原始分母，另列验证状态分布。
- 效率指标使用实际准备＋Agent时间，并单列共同的模型服务启动等固定成本，按官方计时范围计入整次预算，不能默认排除。正式12小时不计Phase2验证，本地GPU与用户等待的完整墙钟另报，不能混用；不可简单把duration减去估计验证时间就当成实测阶段耗时。
- 重复次数、生产源码修改比例、首改时间、主动提交/自动回收、超时与工具错误用于解释，不替代resolved。
- 定位失败、错误修改、验证超时、临时文件污染和读取异常分开记录。扩大面板后若定位成为主要瓶颈，再提升002优先级。
- 小样本净增或单次胜负不宣称统计显著，不外推正式排行榜分数。

如果C过早收尾导致解决率下降，回退C并保留B的独立结果。若只是更早结束而没有增加解决数，报告实测节约时间与不确定性，不宣称代码修复能力提高。

## 实现位置与产物

| 位置 | 工作 |
| --- | --- |
| `notebooks/001-baseline.ipynb` | 逐题持久化、run_id、正确结果字段、计时标签、归档 |
| `notebooks/001-smoke-single-cell.py` | 同步修复，内嵌配置与仓库一致 |
| `notebooks/003-budget-and-context.ipynb` | 复用既有逻辑，组织人工选择的A0/B/C及冻结补丁诊断 |
| `research/003/experiment_manifest.json` | 探索任务、组别、顺序、重复约定、公共参数与指纹，均为项目字段 |
| `research/003/variants/<variant>/agent/` | 各组独立提交目录，仅一个根agent.yaml；单组打包 |
| `results/003/<run_id>/<variant>/` | 官方结果、补丁、验证输出、trace、manifest和分析表 |

保留`agent/`的001配置作为可回溯基线，从同一权威配置生成需要的内嵌文本，避免精简提示词与正式提示词不一致。候选配置不能全混入同一个submission.zip。

后续实现和结果写入`docs/progress/003-budget-and-context.md`，记录修复、诊断、逐题配对、实际资源、证据限制以及是否进入002/004。当前不创建虚构结果。

## 来源

- [001进展与复核](../progress/001-baseline.md)
- [官方wheelhouse](https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse)：本轮只读源码，未安装或执行。
- [官方数据与HARNESS_README](https://www.kaggle.com/competitions/gemma-4-developer-agent/data)
- [9月28日主办方：12小时、并行、LoRA、自生成数据](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743964)
- [主办方对公开subprocess环境的说明](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/742882)
- [社区验证问题报告](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743973)：不是对001原运行的复现。
- [官方依赖与ADK版本说明](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743800)：以wheelhouse为准，不支持ADK2.x。
