# T0 / single_state 四题同加载研究

2026-10-11，在 `main` 直接实现。私有实验：[P005 T0 Single State Pair](https://www.kaggle.com/code/ccjjllt/p005-t0-single-state-pair)。**v3为COMPLETE，八次尝试均有真实结果和轨迹：T0为1/4，single_state为2/4，新增修复1题、无成功题回退。** 四个已暴露开发题各只跑一对，不能据此认定总体修复率翻倍或有排行榜收益。未作比赛提交。

机器可读[分析](analysis-v3.json)、[原始逐题结果](output-v3/analysis_rows.json)、[完整运行原件ZIP](output-v3/run-output.zip)。ZIP SHA256为`9243a2c274d7e8f2995d05a798a558a618251ce464b3142b15142e8b7faa3840`，保留所有轨迹、补丁、官方裁判输出、采样及冻结源文件；展开目录只作本地阅读缓存。

## 逐题结果与实际行为

| 任务 | T0 | single_state | 原始证据解释 |
|---|---|---|---|
| rich_3942 | 未修复，88.22秒 | 未修复，303.82秒 | 两臂零补丁、未运行最终判分；T0上下文硬溢出，候选只读/搜索后超时。 |
| fastapi_14873 | 未修复，309.47秒 | **修复，309.35秒** | 候选修改APIRouter回调初始化顺序，10 passed；T0只有复现脚本，1 failed / 9 passed。 |
| fastapi_5077 | 未修复，308.66秒 | 未修复，308.72秒 | 两臂补丁都只有复现脚本，未改FastAPI源码；官方目标测试失败。不是已实施修复方案被证明错误。 |
| rich_3518 | **修复，101.83秒** | **修复，240.37秒** | 两臂529字符补丁逐字相同，均20 passed；候选多耗138.53秒。 |

**候选四题都没有调用新增技能。** 原始轨迹没有`load_skill`、`run_skill_script`或相应脚本执行；不能把新增修复归因于状态卡、句柄编辑或probe技能。此轮支持的是提示＋技能声明组合的一个探索性结果，实际动作仍来自官方基础工具。

- `fastapi_14873`：题干已给出父类初始化覆盖回调的原因，T0在step 10也提出正确修法。候选step 28（221.8秒）才真正改生产代码，之后验证回调恢复并保留自定义lifespan行为；T0经历六次畸形read_file参数/路径及两次错误lifespan复现API，反复读取，300秒内未完成生产编辑或提交。差异在执行落地，不能称为候选发现新根因。
- `rich_3518`：首次修复后复现通过几乎同时（T0约68.8秒，候选69.8秒）。候选随后扩展到全库Column、ProgressColumn、Table.grid并重复确认，231.7秒才提交；工具计数39比16、LLM调用41比20，没有得到不同或更好的补丁。
- `fastapi_5077`：T0 step 7–14连续八次write_file缺少filepath，浪费约99秒。这些调用未计入结果的tool_calls=11，却消耗了模型调用与时间。候选step 29约210.8秒已确认__wrapped__相关原因，随后继续追查ForwardRef旁支，未落地源码修改。候选还在sandbox里创建了test_annotation.py（不在最终归档补丁中），说明提示自身的“不新增测试”也未被稳定遵守；本仓库没有为本轮新建测试套件。
- `rich_3942`：T0广域grep后输入由4469增至9791，最后成功请求为13838；后一次请求因“至少16385输入＋16384输出预留＞32768”被拒绝，16385不是精确测量。候选有两次压缩后的输入下降，但23次动作均为读取/搜索，仍无补丁。题干仅泛化要求更新Markdown样式，不能假定消除溢出就一定能修复。

## GPU效率与额度

一次服务启动325.387秒，整轮wall time **1660.215秒＝27.67分钟**。前置setup＋服务启动427.658秒，占25.76%；评测阶段1221.361秒。评测阶段四卡利用率采样均值 **93.388%**（121次），包含启动的全会话采样均值 **69.118%**（165次），不能只报前者代表整轮。

评测采样中双请求73次（60.33%）、单请求38次、零请求10次，等待队列均为0；真实观察到最多两路请求并发。约每10.086秒采样一次，这些比例不等于连续busy时间或精确并发时长，也不代表每个生成token都有研究价值。

按官方4×L4的2倍系数，以脚本wall time估算本轮为 **0.92234额度小时**，不是独立账单。北京时间14:08读取账户较启动前增加3.44044h，但期间另有`arc2-011-s0-shared-model`（NvidiaL4，04:51:59Z开始）等工作，不能将该差额归给Gemma。该快照账户剩 **16.6049h**；详见[配额原件](quota-after-v3.json)。前两次失败也不能用脚本时长冒充精确账单。

## 下一轮研究决策

不直接扩大原候选，也不把SFT或新增技能当作这次增益原因。优先两个可以区分的问题：

1. 固定其余条件，仅比较输出预留16384与8192，观察上下文硬溢出及源码编辑是否改善。本轮所有已记录单次completion最多2214 tokens，8192有观察上的余量；历史8192实验也未显示总体成功率提高，因此这里只是待复测的运行边界修正。
2. 另比较精简的行为提示：工具参数错误后一次纠正或换等价工具；直接因果证据成立后尽快完成最小源码修改；原复现通过且相关行为保留后及时提交。先确认fastapi_14873的收益能重复，再检查rich_3518的额外耗时能否消除。保持预留八题和heldout不动。

GPU评测阶段已经较繁忙，下一步重点是减少无效模型调用并继续摊薄启动成本。无需为了提高利用率百分比启动更多GPU任务。本次结果审阅没有再启动GPU。

## 执行设计与额度

一次加载官方CT v2，固定Starter Python 3.12镜像、wheelhouse v30、4×L4张量并行。两个独立Evaluator子进程共享模型HTTP服务，各有独立工作区、会话、技能状态、日志和裁判输出。vLLM限制`max_num_seqs=2`，对应本轮请求并发上限；没有adapter。

按`rich_3942`、`fastapi_14873`、`fastapi_5077`、`rich_3518`顺序执行四波，同题两臂并发，启动顺序AB/BA/AB/BA。两臂采样、官方六工具声明及每题5分钟预算相同；single_state增加技能工具并重写提示，因此是整个机制组合对照。只解析和归档四题子集，选题依据见[task-selection.md](task-selection.md)。

共享prefix cache和并发吞吐竞争可能影响五分钟内能完成的动作；AB/BA不能使缓存条件完全相同。四个已暴露开发题、每题一对，只用于机制探索，不是独立泛化或奖牌收益证据。

v3对安装、编译、模型加载及评测设置2550秒期限，留时间终止后代进程和保存结果；Kaggle会话缩短为2700秒硬限。4×L4按官方2倍计费，单次完整会话上限约1.5额度小时；前两次失败的整个账户增量也仅0.3363h（其中含ARC），三版仍按本轮2小时计划预算安排，不自动启动额外研究。详细使用方法见[载体说明](../../../notebooks/005-t0-single-state-pair.README.md)。

v1启动前账户余额约21.976h，v2提交前约21.884h，原始读数见[quota-before.json](quota-before.json)、[quota-before-v2.json](quota-before-v2.json)。另有独立ARC RTX任务同时运行，账户总差额不能全部归给Gemma；需同时报告本实验wall time、阶段利用率和请求并发样本。排队时长也不能当作模型加载时长。

## v1：安装前退出，已修复

v1载体SHA为`a6eb94d87e06d3031ac017ba94bbb3ca8bd1f3753ed732a995548b7d0f3a8c45`。脚本运行4.120秒，0个attempt，无模型加载；MCP清单的41项`total_bytes`全部是字符串，与`stat().st_size`整数比较导致误报尺寸变化，文件名称没有缺失或多余。

已在生成器统一转换为`int`。手动核对41项总计880,842,253字节、两个本地官方wheel实际大小，以及tokenizer/tasks的字节数类型；没有新增测试文件。v2载体SHA为`d777187981d07e6e3d50c32dee177ebeb35b51d00151a47214bb44d77e52101a`，重推记录见[launch-v2.json](launch-v2.json)。

v1原始[coordinator.log](output-v1/coordinator.log)、[supervisor报告](output-v1/supervisor-report.json)、[提交源文件](output-v1/submitted-pair.py)保留。4.120秒只是脚本wall time，不能冒充该会话精确账单用量。

## v2：旧导入路径不兼容，先转CPU准备

v2安装41个官方wheel后，在`from swegemma.models.discovery import validate_single_declared_model`处退出。该模块/函数不存在于当前v30，之前载体继承此旧接口却未实际走通。脚本wall time为129.901秒，模型仍未加载、0个attempt。原始[coordinator.log](output-v2/coordinator.log)、[supervisor报告](output-v2/supervisor-report.json)、[提交源文件](output-v2/submitted-pair.py)保留。

修正应使用官方`adk_submission.discovery.discover_declared_models`的集合结果。下一次GPU前先以同镜像的CPU会话实际执行安装、全部导入和两臂配置构造；一次性核对代码放临时目录，不建立测试套件或新的门控脚本。v2后账户余额约21.640h，仍含ARC并发消耗，不能把账户差额称作本实验用量。

## 免费CPU实际准备与v3

私有`ccjjllt/p005-pair-cpu-prepare`使用同镜像、同输入，明确`enable_gpu=false`，真实设备列表为空。一次性命令仅放`/tmp`，在两个子进程中构造官方ModelRegistry/EvalConfig/Evaluator/Context，创建隔离sandbox并调用官方`compile_submission`，没有启动服务、模型请求或裁判。

CPU v1实际编译T0成功，single_state因三技能下划线名称被ADK默认规则拒绝。[原始日志](cpu-preparation-v1.log)。已将候选技能目录、frontmatter及引用迁移为`source-edit`、`working-state`、`source-probe`；8个Python脚本字节不变，不开启额外feature flag，不修改冻结T0或历史产物。

CPU v2为COMPLETE，两臂均真实编译为LlmAgent，候选包含SkillToolset，模型请求/裁判运行均为0。[原始日志](cpu-preparation-v2.log)。这证明实际接口构造可用，不代表GPU推理或修复收益。

CPU准备前后账户用量均为35836.848秒，未消耗GPU配额；v3启动前余额20.0453h，之前的ARC任务已COMPLETE。[配额记录](quota-before-v3.json)。v3载体SHA为`49d3ed0c3c1c8e597fcf1256af7eb12a185530ae9c0217c03b19c266cb014e8c`，[启动记录](launch-v3.json)保存期限和来源。
