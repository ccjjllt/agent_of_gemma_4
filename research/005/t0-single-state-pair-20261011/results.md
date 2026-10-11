# T0 / single_state 四题同加载研究

2026-10-11，在 `main` 直接实现。私有实验：[P005 T0 Single State Pair](https://www.kaggle.com/code/ccjjllt/p005-t0-single-state-pair)。v1/v2均在模型加载前失败；免费CPU已实际编译两臂，修正后的v3于北京时间12:24提交，尚无修复成绩。未作比赛提交。

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

## 收到输出后的研究重点

逐题核对官方resolved、新增修复和回退，再读取失败轨迹，区分环境异常、上下文溢出、工具参数重复、编辑未生效、预算耗尽与真实错误补丁。不要把COMPLETE、正常退出或工具返回ok当作修复成功。

同步读取`gpu-telemetry.jsonl`和`supervisor-report.json`，区分setup/server_start/evaluation。evaluation包含工具、环境和裁判，并非纯GPU解码；采样均值也不等于完整GPU忙碌率。以有效修复尝试/额度小时衡量投入，而非单独追求利用率百分比。
