# 003补充：运行载体修正与证据补全

核对日期：2026-09-28。本文件补充[003行为研究计划](003-budget-and-context.md)，将现有实现和官方源码中已经定位的问题落实为修改要求。它不是完成报告；本次没有修改运行代码、安装评估包或运行实验。

## 核对范围与版本

已逐项阅读两个001运行载体、四份Agent配置、任务划分、全部研究计划和001进展，并沿调用链检查以下官方包：

| 资料 | 版本与指纹 |
| --- | --- |
| 官方wheelhouse | 本次元数据为版本25，更新时间2026-09-27T11:49:10.890Z |
| swegemma | 0.2.7；SHA-256 `2b74d402dc95a615f3b36c56fd3f9ba11ddbfb7db8a5a32ec259369a0f559d98` |
| adk-eval-core | 0.1.0；SHA-256 `b1608c4471a2739eca6469735e4e118ff33a20823ce9430719238ea843aa9d1a` |
| adk-submission | 0.2.11；SHA-256 `d14f7d31e724f507322c763bc7c1ed56ac603085b3709afed954e50006b494a8` |
| google-adk | 1.36.1；SHA-256 `1a2f6868c509e3151fb0de3575a7d18b45c338be86f420924dad74e7193631a0` |

重新读取了官方Getting Started v2、比赛内容页、最新讨论，以及主办方对依赖、并行、总时间、LoRA和自生成数据的答复。文件和包版本应在实际运行时再次记录；当前同版本源码不能证明001内存中实际加载的内容完全相同。

下面区分：现有实现已确定的问题、官方实现的条件性风险、尚需原始运行证据确认的现象。旧三题的完整trace和patch仍未取得，不能将静态检查写成远程问题全部已修复。

## 一、先修输入与环境

### R01：安装前导入导致内存与磁盘版本不一致

现有多单元格Notebook在安装前导入torch、评估库和模型服务库，之后又强制重装。`invalidate_caches()`不会替换已经存在于`sys.modules`的对象；单cell在已用过的kernel中重跑也有同类风险。

改动：最初的环境检查只用标准库和distribution元数据；安装完成后再导入GPU/Agent栈。若当前kernel已经加载将被重装的模块，应完成安装后由使用者重启kernel，再进入执行阶段，不用循环重装来代替重启。将安装与执行阶段明确分开，并记录当前执行是否来自重启后的干净kernel。

观察依据：已加载模块路径、distribution版本和本次安装清单对应，模型服务子进程与Notebook使用一致的环境；导入失败立即显示具体异常，不只打印MISSING后继续。

### R02：wheel staging和解包缓存会保留旧版本

两个载体复用`/tmp/wheelhouse`，已有链接就跳过，随后安装目录内全部wheel；更换来源后可能混入旧包或悬空链接。官方`swegemma/harness/container_setup.py`的解包缓存也只按文件名复用，不包含wheel内容指纹。

改动：以来源版本和内容清单创建新的staging目录，只安装该清单内的wheel；检查源目录非空、链接有效和同一distribution没有意外多版本。记录wheel文件名、大小和SHA-256。依赖集合变化时在新的干净运行环境中执行，不能沿用旧缓存后判断新环境仍失败，也不批量删除其他会话缓存。

仅打印版本不够。用`importlib.metadata.version`读取`google-adk`、`adk-submission`、`adk-eval-core`、`swegemma`、`vllm`、`litellm`、`torch`等实际distribution版本；`__import__('google.adk')`返回顶层google，不能据此打印正确ADK版本。

### R03：任务清单缺失或不一致不能静默换题

现有Notebook只打印数据指纹；缺失划分时回退到`tasks[:3]`或`tasks[:24]`，ID不存在时静默跳过。任何非`smoke`的scope还可能直接进入24题。单cell与多cell的smoke顺序也不同。

改动：使用单一有序manifest；核对`tasks.jsonl`与`task_splits.json`的指纹，检查ID唯一且全部存在、repo/commit一致、集合完整互斥、同commit组不跨集合。scope使用明确枚举，未知值报错；不得自动回退、丢题或按首N条替换。启动前固化完整任务ID和顺序。

官方数据更新时先记录差异、形成新的数据与划分版本，不能只更新期待哈希而继续声称与旧实验完全可比。

### R04：配置、结果和打包产物都需要独立归档

现有载体会删除固定`agent_run`目录，复用`results/`和`submission.zip`。重跑可能覆盖配置、trace或误读旧结果。

改动：采用`results/003/<run_id>/<variant>/<repeat_id>/`，其中保存本次配置快照、输入manifest、结果、日志和分析。每题重试使用新的attempt标识；完整批次重跑使用新的repeat目录。已有实验目录不直接删除或覆盖，配置源目录与执行目录不可相同或互为祖先。

官方patch/test/trace文件名只含任务ID，因此每次重试必须对应新的实际results_dir（或新repeat/run），不能只在manifest里增加attempt_id而继续写旧目录。汇总通过manifest引用各attempt原件。

打包产物同样带run/variant标识。先写manifest再开始，归档不依赖整个批次成功结束。

### R05：明确GPU能力与模型身份

无CUDA时现有代码仍把gpu_count置为1；“开发用单/双L4即可”的注释没有实测依据。本项目目前可复现的已记录环境是4×L4。

改动：无CUDA时在加载前停止；记录GPU型号、数量、显存、驱动/CUDA、TP、dtype、模型资源版本及服务启动日志。其他卡数或上下文配置作为独立可行性问题，不自动沿用同一成功承诺。

`validate_single_declared_model`只保证模型声明唯一，不保证等于比赛指定模型。显式核对别名为`gemma-4-31b-it-qat-w4a16-ct`，记录模型config/tokenizer和挂载版本指纹，以及服务实际模型列表，避免错误别名被映射到固定路径而掩盖拼写错误。

### R06：使用已确认的AdapterManifest接口

0.2.11的真实字段为`AdapterManifest.adapters: dict[str, AdapterInfo]`。001应明确检查该映射为空；不能通过任意`__dict__`长度或未知对象默认0来判断没有adapter。

使用已固定版本的官方发现和配置验证机制检查整个Agent配置图，包含根、引用、工具与子Agent。遇到未知API形状记录版本并停止配置加载，不猜测兼容。始终从同一权威配置生成内嵌文本，用文件指纹核对执行快照。

## 二、补齐服务与产物生命周期

### R07：服务启动、复用和释放

已核对`adk_submission.server.BaseInferenceServer`，提供`start()`、`stop(timeout=10.0)`和上下文管理器。现有载体在成功、异常或中断后没有显式释放路径，重复创建对象还可能遇到被旧服务占用的8000端口。

改动：记录本批次服务对象、PID、日志路径和配置；模型启动与执行置于有明确`finally`收尾的范围。复用前核对服务所有权、模型版本和配置，不能只看health成功。只停止本批次创建或明确归属本研究的服务，不使用无差别杀进程命令。

对正常完成、单题错误、使用者中断分别保留结果和释放资源的路径。服务日志使用run专属路径，停止后观察所属进程是否退出；遇到无法清理时记录状态，不冒称已释放GPU。多单元格Notebook需提供清晰的人工结束单元格。

该版本stop()会删除自己的log_path，启动失败也可能先stop再抛异常。必须在正常stop前复制日志到不同的永久归档文件；启动异常保存ServerStartupError.output及已有错误信息，不能把会被删除的log_path当成归档已完成。日志归档、结果保存和停止按这一顺序组织。

启动超时分支可能在调用者收到异常前就删除日志，因此单靠外层except再复制并不完整。需要时对本地服务管理增加仅负责归档的薄封装，使内部stop路径也先保存现存日志，再委托官方stop；不改变模型请求或评分流程。异常output为空且原日志已丢失时明确记录证据缺失，不能声称已保存完整启动日志。

### R08：逐题持久化，不让一次错误抹掉整批

保持逐题`evaluate_task`，每题返回后调用官方`append_task_result(result, output_dir, running_results)`，累计列表必须包含刚返回的结果。读`agent_patch_size`，正文从patch文件读取，保留完整test_output和trace索引。

append不负责保存trace；verify_task部分异常路径虽返回result.trace，却未写trace文件。返回后核对真实文件，若对象存在但文件缺失，使用当前官方`swegemma.harness.verification.save_trace_artifact(trace, results_dir, instance_id)`补存并核对结果。只有索引、空路径或保存失败都不算已归档；异常发生在对象生成前时如实记录缺失。

每个计划任务记录planned/running/completed/error/interrupted/not_run等执行状态；这些是项目字段，不替代官方resolved。官方可能抛`ContainerSetupError`，不能假定每题都有TaskResult。异常逐题记录，中断保留现场，不自动无限重试。

### R09：处理部分写入与安全恢复

官方append顺序为JSONL→patch/test_output→summary，不是一次原子事务。只在JSONL看到ID不足以判断产物完整；当前逐行解析也会因坏尾行使整批汇总失败。

改动：保存本attempt产物清单及完成标记；恢复时核对任务与配置指纹、非空补丁/输出是否存在、字符数及文件指纹。仅完整attempt可以跳过。坏尾行保留为损坏证据，单独报告，不静默丢弃或把旧目录内容算进本轮。汇总从当前选定attempt集合重建。

观察依据：计划题数可与完整返回、明确异常和未运行项逐一对应；没有数据时报告缺失，不生成看似有效的0分汇总。

### R10：打包的必须是实际运行过的单组配置

现有检查只覆盖根文件、扩展名和大小。补充核对全部include、子配置引用、无越界路径和symlink、单一根配置及官方配置验证结果。以冻结单variant目录打包，核对压缩包文件清单与执行快照哈希，并记录zip SHA-256。

Notebook、参考答案、日志、其他候选配置和训练数据留在包外。不要把本地改过的harness打包并期待正式评分采用它。

## 三、补齐工具、时间与提交语义

### R11：记录真正提供给模型的工具和初始上下文

官方`build_agent_prompt`只看图文件是否存在，就可能注入图工具说明，不检查根Agent实际工具列表。001有图目录，但只注册6工具。是否干扰原三题仍需初始prompt证明。

共同配置明确以实际工具schema为准，不调用未暴露工具；保存实际初始prompt和工具schema，检查与6工具设置一致。不能仅在本地隐藏图目录后声称提交已获得同样环境，因为正式评分的图目录不由提交控制。

记录有效生成设置及服务启动配置。`litellm.drop_params=True`下，YAML存在某字段不等于请求实际采用该字段；无法从现有日志核对时标明未知，不声称已验证thinking预算或所有采样参数生效。各组使用同一服务和相同参数路径。

### R12：read_file范围与字符截断

已沿`budget_gated`、具名工具注册和Google ADK的FunctionTool检查，合法的start_line/end_line没有被这些wrapper删除；ADK会过滤不在签名里的字段，因此模型传错键名时可能退回默认范围。

另有确定边界：`workspace.py`先选行范围再截断字符，字符截断后返回的end_line没有按实际文本末尾调整。直接从end_line+1续读可能跳过未显示内容。

实现时保存原始function-call参数、响应start/end/is_truncated、文本长度和当前文件版本。发生截断时缩小窗口、有重叠地重读并核对源码，不能只依赖返回end_line。确需用shell限定行读取时，先记录工具故障证据，并把备用策略共同应用到各组或单独列为实验，不让它只改变一个候选。

### R13：分段时间用公有context接口记录

官方支持`Evaluator.evaluate_task(..., context=ctx)`。每题创建新的`swegemma.context.SwegemmaContext`，参数复用当前运行配置：`docker_manager=evaluator.docker, task=task, graph_dir=config.graph_dir, embeddings_dir=config.embeddings_dir, budget=config.budget, harness=config.harness`。

在调用前、返回或异常后记录外层perf_counter，保留ctx的task_start_time、agent_start_time、agent_end_time、agent_elapsed_seconds、patch_submitted及调用计数。只在对应时间戳存在且口径一致时作差。

Agent结束到整个调用返回，还包括补丁回收、A清理、B准备、验证和清理，不能直接命名为纯pytest耗时。未取得更细记录时保留组合阶段。官方duration仍保存原值；共同启动成本另列。

### R14：免费工具不是截止时间后的保障

get_status和submit_patch不增加工具次数，但仍经过时间检查。run_command近截止时还有5秒最小时限；同步工具可能阻塞事件循环，外层asyncio超时不保证精确在300秒停止。提交内部git操作也需要时间。

20秒只能作为软策略候选，配置的18次×5分钟是预算总额，不是墙钟硬上界。记录最后一次状态时间、模型等待、提交成功响应、patch_submitted以及最终纯文本完成响应。只看到一次submit_patch调用不能判定提交成功。

### R15：提交后停止修改，并分离退出原因与裁判结果

官方成功提交后使用已捕获的submitted_patch；只有未成功提交时才从工作树回收。先提交占位再继续改文件可能丢失后续修改。

共同提示要求完成修改和必要检查后提交，再返回简短纯文本，不继续工具编辑。成功resolved时官方error可能被清空，失败时error也可能携带Agent错误而非裁判根因，因此分别保存Agent结束状态、测试退出码、完整输出、JUnit相关结果和resolved。

## 四、冻结补丁与环境诊断的准确流程

### R16：回放不能用skip_agent_patch代替

`skip_agent_patch=True`会生成空补丁，不会重放原补丁。冻结诊断依次执行：

1. 从官方数据选已用于开发的requests_6592，读取原始Agent补丁，记录字符数与SHA-256。缺失原件则明确标记，不能用新运行替代后仍称旧补丁。
2. 分别构造timeout_seconds=60、300的EvalConfig，通过Evaluator(config)取得相同类型的manager。创建新配置，避免旧flat timeout_seconds覆盖手动修改的harness字段。
3. 调用`swegemma.deduplication.resolve_task_snapshot_paths(config.snapshots_dir, task.instance_id, task.repo)`取得snapshot_path、base_snapshot_path、incremental_patch_path。
4. 调用官方verify_task，snapshot相关三路径各归其位，`patch_path`是快照增量，绝不是Agent补丁；冻结文本传agent_patch，agent_error=None、trace=None、start_time=time.perf_counter()，保持官方fast_path语义。
5. verify_task自行创建和销毁干净B沙箱，不传旧A工作区。保存到独立诊断目录，最后清理manager；保留原Agent结束信息作为旁注，不混入新的裁判判断。

直接verify_task的duration只覆盖此次验证流程，调用次数也不代表原Agent统计，不能与整题时长直接比较。

若001原补丁始终取不到，保留其具体归因未知，不阻塞其余载体修复。可以用修正后A0新产生、另有指纹和命名的补丁执行同样诊断；明确这是新实验，不是001原件回放。先以小批A0取得新证据，再冻结各候选共用的验证设置。

### R17：验证对照要同时看修复前和修复后

如需使用公开参考修复诊断环境，仅限已用于开发的任务。需要看到未修改版本按预期失败、参考修复通过，且官方resolved/JUnit判断有效。空补丁已经通过、零测试收集、关键用例跳过或参考修复也失败，都不能作为模型修复收益证据。

保留测试补丁应用、收集、导入、证书/外部等待、超时、断言失败等类别。不要用exit=0覆盖官方resolved，不修改测试期望或跳过不利用例来获得通过。诊断不覆盖001原结果。

### R18：subprocess与Docker环境差异

当前源码中subprocess跳过editable package和test dependencies安装，继承宿主site-packages并写入宿主路径。对它设置fast_path=False不会获得完整Docker式依赖重建。

记录沙箱后端、实际Python解释器、任务包的导入__file__、宿主与离线依赖、缓存状态；在失败日志中确认运行的是工作区代码还是已安装发行包。ensurepip警告未阻止启动，不等于依赖与导入一致性已经验证。

官方最终B流程会尝试恢复部分测试/配置并应用test_patch，包含pyproject.toml、setup.cfg、conftest.py、pytest.ini等保护路径处理；不能仅凭代码中有恢复步骤就假定每次全部成功。原地脚本通过不能代替B通过；如问题需修改受保护配置，记录正式可表达性限制，不尝试绕过保护或更改评分代码。

### R19：实际输入边界与污染识别

任务文件包含patch/test_patch，subprocess不是与Docker等价的强隔离。“未拼入prompt”不足以证明未读取答案。

研究记录应核对初始输入、工具读取路径与检索来源：仅本题问题、允许的hints、base_commit工作区和匹配图作为任务证据。参考答案、最终B裁判日志、参考修复诊断日志、其他组轨迹和未来提交不进入Agent可用材料；Agent在当前A阶段通过允许工具取得的自身局部检查输出仍是正常观察。发现答案污染应把该attempt标为无效比较，保留证据后在干净输入上重跑。此处是结果有效性要求，不能宣称仅靠提示词实现了文件系统隔离。

## 五、统一到所有比较组

环境、记录、数据与服务生命周期修复共同应用于A0/B/C。共同提示修正包括：只有证据充分才修改、避免将临时复现文件/日志混入补丁、仅调用已注册工具、正确处理截断、修改完成后提交并停止。它们会改变原001行为，因此新A0是修订后的共同基线，001仅保留为历史观察。

get_status检查节奏也共同固定，C只增加软收尾决策。主计划进一步规定非相邻循环、平衡组别次序、重复运行、保留集使用和训练数据暴露登记。

## 实施与结果记录

先改两个现有运行载体的输入、安装、版本、记录和生命周期部分；诊断可取得的冻结补丁，原件缺失时按R16用新A0证据继续；随后冻结共同A0及B/C增量，进入小规模人工对照。所有新增行为对照在主计划中有明确标签，不能作为悄悄改变基线的“杂项修复”。

每项R编号在同轮`docs/progress/003-budget-and-context.md`记录：改动位置、采用的官方版本、实际运行证据、是否仍有未知点。代码修改完成与远程问题已验证解决分开记录；未取得的旧轨迹、单/双卡能力和正式评分差异不能填成已解决。

## 来源定位

- [官方wheelhouse](https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse)：swegemma的evaluate.py、results.py、config.py、context.py、tools/workspace.py、tools/execution.py、harness/agent_runner.py、harness/verification.py、harness/container_setup.py、sandbox/subprocess.py、deduplication.py。
- 同一wheelhouse：adk_submission的server.py、discovery.py、resolvers/tools.py；google/adk的tools/function_tool.py及函数声明生成逻辑。
- [官方运行说明](https://www.kaggle.com/competitions/gemma-4-developer-agent/data)与[Getting Started](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent)。
- [主办方环境说明](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/742882)、[9月28日最新限制](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743964)、[官方依赖范围](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743800)。
- [wheel缓存问题报告](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744029)：缓存键风险已对照源码，但未证明影响了001。
