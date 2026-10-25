# 首批96题与明确补充课程

当前优先准备[S600共同/专科课程](../s600/README.md)。本文件的81题总量和子机制表为S600启动时的历史起点；最新总量见[课程清单](../s600/course-manifest.json)。原96题成员、61正例和35未入池记录保持不变。

2026-10-25。原96个任务身份保持不变，现已全部真实执行官方六工具；90题保存基线/修复双轮结果及完整JUnit，61题完整通过。另有20个逐题审阅的补充train任务通过，合计81条可用示范。原96仍有35题未入池：29题双轮结果完整但未通过，6题验证超时或报告不完整。**尚未完成全部候选的数据准备，未启动GPU训练。**

入口：[原96逐题目录](catalog.md)、[原始来源与机制索引](catalog.json)、[20题明确补充目录](../acquisition/supplemental/catalog.json)、[完整重放总览](../replay/replay-summary.json)、[逐题准入与原始证据](../replay/canonical-results-manifest.json)、[Gemma导出清单](../replay/gemma-sft-manifest.json)、[四份主责课程及共享课程](../replay/adapter-curricula-manifest.json)。补充任务由此前6题和本轮14题组成，没有静默替换原失败记录；81题的最终动作导出与课程清单已同步。

| 主责Adapter | 原96审阅数 | 原96已验收 | 补充已验收 | 当前可用合计 |
| --- | ---: | ---: | ---: | ---: |
| G | 36 | 28 | 0 | 28 |
| S | 20 | 15 | 1 | 16 |
| F | 20 | 11 | 7 | 18 |
| R | 20 | 7 | 12 | 19 |
| 合计 | 96 | 61 | 20 | 81 |

主责标签是能力契约，不是训练混合比例。G的共同修复课程可覆盖全部机制，专家也需要通用回放保持；共享课程复用相同81题，不另算独立数据。G=r128，目标为q/o与MLP共五种投影；S/F/R=r32七投影，BF16计算与导出保持不变。

| 子机制 | 能力契约 | 原配额已通过 | 补充已通过 | 当前可用 |
| --- | --- | ---: | ---: | ---: |
| S1 | 编码、解析与序列化 | 3 / 4 | 1 | 4 |
| S2 | 位置、坐标与跨度 | 4 / 4 | 0 | 4 |
| S3 | 集合、多值与筛选 | 2 / 4 | 0 | 2 |
| S4 | 遍历与不变量 | 3 / 4 | 0 | 3 |
| S5 | 递归、语法与结构 | 3 / 4 | 0 | 3 |
| F1 | 参数与上下文传播 | 2 / 4 | 2 | 4 |
| F2 | 身份、引用与所有权 | 4 / 4 | 0 | 4 |
| F3 | 缓存与失效 | 2 / 4 | 2 | 4 |
| F4 | 资源与退出 | 1 / 4 | 3 | 4 |
| F5 | 事件与协议时序 | 2 / 4 | 0 | 2 |
| R1 | 注解与命名空间 | 0 / 4 | 4 | 4 |
| R2 | 可调用对象与包装 | 3 / 4 | 1 | 4 |
| R3 | 对象协议与描述符 | 1 / 4 | 3 | 4 |
| R4 | 版本与兼容契约 | 1 / 4 | 2 | 3 |
| R5 | 导入、加载与可用性 | 2 / 4 | 2 | 4 |

重点S2、F2、R2、R5各有4条可用示范；原16题仍只有13条通过，另外R2的1条和R5的2条来自明确补充。本轮补齐后F1/F3/F4/R1/R3也各有4条；S3、F5各2条，S4、S5、R4各3条，仍需补足。任务数量不证明泛化收益，不能称四个adapter已训练充分。

## 已准备的训练文件

已导出81题、716条决策（641次工具调用＋75条实际终止回答），共89,774个助手监督tokens，最大动作上下文8,973。主文件位于本地`competition/vfarl/replay/artifacts/gemma-sft/sft-actions.jsonl`；四份主责和共享文件位于`replay/artifacts/adapter-curricula/`，hash与逐题数量见课程清单。采样比例尚未设置，没有执行训练。

每条动作携带主责adapter、子机制、仓库、family、snapshot group和来源指纹metadata；这些不进入policy prompt。所有材料均train_only，同一上游snapshot或修复家族不能靠不同task ID伪装成跨评测独立样本。本池没有创建新的评测划分。

动作由Codex编排、审阅并实际执行，属于脚本指定的off-policy冷启动材料。legacy `human_reviewed_off_policy_demonstration`字段不表示用户人工标注，也不是Gemma当前策略轨迹。没有编造思考、logprob、on-policy rollout或同状态关键动作对。故意复现缺陷的error是正确诊断动作，不能当作rejected action。

## 机制匹配与来源

分类按实际程序契约逐题核对，保留触发、负责生产符号、因果改动、真实区分观察、原验证要求与限制。新MonkeyType148只是stub注解字符串的AST表示契约，归S1；164生成stub确实因缺Optional导入而NameError，归R1。另一个R1维护类型反序列化的限定名解析与type identity，不夸成复杂泛型推断。新Tenacity R2恢复同步/native async/Tornado包装的控制器分派，独立于原PR406的wraps元数据契约。Flake8两题维护插件发现去重和提供方分派，属于R5可用性，明确记录同snapshot。

本轮14题补充对应具体生产契约：Funcy两题维护参数捕获、尾部绑定与关键字优先级，归F1；Funcy过期缓存和TinyDB删除后的查询失效归F3。Ptyprocess阻塞等待、非阻塞子进程回收和Dataset事务退出归F4，其中两个Ptyprocess家族共享snapshot，不作为跨评测独立样本。Boltons描述符、集合相等分派与Addict属性赋值保护归R3；APISpec两题按OpenAPI版本维护引用/可空语义，归R4；MonkeyType237和151分别维护Callable注解导入闭包及`Tuple[()]`零元参数身份，归R1。

原96中81题来自SWE-smith-py train（`77cab9055d42ab4a5c25c89a8f937096db13558e`）、13题SWE-Gym train（`bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb`）、2题SWE-rebench-V2 train（`10483de0f50fe5da545942705a76c6150171af7f`）。补充20题均来自同固定Smith train。共116个经深入审阅候选、50个canonical仓库；宽页检索缓存不计作深入审阅任务。

全部保存原task blob SHA、40位执行commit、独立源码/数据集许可与机制审阅指纹。Smith patch引入缺陷，不能直接当修复答案。来源任务、参考修复和验证资产留collector侧；policy使用允许的原题面/提示，四题原描述为空时采用下述显式版本化的行为题面。排除比赛FastAPI/Rich/Requests/HTTPX四仓与别名，没有摄入129公共开发题或heldout正文。大文件留本地ignored artifacts，不随adapter submission提交。

来源审阅：[G36](../acquisition/general/general-36-review.json)、[S/F40](../acquisition/s2-f2/reviewed-sf40.json)、[R20](../acquisition/r2-r5/mechanism-reviews.json)、[重点16复核](priority16-review.md)、[Moto14复核](../replay/moto-source-cross-review.json)、[Flake8补充](../acquisition/recover-semantics/reviewed-candidates.json)、[MonkeyType补充](../acquisition/recover-r1/reviewed-candidates.json)。原96个候选中81个来自Smith合成任务、Moto占14题，这不代表比赛分布或模型成功率。

本轮审阅：[参数与缓存](../acquisition/expand-cache/reviewed-candidates.json)、[资源与退出](../acquisition/expand-resource/reviewed-candidates.json)、[对象协议](../acquisition/expand-objects/reviewed-candidates.json)、[版本兼容](../acquisition/expand-compat/reviewed-candidates.json)、[注解身份](../acquisition/expand-annotation/reviewed-candidates.json)。

## 题面、来源标签与数据格式边界

TinyDB删除失效、Boltons IndexedSet相等、Ptyprocess `isalive`、Dataset `__exit__`四题的原`problem_statement`为空。已保留原blob、旧adapted task及run001，另建curator-authored最小行为题面和v2 task，并对v2实际执行基线/修复双轮重放。补写内容只定义可观察契约，不含参考修复或required节点名单；来源审阅记录原空文本hash、新文本hash、审阅者、来源依据及实际v2 run，不能称其为发布者原题面。准入时完整profile必须与已审阅catalog相等，并进入稳定来源指纹；未审阅的自报新题面不能入池。原test patch和全部F2P/P2P要求保持不变。

80条示范使用原required ID并通过原官方matcher；Slugify另1条保守映射44项unittest要求到45个原pytest节点，歧义时两项都要求、全82项原文件通过，不标成原字面ID通过。[原映射](../replay/sfr-slugify-s1-node-mapping.json)保持可追溯。

MonkeyType限定名解析题来源把361项标作F2P，实际基线52失败、309已通过，全部ID均找到；原8项P2P通过。它只在固定task/blob/基线XML经过独立复核后按`audited_source_F2P_label_noise`单独入池，原361+8项修后全部通过。默认“全部F2P基线失败”的证据规则不整体放宽，也不声称361个失败被修复。该题原示例还误传模块对象，合法API需要模块名字符串；原题面未偷偷改写，真实诊断使用合法编码API。见[独立来源审阅](../acquisition/recover-r1/reviewed-namespace-candidate.json)与补充目录的显式baseline记录。

MonkeyType151也使用固定task/blob/基线XML独立审阅的同类profile：原4项F2P实际3失败、1已通过，全部ID均找到；修复后原4项F2P和365项P2P全部通过，原标签及要求均未改。实际Python3.11源码兼容环境验证零元Tuple身份，不声称复现原Python3.7故障，也不把已通过的case当负动作。两项profile只解释各自实际基线，不能推广放宽其他题的准入。

PennyLane原嵌套`tests/pytest.ini`让实际JUnit地址相对tests目录。collector仅按真实快照中的原配置hash解释基线的精确类名/参数ID，不修改原required或验收器。h11早期缺目标字段则从hash一致的原task、原版Task.test_files恢复范围。

## 官方协议与运行环境

真实调用原swegemma0.2.11的Context/Budget、六工具、路径保护、verify_task和JUnit matcher。Phase A/Phase B使用独立干净workspace，保存真实事件、输入树/快照/补丁/源码绑定与双轮JUnit。测试只恢复固定上游字节；不新写专用测试、修改断言或缩为required节点验收。

当前是官方SubprocessManager CPU后端，**不是比赛Docker环境等价证明**。现代官方父环境和旧任务依赖分开。Hydra任务Python3.10来自源码CI；原Phase B的PYTHONSAFEPATH=1命令保持，但3.10不实现该变量。Moto两题用source-compatible3.11.8恢复旧stdlib行为，明确区别原Gym3.12环境。MonkeyType依赖物理复制进每份sandbox实际purelib，不改sysconfig或源码过滤规则。PennyLane原三框架的实际设备均为CPU，原3F2P+283P2P完整通过；Linux/Kokkos未在本Mac复刻。原负结果都保留，不能算模型错误。

逐动作prompt由原build_agent_prompt、固定官方31B tokenizer/template和实际ADK声明生成，thinking_budget=0、无新增system prompt。强制空thought块与所有user/工具观察不进loss；真实assistant动作/终止回答才有监督。早期raw仅题面的官方包装增补明确版本化。其他部署prompt/预算需另行导出，人工CPU耗时不当作推理预算收益。

## 仍未准备完的部分

原96还有35题未入池，主要涉及固定skip/xfail与原matcher跨参数/类误配、AnyIO禁用、YAML/Describe/doctest发现契约、旧外部服务及超时。Pydantic恢复锁版pytest/plugin后实际测试失败已消除，仍因原matcher拒收，不以局部改善冒充整题成功。原负结果的逐题原因在准入清单中。

下一步应补弱机制的可完整验收数据，再按授权采当前策略attempt与可恢复checkpoint，实证关键动作替换、同状态续跑与结果/成本优势。当前模型rollout、已验证关键动作对及GPU训练均为0；81条脚本示范只是冷启动实物，不是整套Verified Focal Agent RL数据已经齐备。
