# 首批96题：来源、机制与官方重放

2026-10-24。当前完成96个训练任务的逐题来源和机制审阅，G36、S20、F20、R20，来自43个canonical仓库。S/F/R各自五个子机制均为4题，优先缺口S2、F2、R2、R5各补到4个不同生产符号修补家族。精确数量与本地验证进度以 [manifest.json](manifest.json) 为准；静态审阅完成不等于96份环境已恢复或96条成功轨迹已采集。

入口：[96题目录](catalog.md)、[完整审阅索引](catalog.json)、[重点16题独立复核](priority16-review.md)、[实际重放结果](../replay/replay-summary.json)。四个adapter的rank、投影和BF16配置保持不变。

当前实际重放10个独立任务，7题完整通过本地官方验收，3题保留为reserve审计。最终96题中7题已有通过示范，89题尚未重放。共有83个主要工具事件，另保留27个采集/环境尝试事件；这些都不是模型采样。7条正例导出59个助手决策（58个工具动作、1个实际人工终止回答），共6,466个loss tokens，最长逐动作上下文7,659 tokens。

| 已通过任务 | 主机制 | 完整保留并通过的原验证节点 |
| --- | --- | --- |
| Tenacity425 | G 局部参数契约 | 1 F2P + 122 P2P |
| Apispec 53fevqi2 | S1 RFC表示顺序 | 2 F2P + 567 P2P |
| Tomli238 | S2 结构化位置 | 2 F2P + 14 P2P |
| h11 h6ghpwe2 | F5 协议切换 | 1 F2P + 75 P2P |
| Boltons375 | F2 sentinel身份 | 1 F2P + 411 P2P |
| PySnooper urhyaplv | R2 方法包装 | 8 F2P + 70 P2P |
| Jinja zwhpkn1s | R5 PEP451/ZIP loader | 1 F2P + 739 P2P |

## 逐题怎样确认能力匹配

每题保留Agent可见触发、故障行为、负责生产符号、因果改动、能区分错误与正确实现的观察，以及原始验证规格。分类依据是需要维护的程序契约；出现Unicode、async、type等字样不构成归类证据。观察若只来自静态推导或来源轨迹，明确保留该证据等级，不改写为本地执行结果。

| 主责 | 任务数 | 匹配能力与使用方式 |
| --- | ---: | --- |
| G | 36 | 通用定位、局部规则、API与集成修复；共同课程覆盖S/F/R机制，36是主责配额，不限制G只学这一组 |
| S | 20 | 编码解析、坐标跨度、集合多值、遍历不变量、嵌套语法结构，各4题 |
| F | 20 | 参数传播、引用所有权、缓存失效、资源退出、事件时序，各4题 |
| R | 20 | 注解命名空间、callable包装、对象协议、版本契约、导入可用性，各4题 |

错误定位、无效编辑恢复、利用工具反馈与预算控制由四个adapter共同学习。F2不能用普通列表/元组转换凑数，R5不能用pip安装失败凑数。同一修复家族的多条attempt不能增加独立任务数。跨组审阅按生产符号和因果契约检查家族重叠；96个不同family字符串本身不是独立性的证明。

三份原始逐题审阅保存全部证据与取舍：[G36](../acquisition/general/general-36-review.json)、[S20/F20](../acquisition/s2-f2/reviewed-sf40.json)、[R20](../acquisition/r2-r5/mechanism-reviews.json)。G36已交叉复核，其中Result121补充了`.value`从property变成bound method的descriptor机制；Safety508保留整个headless能力恢复的G集成范围。相关补充没有覆盖原始来源题面。

## 来源与代码身份

| 任务池 | 题数 | 固定revision | 原始patch含义 |
| --- | ---: | --- | --- |
| SWE-bench/SWE-smith-py，train | 81 | `77cab9055d42ab4a5c25c89a8f937096db13558e` | 引入缺陷；不能直接作为修复答案 |
| SWE-Gym/SWE-Gym，train | 13 | `bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb` | 参考修复，留在collector侧 |
| nebius/SWE-rebench-V2，train | 2 | `10483de0f50fe5da545942705a76c6150171af7f` | 参考修复，留在collector侧 |

96题均保存完整原始task、下载blob的SHA256、精确任务ID、完整40位执行commit及逐仓许可文本证据。上游代码许可与数据集/轨迹许可分别记录，不由其中一项替代另一项。Smith的bug branch执行SHA与upstream snapshot SHA分开保存；真实issue任务在原始base_commit执行。原生轨迹仅为审阅线索，不是目标协议的成功过程。

这是为补机制缺口而配额采集的试批，81/96是合成故障，Moto占14题；不能把它的标签比例、成功率或短补丁分布当作比赛隐藏任务分布。后续须根据真实环境恢复率、有效轨迹产率与跨仓修复收益，调整合成/真实issue和仓库配比。

当前所有题仅用于训练：排除了FastAPI、Rich、Requests、HTTPX及fork/别名，不摄入129个公共开发题或隐藏/预留评测题。先按允许train清单取正文，没有为查重读取heldout内容。旧来源材料只复用精确身份匹配的原始正文和缓存，不继承旧研究结论。

全部代码commit已固定，依赖与镜像digest尚未全部实例化。13个Gym任务另从官方README指向的SWE-Bench-Fork固定commit `242429c188fcfd06aad13fce9a54d450470bf0ac`取得Moto环境规格，原声明为Python3.12、`make init`、`pytest -n0 -rA`；这说明来源环境定义可追溯，不代表13份镜像已经在本地跑通。只用AST读取声明数据，没有执行下载的安装代码。见 [环境来源](../acquisition/general/sources/gym-environment-source.json)。

## 对齐比赛的工具、对白与验收

重放正常导入原版`swegemma 0.2.11`、`adk-eval-core 0.1.0`、`adk-submission 0.2.13`，通过实际Context/Budget与六工具执行。真实Google ADK声明单独捕获，参数名及required与wrapper签名核对；没有把源工具改名后伪装成官方工具。

当前使用官方`SubprocessManager`：本机Docker daemon未运行，所以它是官方代码支持的CPU本地后端，**尚不能证明比赛Docker环境等价**。Phase A与Phase B使用独立干净workspace，Phase B重建基线、应用经过官方路径保护过滤的生产补丁，再调用原版`verify_task`和JUnit核对。工具/命令/验证时间分别记录；人工脚本耗时不能当作Gemma推理时间或预算训练成绩。

外部训练任务有明确F2P/P2P，而129个官方公共任务通常使用test_patch fallback。数据适配只从原始F2P/P2P取目标文件并集，保留全部节点，不修改官方验证器。若来源bug branch移除了测试，只恢复精确upstream commit中的原有验证文件，保存hash；不编写新的测试套件。必需节点必须通过；pytest exit 0、工具编辑成功或局部复现改善都不能替代整题resolved。

成功示范进一步由固定官方31B tokenizer/chat template渲染成Gemma对白，使用实际六工具声明。原始`function.arguments`JSON字符串先解析成对象，避免模板重复加括号；原始事件不变。初版工具重放的首条user只有来源题面，canonical导出另调用原版`build_agent_prompt`，按相同题面/预算加入官方工具规则与提交指令；这项collector侧prompt增补明确版本化，原始消息不改，不声称模型在增补后prompt下实际采过样。没有加入参考修复、required列表或事后结果。

训练输入按每个真实动作导出推理prefix，实际调用`add_generation_prompt=True`，包括模板在关闭thinking时补的空thought控制块；这个强制prefix只作条件、不进loss，不是假造思考。loss覆盖经审阅的assistant工具动作及确实记录的人工终止回答，user/system/工具观察不进loss。原始题面、动作prefix与后续验证严格分开；没有编造thinking、策略logprob或模型回答。新Jinja示范已在实际submit后记录官方要求的text-only终止回答；旧6条没有这项监督，保持原样，不补写后冒充原始轨迹。详见 [官方任务prompt导出器](../replay/export_prompts.py)、[对白导出器](../format_replays.py) 和 [导出清单](../replay/gemma-sft-manifest.json)。

这些是**人工审阅的off-policy冷启动示范**。首次命令故意复现缺陷而返回error，属于正确诊断动作，不能充作FocalRL的rejected action。当前没有模型on-policy rollout、同状态已验证关键动作对，也没有GPU训练。本版导出固定`thinking_budget=0`、`include_thoughts=False`、无新增system prompt，供直接工具调用冷启动；部署若使用其他system prompt或thinking预算，需一致地另行版本化导出，不能把本版称为已经对齐4096预算训练。

## 实际遇到的来源与验收问题

- 旧Smith轨迹种子的18个非空`row.patch`跨仓错配，该列整列隔离；本批以精确匹配的任务表bug-introducing patch和代码身份为准，不复用错误答案列。
- parse审计题原P2P把生产`parse.py`列为doctest目标，触发官方测试文件保护，实际生产补丁被过滤为0字节；保留真实失败，未放入96题。
- Freezegun492的修复确实消除了staticmethod绑定错误，但原required Maya节点在当前本地Python3.12中skip，来源又将该依赖限定在Python<3.12，官方验证器正确判为未解决。492/493同家族和其他含相同必需依赖矛盾的候选转reserve；没有删required或把skip算pass。
- Pinyin318的生产doctest适配尚无执行证据，移到reserve；它没有被描述为实际重放失败。G改用Moto7111，S3/F4的Freezegun候选分别改用Deepdiff403和Tornado3315。
- Dask9573真实修复后pandas required用例通过、cudf可选用例skip，但官方matcher去参数后合并了同名函数节点，仍判整题未解决；原baseline两个用例均skip，也没有改写成测试failed。真实计算/导入改善只作R5机制证据，不能收为完整成功或因模型错误而失败。保留原要求和负结果，不修改官方matcher或安装GPU依赖来凑通过。最终R5配额改用Jinja PEP451/ZIP loader，原740 required全部通过。
- Jinja第一次工具重放读到了先前安装的同名发行版，预期故障未出现，第3个事件即停止并保留。后续把sandbox的workspace/src绑定到路径最前，并在实际命令核对`__file__`和源码hash后重放成功。旧src示范Apispec/Tomli也已复核原freeze与真实前后观察；证据与边界见 [源码绑定复核](../replay/historical-source-binding-review.json)。没有将首次意外正常当成修复成功。
- 部分Tornado任务仍有unittest式节点，需保留原ID并明确适配JUnit匹配规则后再重放。部分源题面复现不充分或与缺陷不吻合，collector审阅保留这些限制，policy导出不偷偷补入金补丁解释。

## 后续采集顺序

先逐repo恢复剩余环境与原验证资产，优先补重点子机制的真实六工具成功过程；每题记录原始失败和修复后完整验收。环境不相容先解释/替换，不能以来源success标志补齐本地成功数。

在冷启动材料和部署对白一致后，再按授权采当前策略attempt，保存真实行为概率和预算消耗。从实际策略错误前的可恢复状态分叉，重放原动作与替代动作，双方继续至完整验收；短段改善和完整结果分别计数。恢复不能归零原prefix成本。当前96题是采集产率试批，192次基线attempt、最多128个checkpoint/512条continuation仍是后续计划，未执行。

原始大文件在本地`competition/vfarl/artifacts/pilot96/`，不随比赛submission提交。`tasks.privileged.jsonl`含金补丁/完整验证；`policy-tasks.jsonl`仅含原始题面、执行commit及工具契约，不含因果标签或验证日志。清单记录两份文件的hash。全部工作无GPU、无付费模型调用，没有新增专用测试文件。
