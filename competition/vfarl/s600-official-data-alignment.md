# S 600题：官方数据与对白规范复核

2026-10-25，Asia/Shanghai。用户最新指示先准备题目，本轮不启动或继续探查GPU训练。本说明记录数据准备约定与阶段起点；当前已完成数量和实物哈希以[S600课程manifest](s600/course-manifest.json)为准。

## 600题如何计数

采用用户指定的400共同任务＋200个S专科任务，合计600个独立任务。共同与专科名单不交叉；同一问题的多条轨迹、恢复分支、动作窗口不增加独立题数。相同生产符号/行为契约的变异家族及相同snapshot继续保留分组，不能仅换task ID制造跨评测独立性。

本次复核起点是81个已验收示范，其中16个S专科、其余65个可作为共同课程种子。若按该分配，仍缺335共同＋184专科，共519个独立任务。该分配不意味着共同课程永远排除S机制；之后可加入与200专科不重叠的S任务，以覆盖共同修复所需的广泛机制。

S专科200题可先按五机制各40题规划；现有S1/S2/S3/S4/S5为4/4/2/3/3，缺36/36/38/37/37。这是采集配额，不是每种只靠关键词挑40题。每题需要明确触发、生产符号、错误行为、因果改动和执行区分观察。

## 当前官方来源

2026-10-25实际打开官方[Kaggle模型与执行规则](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview/model-selection-budget-and-harness-rules)及[比赛规则](https://www.kaggle.com/competitions/gemma-4-developer-agent/rules)，通过后台浏览器读取已加载正文。web文本抓取对这两个动态页面返回零行，搜索索引还保留“仅支持31B”的旧描述，因此以下以当前浏览器正文为准，不把旧索引当最新规则。

模型页当前允许31B、12B、E4B、E2B四个官方W4A16 variant；同一submission的所有agent/subagent仍使用同一个variant，可以挂不同PEFT LoRA。当前项目固定31B，不因此更换模型。官方工具页列出九种预定义工具，本轮课程使用其中真实的六工具子集：run_command、read_file、edit_file、write_file、get_status、submit_patch。官方也支持声明式agent_tool，但六工具课程没有委派动作，不能造一个handoff回包。

官方[Evaluation](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview/evaluation)正文确认按完整任务的补丁验证结果计PASS/FAIL，总分为通过任务比例；没有单题按测试数累计的部分分。所有题生成补丁的总预算12小时，包含sandbox setup，排除patch validation。数据中的脚本耗时不能当作模型推理预算优化成绩。

官方Rules的External Data and Tools条款允许外部数据，但要求公开可及或符合其合理可及标准；源码、数据集及生成材料的许可和获取方式仍需逐来源留证。这不是对任何具体数据集许可的法律结论。项目继续仅用明确train来源，四比赛仓/129公共题和heldout用途不改变，不将其转入600题课程。

## 原版验收及本地边界

本机正常导入版本为swegemma0.2.11、adk-eval-core0.1.0、adk-submission0.2.13、google-adk1.39.1。实际安装的verification.py SHA256为`99e2a5fdab2be2b5e0388de29080da9d6e7003cf8b53299cda790464f4ecd2aa`，与已保存官方wheel提取件相同。

原验收在全新Phase B重建源码，应用有效生产补丁及原验证资产，运行原目标测试文件范围；先要求pytest成功退出，再检查JUnit有实际通过且没有failure/error。原FAIL_TO_PASS/PASS_TO_PASS要求均需匹配通过。不能将文件范围缩成只运行required节点，也不能改断言或官方matcher取得正例。原测试缺失时只按固定上游字节恢复，项目不新写专用测试套件。

训练正例还需要真实基线因果失败证据。已单独审阅的节点映射/基线差异profile继续逐题绑定固定task和XML，不能把它们推广成全局放宽规则。空原题面需要明确披露curator新版本，并在该实际题面下重新执行；原task与旧运行留存，完整profile与已审阅catalog相等并进入稳定来源指纹。

当前重放后端是官方SubprocessManager的本地CPU实现。macOS、源兼容Python/依赖配置等差异照实保留，不宣称比赛Docker/Linux环境等价。仅来源成功标签或teacher patch存在，不等于本工具环境中的已验收示范；候选、已审阅、已执行、可训练分别统计。

## 白名单输入、模板与loss

初始prompt继续由原`build_agent_prompt`生成，仅提供repo、problem_statement、hints_text、base_commit及真实预算/工具名。参考patch、test_patch、原required列表、未来验收结果、离线机制标签和来源审阅元数据不得混入策略前缀。

使用固定官方31B CT v2 tokenizer/template及实际ADK六工具声明。每个动作使用`add_generation_prompt=True`的真实生成前缀，保留当时完整可见历史和真实观察，不截历史后伪称完整状态。当前冷启动导出为thinking_budget=0、include_thoughts=False、无新增system；强制空thought控制和所有user/tool观察均为prompt-only，只有已审阅且有执行证据的assistant动作/实际终止回答参与loss。不补写thinking、final、logprob或偏好对。

当前官方包的`build_submission_limits()`实际返回解包总尺寸3,221,225,472 bytes上限、输出token范围1..32,768、thinking预算0..32,768。官方README给出的总上下文为32,768（prompt＋reasoning＋output），不能将输出上限误当可额外叠加的上下文。当前数据导出拒绝超过总上下文的动作，训练阶段若容量更小应保留真实完整状态并如实排除超长样本，不能靠静默截断凑齐600题。

600份冷启动示范仍不是完整Verified Focal Agent RL材料。后续同状态关键动作比较需要可恢复状态、相同前缀/剩余预算、真实替换及双方完整续跑；on-policy优化还需真实策略身份和采样logprob。准备题目阶段不把这些未采集字段补成假数据。

本轮只读取通用框架、规则与现有数据流程，无GPU或付费模型调用、无训练/评测启动、无新测试文件。另有一处读取边界：打开既有`competition/scoring.md`查通用说明时，其后半包含此前公共任务研究片段；本轮未打开129任务文件或heldout，也未将这些片段用于600题的题面、标签或训练材料。
