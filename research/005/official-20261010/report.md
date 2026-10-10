# 005：比赛官方信息与研究空间更新

读取日期：2026-10-10（Asia/Shanghai）。方法：Kaggle MCP 只读取得比赛元信息、全部说明页、前50条排行榜、公开数据清单摘要、讨论及官方 wheelhouse；下载两份合计186,048字节的官方wheel，仅解包阅读源码，未安装或执行。此前本地研究停在2026-10-01，本文件区分官方事实、参赛者报告和研究推断。

## 1. 冲牌与奖金是三个不同目标

| 目标 | 当前已知条件 | 对研究的意义 |
|---|---|---|
| 主赛金牌 | MCP显示2234队，awards_points=true；Kaggle常规金牌公式为前10名加每500队1名，若最终仍2234队约为前14名，实际按最终参赛规模、排名及官方发牌计算 | 当前公开榜名次与分数不是最终金牌线 |
| 主赛现金 | 第一$37,000、第二$18,000、第三$10,000 | 拿金牌不自动等于拿主赛奖金；需要研究可能超越公开基线的能力增量 |
| 论文赛现金 | Overall Best Paper $15,000；Best New Resource $10,000；Best New Application $10,000 | 可把同一研究沉淀为可复用资源与机制论文；论文赛不发竞赛积分或奖牌 |

主赛最终截止为2026-12-02 23:59 UTC，即北京时间**12月3日07:59**。论文赛截止2026-11-12 23:59 UTC，即北京时间**11月13日07:59**。论文限3000词，Novelty/Quality/Relevance/Verifiability/Clarity五项等权；可以研究已有机制的新性质，图与embedding并非必选。论文赛参赛不要求先参与主赛。当前只是研究规划，不创建公开Writeup或提交。

来源：[主赛说明与奖金](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview)、[主赛时间表](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview/timeline)、[官方金牌规则](https://www.kaggle.com/progression/competitions)、[论文赛](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper)。本地原件：[snapshot.json](snapshot.json)、[paper-pages.json](paper-pages.json)、[paper-overview.json](paper-overview.json)。

## 2. 榜单变化及可比较范围

2026-10-10本次MCP返回的前50条中，前二为0.24、第三0.22、第4–10条0.20、第11–40条0.18。10月1日本地快照榜首为0.17。榜首对应提交时间10月7–8日，不能将检索日期写成提交日期。使用返回顺序描述名次，同分最终处理由Kaggle决定。

这表明公开竞赛表现已有进展，不证明同一方案从0.17提升到了0.24，也无法识别领先队伍的训练方法。公开代码与排行榜的对应关系必须通过Notebook版本评分或提交归属确认；标题、点赞、作者当前名次都不足以确认代码分数。约60题公开集意味着一个额外成功约1.7个百分点，仅是分辨率估算；确切分母、任务组合和最终私榜表现不能由显示两位小数反推。不要据此预言金牌阈值。

来源：[官方排行榜](https://www.kaggle.com/competitions/gemma-4-developer-agent/leaderboard)，保存于snapshot.json；公开源码逐项审查另见[公开方案报告](../public-solutions-20261010/report.md)。

## 3. 官方数据对研究重点的影响

- 数据页仍为129个公开开发任务，FastAPI/Rich/Requests/HTTPX四仓库；公开patch/test_patch供本地研究与训练，隐藏阶段不提供答案。
- MCP文件摘要：524文件，129个快照、127个npz、129个json、124个依赖wheel；tasks.jsonl仍为1,984,455字节。这里只能确认元信息，没有重新下载或证明内容哈希相同。说明页包含hard-link布局，文件API计数与说明页物理命名数量不宜混用。
- 隐藏集约120题，公榜私榜约各半，来自私有仓库。更强模型参与过任务可解性筛选。这使“记住公开仓库修复答案”价值有限；应投入通用语义推理、工具使用和状态保持。
- 10月5日主办方明确：隐藏私有仓库任务的修复均不涉及async def，私有仓库async函数占比最多1.8%，不会因此更新图。修复公开FastAPI async覆盖不是当前隐藏分数的强先验。这个官方说明不代表隐藏任务只涉及某种应用领域，领域分布问题未获得进一步答复。

来源：[官方数据](https://www.kaggle.com/competitions/gemma-4-developer-agent/data)、[主办方async说明](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/745998)、[领域迁移提问（无答复）](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/745796)。本轮未读取本项目28题heldout内容。

## 4. 旧版参数结论不能作为当前研究禁区

官方wheelhouse已从历史版本更新到**v30，2026-10-09**；目前包含swegemma0.2.11、adk_submission0.2.13、google_adk1.36.1、vllm0.19.1、transformers5.13.1。这里只研究会改变算法选择的接口，不展开环境排错。

| 旧结论/风险 | 当前官方源码事实 | 研究含义 |
|---|---|---|
| thinking_budget的4096和1024只映射为True | generation.py:411–458优先根据预算决定enable_thinking，并把正值传为extra_body.thinking_token_budget；schema.py:169允许0；同时转发seed | 重新开放“思考预算”作为可控研究变量。源码转发不等于当前服务端确切执行效果已经实测 |
| include_thoughts=false可视作关thinking | 正预算的优先级高于该标志；该标志还控制思考内容保留/转发 | 研究明确写thinking_budget=0与正预算，不将“隐藏思考文本”和“关闭思考生成”混为一谈 |
| reasoning_content最终是否被模板采用未知 | generation.py:65–76将reasoning_content同时填入reasoning，195起安装桥接 | 历史长程轨迹的状态丢失可能与旧桥接混杂，不能直接据旧失败断言模型的固有记忆极限 |
| 未注册/坏工具参数可能结束整题并丢补丁 | agent_runner.py:244–280有ToolErrorPlugin，包含技能参数清洗与结构化错误回包；1045起异常后仍回收已提交或工作树补丁 | 减少把错误恢复研究变成补旧harness漏洞；更值得研究错误回包后如何选对下一步 |
| 图查询只能匹配全名，关键词必空 | graph_utils提供名称解析；retrieval_utils.py:781后先解析名称，831–858有词法fallback；图工具提示也更新 | 旧关键词空结果数据不能直接预测新版本；仍不等于官方embedding具有优秀语义区分度 |
| workspace树先被文档淹没 | agent_runner.py:299–312改为剪枝、按深度排序、最多200行；工具提示按实际声明过滤 | 不值得再大量开发一份专门补旧树布局的框架 |

源码归档：[generation.py](wheels-source/adk_submission/resolvers/generation.py)、[schema.py](wheels-source/adk_submission/schema.py)、[agent_runner.py](wheels-source/swegemma/harness/agent_runner.py)、[retrieval_utils.py](wheels-source/swegemma/graph/retrieval_utils.py)；[wheel指纹](wheels-source/manifest.json)。下载的两份wheel为66,528和119,520字节。以上是公开包源码事实，不声称已直接观察正式评分服务内部版本。

来源：[官方wheelhouse及版本说明](https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse)、[参数变化报告](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747914)、[工具错误讨论](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/745028)、[树与检索讨论](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/745220)。参赛者报告中的速度、失败率不视为本项目实测。

## 5. 三项值得保留的社区研究线索

1. **上下文摘要可能丢失动作证据。** 10月8日社区报告ADK默认摘要只收part.text，不收函数调用与回包，并请求开放摘要提示配置。这与本地“读过却重复读/不编辑”有机制关联；但该帖不是官方承诺。提交schema确实有output_key、include_contents以及SequentialAgent/AgentTool，可研究明确的证据包与一次串行阶段切换，无需等待自定义host压缩器。
2. **图的结构信号与语义信号应分开。** 10月4日作者对127图的旧版审计报告：词法TF-IDF top10命中改动定义52%，embedding9%；观察到调用边缺失、方向混合、class-members混入calls。作者数据/脚本已公开，但本轮未复算。10月8–9日名称解析已改变，不抹去图数据本身的限制；建议图用来扩展有根据的邻域，文本负责从issue找入口。
3. **上下文窗口与思考预算是两个旋钮。** 10月9日社区报告14336摘要阈值+16384输出预留只剩2048 headroom，工具回包可在摘要触发前导致溢出。本项目0C不能证明“保留16384永远最优”，只证明当时该小样本下8192未晋升。新研究可保持模型输出上限与思考预算分开，而非把所有“少想/少输出”混成一个实验。

来源：[摘要信息丢失](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747186)、[127图审计](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/745731)、[溢出报告](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747791)。本轮只引用其问题与方向，不将社区诊断代写为本项目已证因果。

## 6. 时间预算及算力策略

主赛约120题共享12小时，包含沙箱准备，不含最终补丁验证。因此5分钟是本项目控制变量，不是比赛强制限制。若每题都用满5分钟，名义生成10小时，还留2小时准备；7.5分钟全跑满则15小时。不能从某份7.5分钟公开代码得分推断其最坏预算可行，实际短题早停、任务数与服务版本会影响总量。

研究建议：先优化“完成一个有依据的修复事务所花时间”，随后比较固定短预算与按证据状态分配思考的策略。不要把所有任务强迫快速编辑，也不要让长思考吞噬简单工具动作。当前允许每个LlmAgent不同sampling/adapter，但所有agent必须使用同一指定31B基座；小模型router、额外embedding模型和换大基座不属于当前主赛方案。

官方L4x4现按旧T4x2/P100的**两倍额度速率**消耗，并禁用互联网。过去“7小时Agent上限”不能当7小时额度预测。10月8日官方已延长排队超时、补处理受影响提交，并限制同时pending提交；不把每日1次等同每天一定有一个可用反馈。采用少数信息量高的变体，正式分用于跨分布验证。这里不进行任何算力申请或提交。

来源：[评估](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview/evaluation)、[加速器说明](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview)、[官方队列公告](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747519)。

## 7. 对历史研究决策的修订

- 保留P003 B无净增、0C保留16384等历史事实；不把旧成绩改写成新包条件的成绩。
- T2继续作为低成本编辑基础，不能凭机械错误减少就认定能冲牌。
- 把**根因假设、当前状态保持、短而完整的修复动作训练**提高到核心研究位置。
- 图不列主攻，但不再用旧名称解析错误否定当前图工具；先从明确符号扩展邻域。
- 思考预算已有新接口，值得独立研究；无须为此重做整套旧环境诊断。
- 奖金路线同时考虑主赛前三与论文资源奖；论文候选需回应新公开robust-edit工作，单独重做句柄编辑不足以构成充分差异。
