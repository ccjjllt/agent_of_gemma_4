# P007进展：高分源码升级、独立选题与训练目标重排

2026-10-17，北京时间。用户要求结合比赛数据、高分Notebook与前沿研究继续深挖金牌和奖金路线。本轮完成研究与准备，没有GPU、训练、教师调用、新提交或heldout内容读取，没有新增测试文件。

## 新得到的事实

- 官方API确认仍只有submission57091482，COMPLETE/0.13。账号最新Notebook清单未见本项目晚于P006的新实验。10月16日Git提交是此前成果入库，不是新增模型收益。
- 榜单前100已全部≥0.20：0.24两队、0.22二十七队、0.20七十一队。账户周额度刷新后查询剩29.3457小时，为共享瞬时余额。
- 核实并冻结flexonafft V10 / script357955943的**0.22完整四文件**，原8分钟无改动。0.20 Jvn V1与旧0.18 flex V9四文件逐字相同，只保留身份对照。
- 新31题同预算研究的BudgetFit/BudgetAware都9/31，附加工具臂都8/31；运行硬件为A100，不能外推官方L4成本。额外声明工具并未形成收益证据。
- 公开QLoRA V4确实训练804.7秒、loss下降，但精确版本公榜0.03；训练基座、目标和Agent预算等多项同时不同，不能单归因LoRA，也不能称已获修复增益。
- 官方确认LoRA可作用于前向，简单贪心输出相同不足以否定；隐藏测试回放明确不开放。wheelhouse仍v31、模型/data元信息未变，12小时规则未改，不为版本名重跑历史。

## 形成的具体准备

[原样0.22候选](../../research/007/candidates/public022_v10/README.md)已保存，来源/许可与agent目录分开，未打包运行或提交，作者分数不记为本项目成绩。

只读元数据冻结了[公共12题身份](../../research/007/evaluation-design-20261017/public-dev12-manifest.json)，来自未在归档trace中命中的dev_future问题，每repo4题、12个快照；原heldout与预留8题及同快照排除。没有读取新题内容或宣称环境可运行。

复用已有SWE-smith分片，仅投影来源列，排除比赛四repo和此前抽读20个repo后，发现184条记录、183个真实PR、36个新repo。固定哈希选择[8个外部候选仓库](../../research/007/evaluation-design-20261017/external-pr8-manifest.json)及替补顺序，整仓排除未来训练；buggy实例、任务与环境尚需准备。不是已取得183条训练样本或8题benchmark结果。

精读SWE-Tester、TestPrune、From Plan to Action、Kimi-Dev、QATFactory、Agent Skills Can Be Harmful六项一手研究，记录实际模型/预算/反证与来源边界。新排序聚焦“有效行为证据→正确生产修改”，同时把12B真实更新、官方CT数值作用和整题收益分开。

## 决定与未完成项

下一轮优先原样0.22参考和单一行为验证改动，使用新问题观察净增/回退；12B只先安排一个真实样本的更新、导出和CT作用观测，之后才决定短SFT。建议首轮总额度上限8h（4.5h公共对照、1.5h12B、2h外部配对），不是本轮已用额度，也不视为全周自动授权。

尚未复现0.22、尚未运行新公共/外部题、尚未下载完整12B权重或得到真实SFT收益。本项目正式分仍0.13。候选强弱、未来成绩和奖牌均不预报。

来源与下一步：[007计划](../plan/007-public022-and-transfer.md)、[官方更新](../../research/007/official-data-20261017/report.md)、[高分原件](../../research/007/high-score-notebooks-20261017/report.md)、[前沿研究](../../research/007/frontier-20261017/report.md)、[选题依据](../../research/007/evaluation-design-20261017/report.md)、[状态与部署证据](../../research/007/status-20261017/report.md)。
