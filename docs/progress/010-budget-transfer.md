# P010进展：复用真实adapter研究响应预算

2026-10-18。P009已完成真实训练与完整Agent诊断，base/adapter均1/3；其失败轨迹显示重复长思考、思考结束后继续循环，以及尚未编辑便耗尽墙钟。详见[P009结果](../../research/009/execution-20261018/report-v1.md)。

据此启动P010，复用P009已导出的相同adapter，不重新训练。双方的thinking_budget改为1024、max_output_tokens改为4096，其余提示、六工具、三题、模型、时间/调用预算和src导入修正不变。这是两个预算参数的联合处理；缩短输出也可能截断有效工具参数，最终看真实修复和回退。

- Kernel：[ccjjllt/p010-12b-budget-agent-pair](https://www.kaggle.com/code/ccjjllt/p010-12b-budget-agent-pair)，V1已完成，2026-10-18 UTC 10:30回读为`COMPLETE`，六次任务原件已下载。
- 提交时间：2026-10-18 UTC 08:41（北京时间16:41）。
- GPU：4×L4，平台1800秒/30分钟，监督器1680秒、1620秒停止评测；最多六次完整尝试，不自动重试。
- Adapter SHA256：`c7845c23852dbc701a267eaa58be87464af90e25c62bfffb0286ac1033945ba2`；从固定P009 V1输出的training/adapter加载并核对。
- 启动前共享账号已用27,061.109秒、reserved=0，剩约22.4830配额小时。本次无训练和正式submission。

实际结果为 **base 2/3、adapter 1/3**；新增成功来自base的Requests修复。adapter仍仅修复Rich，FastAPI只添加导入、未完成主体修改，Requests零编辑。缩短思考预算得到执行，但没有证明训练收益。这三题已经反复用于诊断，不能把比例当排行榜预估或独立泛化结果。详见[逐题结果及机制](../../research/010/budget-pair-20261018/result-review.md)。

监督器总耗时1117.63秒，六次任务均完成、无外层超时；评测期四卡平均利用率94.60%。账号同期配额增加2246.176秒（0.62394小时），这是共享账号累计差值；截至UTC 10:31尚余21.8591配额小时。原件、失败轨迹和配额均已保存。

下一步转入P011批量数据与连续SFT，不再追加同样的三题GPU探针。现有训练仅8个独立问题、4次优化更新；另4条CPU事务尚未训练。新批次按独立issue扩量，先准备128条，进一步扩为512条；两者分别对应global batch4、2epochs下64和256次更新。候选来源计数不等于已制成训练数据，实际规模与启动时刻以P011产物和Kaggle回执为准。[研究条件与载体](../../research/010/budget-pair-20261018/README.md)、[实际作业回执](../../research/010/budget-pair-20261018/job-request.json)保留。
