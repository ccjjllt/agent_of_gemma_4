# P010进展：复用真实adapter研究响应预算

2026-10-18。P009已完成真实训练与完整Agent诊断，base/adapter均1/3；其失败轨迹显示重复长思考、思考结束后继续循环，以及尚未编辑便耗尽墙钟。详见[P009结果](../../research/009/execution-20261018/report-v1.md)。

据此启动P010，复用P009已导出的相同adapter，不重新训练。双方的thinking_budget改为1024、max_output_tokens改为4096，其余提示、六工具、三题、模型、时间/调用预算和src导入修正不变。这是两个预算参数的联合处理；缩短输出也可能截断有效工具参数，最终看真实修复和回退。

- Kernel：[ccjjllt/p010-12b-budget-agent-pair](https://www.kaggle.com/code/ccjjllt/p010-12b-budget-agent-pair)，V1已push，随后实际回读为`RUNNING`。
- 提交时间：2026-10-18 UTC 08:41（北京时间16:41）。
- GPU：4×L4，平台1800秒/30分钟，监督器1680秒、1620秒停止评测；最多六次完整尝试，不自动重试。
- Adapter SHA256：`c7845c23852dbc701a267eaa58be87464af90e25c62bfffb0286ac1033945ba2`；从固定P009 V1输出的training/adapter加载并核对。
- 启动前共享账号已用27,061.109秒、reserved=0，剩约22.4830配额小时。本次无训练和正式submission。

当前尚无P010修复结果，不提前判定有效。[研究条件与载体](../../research/010/budget-pair-20261018/README.md)、[实际作业回执](../../research/010/budget-pair-20261018/job-request.json)已保存。
