# P009：多仓库真实SFT与完整Agent修复对照

2026-10-18。用户授权继续研究，使用`ccjjllt`账号Kaggle GPU，优先有效利用率，不建设测试或门控平台。本轮将CPU数据工作与一个有界GPU联合实验连接起来；最终效果以实际训练和Agent产物为准。

## 已完成的研究证据

**1. 纠正12B部署判断。** 主办方在[官方讨论748287](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748287) comment3536939已说明scorer采用`Gemma4ForCausalLM/model_type=gemma4`覆盖。P006原件已有这条证据，P008复核遗漏，导致把V6覆盖配置误判为自定义部署障碍。本轮明确更正，不再重复启动已知失败的默认Unified后端。V6仍未证明完整任务收益；本轮直接做完整任务对照。

**2. Requests假阴性已得到因果重放支持。** 对`requests_7205`原503B Agent补丁，正确源码路径下，原官方测试文件从`1 failed / 205 passed / 13 skipped`变为`206 passed / 13 skipped`；保持补丁、改导入同快照未修副本，原失败再次出现。未改变断言或读取gold patch。macOS/Python3.11与原Kaggle环境不同，目标文件preimage精确匹配，不能将整快照说成完全同一环境。P007原结果不变，另记本次重放。[完整证据](../../research/009/public-replay-20261018/README.md)。

**3. 首批真实训练数据已冻结。** 从预先分配train的仓库生成8个独立问题、7个repo的真实局部事务，8条编码无截断：35,992输入、6,147监督tokens，最长6,089。题目来自apispec两题，以及pydantic、tenacity、mypy、dataset、deepdiff、yamllint各一题。每条实际执行官方工具并观察修前失败/修后通过；tenacity是明确的静态构造器类型合同。没有把它们称为完整benchmark成功，`task_resolved=null`。

训练输入SHA256为`0aca8fc273f6f58e94c641f52b7bf3e4dc039c7275a9b719010ad6300c8a72f5`。tenacity无独立收益证据的copy动作仅保留上下文、已mask；alive-progress组件候选和两次失败尝试不入训练。外部选择8repo、比赛4repo、pydicom等均整仓隔离；原heldout与预留题未读取。来源许可证和已有公开教师出处逐条保留，没有新教师调用。[数据与边界](../../research/009/training-data-20261018/report.md)、[独立定点复核](../../research/009/public-replay-20261018/training-transaction-review.md)。

## 已启动的联合GPU实验

- 账号：**ccjjllt**；Kernel：[p009-12b-sft-agent-pair](https://www.kaggle.com/code/ccjjllt/p009-12b-sft-agent-pair)，V1于UTC **07:24** push成功，随后状态`RUNNING`。
- 资源：明确`--accelerator NvidiaL4`，4×L4；平台上限2640秒/44分钟，监督器42分钟，先训练再推理。
- 训练：复用已有7.71GB 12B NF4缓存；四rank DDP、r8/alpha16、LR2e-5、microbatch1、全局batch4、两轮、计划4个连续optimizer steps。实际最长6089，配置上限6144。保留独立adapter和最后optimizer checkpoint；不重复训练旧click探针。
- 对照：官方12B CT v2、v31、TP4、32768上下文、官方架构覆盖、maxseq4；单次启动共享服务，最多4个独立Evaluator。双方同六工具及相同public022提示、采样、8分钟/28调用预算，仅adapter不同。
- 固定诊断题：`rich_3067`、`fastapi_14463`、`requests_7205`，每臂各一次，共6次。均为已暴露、事先选定的开发诊断，不能估计隐藏泛化或公榜分数；12B六工具配方不继承来源0.22。
- 环境：双方的agent与verification统一将存在的`workspace/src`放在导入路径首位；保存原官方源码、实际修改与diff。这个本地研究修正不会进入正式submission。
- 利用率：训练阶段四卡同步有效梯度，推理阶段共享TP4服务滚动并发，10秒采样GPU利用率/显存/功率与vLLM队列。实际利用率待产物，不能因配置四卡就宣称已拉满。

当前正式提交57340172仍`PENDING`，已确认正式分数仍为上一版0.13。GPU研究作业不是一次新的正式submission。

GPU前共享账号配额：已用23,601.973秒，未预留，剩余84,398.027秒（约23.4439小时）。本阶段先将总配额支出控制在4小时内，不预支整周；44分钟四L4作业按既往倍率估算上限约1.47配额小时，最终以账号读数为准。执行清单见[job-request](../../research/009/execution-20261018/job-request.json)和[配额原件](../../research/009/execution-20261018/quota-before.json)。

## 执行状态与结果

V1已经启动，尚未回收实际训练/评测结果。官方实时日志接口曾返回500，普通日志暂为空；这只描述观测接口，不据此判定GPU作业失败。完成后补入真实optimizer步数、各rank峰值、adapter身份、每题新增修复/回退、耗时和配额。
