# P005 single_state 首次正式提交

2026-10-12，北京时间。用户授权操作此前建议的无 adapter 正式提交。

**提交ID `57091482`，2026-10-12北京时间09:42:48接收；10月13日后续复查已为 `COMPLETE`，公开分 `0.13`，私榜未返回。** [最终成绩API原件](final-score-20261013.json)。官方入口：[本队提交记录](https://www.kaggle.com/competitions/gemma-4-developer-agent/submissions)。下文保留提交与排队时的历史观测。

## 提交内容

- 使用 2026-10-11 v3 四题配对中冻结的 single_state Agent，未改变提示、技能、采样、每题预算，也未加入 adapter。
- 将原 `agent-bundle-005-pair-single_state.zip` 按字节复制为本目录 `submission.zip`；41,837 字节，15 个文件，根目录为 `agent.yaml`。
- SHA256：`1f5d0001e6bbfcf499cea0c91e94e97cb1e331cf4e4147c9f745acd48df429e1`。
- 一次性手动核对 ZIP、原 manifest 和当前候选源码的所有文件内容一致，配置引用完整，没有任务数据、模型权重、额外路径或 symlink。没有新增测试文件。
- [准备清单](submission-manifest.json)记录来源；[官方回执与额度](submission-receipt.json)记录实际提交。

## 实际操作与结果

CLI 使用标准文件名 `submission.zip` 完成上传与 CreateSubmission，退出码 0，返回 `Successfully submitted`。MCP 随后查得对应 ID、描述及提交时间；CLI 提交列表确认 `SubmissionStatus.PENDING`，两个分数字段均为空。历史 10 月 1 日的 HTTP 400 原因仍未确定，本次成功不能单独证明旧错误由文件名造成。

提交次数从 today/lifetime/remaining `0/0/1` 变为 `1/1/0`，本次消耗一个正式提交名额。没有额外启动 Notebook、训练或开发评测 GPU 会话；官方评分由平台处理。

本地四题 2/4 仅是已暴露开发题上的探索结果，不是本次官方分数预测。分数需等待 `57091482` 实际评分；不因等待而重复提交。

## 2026-10-13 状态复查

北京时间 11:23 查询：CLI 仍显示 `57091482 / SubmissionStatus.PENDING`，publicScore/privateScore 为空，距提交约25小时41分钟；MCP元数据仍未提供评分阶段、错误或分数。只读查询不能确定它已执行多少题，也不能单凭 PENDING 断言尚未开始模型运行。

Kaggle官方STAFF/ADMIN Dustin在[队列公告](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747519)说明L4供给不足、提交积压；北京时间10月11日00:01报告[当时排队约33小时](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747519#3536532)，11:24进一步说明[排完队后还需实际运行时间](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747519#3536694)。Ryan Holbrook于10月10日20:26明确[排队不计入12小时运行限制](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747519#3536449)。这支持平台队列延迟是当前最可能解释，但33小时是历史估计，不是本提交的当前ETA。本次未发现10月12–13日新的官方耗时承诺。

此次没有取消、重复提交、启动GPU或发送讨论消息；原提交回执保留提交当天的观测。

## 2026-10-13 出分后的研究

用户报告0.13后，官方API已复核status=COMPLETE、public_score=0.13。未获得隐藏逐题轨迹，不能从总分推断具体失败机制；本轮没有重交。已更新比赛规则/高分Notebook/前沿研究，形成[006研究路线](../../../docs/plan/006-from-013-to-competitive-agent.md)。
