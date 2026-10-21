# P014 真实提交 57438326

**2026-10-21 更正：该提交因整体运行超时失败，没有分数。** 原始API为 `status=COMPLETE`、`public_score=""`，但 `error_description` 明确为 `Your submission notebook exceeded the allowed runtime`。COMPLETE在这里仅表示处理结束，不能当作评分成功。详见[失败原件](failure-20261021/submission-full.json)和[修复说明](failure-20261021/README.md)。下文PENDING为上传时历史快照。

2026-10-20 **20:46:56.930 北京时间**，使用既有参赛身份 `ccjjllt` 成功提交；上传进程退出0，CLI返回 `Successfully submitted`，列表随后返回 **57438326 / PENDING**。分数尚未返回，不宣称超过当前0.15，也没有奖牌证据。官方状态见[比赛提交页](https://www.kaggle.com/competitions/gemma-4-developer-agent/submissions)。

- 文件：`submission.zip`，3089 bytes；与新八题实际运行的 `p014_single_v1` 完全一致。
- SHA256：`17435502e614518e29bd74077e1f1a05515c86708a269375b20b6d9f8c7ab3c9`。
- 31B CT v2，无adapter，8min/28calls/80turns，temperature0.2、top_p0.95。
- 同源码/预算的新开发面板，候选8/8、基线8/8；W=0、L=0。只有冻结F2P见证，无独立P2P；本地收益没有得到证明。
- 八次候选真实attempt均经过官方 `run_agent_sandbox/compile_submission`；提交前没有悄悄修改prompt，也没有另开GPU打包作业。
- 提交理由：候选实跑可用、没有观察到该面板回退、已有明确执行习惯改动，按已授权的探索路线获取官方反馈；不是已确认提分方案。
- 上传前 `numAllowedNow=1`；上传响应为当日剩余0次。不要重复上传。

原始[上传回执](upload-receipt.json)、[提交列表](submissions-after-upload.json)、[单次提交记录](submission-record.json)、[逐文件指纹与选择依据](submission-manifest.json)均已保存。原0.15提交 `57340172` 与ZIP继续保留。

最新只读查询：北京时间 **20:49:53**，仍为 **PENDING**，publicScore为空。原件见[最新提交列表](submissions-latest.json)。

后续读取完整 `api.competition_submissions(...)` 对象，同时保存 `status`、`public_score`、`private_score`、`error_description`。CLI默认字段不含错误描述；COMPLETE且无分数不能认定成功，有error_description应按失败处理。修复版使用独立目录和新回执，保留本失败包。下一轮收益研究见[执行交接](../execution-20261020/README.md)。
