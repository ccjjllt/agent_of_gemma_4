# 新任务小轮实际执行

所有题来自S600 train；四道研究验证题不参加focal采集或更新。这些小样本结果不构成比赛heldout成绩。

2026-10-27T04:44:27.738347+00:00：开始固定12＋4研究轮；平台输出只按实际执行结果报告，不扩充100题。

2026-10-27T04:45:04.973654+00:00：目标账号CPU运行时实际完成，Unsloth依赖已组装；归档SHA：0a8e277010c63701b1a163869bd6998732c18307770d7297a5cea524391879a0。CUDA尚待实测。

2026-10-27T04:45:04.973820+00:00：复用本轮已实际完成的CPU验收原件：competition/vfarl/l4x4/next-round/cpu-v4-evidence/vfarl-prevalidate/prevalidation.jsonl；不重复验收。

2026-10-27T04:45:04.977308+00:00：新题Linux CPU实际可用：16/16，源作业COMPLETE。

2026-10-27T04:45:12.826390+00:00：GPU作业已提交并回读源码/私有设置/输入：jiyujia/vfarl-newtasks-collect-auto-r3/1。


北京时间12:57已确认真实采集：两worker均完成turn0/1/2共6次policy_action，finish_reason均tool_calls，并进入turn3请求。首动作55.216/45.294秒；随后24.080–34.388秒。原始日志observer-20261027T045641Z.log。尚未取得完整任务奖励、异奖组或更新，不宣称加速比/泛化收益。后台驱动继续collection-only；至少2个真实异奖组且余额≥6h时，才续办同一未训练batch的一次RL更新与4题独立前后对照。

北京时间14:48状态实读：auto-r3/1仍RUNNING；两worker无退出，四L4均可见，无worker_failure/collection_error。14:47:43显存MiB/利用率分别卡0=9668/0%、卡1=12986/48%、卡2=10068/0%、卡3=13586/22%，每卡总23034MiB；多个相邻观测可见各卡利用率轮换，不能把瞬时0%判作掉卡。Funcy与Bottle两个新任务均已完整形成rewards=[1,0,1]的真实工具失败后分叉；第三题TextFSM正在分叉，另一采集器进入python-string-similarity新题。尚未提交RL更新，完整训练候选仍待采集结束后prepare_rows核对。账号实读GPU已用11.69480h、剩18.30520h。原始日志status-20261027T064829Z.log、配额quota-status-20261027T064835Z.json。

北京时间14:54用户指出首步更新等待过长。确认原调度的采集停止目标为3异奖组、更新最低为2组；还需等全部在途分支与每个worker两次actor-before评分结束，驱动只在整个collection Notebook COMPLETE后切换。实时API输出列表为0，已知carrier/focal路径下载均404，无法从日志摘要重建精确行为token/logprob，当前不取消以免丢失真实batch。已修改尚未冻结的续办runner：先一次真实optimizer，之后用原始冻结base＋fresh seed0 adapter做4题未更新策略控制，再做4题更新策略；固定任务/种子/预算不变，measurement_order和actual_update_before_baseline明确记录墙钟顺序。一次性AST解析通过，当前采集绑定的actor/tool/source_files均未变化。未来采集目标下调为2组，auto-r3既有3组目标不热改、不重采。
