# 10月17日项目、榜单与训练部署证据

项目HEAD为`46b270f`，10月16日的提交将P006产物入库；本地没有P007新模型实验。远端账号最新30个Notebook中，最后一个本项目GPU实验仍为10月13日`p006-reference-pair`；本轮只看到一条正式submission，`57091482`，COMPLETE/0.13。[提交原件](submissions.json)、[账号Notebook清单](account-notebooks.json)。清单有一条不返回身份的Private Notebook，因此不把检索结果说成穷尽所有私人作业。

本次榜单前100：0.24为2队、0.22为27队、0.20为71队；10月13日相应为2、6、26，另66队0.18。可见竞争密度已提高，不能继续把0.18当足够竞争的目标。榜单只是当日公榜，不是金牌线或奖金保证。[原件](leaderboard.json)、[官方排行榜](https://www.kaggle.com/competitions/gemma-4-developer-agent/leaderboard)。

GPU周额度已刷新：已用2355.5126401秒，总108000秒，预留0，剩29.3456909额度小时；下次刷新北京时间10月24日08:00。它是共享账户快照，不是本项目独占预算。[配额原件](quota.json)。本轮没有消耗新GPU。

## 老部署证据必须准确命名

重新读取`research/005/ct-adapter-probe-20261010/result.json`：它确认31B诊断adapter文件、manifest、vLLM注册、请求确实选择`diag_ddp_step4`、返回一次get_status调用。**没有base/adapter同输入概率对照，也没有修复对照**。因此“已加载并路由”成立，“已经证明训练改变修复能力”不成立。12B更不能继承31B的这一观察。

[vLLM官方LoRA文档](https://docs.vllm.ai/en/latest/features/lora/)明确请求需要使用adapter对应model ID；`/models`存在只确认注册，效果要另看。官方本赛10月15日答复也说明同一贪心文本不足以证明无作用。下一次使用同一模型/模板/输入，比较base、真实adapter及同结构零delta控制，先量目标token/action分布，再看真实工具执行和修复结果。分布差异证明数值作用，不代表正收益；文本相同也不等于零作用。

本轮只读vLLM公开源码作为命名参考，**没有把上游v0.19.1等同比赛patched wheel**。不盲目改PEFT键、不合并或替换官方CT基座，也不再扩大合成诊断训练来代替真实数据。

## 新社区机制反证

[Found, not fixed](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/749198)，10月14日作者报告：两配置共同69题中，无生产修改25→7，实际解决26→30。两次运行有九项条件差异，不能归因于编辑deadline；大量失败已经修改了参考涉及的文件。此前“更早编辑即可解决”的说法需要收窄为“有效行为证据支持的正确修改”。本轮只读讨论正文和汇总，不读其任务或参考patch。原文已归档[topic-749198.json](topic-749198.json)。

它与本项目E2的DEBUG-only生产修改、弱断言、错误复现API相互呼应；不能用首次edit时间、流程遵从度或非空patch替代整题修复。这是下一轮数据目标与单变量候选的依据，不是当前正式0.13的隐藏逐题归因。
