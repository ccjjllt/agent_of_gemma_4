# 有界响应片段作为显式焦点动作

2026-10-28。本轮没有GPU、训练或模型rollout。新配置 `training/configs/vfarl_l4x4_live_pair_segment.yaml` 显式设 `rl.focal_unit: bounded_response_segment`；原配置与缺省值仍为 `complete_response`。新模式只接受之后真实采样器生成并由live-original-pair收集的记录，不能给旧length记录补字段后称为新证据。

## 已证实的筛选问题

当前live episode在单响应达到length上限时，会把实际生成token追加prefix、扣除实际token与turn，再按已有预算条件加入真实续写反馈，最后验收真实后续patch。旧prepare_rows仍丢弃这段finish_reason=length的焦点响应；配对因而只余一行或消失，分母也失去该组。固定两题cohort避免了异奖早停，但并未消除这种动作相关筛选。

`action-completion-censoring.json` 保存9/81种有限结果的完整精确枚举。全部终局reward在模型中已知，长度结束的动作不会被虚构为成功或失败：

- 一状态真实终局回报梯度−0.06；旧筛选期望上升方向+0.02。把完整观察到的有界片段作为动作后为−0.03，等于两样本组均值baseline应有的二分之一梯度。
- 两状态真实平均终局梯度+1.005；旧完整组分母期望方向−0.04633125。仅保留旧筛选分子而改固定预声明质量时为+0.129975，说明随机完整组分母还有额外影响；显式接纳全部已验证有界片段后为+0.5025。

这些数字采用固定token scale=1表达，生产128只将数字等比缩小。它们是有限MDP机制证据，未证明真实模型收益。旧R3的两条实际焦点动作均tool_calls，99/97token；此新机理不能解释之前观察到的负更新。

## 新动作边界与最小贯穿

动作是从已捕获prefix出发，直至自然tool_calls/stop或相同预声明token cap的真实采样response segment。length反馈是之后的环境转移。mask、forced control、实际old/reference概率、真实后续验收reward均沿用原件；不把续写token并入本次焦点loss。

native与batched真实循环分别记录requested_max_new_tokens/effective_max_new_tokens。collector记录现场snapshot的完整pre-action ledger、prefix与ledger哈希及实际context/episode/turn cap。prepare在新模式校验：实际length token数必须等于effective cap；requested/effective必须等于该prefix/ledger下的预算公式；两成员的边界与cap必须相同；原始成员的cap还须匹配其action日志证据。只有prospective live original及其真实恢复partner可构成新模式完整组。context_limit、缺cap、短于cap却标length、来源/预算不匹配都不变成可训练动作。

prepare的unit从RL配置贯穿collector评分、collector组判定、resident、native/Unsloth learner和runner预检查。配置文件由现有build_runner的training-config机制随源打包，不需增加新的runner开关。

## 手动机制观察

`bounded-segment-mechanism.json` 保存一次性CPU观察。真实native/batched采样循环用CPU模型及空CUDA记账运行：requested/effective为4/4与10/6，分别观测length/context_limit。没有真实GPU或实际模型forward证据。

实际episode函数在受控policy/tool/snapshot/verifier stub中跑过：先真实经过该函数的失败反馈路径，再采样到cap、追加length续写反馈、续写并提交；恢复partner从同prefix/ledger出发，首段cap相同。新模式保留两动作及其已知有限终局reward；旧模式保留0个训练动作。缺cap、异cap、改prefix/ledger、未到cap的length均拒绝；unknown partner不获得虚构reward或完整组质量。此stub不证明真实环境恢复、31B概率或奖励收益。

## 剩余边界

仍要求真实reference、终局验收及同状态恢复完成。原始episode验收unknown时不会启动配对，partner unknown时不形成训练组；这些可与动作有关，新unit并未修复该覆盖限制。不得补0或称全任务无偏梯度。相同后续seed只提供配对随机性，不能消除off-policy continuation、截断TIS、PPO/KL近似等既有边界；没有有效焦点状态的题也不增加完整action质量。
