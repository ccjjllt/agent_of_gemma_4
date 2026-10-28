# MySQL copy 契约的边界增强

2026-10-28，固定100题原索引42，F／F2。当前题目是 question-revision-r2 的 `kayak__pypika.1c9646f0.func_basic__x7p5x68a`；没有改题面、奖励、历史示范、selection 或 curriculum。

原 mutation 使 `MySQLQueryBuilder.__copy__` 返回 None，原 oracle 的44个F2P确实发现它。历史 submitted.patch 恢复返回值并隔离 duplicate updates，是有效的目标修复。本轮另发现同一公开 copy／immutable builder 契约的相邻边界：`_modifiers` 列表仍由原 query、副本及分支共享。它是 snapshot 原有缺口，不是历史 patch 新引入回归，也不是 OSA 那种只打印常数便过验收的弱 oracle，不据此暂停RL。

契约来自真实 source-snapshot：`pypika/utils.py:42-59` 的 @builder 文档明确 make build functions immutable，并实际 copy.copy(self)；`pypika/queries.py:794-795` 说明返回 query 的副本；`pypika/dialects.py:165-173` 的 modifier 使用 @builder 后 append 到列表。原 task 的同状态、distinct instance、正常链式构造要求及 r2 bounded direct-copy 澄清均保持不变。原 test_patch 的 test_modifier_select／test_multiple_modifier_select 只断言最终生成SQL，没有保留旧分支来观察反向修改。

`manual-observations.json` 记录真实源码三态（只临时 materialize 生产源码、原 patch 用 git apply 应用）：baseline direct copy 是 None，正常构造抛 AttributeError；历史 reference 后 direct copy 正常，但给副本添加 SQL_NO_CACHE 会把原 query 从 ``SELECT SQL_CALC_FOUND_ROWS * FROM `users` `` 改成两个 modifier，并且三个列表 identity 相同；candidate 只在原 reference 上增加 `newone._modifiers = copy(self._modifiers)`，原 query 和兄弟分支保持各自SQL。duplicate update 分支作为对照，原reference与candidate均保持隔离。没有扩大为无依据深拷贝。

`teacher-r1/` 是新执行的官方六工具示范：7次工具调用和1条真实终止回答。原88个 source-inventory 文件的内容和mode完整恢复，tree SHA `fe0eb4d3245044a57e7d906c4aef31eabb10f6fd07e5a3b8f0160a7fed905408`。snapshot内额外的post-setup oracle文件未冒充原始source。恢复原整数60分钟预算、90秒命令上限及180秒verification上限；调用未改动的 run_official.replay API，避免CLI把60转成60.0后改变prompt。真实新prompt SHA `ed88fa11c2b0ad24aa95cd4c71affc94762e2b01c322112516d2e4ff29a37623` 与r2逐字一致；新task SHA `c770113fcd9fd1ffa4ec37ef6f34afa18069201510f1ff14fb11f58cc4be0723`。

原1061个验收节点全部保留：baseline 44失败／1017通过；生产candidate后1061通过。真实总用时9.481秒。六工具stdout中的最终生产源码SHA与手动candidate相同。canonical原normalizer准入无缺项；单题source-catalog独立披露原文与新澄清来源，原catalog未改。

可用新增量是 [gemma-sft/sft-actions.jsonl](gemma-sft/sft-actions.jsonl)：8条决策，1333个监督tokens，最大上下文4060；SHA `3c3e7777866b25781ad3b14fe7aad2a6981935e345364fce63c89607057add55`。这是Codex指定并实际执行的repair-enhancement off-policy示范；没有模型推理、行为logprob、GPU或训练更新。没有重写旧teacher或追认旧奖励。macOS CPU成功不等于Kaggle/Linux等价或模型独立修复收益。

初次机制证据组装使用了错误metadata键，在写summary前停止；修正为实际官方prompt键并重新执行三态观察。其他初次依赖/方法名查找失败也保存在manual-observations，不冒充成功行为。完整hash绑定见 enhancement-summary.json。
