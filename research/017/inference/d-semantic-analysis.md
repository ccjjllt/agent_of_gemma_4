# P017 D：真实轨迹的工具、修复与结束行为

2026-10-24。基于已回收的D全24次运行，只读原题、模型响应、实际工具回包和评测产物；没有调用模型、运行新增测试、修改Agent或读取heldout。Base与Step68均为6/12，冻结判据的新增W=1（Faker1741）、回退L=1（Arrow986）、净增0。下述语义诊断不改写这一主指标。

当前最明确的行为证据是重复操作、工具失败后不恢复、已有验证反馈未被正确利用，以及完成修复后的结束不稳定。**没有发现实际调用来源工具`bash / str_replace_editor / submit`。** 因而“来源工具名污染导致主要失分”没有得到这批D的支持；这不等于已经证明协议迁移完全无问题。

## 证据与统计口径

源归档：[p017-weight-ab-output.zip](evidence-d-v1/p017-weight-ab-output.zip)，8,831,294字节、605成员，SHA256 `4bbb0d6ee65cb3f616530860d01b728d8650e4be30bff89893d43c525bbef1b7`；内部根目录为`weight-ab-d-20261024T061136Z-7cbb50ca`。下文`003`等指其`attempts/003-arrow-py__arrow-986-adapter/`。

[behavior-evidence.json](evidence-d-v1/semantic-excerpts/behavior-evidence.json)保存24次计数、重点轨迹索引、原始wire与ADK成员SHA、请求及回包行号和少量原文。另提取Arrow986、Faker1741两臂的原题、scored.patch与F2P/P2P结果；未解包整个归档。具体指标来源为`analysis-rows.json`，修复判断以隔离judge的`f2p-execution.json / p2p-execution.json`为准。

- 工具调用只计每个真实`wire_response.choices[0].message.tool_calls`一次，不扫描历史messages重复计数；工具结果按调用ID关联`raw-adk-events.jsonl`。已核对每次运行wire响应ID、ADK事件ID与工具回包ID唯一。
- “完全重复”指同一次运行内工具名与解析后参数完全相同，首次之外的调用次数；“相邻重复”指连续两次工具调用相同。修改后重跑同一验证可能合理，故不把全部重复自动当失败。
- 请求序号是wire请求序号，**包含历史摘要请求**；摘要不计为普通Agent final。时间约数相对该次运行首个wire请求，以工具返回或响应返回时刻标记，不是完整作业墙钟。
- 服务等待与取消另见[d-runtime-analysis.md](d-runtime-analysis.md)。无响应请求没有可见的下一动作，不能断言其本来要生成final、submit或继续思考。

## 全24次工具行为

| 指标 | Base，12次 | Step68，12次 |
|---|---:|---:|
| 实际结构化工具调用 | 232 | 236 |
| 来源三工具或其他未注册工具名 | 0 | 0 |
| 工具参数无法解析为JSON | 0 | 0 |
| 参数键/类型与请求schema不一致 | 2次，2题 | 3次，1题 |
| 完全重复调用 | 16 | 89 |
| 相邻重复调用 | 1 | 31 |
| 工具回包`status=error` | 26 | 52 |
| `submit_patch`调用 | 6 | 15 |
| 普通Agent文本响应，无工具调用 | 5 | 2 |

工具分布为Base：run_command 118、read_file 87、edit_file 14、write_file 7、submit_patch 6；Step68：90、79、35、17、15。两臂都没有调用get_status。工具总数包含submit和被预算拒绝的调用，所以不等同于context的计费工具数。`status=error`混合了预期的故障复现、grep未命中、模型命令错误和预算拒绝，**不能直接作为协议错误率**。

| D问题 | Base完全重复 | Step68完全重复 |
|---|---:|---:|
| sqlglot1001 | 0 | 23 |
| Arrow986 | 1 | 1 |
| pyupgrade530 | 0 | 0 |
| Faker2190 | 4 | 26 |
| sqlglot2047 | 4 | 1 |
| Arrow1035 | 1 | 3 |
| pyupgrade133 | 1 | 7 |
| Faker1741 | 1 | 4 |
| sqlglot3401 | 1 | 1 |
| Arrow906 | 1 | 4 |
| pyupgrade656 | 2 | 18 |
| Faker1869 | 0 | 1 |

具体协议异常较集中。Base sqlglot2047第10请求将`end_line`传成字符串`100<|"|>`，收到明确`FileReadError`；同题第8、9请求还把`,start_line:70,end_line:100`拼进filepath，收到两次路径不存在。这两次虽然JSON键类型合规，参数语义仍错误。Base Arrow906第2请求、Step68 Arrow1035第4—6请求把键写成`start_line"`；运行时忽略未知键，实际返回1—150行并标`status=ok`，没有读到要求的151行以后。Step68重复三次后改用`tail`，Base下一次即改正键。

此外，Step68 Arrow986第21请求与sqlglot3401第16请求把目标六工具的`<|tool_call>call:...`文本留在reasoning字段，没有生成有效`tool_calls`或普通content；Base未见此种空动作响应。这是可见的序列化/解析兼容异常，不能混作成功工具调用，也不是来源三工具名调用。

重复的主要证据不是上述小量schema错误：Step68 sqlglot1001第5—28请求连续24次以完全相同的`old_string == new_string`调用edit_file，全部返回`No changes to apply. old_string and new_string are identical.`，没有有效源码修改。Faker2190和pyupgrade656还有明确重复环，见下文。三题合计贡献67/89次完全重复，不能把这个集中现象解释成每道题都同样退化。

## Arrow986：回退来自编辑失败后仍提交与错误汇报

原题要求humanize对重写`_format_timeframe`的语言输出整数，例如保加利亚语35天、916分钟。两臂均找到`SlavicBaseLocale`。证据：[Step68原题](evidence-d-v1/semantic-excerpts/003-arrow-py__arrow-986-adapter/agent-task.json)、[Step68补丁](evidence-d-v1/semantic-excerpts/003-arrow-py__arrow-986-adapter/scored.patch)、[Base补丁](evidence-d-v1/semantic-excerpts/004-arrow-py__arrow-986-base/scored.patch)。

Base第19请求约175.8秒将该类`delta = abs(delta)`改成`delta = trunc(abs(delta))`；第20请求实际重跑复现，输出从35.0/916.666…变成35/916。之后删除repro、提交、普通final，正常结束；隔离判据F2P 2项与P2P 208项均通过。

Step68第9请求约58.8秒已读到该实现，第11请求约81.2秒复现浮点输出。随后继续枚举各locale，直到第29请求约175.5秒才进行唯一生产源码编辑：把全文件`return form.format(delta)`替换为`return form.format(trunc(delta))`。工具明确返回：

> Target string occurs 5 times in arrow/locales.py. Please provide more surrounding context.

同一回包提示已用28/28次工具。模型未恢复编辑，第30请求提交，第31请求普通final声称多个locale已修改、复现已变成整数并“complete and verified”。实际最终补丁只有新增`repro.py`，没有`arrow/locales.py`修改，且唯一复现运行仍输出浮点。隔离F2P两项均失败、P2P 208项通过。这里可以明确判断过度探索、未处理多处匹配错误、提交内容与结果汇报不一致；没有依据称为工具名迁移失败。

## Faker1741：新增判据通过，但完整范围要求未满足

这是冻结主指标中的新增通过，保留W=1。原题自己的示例就是`fake.pyfloat(min_value=1.25, max_value=1.75)`，明确期待“pyfloat returns a float between min_value and max_value”；并非诊断时新增的输入假设。原源码接收这两个参数，内部`_safe_random_int(min_value: float, max_value: float, ...)`也明确标注浮点类型。题面没有精确规定端点开闭，但下述值严格小于最小值或大于最大值，对两种区间解释都越界。证据：[原题](evidence-d-v1/semantic-excerpts/015-joke2k__faker-1741-adapter/agent-task.json)、[Step68补丁](evidence-d-v1/semantic-excerpts/015-joke2k__faker-1741-adapter/scored.patch)、[F2P](evidence-d-v1/semantic-excerpts/015-joke2k__faker-1741-adapter/f2p-execution.json)、[P2P](evidence-d-v1/semantic-excerpts/015-joke2k__faker-1741-adapter/p2p-execution.json)。原函数内容还保存在behavior-evidence的`faker1741_original_function_source`（Base请求2、3的真实read_file返回）。

Base只写/修改repro.py，将复现上界改为2.75后打印一个数，没有改生产函数；最终原F2P 1项失败、P2P 61项通过。Step68第4请求约260.9秒把`int(max_value - 1)`改为`int(max_value)`，避免了整数部分相同情况下的空range；原F2P 1项及P2P 61项均通过。

但是Step68已有反馈直接否定了完整需求：第5请求用题面原输入打印`1.904952`，第6—9请求连续四次打印10个样本，其中包括`1.9`、`1.181`、`1.13491`、`1.0`、`1.97`。没有断言或对返回值做范围判断，也没有再次编辑。第10请求在最后一次样本输出后发出，约19.2秒后因会话时间耗尽被取消；无submit、无普通final，工作树回收的补丁通过冻结判据。

因此“工具命令exit 0”没有在该轨迹中转化为“业务结果符合题意”的判断。此项应写成**原判据新增通过、完整语义仍有已观察缺口**，不能静默改W/L，也不能宣称完整pyfloat修复。没有复跑reference/gold补丁，故其是否也有同类缺口未知；本报告不推断Step68比reference更差或更好，也不新增判据来追改这次主结果。

## Faker2190：提前提交后真实循环，非单纯最后一轮边界

Step68第10请求约54.4秒运行脚本，补全`en_PK.last_names`字符串项之间的逗号；第11、12请求验证得到完整姓氏，第13请求已成功删除脚本。第15请求约68.3秒首次`submit_patch`成功，11210字节、1个文件。第16请求是历史摘要：tools为空、只有汇总指令，其“已完成”文字不是Agent final。

摘要返回后，第17请求仍在末尾保留首次submit及成功回包；模型却再次`rm fix_last_names.py`，收到“No such file or directory”。第18请求再次submit，随后反复“删除已删除的脚本→重新submit”，直到第40请求提交、第41请求的rm被`Turn budget exhausted (40 turns)`拒绝。**总共13次成功提交，12次冗余重提；不存在提交后的普通Agent final。** 41个wire请求含1个摘要，恰为40个Agent请求。

该例已排除“首次submit恰好位于最后可用回合”的解释。循环出现在压缩之后，但后续上下文仍有成功提交的直接证据；不能仅按时间先后认定压缩造成遗忘。修复后的F2P 1项与P2P 134项均通过，故应同时记录“判据通过”和“结束行为失败”。Base该题有一次submit并普通final退出，没有同样循环。一般提交/结束契约、官方源码位置与SHA见已更新的[submit-contract-notes.md](submit-contract-notes.md)。

## 已留下通过补丁却超时：最后请求并不都处于同一步

下表是Step68，时间相对首wire请求。所有未返回请求均没有可见的后续动作；“正在准备submit/final”只能是猜测。请求取消的精确耗时见运行分析。

| 问题 | 已完成的编辑与验证 | 最后一个未返回请求之前的状态 | 结束/判据 |
|---|---|---|---|
| Arrow1035 | #9约71.5秒首次编辑，#10加Decimal类型，#12约121.6秒修正先前误改的datetime import；#14约133.4秒实际验证Decimal timestamp成功 | #15—17又读factory/arrow源文件；#18在约148.7秒读文件成功后发出，等待151.3秒被取消 | 无submit/final；最终生产补丁与Base相同，F2P 1通过，P2P 47通过/1跳过 |
| Arrow906 | #4、6—9重复同一grep；#10约221.1秒将None分支改为抛TypeError | #11直接跟在成功edit回包之后，等待78.9秒被取消；没有任何修后执行验证 | 无submit/final；回收补丁F2P 1、P2P 42通过 |
| pyupgrade656 | #8约67.8秒让open_mode处理io.open；#9看到两个插件均产生修改 | #10、12…24共8次写入完全相同repro.py，并各自执行，未推进生产修改；#26约282.4秒是未返回的历史摘要请求 | 无submit/final；F2P 1、P2P 28通过；保留多余repro.py |
| Faker1741 | #4约260.9秒编辑；#5—9反馈已经包含越界数值 | #10在第四次十样本输出后发出，无返回；不能称验证已完成 | 无submit/final；原F2P 1、P2P 61通过，语义缺口如上 |
| Faker1869 | #18约111.7秒用numerify补足法国邮编位数；#19约113.8秒脚本报告全为5位，#20成功删除脚本 | #21约129.2秒git diff成功；#22随后等待170.8秒被取消 | 无submit/final；F2P 1、P2P 331通过 |

Arrow1035的三个错误`start_line"`键没有造成最终判据失败，但浪费读取机会；第一次import编辑还引入了额外问题，压缩后的#12进行了修复，说明并非“压缩后必然不会恢复”。Base约72.2秒完成最终编辑，实际验证整秒/小数timestamp及异常输入，约120.2秒提交、139.6秒普通final结束。

Arrow906的Base也有一次错误键，随后改正；先删除None特殊分支后出现SyntaxError，读回源码把悬空`elif`改成`if`，再运行确认None抛TypeError、无参数get仍有效，约171.0秒提交、179.5秒普通final结束。这是与Step68“编辑后即进入未返回请求”不同的恢复/验证证据，不能仅比较两者最终F2P均通过。

pyupgrade656的Step68重复脚本只是检查两个visitor是否产生修改，未演示完整CLI一次收敛；相同内容写8次是有证据的冗余操作，不是多种边界检查。Faker1869则在编辑、验证、清理、diff均完成后遇到长时间无响应，不能像Faker2190那样定性为可见的提交后循环。此处API等待可能混合排队、生成、服务及取消行为，现有客户端轨迹不足以拆因。

## 对下一步研究的约束

这批证据支持把六工具适配的重点细化为：根据真实工具错误调整下一步；确认diff实际包含生产修改；将验证输出与题面要求比较；避免相同编辑/脚本重复；在成功提交后可靠结束，并保留压缩前后的任务状态。仅替换来源工具名不足以覆盖已观察问题。数据应来自独立训练任务，不得把本报告的D题、答案或轨迹回流训练。

目前不能从这12个已熟悉问题推出总体能力提升、协议迁移已失败或训练后段过拟合。不能把6题适配种子缺少普通final监督推广成537题主训练也有同样缺口；也不能把所有timeout归因于模型输出冗长。Step30、独立C和必要的关键差异复核应按冻结计划推进；这份报告提供失败类型和证据，不另建Agent或判据来追求D分数。
