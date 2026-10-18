# P009 首批真实训练事务：8个问题、7个仓库

2026-10-18。本轮产出已冻结，可供本次短SFT使用。数据SHA不再覆盖；后续变化必须另起版本。没有启动GPU、提交或付费教师，没有读取heldout、预留题或外部选择题正文，没有运行仓库测试套件或新建测试文件。

**最终输入：[encoded-train-12b.jsonl](encoded-train-12b.jsonl)，473,829 bytes，SHA256 `0aca8fc273f6f58e94c641f52b7bf3e4dc039c7275a9b719010ad6300c8a72f5`。8个独立issue、7个repo，35,992输入tokens、6,147监督tokens，最长6,089；全部完整保留历史，8,192上限没有丢弃或截断。**

这是一批实际CPU执行的局部修复示范，不是完整issue benchmark成功集，也不是模型自主解题成绩。每条的task_resolved都为null；本轮完整benchmark运行数为0。

## 1. 先分配训练仓库，再读内容

从P007已冻结元数据的剩余repo顺序选前8个，主代理确认没有启用替补后，正式分配为train：apispec、pydantic、tenacity、mypy、dataset、alive-progress、deepdiff、yamllint。未来外部选择替补必须跳过这8个repo。

整仓排除原外部8repo、比赛4repo、pydicom；click容量样本和MarkupSafe candidate均未回流。已有heldout/预留边界不变。分片仍是已缓存77,754,558-byte SWE-smith tool分片，revision `08e109b4a59eaeebf80e4675cd125d42e7ac99a4`，SHA `e41ffbd3f9d1571917ac5b9f4d0108ded4799d9776f597a054203467135a5715`。

首轮alive-progress只证明局部组件，按主代理意见保留candidate。随后按已分配train的repo顺序进行第二轮，apispec只剩PR950一个新的真实PR；先登记元数据再读取正文。最终读取9个train问题，产出8个主训练问题；**没有把同issue多片段或失败重试算新问题，也没有拿选择题补数**。精确用途见[data-usage.json](data-usage.json)。

## 2. 八条实际观察与编码长度

| 问题 | 修前观察 → 修后观察 | 输入 / 监督tokens |
| --- | --- | ---: |
| apispec PR851 | query参数deprecated留在schema中 → true/false均移到parameter层；缺省与component property行为保留 | 4020 / 634 |
| pydantic PR6809 | 自定义序列长度正好3被MaxLen(3)拒绝 → 长度2/3通过、4继续拒绝 | 3259 / 550 |
| tenacity PR384 | 实际mypy对正确构造器callback报3个arg-type → 0；str返回的错误callback仍被拒绝 | 6089 / 827 |
| mypy PR15392 | TypedDict空key触发INTERNAL ERROR → 正常missing-key诊断；有效key仍无错误 | 5139 / 691 |
| dataset PR423 | Alembic1.12.1添加新列因schema位置参数TypeError → SQLite真实upsert新增列且原值、行数保留 | 2971 / 348 |
| deepdiff PR496 | NumPy bool数组hash报memoryview异常 → 1D/2D数组稳定hash，true/false区分，整数数组hash保持一致 | 4875 / 687 |
| yamllint PR618 | check-keys配置未知 → 实际linter正确区分未引用key、错误引号、正确key与value，并保留默认忽略key | 5582 / 1622 |
| apispec PR950 | plain Dict缺显式additionalProperties → 输出{}；Raw/Integer Dict与标量输出保持 | 4057 / 788 |

以上全部调用的是对应固定源码中的实际API，**tenacity明确例外为真实mypy静态分析其源码注解**，不宣称该buggy分支运行时导入已经修复。每条的修前与修后执行同一条inline命令，没有用修改判据代替修复。

第8条PR950的完整证据：[transaction](outputs/apispec-pr950/attempt-01/transaction.json)、[events](outputs/apispec-pr950/attempt-01/events.json)、[production.patch](outputs/apispec-pr950/attempt-01/production.patch)。原4例只有plain Dict的输出不符合要求；生产修改后4例通过，官方submit_patch捕获一个生产文件。用户原问题关于OpenAPI缺省additionalProperties的解释不作为本报告的语义结论；这里验证的是请求的显式schema输出及邻接行为。

[training-data-summary.json](training-data-summary.json)保存逐题精确长度；[编码报告](encoded-train-12b.jsonl.report.json)为8/8写出、0超长跳过。每题不同的输入长度来自完整原始问题、实际读文件和新执行回包，不是删历史凑窗口。

## 3. 实际工具语义与监督边界

数据生产器[produce.py](produce.py)复用P006的原样v31 swegemma工具实现和LocalTransport。read_file/edit_file/run_command/submit_patch的回包均为本次实际执行结果；文件/进程传输在macOS本地临时目录，未冒称Docker环境等价。新建临时Git baseline用于真实捕获生产补丁；源镜像按完整buggy commit固定，未把profile短SHA当作buggy SHA。

没有运行任何外部轨迹里的shell安装/复现脚本，也没有把bash或str_replace_editor简单改名。源中的编辑建议被显式提取为参数，随后实际调用官方edit_file；所有后续观察重新执行。dataset采用schema关键字的最小修复，避免沿用源轨迹中吞掉TypeError的回退；alive-progress也未照搬源轨迹的大量无关修补。

训练只监督有证据支持的成功生产修改、成功的修后行为命令、submit_patch。修前失败命令与回包保留在完整上下文，不计loss。所有tool输出、问题、源码和角色前缀均不计loss，thinking不监督。

tenacity特别处理：
- attempt01因实际源码缺RetryBaseT，修前修后均import失败；整条排除。
- attempt02遇到timeout与repo mypy配置的files/-c冲突；整条排除。
- attempt03用真实mypy、明确的命令配置分析源码，构造器3个arg-type→0，错误callback仍被拒绝。
- copy签名虽然实际修改且保留在历史里，但原before没有独立copy失败，**ordinal3被mask**；最终target_assistant_steps为[2,4,5]，监督量从1,447降为827。没有重写已发生的工具观察，也未把运行时import问题说成已修好。

这批主输入包含8条局部修复事务，没有经验证的edit匹配失败恢复链；不能仅凭它训练后有数值变化就声称已学会工具恢复。独立agent已经只读复核实际动作、来源cwd/PYTHONPATH、修前后区分和mask，未靠重复跑测试增加“通过数”。

## 4. 未纳入样本与环境边界

[alive-progress事务](outputs/alive-progress/attempt-01/transaction.json)实际执行原源码AST工厂函数及真正refresh_secs表达式，0.25拒绝→接受、整数/0/负值/字符串/上界与其他整数选项行为保持。它不是字符串包含检查，但绕过模块初始化和public alive_bar路径。主代理决定本次保留candidate，allow_parameter_update=false。

源码下载保留完整branch→commit stdout、archive/file SHA和许可证，见[source-manifest.json](source-manifest.json)。alive-progress归档超出20MB小下载范围，丢弃该响应后只取固定commit的configuration.py与LICENSE；其余8个训练问题的archive已保存。源码解包排除了tests/test/docs目录，没有为了凑验证通过安装或跑整个仓库。

必要依赖仅装入/private/tmp/p009-replay-env，使用固定PyPI wheels；没有执行仓库setup.py或安装脚本。实际API合同只用inline Python或mypy -c；实现中不创建测试文件。编码依赖后续安装导致typing_extensions从4.12.2升到4.16.0，执行时序及两阶段版本均记录在[environment-and-assets.json](environment-and-assets.json)，没有隐去环境变化。

模型编码使用已存在的12B CT v2 tokenizer与17,466-byte模板：
- tokenizer.json SHA256：`cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`。
- 模板SHA256：`36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0`。
- 实际encoder tokenizer指纹：`d001718145f4980b65e70b5ffa1dfce93b3eca5a5d8de2446b8c483f2b4b8927`。
- 协议为p005-harness-tools-v1，未下载权重，CPU环境也未安装torch。

## 5. 许可、来源与可复现资产

公开轨迹的数据卡为MIT，原始教师来源按每条model字段保留，均为此前公开的Claude3.5/3.7来源；本轮没有调用这些教师或复制其推理文本。源码许可分别保存：apispec/pydantic/mypy/dataset/deepdiff为MIT，tenacity为Apache-2.0，yamllint为GPL-3.0-or-later。**不能把混合源码材料统一再标为MIT或Apache**；分发相关源码/示例时保留对应许可证与来源。数据集许可标签不替代第三方来源条款判断。

主资产：
- [transactions-train.jsonl](transactions-train.jsonl)：8条可追溯的完整官方工具事务。
- [encoded-train-12b.jsonl](encoded-train-12b.jsonl)：本次冻结训练输入；不要传provisional文件或source-trajectories整体。
- [data-usage.json](data-usage.json)：repo保留、精确admitted样本、candidate、失败attempt与文件指纹。
- [output-manifest.json](output-manifest.json)：输出、实现和计划指纹。
- [transaction-plans.json](transaction-plans.json)：本次审阅后的生产编辑与inline命令；不是测试套件。
- [replay-summary.json](replay-summary.json)：全部11次事务尝试，包含2次tenacity失败和1条组件candidate。
- [assemble.py](assemble.py)：本次事务组装和CT编码实现；当前输出已冻结，后续扩量必须另起数据版本，勿覆盖已运行输入。

首批比此前单click容量样本增加了独立问题和行为覆盖，但仍远低于32issue/8训练repo准备目标。它足够支持一个小规模、可归因的连续更新实验；是否提升真实修复，仍须由同基座base/adapter在隔离任务上的实际结果回答。

## 6. 冻结数据的监督token组成（只读补充）

GPU版本冻结后，使用现成CPU tokenizer环境做了一次性统计，没有改数据或安装新依赖。对8条**完整对话**再次调用编码器的官方CT render，确认render token IDs、完整文本tokenization IDs与冻结input_ids三者完全一致；随后把每个labels非-100且位置大于0的真实token，依据offset_mapping归入其唯一supervised span。没有用字符跨度长度代替token数；未分类或多义归属为0，合计精确为6,147。

| 监督类别 | 字段/动作数 | 实际监督tokens | 占比 |
| --- | ---: | ---: | ---: |
| edit_file工具动作 | 11 | 3,366 | 54.758% |
| run_command修后验证动作 | 8 | 2,426 | 39.466% |
| submit_patch工具动作 | 8 | 64 | 1.041% |
| assistant正文 | 27段 | 291 | 4.734% |
| 合计 | 27个工具动作及其正文 | **6,147** | **100%** |

工具类别包含该动作的工具名、序列化参数及tool-call界定token；不包含角色前缀、工具观察或未监督的修前命令。edit_file的3,366包括old_string和new_string等参数，不能全部称为“新修复代码tokens”。assistant正文是动作旁的简短说明，没有监督源教师推理。

| 问题 | edit_file | run_command | submit_patch | 正文 | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| apispec PR851 | 192 | 403 | 8 | 31 | 634 |
| pydantic PR6809 | 274 | 237 | 8 | 31 | 550 |
| tenacity PR384 | 472 | 316 | 8 | 31 | 827 |
| mypy PR15392 | 411 | 241 | 8 | 31 | 691 |
| dataset PR423 | 84 | 212 | 8 | 44 | 348 |
| deepdiff PR496 | 344 | 294 | 8 | 41 | 687 |
| yamllint PR618 | 1,177 | 386 | 8 | 51 | 1,622 |
| apispec PR950 | 412 | 337 | 8 | 31 | 788 |

**解释：验证脚本占比明显，但不是全体目标的多数。** 整体39.47%在学习修后run_command；apispec PR851和dataset两题中该项超过各自监督量的一半。这8条修后命令与已在上下文中出现的修前命令相同，因此其中有相当部分是重复已有验证脚本的输出学习，不能把总loss下降直接解读为补丁生成能力提升。另一方面，yamllint一题占总监督量26.387%，并贡献34.967%的edit_file目标；单题权重差异也会影响实验解读。

这些是token数量口径下的组成比例，不是实测梯度范数或参数更新贡献；不同batch、token loss及更新顺序仍会影响实际贡献。本次训练维持已冻结监督。下一版若研究目标更集中于生产修复，可保留完整验证历史，只调整修后重复命令的loss权重/是否监督，或按issue/动作归一化权重，与当前配置作单一变量比较；不删回包、截断历史或把验证失败移走。任何此类改动另起数据版本，并仍以同基座的实际修复和回退判断收益。

统计前后冻结输入SHA均保持为 `0aca8fc273f6f58e94c641f52b7bf3e4dc039c7275a9b719010ad6300c8a72f5`。本补充没有新增检查脚本或测试文件。

