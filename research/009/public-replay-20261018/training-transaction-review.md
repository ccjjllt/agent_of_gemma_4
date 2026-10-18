# 2026-10-18 指定 train 事务定点复核

仅查看 data_status 明确分配 train 的来源和新 CPU 实际事务，不读取其他选择题/预留题，不重新运行套件或模型。第一批范围为 `research/009/training-data-20261018/outputs/{apispec,pydantic,mypy,dataset,alive-progress}/attempt-01/{transaction.json,events.json,commands.json}`，split 身份见该研究目录的 `data-usage.json`。

## 第一批判断

| 事务 | 实际执行与判别 | 纳入建议 |
|---|---|---|
| apispec | 通过真实 APISpec/MarshmallowPlugin 生成参数；deprecated=true/false 原在 schema、修后在 parameter；缺省不新增，组件属性仍保留 | 可纳入局部公共 API 修复事务 |
| pydantic | 真实 BaseModel + Annotated[Seq,MaxLen(3)]；长度 3 由拒绝变接受，长度 2 仍接受、4 仍拒绝 | 可纳入局部公共 API 修复事务 |
| mypy | 真实 mypy.api.run 内联 TypedDict 程序；空 key 原 INTERNAL ERROR/退出2，修后正常 missing-key diagnostic/退出1，合法 key 退出0 | 可纳入局部公共 API 修复事务；实际 stdout 含预期诊断，未把仅“无异常”当通过 |
| dataset | 真实内存 SQLite 经 dataset.upsert/Alembic 增列；原参数 TypeError，修后 new=11、old=7、行数仍1 | 可纳入局部公共 API 修复事务 |
| alive-progress | 从实际 configuration.py 提取未改写 AST 工厂和实际 refresh_secs 表达式执行；0.25由ERROR变0.25，0/1保留，负数/字符串/超界拒绝，整数选项仍拒绝小数 | 仅组件局部证据，绕过模块初始化和公开 alive_bar 路径；本轮主SFT宜单列/降权或暂不纳入完整验证动作 |

前四条命令在实际临时源码 workspace 下运行，`PYTHONPATH=src:.` 优先源码；`produce.py` 用同一个 command 对修前/后执行，事件显示精准目标行为由 fail→pass。它们没有完成完整基准套件，`task_resolved=null` 与 `official_suite_ran=false` 的表述正确。

alive-progress 不是自造替身函数：它执行了文件中真正的 AST 函数与配置表达式，但没有证明公共进度条初始化和实际刷新链可用。事务的 validation_scope 已披露这一点，不应把它提升为与前四条相同层级的公共 API 成功。

## 失败动作与监督边界

前四条 `target_assistant_steps=[2,3,4]`，依次为成功 edit、成功修后检查、submit；alive-progress 为 `[2,3,4,5]`，多一次成功 edit。baseline 失败检查是 ordinal 1，未被选为监督目标，read ordinal0也未监督。encoder `training/encoding/gemma_tool_chat.py` 使用零基 assistant ordinal，与这里一致。五条 targets 均未命中 error observation。

失败 attempt 即使保留前面已成功编辑的 target ordinal，也必须由 admitted-status 整条筛掉，不能单凭目标列表非空纳入。本复核只确认事务层的目标选择；最终 token 编码后的实际 mask 由数据生成分支记录，不把本次只读检查冒称为已验证全部训练 label。

证据路径和发现已同时反馈数据分支与主代理，由主代理决定实际纳入样本。未修改原事务或生产源代码。

## 后续三条实际事务

仅继续读取数据分支明确交付的 `deepdiff/attempt-01`、`yamllint/attempt-01`、`tenacity/attempt-03`，没有运行新检查。

- **DeepDiff：可纳入局部真实 API 修复。** 真正的 DeepHash 对嵌套一维/二维 NumPy bool 原报 memoryview hash ValueError；修后稳定生成 hash、true/false hash 不同，integer-array hash 与修前相同。目标 ordinal `[2,3,4,5]` 均对应成功 edit/验证/submit。
- **yamllint：可纳入局部真实 API 修复。** 真正 YamlLintConfig/linter 对六个 inline YAML 配置执行，原新选项无法识别，修后 plain/wrong-quote key 报对应问题，正确 key、默认忽略 key、独立 key quote-type 正常，unquoted value 仍失败。事件中的实际诊断与目标一致；验证谓词主要数错误数量，不等于覆盖所有 rule combinations。目标 `[2,3,4,5,6]` 均成功，GPL-3.0-or-later touched-code 来源需按数据分支原标记保留。
- **tenacity attempt03：可纳入静态构造器注解修复，不能称 runtime 成功。** 真 mypy API 读取 pinned 源码注解，正确 callbacks 从三个 constructor arg-type errors 变为零；错误 str 返回仍产生一个对应错误。`--follow-imports=silent`/`--ignore-missing-imports` 没有让负例失效，但 runtime import 的缺 RetryBaseT 问题未修复，scope 已披露。前两次 attempt 不应纳入。

tenacity 的一个边界需特别保留：`copy()` 参数原先有 `Union[..., object]`，修前输出没有 copy 错误，错误返回的 guard 也只在构造器行报错。因此本次没有独立证据证明 copy 签名修改带来行为改善。可将 scope 收窄为 constructor 判别、copy 仅伴随修改；若首轮只监督有直接行为证据支持的动作，可将 copy edit ordinal3 留在上下文而不作为 target，保留 constructor edit2、成功验证4、submit5。已向主代理提出，由主代理决定，不直接修改事务。

综合：支持前四条加 DeepDiff/yamllint/tenacity 静态事务作为七条首轮候选，alive-progress 暂留组件候选。所有这些结论均是局部行为/动作质量判断，不是七个完整基准任务已经通过；task_resolved=null 应保持。

## 最终冻结增量：独立 PR950 与八条编码

主代理随后授权只读新增 `outputs/apispec-pr950/attempt-01/` 和最终明确 train 的 transactions/encoded metadata。PR950 与 PR851 是**同仓库的不同问题**：851 修 query parameter 的 deprecated 层级（openapi.py，buggy commit 82d39f5…），950 修 plain Dict 的显式 additionalProperties 输出（field_converter.py，buggy commit 322c9b2…）。不是对同一个问题重复切片；同仓库的相关性仍在，因此最终准确计数是 **8 issues / 7 repos**。

PR950 实际通过 APISpec/MarshmallowPlugin 构造 OpenAPI3.0.2 组件。原命令四例中只有 plain Dict 缺 additionalProperties 而失败，Raw-valued Dict、Integer-valued Dict 和 Integer scalar 均通过；生产编辑后同命令四例全通过，submit_patch 输出1241B。它监督 ordinal `[2,3,4]`（edit、修后检查、submit），均对应成功真实观察；修前失败ordinal1没有监督。

最终 `transactions-train.jsonl` 包含八个不同 issue，全部 split=train、task_resolved=null；alive-progress 没有纳入。`encoded-train-12b.jsonl` 读取实际数组计数为 **35,992 input tokens / 6,147 labels != -100**，与编码报告、汇总一致，最大6089 tokens，无历史截断。Tenacity 最终 `target_assistant_steps` 和 `supervised_spans` 均仅为 `[2,4,5]`，copy ordinal3已作为上下文屏蔽。

冻结指纹与 output-manifest 一致：transactions SHA256 `10d4dd8daa770a2f59e1a21514a5bad06499d6c8b5bedf7cbaa5ba562ad6bd3f`；encoded SHA256 `0aca8fc273f6f58e94c641f52b7bf3e4dc039c7275a9b719010ad6300c8a72f5`。这次只读核对未发现需要阻止已排队训练的实质数据质量问题；没有重跑测试、编码或修改最终数据。
