# P017：submit_patch 与 Agent 结束契约

2026-10-24。记录已缓存官方源码的实际行为，并补入已回收D完整轨迹；没有修改Agent或运行模型、没有联网或GPU。缓存版本为swegemma 0.2.11、google-adk 1.36.1，配套adk-submission 0.2.13。三个swegemma解包源码与本地官方wheel逐字相同；ADK内容直接从wheel成员读取，没有执行库代码。

**submit_patch成功只表示补丁已捕获，不会立即结束ADK内部循环。**

- `research/006/official-data-20261013/wheels-source/swegemma/tools/execution.py:76–116`：`submit_patch`执行`git add -N .`和相对baseline的diff，剔除受保护文件，写入`ctx.submitted_patch`、置`ctx.patch_submitted=True`，返回`status=ok/patch_size/files_changed`。工具允许多次提交，以最后一次保存的补丁为准；没有设置ADK的`end_invocation`或`skip_summarization`。空diff也会置已提交，故不能仅以该标志判断修复质量。
- 同文件`:164–174`的工具封装仅返回解析后的dict，没有附加终止动作。
- 装饰器为`count_tool_call=False`。`swegemma/context.py:468–525`的工具数、turn、cost、token检查都以该开关为条件；`:527–547`的会话时间检查仍生效，`:549–550`也不会增加工具计数。因此提交可以在普通工具/turn计数已满时执行，但不会绕过ADK的下一次模型调用限制。

**普通final通过事件结构结束循环；EOS/STOP本身不是唯一终止判据。**

- google-adk wheel成员`google/adk/events/event.py:83–98`：通常在事件非partial、没有function_call/function_response、没有尾随代码执行结果时认定final；另有skip_summarization/long-running特殊分支。该函数不检查`finish_reason`。
- `google/adk/models/lite_llm.py:110–123`将`stop`、`tool_calls`、`function_call`都映射为STOP。因此看到STOP不能单独断言是普通final；需检查真实parts/tool_calls。`:1694–1740`将普通文本、thought及工具调用转换为对应parts。
- `google/adk/flows/llm_flows/base_llm_flow.py:862–875`在最后事件为final、partial或无事件时结束内部flow；`:1172–1206`对普通工具调用先产出function_response，普通submit工具回包本身没有final结束动作，所以通常还会进入下一次模型调用。
- `swegemma/harness/agent_runner.py:806–864`消费内部runner事件。若模型作者给出非thought文本、没有工具调用/回包、事件为final且patch已提交，则直接break；这会关闭生成器，因而捕获到终端consumer-close不一定是模型响应失败。
- 同文件`:920–922`：内部runner正常返回后，只要patch已提交，外层即退出。因此有非空普通final是一个明确的提前结束路径，但不是所有正常结束情形的必要条件。
- 若未提交，`:973–1024`按普通结束/长度截断等情况发有限次continuation nudge；不能把任何EOS都理解成整个比赛任务已经结束。

**“提交成功＋修复通过＋40turns错误”一般存在纯预算边界路径；D中Faker2190已排除仅由此路径解释。**

源码允许如下序列：最后一个可用模型调用产生`submit_patch` → 工具成功保存补丁 → ADK准备再次调用模型生成普通final → 在发起下一模型请求前抛出调用上限异常。依据是：

- `google/adk/flows/llm_flows/base_llm_flow.py:1287–1295`在`llm.generate_content_async`之前递增/检查次数。
- `google/adk/agents/invocation_context.py:82–98`先递增，超过`run_config.max_llm_calls`就抛`LlmCallsLimitExceededError`；达到上限的最后一次调用本身允许完成。
- `swegemma/harness/agent_runner.py:1026–1038`捕获该异常并写turn-budget error，未清除patch标志。`:1048–1103`仍取已提交补丁，只有从未提交时才尝试回收工作树diff。因此预算错误与最终补丁通过判据可以同时成立。

以上保留为通用源码契约；单凭`patch_submitted=true/resolved=true/40turns error`不能区分预算边界与提交后循环。但D完整输出已经给出了Faker2190的直接证据：`007-joke2k__faker-2190-adapter/raw-wire.jsonl`第15个wire请求首次成功`submit_patch`，第16个是无工具的历史摘要请求，第17个起重新执行清理脚本和提交。第17个请求的最后两条消息仍包含首次提交及其`status=ok/patch_size=11210/files_changed=1`回包，随后模型仍选择`rm fix_last_names.py`，并在收到文件不存在的反馈后再次提交。

总计13次成功提交（12次冗余重提），清理与重提交替到第41个wire请求；该请求的`run_command`回包为`BudgetExceeded: Turn budget exhausted (40 turns)`，最终Agent记40turns错误。41个wire请求包含1次摘要，所以不能当作41个Agent回合。该例没有普通Agent final；摘要中的完成说明不是Agent final。其原F2P 1项及P2P 134项均通过，已提交补丁保留，因此修复判据通过与结束行为失败同时成立。

这证明了**提前成功提交后的真实循环**，不等于已经证明三工具到六工具的迁移是原因。循环发生在摘要之后，但后继上下文仍保留成功提交回包；不能仅以时间先后认定压缩造成遗忘，也不能由薄弱适配种子的final覆盖推断整轮主训练缺少final。完整语义解释、请求序号及原始成员SHA见[d-semantic-analysis.md](d-semantic-analysis.md)和[behavior-evidence.json](evidence-d-v1/semantic-excerpts/behavior-evidence.json)。源zip SHA256为`4bbb0d6ee65cb3f616530860d01b728d8650e4be30bff89893d43c525bbef1b7`。

**源码身份。** 以下路径均相对项目根；ADK行号是对应wheel内成员行号。

| 文件/成员 | SHA256 |
|---|---|
| `research/006/official-data-20261013/wheels/swegemma-0.2.11-py3-none-any.whl` | `9ad876581f3acde8ec3c369858f0834e2b9f41a3cffce3ca42f09cb090aea9d3` |
| `research/006/official-data-20261013/wheels/google_adk-1.36.1-py3-none-any.whl` | `1a2f6868c509e3151fb0de3575a7d18b45c338be86f420924dad74e7193631a0` |
| `wheels-source/swegemma/tools/execution.py`（同上official-data目录） | `ff4df981ae1070334665ce8ab726c231e139ac5cf34bb5ebc4660eada28a7b2d` |
| `wheels-source/swegemma/context.py`（同上） | `272168834495f424dac54cfcc3d470c1b8e42b75a1125e7501586eb5f4fb73c0` |
| `wheels-source/swegemma/harness/agent_runner.py`（同上） | `4d7eccabd203c26f7c39e882684d2fd851f383f6a5c76c4a75b0e559cd5d6a18` |
| ADK成员 `google/adk/events/event.py` | `120808bf6ce77d6c2322881b335bcdee12e2ac8a94d9fbbfc38353f3764d946a` |
| ADK成员 `google/adk/flows/llm_flows/base_llm_flow.py` | `b20beb63a63aa143ef49c6246a1efed75e0dfcb9d8fae45b46c390d0cacab1fd` |
| ADK成员 `google/adk/agents/invocation_context.py` | `17d68b066d7ba50da434d572f87d8d2a5abe14c90b3767f81aa8f959fbcae0c5` |
| ADK成员 `google/adk/models/lite_llm.py` | `2f28c6afa110006e0fdec46c9ac7c6b614d74c81e0d0bf6fda2b0d6353111098` |
