# P017 D：真实请求长度、API耗时与取消请求分析

记录时间：2026-10-24 15:05:27 +0800。只读本轮真实结果 ZIP；未调用模型、未联网、未启动 GPU，未读取 heldout，未新增测试或检查脚本。

**结论：D旧12题的冻结判据没有净新增通过；性能证据也不支持简单断言“Step-68 已返回输出更长，所以超时更多”。** 两臂都是6/12（W1/L1）。唯一新增Faker1741虽通过冻结判据，其原题浮点范围仍有已观察的严格越界；此处resolved只表示冻结判据通过，语义缺口见[独立语义报告](d-semantic-analysis.md)，不追改W/L。Step-68已返回 completion 更少，但9个最终没有返回的请求累计等待658.52秒，Base的7个仅52.91秒。取消请求没有usage，这个缺口使完整输出量不可比较。pyupgrade-530的六次调用并不短：两臂都存在一次4K以上completion的慢响应，少量调用已经耗尽五分钟。

**1．来源与计数方法**

- [完整结果 ZIP](evidence-d-v1/p017-weight-ab-output.zip)：8,831,294 bytes，SHA256 `4bbb0d6ee65cb3f616530860d01b728d8650e4be30bff89893d43c525bbef1b7`。run目录为 `weight-ab-d-20261024T061136Z-7cbb50ca`。
- 每题读取 ZIP 内 `attempts/<attempt_id>/raw-wire.jsonl`、`raw-model.jsonl`、`raw-adk-events.jsonl`、`context_metrics.json`；[analysis-rows.json](evidence-d-v1/weight-ab-d-20261024T061136Z-7cbb50ca/analysis-rows.json)提供修复与预算状态，[服务日志](evidence-d-v1/weight-ab-d-20261024T061136Z-7cbb50ca/vllm-server.log)提供独立运行采样。
- 以唯一 `wire_id` 配对 request/response/error。token仅累加当前 `wire_response.response.usage`，不重新数请求中携带的旧assistant历史；当前completion的字符分解仅来自本次choice.message。所有488个已返回响应均有prompt/completion/total usage，没有对缺失生成token做估算。
- prompt总量是多次实际请求报告的输入工作量，包含真实重复历史、摘要请求和可能命中prefix cache的token，不是独立训练数据量或全部实际重算量。usage没有可用的缓存token细分，也没有reasoning-token细分。
- 504个model_request全部记录`stream=false`。API耗时为request timestamp到完整response timestamp；没有返回的请求则到`wire_error(CancelledError)`。每个attempt内这些区间无重叠。**这包含服务排队、prefill、生成、传输及客户端等待，不能称为纯解码或纯思考耗时；没有TTFT、首token时间或分阶段服务器计时。**

**2．按权重聚合：usage覆盖的是已返回部分**

| 指标 | Base | Step-68 |
|---|---:|---:|
| 实际wire请求／完整响应 | 250／243 | 254／245 |
| 无返回、最终CancelledError | 7 | 9 |
| 已返回prompt tokens | 2,103,471 | 2,003,982 |
| 已返回completion tokens | 65,762 | 55,405 |
| 已返回total tokens | 2,169,233 | 2,059,387 |
| completion中位数／P95／最大 | 92／946／4,138 | 77／647／4,521 |
| 已返回API耗时之和，秒 | 2920.18 | 2509.81 |
| 无返回请求等待之和，秒 | 52.91 | 658.52 |
| 全部API区间之和，秒 | 2973.09 | 3168.33 |
| Agent elapsed之和，秒 | 2986.08 | 3180.56 |
| Agent elapsed减API区间，秒 | 13.00 | 12.23 |
| context计费工具数／Agent LLM计数 | 225／237 | 220／240 |
| 成功／session timeout／turns超限 | 6／7／0 | 6／9／1 |
| patch_submitted=true | 6 | 3 |

实际结构化工具调用另为Base 232、Step-68 236，包含submit和预算拒绝的调用；上表context计费口径不能替代它，详见[工具统计](d-semantic-analysis.md)。

各attempt时间在四worker下会重叠，不能把上表累计秒数当成作业wall；本轮实际wall为2,428.60秒。API区间覆盖Agent elapsed约99.56%／99.62%；剩余约13.00／12.23秒仅是非API区间残差，包含工具、调度、序列化等，不能全部标成工具执行时间。

已返回completion的Step-68/Base差为−15.75%，但这不是完整生成量下降证据：更多长请求被300秒截止线截住，没有response/usage。Step-68的Agent累计时间多194.48秒，算术上由“已返回请求少410.37秒、无返回请求多605.61秒、其他残差少0.77秒”组成；这是时间分解，不能直接解释其服务端成因。

在已返回请求上，completion tokens与端到端API耗时的Pearson r为Base 0.9906、Step-68 0.9943。长返回和长等待高度伴随，但该相关性没有纳入16个被取消的请求，且受到题目、请求类型、并发、cache等共同影响；不把`tokens/该耗时`报告成纯解码TPS。488个返回finish_reason均为tool_calls或stop，没有length；取消请求无finish_reason。

**3．pyupgrade-530：仅六次调用为什么仍耗尽五分钟**

| 请求序号 | Base prompt／completion | Base API秒 | Step-68 prompt／completion | Step-68 API秒 |
|---|---:|---:|---:|---:|
| 1 | 3,138／695 | 40.04 | 3,138／1,702 | 85.87 |
| 2 | 3,882／47 | 1.96 | 5,201／47 | 2.21 |
| 3 | 5,592／90 | 4.64 | 6,911／48 | 3.07 |
| 4 | 5,948／136 | 6.84 | 7,225／295 | 15.42 |
| 5 | 7,183／1,818 | 83.92 | 8,057／4,435 | 164.82 |
| 6 | 9,055／4,124 | 148.82 | 13,052／83 | 3.66 |
| 7 | 无response/usage | 13.65 | 无response/usage | 24.85 |

前六个响应分别合计6,910／6,610 completion tokens，API耗时286.22／275.05秒；第七请求再等13.65／24.85秒后被取消。两臂Agent elapsed都是约300秒，非API残差分别只有0.14／0.12秒，且此题没有摘要请求。不能把“6 calls”误判成模型调用很少、时间应充裕，或归咎于长时间运行测试。

当前响应组件还表明已返回慢请求中确有长reasoning正文：Base第6响应4,124 completion tokens、148.82秒，reasoning_content为12,706字符、工具参数59字符；Step-68第5响应4,435 tokens、164.82秒，reasoning_content为12,877字符、工具参数1,258字符。这里字符只是组件大小，不是reasoning token数；没有用后续请求中的旧历史重复计算生成。第七请求未返回，无法知道其输出内容，也无法知道它原本准备调用哪个工具或结束任务。

**4．16个最终取消请求：按题定位，保留缺失usage**

表中请求号是该attempt的wire请求顺序，包含摘要调用；“上一返回”列的P/C为上一个已返回响应的usage。所有取消项均为`CancelledError`，其自身prompt/completion/total没有返回usage，不能把上一响应的数字套到取消项。

| attempt | 取消请求号／类型 | 等待秒 | 上一返回：请求号，P/C，API秒 | 修复通过 |
|---|---|---:|---|---|
| `001-tobymao__sqlglot-1001-base` | #25／Agent | 2.02 | #24，12,441/57，2.23 | 否 |
| `002-tobymao__sqlglot-1001-adapter` | #29／Agent | 71.58 | #28，13,744/160，9.54 | 否 |
| `005-asottile__pyupgrade-530-base` | #7／Agent | 13.65 | #6，9,055/4,124，148.82 | 否 |
| `006-asottile__pyupgrade-530-adapter` | #7／Agent | 24.85 | #6，13,052/83，3.66 | 否 |
| `009-tobymao__sqlglot-2047-base` | #32／Agent | 13.97 | #31，11,330/1,577，61.09 | 否 |
| `010-tobymao__sqlglot-2047-adapter` | #12／Agent | 1.49 | #11，2,539/835，32.45 | 否 |
| `011-arrow-py__arrow-1035-adapter` | #18／Agent | 151.27 | #17，8,575/201，8.72 | 是 |
| `013-asottile__pyupgrade-133-base` | #30／Agent | 6.87 | #29，3,936/767，33.71 | 否 |
| `014-asottile__pyupgrade-133-adapter` | #29／Agent | 122.86 | #28，2,169/804，34.04 | 否 |
| `015-joke2k__faker-1741-adapter` | #10／Agent | 19.20 | #9，12,075/95，3.94 | 是 |
| `016-joke2k__faker-1741-base` | #13／Agent | 14.99 | #12，12,777/4,138，170.73 | 否 |
| `017-tobymao__sqlglot-3401-base` | #17／Agent | 1.12 | #16，14,271/660，22.84 | 否 |
| `019-arrow-py__arrow-906-adapter` | #11／Agent | 78.90 | #10，5,804/4,205，165.41 | 是 |
| `021-asottile__pyupgrade-656-base` | #19／Agent | 0.28 | #18，5,159/85，3.45 | 是 |
| `022-asottile__pyupgrade-656-adapter` | #26／摘要 | 17.55 | #25，14,541/41，2.02 | 是 |
| `023-joke2k__faker-1869-adapter` | #22／Agent | 170.83 | #21，13,649/85，3.49 | 是 |

Step-68最大的四个pending分别是faker-1869 170.83秒、arrow-1035 151.27秒、pyupgrade-133 122.86秒、arrow-906 78.90秒，合计523.86秒，占该臂无返回等待的79.55%。其中faker-1869、arrow-1035、arrow-906的最终回收补丁通过，pyupgrade-133失败；语义上是否已经完成有效验证、为何继续调用，应对照独立轨迹/补丁审查。这里只证明最后请求长时间未返回，不把它直接叫作长思考、死锁或确定的队列拥堵。

Step-68六个成功里，五个在session timeout后回收补丁通过，一个faker-2190在40-turn budget结束后仍通过。五个timeout成功分别为arrow-1035、faker-1741、arrow-906、pyupgrade-656、faker-1869；pyupgrade-656的最后pending是摘要调用，其余四个是正常带六工具的Agent请求。故“冻结判据通过的补丁”与“顺利结束”必须分列，不能把pending的未知输出推断为正常final，也不能因未submit把通过判据的补丁改判失败。

**5．输入长度与压缩：从真实wire识别，不能只看ADK action字段**

| 已返回的正常Agent请求输入 | Base | Step-68 |
|---|---:|---:|
| prompt最小／中位数／P95／最大 | 2,331／8,687／13,999／16,114 | 2,331／8,496／13,647／15,314 |
| 已返回prompt超过16,384 | 0 | 0 |
| prompt+completion最大 | 16,915 | 15,394 |
| 明确历史摘要请求／响应 | 6／6 | 6／5 |
| 已返回摘要prompt／completion总量 | 22,253／4,137 | 10,822／3,629 |
| 摘要API总秒（含取消） | 179.30 | 171.12 |

全部raw-adk-events的actions.compaction都没有非空记录，但wire中有12个明确的“summarize conversation history”请求，且不携带工具。11个已返回摘要的正文都精确出现在下一次Agent请求；这是实际发生与实际应用的证据。不能据ADK action字段为空断言没有压缩，也不能将prompt突然缩短本身当成已证实的摘要丢信息。

| 摘要所在attempt | 请求号／API秒 | 摘要completion | 前一Agent prompt → 后一Agent prompt |
|---|---|---:|---|
| `001-tobymao__sqlglot-1001-base` | #10／24.70 | 577 | 14,871 → 4,928 |
| `007-joke2k__faker-2190-adapter` | #16／26.28 | 612 | 14,759 → 4,470 |
| `008-joke2k__faker-2190-base` | #19／24.03 | 598 | 14,974 → 5,151 |
| `011-arrow-py__arrow-1035-adapter` | #11／35.56 | 810 | 14,474 → 4,193 |
| `009-tobymao__sqlglot-2047-base` | #19／30.13 | 655 | 16,114 → 4,816 |
| `010-tobymao__sqlglot-2047-adapter` | #11／32.45 | 835 | 15,314 → 下一请求无返回usage |
| `013-asottile__pyupgrade-133-base` | #29／33.71 | 767 | 15,211 → 下一请求无返回usage |
| `014-asottile__pyupgrade-133-adapter` | #28／34.04 | 804 | 14,491 → 下一请求无返回usage |
| `018-tobymao__sqlglot-3401-adapter` | #13／25.23 | 568 | 14,989 → 3,990 |
| `020-arrow-py__arrow-906-base` | #16／29.39 | 722 | 14,551 → 3,333 |
| `021-asottile__pyupgrade-656-base` | #16／37.34 | 818 | 15,308 → 4,802 |
| `022-asottile__pyupgrade-656-adapter` | #26／17.55 | 无usage | 14,541 → 摘要未返回，无后继 |

有返回usage的prompt没有超过16K，只有Base faker-1741第12次请求的prompt+completion达到16,915（12,777+4,138）。16个未返回请求缺服务usage，不能声称全部实际上下文都≤16K；D也没有足够依据支持Step-68在>16K输入上退化的结论。下一轮可专门保留已冻结的长历史/压缩后状态证据，不能把此轮短prompt结果泛化成长上下文能力。

**6．服务日志能排除哪些简单解释，不能回答什么**

vLLM日志有190条周期运行指标，Running最大4，Waiting在全部采样点均为0，GPU KV cache峰值14.0%，末尾prefix cache hit rate为91.9%。服务端自行报告的全服务Avg generation throughput中位77.05 tokens/s，范围0–106.3；它是不同活动请求数下的整体周期指标，不能分摊为某一臂或某次请求的纯解码速度。采样Waiting=0削弱“持续等待队列堆积”的解释，但没有排除采样间排队、共享GPU计算争用、prefill或单个生成很长。

所有24条结果route_verified和judge_valid均成立；没有证据支持把这批超时当作adapter未加载。另一方面，D是四worker共用服务，不是单Agent运行基准。C1已按单worker、16道训练外新题冻结并启动，是另一组独立权重对照：C内Base/Step-68可配对，D/C差异同时包含题目和并发变化，不能全部归因于降低并发。

**7．交接给下一步的具体判断**

优先将“已形成有效补丁后如何停止”“摘要后状态和预算处理”“单次长返回”作为可检验的行为线索，同时保留shared-service性能限制。不要现在把总completion变少当作效率提升、把所有pending当长思考、把所有超时当未修复，或因此直接扩大训练rank/epoch。请求级计时与usage已足够定位上述差异；TTFT、取消时已生成token和每请求prefill/decode拆分当前缺失，不能从日志补造。

补充训练背景见[537题监督审计](training-supervision-evidence.md)：全量元数据与哈希匹配producer独立证明普通final监督0/537、每条末尾submit无回包；验证/恢复链的全量覆盖仍未知。结束协议差异已证实，但这份时间统计与D轨迹都没有建立“final缺席导致循环/长pending”的因果关系。

本报告没有新增模型请求、tokenizer估计或网络下载；全部数字来自当前choice usage、原始时间戳、结果索引和服务日志。
