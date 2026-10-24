# P017 C1：单活动Agent的耗时、长度与结束状态

分析完成时间：2026-10-24 17:48:07 +0800。只读已回收C1结果；未联网、调用模型、启动GPU、安装依赖或写测试。仅写本报告。

**C1实际16题同期比较为Base14/16、Step-68 9/16，W0/L5，净新增−5（−31.25个百分点）。Step-68的Agent耗时中位数达到300秒，11题session timeout、1题turns超限；Base只有2题session timeout。** 这轮在单活动Agent条件下仍有明显退步，不能把D的问题全部归于四worker共享服务。也不能把C与D的分数差全部归于并发，因为题目同时改变。

**1．范围与计量依据**

- [完整ZIP](evidence-c1-v1/p017-weight-ab-output.zip)：13,912,723 bytes，SHA256 `4732a49c99dee59ccd515e7b706ab204d7d01d89952d102fd76c858d8d2e8bb4`，798成员。
- [summary](evidence-c1-v1/weight-ab-c-20261024T070519Z-05d256b1/summary.json)、[analysis-rows](evidence-c1-v1/weight-ab-c-20261024T070519Z-05d256b1/analysis-rows.json)及32个真实`context_metrics.json`。16题含13题有P2P、3题F2P-only；本文保留原judge范围，不把F2P-only说成回归全覆盖。
- 对每次真实`wire_request`按wire_id关联当前`wire_response`或`wire_error`。753次请求、740个唯一response ID，740个返回都含prompt/completion/total usage，无重复计数、无缺失usage的返回响应。13个无返回请求均以CancelledError结束，未提供usage；没有估计其已生成token。
- prompt是每次服务实际报告的输入量，可重复包含旧历史和cache命中；completion只来自当前响应，不扫描请求messages中的旧assistant文本当新生成。历史摘要请求独立标注并包含在总API/token成本内。
- 所有753个model_request均为`stream=false`。API区间从wire request到完整response，取消项则到wire_error；这是客户端可见的端到端等待，包括排队、prefill、生成、传输及框架等待。**没有TTFT、逐token时间、取消时已生成tokens或每请求prefill/decode拆分，不称为纯解码TPS或纯思考时长。**
- 结束错误使用每题原始`agent_error`，提交使用实际`context_metrics.patch_submitted`。FINAL配对项的`error=null`不覆盖Agent预算错误；无agent_error也不能直接当成出现了普通final。

**2．32个实际Agent的聚合与分布**

| 指标 | Base，16次 | Step-68，16次 |
|---|---:|---:|
| 冻结判据resolved | 14 | 9 |
| session timeout／turns超限／无agent_error | 2／0／14 | 11／1／4 |
| 判据通过且伴Agent错误 | 1 | 5 |
| patch_submitted=true | 14 | 6 |
| 有普通非工具文本响应的题数 | 13 | 4 |
| Agent累计秒 | 3124.77 | 4117.02 |
| Agent均值／中位数，秒 | 195.30／195.17 | 257.31／300.01 |
| Agent最小／P90／最大，秒 | 75.26／300.00／300.01 | 65.18／300.05／300.10 |
| context计费工具／Agent LLM计数 | 358／410 | 280／308 |
| wire请求／完整响应 | 425／423 | 328／317 |
| 已返回prompt tokens | 3,548,218 | 2,711,768 |
| 已返回completion tokens | 101,503 | 95,976 |
| 已返回total tokens | 3,649,721 | 2,807,744 |
| 已返回API区间累计秒 | 3076.05 | 3289.25 |
| 无返回请求等待累计秒 | 14.89 | 802.00 |
| 全部可见API区间累计秒 | 3090.94 | 4091.25 |
| Agent减API区间残差，秒 | 33.83 | 25.77 |

普通非工具文本响应的独立定义是：非摘要请求的当前choice有非空content且没有tool_calls；Base13题各1次、Step-68 4题各1次。它不自动等于汇报真实或完整修复，但已足以说明“Base14次无agent_error”不能直接改写成“14个普通final”。context计费工具数不等同于全部结构化调用数，后者由语义审计另计。

32个context中的Agent区间无重叠，总计7241.79秒；作业wall为9180.48秒，差额1938.69秒包含服务启动、sandbox准备、host判分、调度与收尾，不当成模型耗时。Agent内部API区间覆盖Base约98.92%、Step-68约99.37%；剩余残差包含工具与客户端开销，不能全部称工具执行时间。300秒上限使耗时分布受截尾影响，不是无限预算下的自然解题时间。

**3．逐题Agent耗时与结束状态**

表中“超时/轮次/无错”来自原始agent_error；提交Y/N来自context；通过/失败只沿用冻结判据。

| 问题 | Base：秒／状态／判据 | Step-68：秒／状态／判据 | Base→Step-68已返回completion | 提交B/S |
|---|---|---|---:|---|
| networkx__networkx-5444 | 75.26／无错／通过 | 65.18／无错／通过 | 2,181 → 1,484 | Y/Y |
| hgrecco__pint-2089 | 229.37／无错／通过 | 300.01／超时／通过 | 7,258 → 6,493 | Y/N |
| hynek__structlog-586 | 225.91／无错／通过 | 180.13／无错／通过 | 7,536 → 5,130 | Y/Y |
| Fatal1ty__mashumaro-128 | 130.70／无错／通过 | 261.37／无错／通过 | 4,162 → 7,823 | Y/Y |
| reata__sqllineage-45 | 300.00／超时／通过 | 300.05／超时／通过 | 10,245 → 5,100 | N/N |
| networkx__networkx-7024 | 119.01／无错／通过 | 300.04／超时／通过 | 3,815 → 5,097 | Y/Y |
| RDFLib__rdflib-1117 | 188.37／无错／通过 | 300.01／超时／失败 | 6,000 → 5,817 | Y/N |
| facelessuser__soupsieve-119 | 252.37／无错／通过 | 300.03／超时／失败 | 8,400 → 6,967 | Y/N |
| Fatal1ty__mashumaro-185 | 300.01／超时／失败 | 300.01／超时／失败 | 9,181 → 5,991 | N/N |
| reata__sqllineage-38 | 241.83／无错／通过 | 300.02／超时／通过 | 8,109 → 6,930 | Y/N |
| networkx__networkx-6908 | 174.51／无错／失败 | 300.00／超时／失败 | 5,623 → 8,819 | Y/N |
| RDFLib__rdflib-1456 | 232.56／无错／通过 | 300.10／超时／失败 | 7,480 → 6,026 | Y/N |
| facelessuser__soupsieve-263 | 198.99／无错／通过 | 109.48／轮次／失败 | 6,623 → 3,061 | Y/N |
| hynek__structlog-198 | 191.35／无错／通过 | 200.58／无错／通过 | 6,185 → 5,762 | Y/Y |
| Fatal1ty__mashumaro-159 | 108.45／无错／通过 | 300.03／超时／通过 | 3,619 → 7,501 | Y/Y |
| reata__sqllineage-19 | 156.07／无错／通过 | 300.00／超时／失败 | 5,086 → 7,975 | Y/N |

五个回退中，RDFLib1117、soupsieve119、RDFLib1456、sqllineage19的Step-68都到300秒；soupsieve263在109.48秒即用尽40turns、没有pending。该题不能归因为长时间服务等待：40个响应全部返回，累计只3,061 completion tokens，应该结合重复动作/工具反馈审查。无错的Base networkx6908仍未修复，进一步说明运行状态与修复结果不可合并。

**4．返回输出呈现更重尾部，未返回部分仍未知**

| 已返回completion分布 | Base | Step-68 |
|---|---:|---:|
| 中位数／P95／最大 | 126／845／2,575 | 76／1,328／5,109 |
| 单响应≥4,096 tokens的次数 | 0 | 10 |
| 这类响应累计API秒 | 0 | 1,392.34 |

Step-68总已返回completion略少，却有10次4K以上的完整响应；这10次占其已返回API时间约42.33%。该差异来自同一C题集，足以提出“少数很长响应耗尽预算”的具体诊断，但不等于完整输出量更少或全体请求都变长。13个pending没有usage，尤其Step-68的802.00秒不能折算成假想生成token。

| Step-68慢响应 | wire请求号 | completion | API秒 | 当前reasoning字符／工具参数字符 | 工具 |
|---|---:|---:|---:|---:|---|
| hgrecco__pint-2089 | 8 | 4,202 | 133.70 | 13,300／231 | run_command |
| Fatal1ty__mashumaro-128 | 13 | 4,107 | 134.86 | 14,712／2 | submit_patch |
| facelessuser__soupsieve-119 | 31 | 4,230 | 136.56 | 15,809／298 | edit_file |
| Fatal1ty__mashumaro-185 | 15 | 4,139 | 134.65 | 15,634／85 | read_file |
| reata__sqllineage-38 | 10 | 5,109 | 162.75 | 15,632／4,363 | edit_file |
| networkx__networkx-6908 | 8 | 4,131 | 135.69 | 13,145／77 | run_command |
| networkx__networkx-6908 | 9 | 4,131 | 139.09 | 13,145／77 | run_command |
| RDFLib__rdflib-1456 | 11 | 4,200 | 135.76 | 13,046／284 | run_command |
| Fatal1ty__mashumaro-159 | 22 | 4,107 | 134.52 | 13,480／2 | submit_patch |
| reata__sqllineage-19 | 16 | 4,482 | 144.77 | 13,947／1,766 | edit_file |

字符计数仅说明当前response的组成，不伪装成reasoning tokens。mashumaro128/159的submit参数仅`{}`两字符，但调用前当前reasoning正文分别14,712/13,480字符，对应4,107 completion与约134.5秒API区间；soupsieve119和RDFLib1456同样出现长reasoning正文。这里确有长生成伴随慢返回的证据，未返回请求的内容却不可见，不能一并贴上“长思考”标签。
已返回请求completion与API耗时的Pearson r为Base 0.9968、Step-68 0.9993。没有TTFT/decode拆分，相关性与`tokens/API秒`都不应当作纯解码速率或单因素因果；返回finish_reason均为tool_calls/stop，没有length，取消项无finish_reason。

**5．13个取消请求及其最后已知状态**

请求号含内部摘要。上一响应P/C取其自身usage，不代替pending的未知usage；“最后工具”仅是上一响应的工具名，不代表工具成功或pending原本想做的动作。

| attempt | pending号／类型 | 等待秒 | 上一响应号：P/C，API秒，工具 | 判据通过／已提交 |
|---|---|---:|---|---|
| `003-hgrecco__pint-2089-adapter` | #16／Agent | 70.85 | #15：6,674/73，5.65，run_command | Y/N |
| `009-reata__sqllineage-45-base` | #24／Agent | 0.43 | #23：8,843/197，5.93，edit_file | Y/N |
| `010-reata__sqllineage-45-adapter` | #16／Agent | 125.11 | #15：14,247/30，1.39，run_command | Y/N |
| `011-networkx__networkx-7024-adapter` | #30／Agent | 115.79 | #29：13,205/32，1.30，submit_patch | Y/Y |
| `014-RDFLib__rdflib-1117-adapter` | #12／Agent | 100.33 | #11：4,285/213，8.21，edit_file | N/N |
| `015-facelessuser__soupsieve-119-adapter` | #32／Agent | 61.24 | #31：9,395/4,230，136.56，edit_file | N/N |
| `017-Fatal1ty__mashumaro-185-base` | #37／Agent | 14.46 | #36：11,991/148，4.80，read_file | N/N |
| `018-Fatal1ty__mashumaro-185-adapter` | #16／Agent | 87.45 | #15：11,662/4,139，134.65，read_file | N/N |
| `019-reata__sqllineage-38-adapter` | #18／Agent | 72.22 | #17：13,782/140，4.78，run_command | Y/N |
| `022-networkx__networkx-6908-adapter` | #10／摘要 | 0.14 | #9：16,287/4,131，139.09，run_command | N/N |
| `023-RDFLib__rdflib-1456-adapter` | #12／Agent | 94.72 | #11：6,112/4,200，135.76，run_command | N/N |
| `030-Fatal1ty__mashumaro-159-adapter` | #29／Agent | 41.34 | #28：8,928/36，1.31，run_command | Y/Y |
| `031-reata__sqllineage-19-adapter` | #23／Agent | 32.80 | #22：3,035/40，2.03，run_command | N/N |

回退中的四个pending等待分别为RDF1117 100.33秒、soupsieve119 61.24秒、RDF1456 94.72秒、sqllineage19 32.80秒。soupsieve119在这之前的第31响应已经用136.56秒生成4,230 tokens；RDF1456第11响应用135.76秒生成4,200 tokens，因此应把已返回长响应和最后未知等待分开。

Step-68判据通过且timeout的5题是Pint2089、SQL45、networkx7024、SQL38、mashumaro159；其中networkx7024、mashumaro159已submit，其他三题未submit。networkx7024最后一条已返回响应就是submit，随后仍有115.79秒未知等待；不能仅据此认定pending正在生成正常final。普通结束契约、语义修复完整性与重复行为由独立轨迹审查补充。

**6．输入长度与真实摘要请求**

| 指标 | Base | Step-68 |
|---|---:|---:|
| 已返回Agent prompt最小／中位／P95／最大 | 2,307／8,588／13,743／15,633 | 2,307／8,604／13,797／16,817 |
| 已返回Agent prompt >16,384 | 0 | 2 |
| 已返回total >16,384／最大 | 0／16,274 | 5／20,418 |
| 明确摘要请求／返回／进入下一请求 | 13／13／13 | 10／9／9 |
| 摘要API累计秒（含取消） | 321.77 | 237.38 |

摘要识别依据是无工具schema且请求明确要求summarize conversation history，不以prompt长度下降猜测。22个已返回摘要的正文均实际进入下一Agent请求，客户端区间显示摘要与普通Agent请求顺序执行。Step-68 networkx6908的第10请求是未返回摘要，0.142秒后到达会话截止；它之前两次普通请求各返回4,131 tokens并分别耗135.69、139.09秒，不能把总超时归因于那0.142秒摘要。

Step-68两个超过16K的正常输入都在判据通过题：mashumaro128第14请求prompt16,817、completion480；mashumaro159第23请求prompt16,635、completion34。它们不足以证明长历史安全或失效，更不支持把本轮五回退归因于>16K；未返回请求没有服务usage，不能断言其长度。最大total20,418来自networkx6908第9请求（prompt16,287+completion4,131），不要把total与输入长度混为一谈。

| 摘要所在attempt | 请求号／API秒 | completion | 前Agent prompt → 后Agent prompt |
|---|---|---:|---|
| `003-hgrecco__pint-2089-adapter` | #14／31.28 | 864 | 15,198 → 6,674 |
| `004-hgrecco__pint-2089-base` | #22／24.94 | 845 | 15,633 → 5,941 |
| `005-hynek__structlog-586-base` | #14／24.60 | 860 | 14,652 → 6,455 |
| `006-hynek__structlog-586-adapter` | #11／22.32 | 686 | 15,544 → 6,223 |
| `008-Fatal1ty__mashumaro-128-base` | #20／27.68 | 976 | 14,585 → 3,551 |
| `009-reata__sqllineage-45-base` | #14／23.05 | 718 | 14,533 → 3,833 |
| `011-networkx__networkx-7024-adapter` | #13／32.66 | 945 | 14,466 → 7,231 |
| `013-RDFLib__rdflib-1117-base` | #9／22.42 | 725 | 14,969 → 6,999 |
| `014-RDFLib__rdflib-1117-adapter` | #10／38.10 | 1097 | 14,932 → 4,285 |
| `015-facelessuser__soupsieve-119-adapter` | #11／19.72 | 625 | 14,358 → 4,055 |
| `016-facelessuser__soupsieve-119-base` | #21／25.99 | 890 | 15,396 → 5,129 |
| `017-Fatal1ty__mashumaro-185-base` | #11／23.34 | 819 | 15,245 → 7,492 |
| `017-Fatal1ty__mashumaro-185-base` | #22／26.75 | 914 | 15,090 → 4,512 |
| `018-Fatal1ty__mashumaro-185-adapter` | #10／20.50 | 620 | 15,468 → 7,289 |
| `020-reata__sqllineage-38-base` | #29／28.42 | 940 | 14,673 → 3,226 |
| `021-networkx__networkx-6908-base` | #22／21.01 | 656 | 14,714 → 3,429 |
| `022-networkx__networkx-6908-adapter` | #10／0.14 | 无usage | 16,287 → 摘要未返回，无后继 |
| `023-RDFLib__rdflib-1456-adapter` | #10／22.79 | 669 | 14,762 → 6,112 |
| `024-RDFLib__rdflib-1456-base` | #12／22.53 | 744 | 14,960 → 6,105 |
| `025-facelessuser__soupsieve-263-base` | #21／24.10 | 784 | 14,375 → 3,324 |
| `028-hynek__structlog-198-base` | #16／26.94 | 948 | 15,179 → 5,266 |
| `030-Fatal1ty__mashumaro-159-adapter` | #24／23.53 | 725 | 16,635 → 8,043 |
| `031-reata__sqllineage-19-adapter` | #21／26.35 | 717 | 14,451 → 3,035 |

**7．单活动Agent与服务器并发：只陈述观察到的边界**

32个真实Agent区间的并发峰值为1；753个客户端wire请求区间（包含摘要、以取消时刻结束pending）全局峰值同为1、没有可见重叠。服务器782条周期采样的Running最大1、Waiting最大0、KV峰5.1%，没有采样点显示Running>1或正等待队列。全服务Avg generation throughput中位30.8、最大37.7 tokens/s，是服务器周期统计，不是逐请求纯解码速度。

所有13次取消到下一attempt首请求间隔30.84–71.25秒；剔除两端各1秒后，这些间隙共有25条服务器采样，Running均0。没有观察到摘要并行或取消请求持续到下一Agent的证据；但客户端取消并不提供服务器实际终止时间，服务器采样也不是连续追踪，**不能升级为每一个瞬间完全无API并发/无残留计算的证明**。服务仍配置max-num-seqs=4，单worker只限制活动Agent。

C1已在单活动Agent条件下观察到Step-68退步及更多长返回/取消等待，因此不能把所有问题归咎于D的四worker设置。另一方面，C1和D任务来源、环境和难度同时不同；本轮可以进行C1内Base/Step-68配对，不能把D→C的分数或时长变化当成纯并发干预。

**交接结论。** 保留C1主结果W0/L5及其覆盖边界；回退分析优先结合四个timeout回退与soupsieve263快速turn耗尽的不同机制。性能证据支持分别检查长reasoning返回、最后未返回等待、摘要后的状态恢复与提交后结束；没有证据允许从pending猜测其下一动作，或用无agent_error替代普通final。此报告不改变任何冻结Agent、权重、数据或原始结果。
