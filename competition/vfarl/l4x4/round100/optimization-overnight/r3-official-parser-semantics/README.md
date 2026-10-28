# R3真实坏语法与官方非流式解析组件差异

只读固定train R3的41个actions文件、501个真实raw_text及其既有episode；不执行任何原工具/命令、不读heldout/预留题、不改历史reward或parser/collector、不启动模型/GPU。原文件SHA全部仍等于此前download绑定。新增文件仅为源码来源与机制记录。

结论：R3的10条确实是严格native坏语法，但“严格parser异常→立即终止”不是缓存官方Gemma4非流式parser core的同一语义。全501按真实reasoning→content→tool组件顺序对照：483条strict接受的call全部name和JSON解码参数相等，8条strict无call仍无call；只有10条坏语法不同，8条被容错提取成调用、2条保留普通文本，没有多call。真实CT tokenizer对501条completion IDs均解码复现原raw；直接解析全raw与先actual reasoning的结果在本批501条也完全一致。后者是本批观察，不能省略未来serving的reasoning阶段。

8条容错提取中，4个read_file保留了错误的start_line引号键，ADK FunctionTool会过滤未知参数而丢失请求范围；1个filepath吞入逗号和行号字段；2个edit_file缺必需old_string且new_string吞入旧字段；只有一个submit_patch提取为真正空参数调用。没有一条坏edit构成正确的可执行生产修复，不能补gold再称原raw有效。提取call不等于语法合法、参数正确或修复成功。

| 当前fixed100索引/题 | attempt:actions行（1-based） | 历史working-tree验收reward | 官方core解释 |
| --- | --- | --- | --- |
| 10 Moto6159 | original:29 | 未验收 | 无call，保留原文本→预算内nudge候选 |
| 22 Tomli qlgy | original:15 | 1 | read_file行号键带引号；ADK过滤后丢失范围 |
| 26 Tenacity rlm5 | original:4 | 0 | read_file行号键带引号；ADK过滤后丢失范围 |
| 63 Prettytable r791 | original:10 | 1 | read_file的filepath吞入后续字段 |
| 6 Bottle r3tamcap | branch-0:6 | 1 | 无call，保留原文本→预算内nudge候选 |
| 9 Moto5348 | original:25 | 未验收 | edit_file错切字符串，缺old_string |
| 23 DVC7628 | original:3 | 未验收 | read_file行号键带引号；ADK过滤后丢失范围 |
| 25 Tenacity pvng | branch-0:5 | 0 | read_file行号键带引号；ADK过滤后丢失范围 |
| 25 Tenacity pvng | branch-1:16 | 1 | submit_patch{}；只提交已有修改 |
| 25 Tenacity pvng | original:11 | 0 | edit_file错切字符串，缺old_string |

10个历史episode均patch_submitted=false、malformed_action终止且没有harness-feedback；既有working-tree fallback带来4个reward1、3个reward0、3个未验收，不能把语法终止统一当修复失败。index25 original第11行（0-based action10）的坏edit实际提取filepath/new_string/伪键`def __call__(self, retry_state`，old_string缺失；branch1末尾坏submit则只会提交已存在且历史验收为1的工作树。两者不是同状态重放或新修复收益。

官方swegemma0.2.11 run_agent_sandbox在一轮无提交、未触及outer的time/tool/turn限制且没有逃逸异常时会追加最多3个连续user nudge；见agent_runner.py约780–1024。带`<|tool_call>`的无call普通文本会命中其截断提示启发式，尽管本例并非真实length截断。ToolErrorPlugin只处理已到达工具查找/运行阶段的错误；ADK模型/JSON解析异常未被on_model_error处理时仍上抛。故不能笼统说“官方格式错误总会反馈”，也不能说“官方总是立即终止”。独立harness/ADK/预算源码报告为`../r3-official-request-path-r1/{README.md,source-evidence.json}`，JSON SHA256 `660d313fc34b63539919380dd494ddccd379c1a3a0ce84462ca181459fede42e`。cost/token仅在budget-gated工具处检查，不是outer无条件break；get_status/submit_patch不消耗tool-call，且绕过consumable turn/cost/token检查，仍检查time。

来源链：实际CPU准备任务从官方wheelhouse的vllm wheel抽取Gemma4源码，缓存JSON及summary/producer均仍存在；官方文件清单列vllm0.19.1。缓存native member SHA682b2152…与Gemma4 reasoning member SHA64a1d3…分别与官方v0.19.1 tag（commit b1388b1fbf5aaef47937fabe98931211684666a6）逐字相同。缓存缺basic/serving，现补的这三文件明确是同版上游参照，不能冒充缺失wheel成员字节或现行线上server取证。实际setup_vllm_server脚本本地缺失；启用gemma4双parser由官方HARNESS_README＋VllmServer配置源码联结，线上实例未连接。

`upstream-v0.19.1/provenance.json`记录每个官方URL、Git blob SHA、SHA256和tag commit。serving.py1374–1395先extract_reasoning，再engine_serving.py881–957从content解析tool；basic_parsers.py约149–177提供实际非流式reasoning行为。两个cached核心原样执行，另用这份上游basic/helper补完整组件顺序。

执行范围：真实CT AutoTokenizer与完整真实vocab；实际parser函数/类方法AST、regex/value parsing没有改。外部ToolParser/ReasoningParser构造仅替成真实tokenizer/vocab赋值；三个vLLM协议容器用惰性keyword属性对象、id=None，请求用auto/None对象，两分支结果501一致；未执行真实协议Pydantic验证、HTTP、LiteLLM、ADK Runner、工具或streaming。补future annotations，取staticmethod helper到模块作用域，logger转Python logger。完整替身边界在JSON，不能称端到端官方重放。

- `actual-raw-differential.json`：10条原文本、原行/原文件/episode SHA、真实CT解码、官方参数及历史结果。
- `all-501-upstream-chain-differential.json`：501条逐行SHA绑定、全部对照计数、10条完整差异及raw/实际reasoning链对照。
- `prospective-design.md`：仅提出显式新profile及历史/轨迹/预算边界，没有实现或收益声明。
