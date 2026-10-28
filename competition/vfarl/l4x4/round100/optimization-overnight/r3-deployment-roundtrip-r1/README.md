# R3 原始 token 与结构化消息回渲染的静态差异

2026-10-28。只读固定100的三条 R3 正确工具动作，使用实际缓存的 CT AutoTokenizer、真实原工具输出和显式 kwargs 做静态渲染；没有 HTTP、ADK Runner、模型、工具执行或新训练数据。来源及源码片段、原行 SHA、token/bytes hash 在 `static-roundtrip.json`。思维文本只保留在原日志，本目录仅记录其长度和 SHA。

**当前证据不能说明官方部署丢弃 thought。** 实际 CT/2 模板 SHA `94899c0f…` 根本不引用 `preserve_thinking`。它按 `reasoning`／`reasoning_content`、该消息有 tool_calls、且位于最后一个 user 消息之后这三个条件渲染 thought。本次显式省略／设 True／设 False 的三种 `preserve_thinking` 输入，对每个静态变体都产生完全相同的字节与 IDs。因此不能把 policy.render_prompt 传 True 与假设的模板默认 False 当作已证实的差异。

源码区分如下：

- VFARL R3 冻结 policy 以 enable_thinking=True、preserve_thinking=True 建初始 prefix；后续直接追加原 completion IDs，再只序列化真实工具 response suffix。模型已采样的 thought／native call 字节不重写。
- 固定 google-adk 1.36.1 对同一 agent 含 function_call 的历史事件保留完整 content。LiteLlm 把 thought Parts 收集为 reasoning_content，adk_submission patch 在 include_thoughts 不为 False 时再补 reasoning 字段。include_thoughts=False 才会显式移除 response 中的 thought Parts，并删除发出消息的两个 reasoning 字段。其他 agent 转写为 user context 时另有排除 thought 的分支。
- 官方工具响应先把 function.arguments 做 json.loads，下一请求又对 ADK 参数映射做 json.dumps。原 native 调用的字节格式不属于这条链的保留契约。真实 CT 模板还会对 mapping 参数 dictsort；字符串参数走另一个分支。
- 新 user nudge 会改变模板的 last_user_idx，因此即便字段还在，较早 thought 也可能不再渲染。这是模板条件，不是本次实测的官方会话。

三条动作均以原 first prompt 加原 completion 和真实工具 suffix **逐 ID 重建了下一条已采样动作的 prompt**。随后用解析后的同一工具名／参数、实际 reasoning 字段和真实 response 建立结构化候选；下表统一 enable_thinking=True、arguments 显式为 mapping。

| 固定索引／动作0 | 历史 raw-append 下一输入 | 保留 reasoning 字段的结构化回渲染 | 显式删除 reasoning 字段 |
| --- | ---: | ---: | ---: |
| 25 Tenacity wait_combine，run_command | 2316 token | 2317 token | 2245 token |
| 10 Moto6159，read_file | 1824 token | 1825 token | 1749 token |
| 6 Bottle overlay，run_command | 1434 token | 1435 token | 1366 token |

保留 reasoning 的三例均增加一个 token。模板会在 thought 结尾插入固定换行，原 thought 内容仍存在。Moto6159 的正确原调用参数顺序是 filepath、start_line、end_line，mapping 回渲染变为 end_line、filepath、start_line，语义参数相同而工具片段字节不同。两条单参数 run_command 的工具片段保持原字节。这里的长度差不是 token 节省或模型收益估计。

JSON 还保留了 arguments 显式为 JSON-string 的变体，因为 ADK 下一请求确实先产生 JSON-string，而本次尚未独立闭合 vLLM 下一请求 preprocessing 将它转换为 template mapping 的源码链。这些是透明的 renderer 输入条件，不冒充真实 wire payload。结构化静态消息使用 `static-render-link` 仅连接 call/response，未声称它是官方实际 call ID。

reasoning 分离使用未改写的 cached Gemma4 方法及同 tag 上游 BaseThinking 方法，仅由 AST 选取纯方法／property，未构造 vLLM 协议对象；不是手工截掉 thought。原 source bundle 中的 NativeArguments／parse_call／tool_suffix 也按冻结函数执行。源码绑定 `ccaa9e2f…`（R3 run）和 `5d50dece…`（R3 policy），CT token/template 与 source manifest 一致。

官方 setup_vllm_server 原脚本、实际 effective template/kwargs 和整条消息 preprocessing 仍有来源边界。这里支持“raw token 续接与 parse→serialize→template 回渲染可能产生不同输入”这一部署候选机制；不支持“官方已丢 thought”、新比赛实跑、恢复率或终局收益结论。collector、历史 token、奖励和当前主 SFT 均未修改。
