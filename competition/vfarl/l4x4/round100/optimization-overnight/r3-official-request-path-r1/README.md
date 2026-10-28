# 官方请求、续行与终止路径的源码边界

2026-10-28，只读本地归档源码和发行 metadata。没有请求模型、执行工具、重放轨迹、联网或修改旧记录。精确 wheel/member SHA、已确认与 wheel 字节相同的本地源码、行号和关键片段保存在 `source-evidence.json`。

**官方普通 agent 路径在固定 ADK 源码中使用 nonstream。** swegemma 0.2.11 的 `agent_runner.py:801` 只传 `RunConfig(max_llm_calls=remaining_turns)`；google-adk 1.36.1 的 `run_config.py:225` 默认 `StreamingMode.NONE`，`base_llm_flow.py:1292–1295` 仅在 SSE 模式传 `stream=True`。`LiteLlm.generate_content_async:2466–2468` 因而走非流式 `acompletion` 和完整 ModelResponse 转换。`adk_submission` 的 thought patch 原样转交 stream 参数。`run_async` 是异步事件接口，并不等于 SSE。这里是源码调用链，未抓取 HTTP 请求，也未证明 R3 自定义 collector 曾经过这条链。

**Gemma4 parser 的配置有文档和构建源码链，缺 setup 原脚本。** 官方缓存 `competition/sources/HARNESS_README.md:163–170,202` 明确 `setup_vllm_server` 设置 `enable_auto_tool_choice=True`、`tool_call_parser="gemma4"`、`reasoning_parser="gemma4"`。`adk_submission/server.py:677–723` 将这些字段变成 vLLM CLI flags；`create_model_registry:472–489` 创建指向本地 `/v1` 的 `LiteLlm(openai/<served-model>)`，编译器与 swegemma adapter resolver 再绑定实际 adapter 名。通用 `VllmConfig` 的 parser 默认值是 **hermes**，不能单凭构建类默认值声称启用了 Gemma4。本地没有官方 `scripts/inference.py` 的实际字节，不补造这个来源。

**没有 tool call 并不必然使官方任务立刻结束。** ADK 将无 function-call/response 的完整事件视为 final，结束当前 invocation；外层 swegemma 在尚未 submit 且未触及停止条件时，保留同一 session，添加一个 user continuation nudge 再调用 runner。最多发三次连续 nudge；第四次连续无调用的完成仍会处理，但不再发第四次 nudge。观察到已解析的 function_call 且工具预算尚余，会重置计数，即便其后工具返回 error。文本包含 `<|tool_call>` 就选“token limit／closing cut off”提示，不要求 finish_reason 确为 LENGTH；其他 LENGTH/MAX_TOKENS 和一般结束各有不同提示。因此不能把现有 nudge 当成新设计的端到端恢复收益。

**成功提取 call 仍有后续语义边界。** LiteLlm 将返回参数交给 `json.loads`，非法 JSON 可能在工具事件生成前抛错。进入 `FunctionTool` 后，未知参数会被过滤，缺必需参数返回 error 供下一轮处理；swegemma `ToolErrorPlugin` 也会把工具查找／运行异常变成 error response。模型/转换异常若不可由 `ModelRetryPlugin` 识别为瞬时错误，外层设置 agent_error 并结束，不保证进入 nudge。

外层明确停止条件是 submit、wall timeout、tool-call 预算、LLM-turn 预算（turns 未设置时上限 500）、agent_error、连续三次 nudge 已用完。每次 invocation 都获得剩余 turn 数。非 partial 模型事件更新调用／usage 计数，另有至少一次调用的兜底与 session 事件核对。工具预算装饰器独立检查 tool/turn/cost/token/time；`get_status` 和 `submit_patch` 不消耗 tool call，其 `count_tool_call=False` 也绕过 consumable turn/cost/token 检查，仍检查时间。本源码的 cost/token 检查不是外层无条件 break。未 submit 时，退出阶段还可捕获并过滤 working-tree diff 交给评估，所以 malformed submit 或无 call 不能直接决定 reward。

本次依据实际归档的 swegemma **0.2.11**、google-adk **1.36.1**、adk_submission **0.2.13**、adk_eval_core **0.1.0**；没有拿本机 ADK 1.39.1 cache 冒充 fixed wheel。adk_submission 当前归档与旧 v30 归档 ZIP 封装 SHA 不同，但所有 member payload 已确认相同。ADK 发行 metadata 仅在 extras 中声明 LiteLLM 范围，不能据此声称本次掌握了实际安装的 LiteLLM 版本／源码。

既有 vLLM CPU 源码 JSON 只有八个 Gemma4 文件，未包含 reasoning base 或 serving 源码。固定 wheelhouse 文件列表列出 vLLM 0.19.1，但当时 CPU extractor 未保存所选完整 wheel 文件名和 SHA；member SHA 不冒充 whole-wheel SHA。另已只读复核 `../r3-official-parser-semantics/upstream-v0.19.1/` 的上游 tag 参照，commit `b1388b1fbf5aaef47937fabe98931211684666a6`：tool/reasoning 两份 Gemma4 文件与实际 wheelhouse cache 字节相同；新取得的 basic/serving 文件仅有同 tag 上游来源。其 `serving.py:1374–1396` 明确先 `reasoning_parser.extract_reasoning(output.text)`，再仅把 content 交给工具解析。这支持该发行版的源码顺序参照，仍不是完整同 wheel 恢复或服务实测；人工裁切 thought 也不能冒充真实 serving 全链。历史文本被 parser 接受，最多是兼容性机制证据，尚不能推出工具执行、恢复动作或最终收益。
