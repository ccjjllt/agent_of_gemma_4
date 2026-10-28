# 下一轮工具参数 JSON-string→mapping 的独立补链

2026-10-28。原 `r3-deployment-roundtrip-r1` 保持原样，SHA 仍为 `6fe0c7dc5a9a2c91e850a933dbaca2accf2a6adfccef7042932549bb90ec524e`。本目录只补同 commit 的源码链和三例静态 CPU 函数观察；没有 HTTP、ADK Runner、模型或工具执行，没有修改 collector／历史数据。

**负责转换的确切层是 `vllm.entrypoints.chat_utils._postprocess_messages`。** 在 commit `b1388b1fbf5aaef47937fabe98931211684666a6` 的 [chat_utils.py:1555–1578](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/entrypoints/chat_utils.py#L1555)，它遍历 assistant tool_calls；function.arguments 非 dict/list 时执行 `json.loads`，空值设为 `{}`。同步和异步 parse_chat_messages 都在返回 conversation 前调用它。

普通 HF serving 输入链为：

1. 已有 `OpenAIServingChat.render_chat_request` 委托 `OpenAIServingRender.render_chat`。
2. 新取的 `render_serving.py:234–255,500–541` 调用 preprocess_chat，再调用 renderer.render_chat_async。
3. `renderer_base.py:898–923` 调用 HFRenderer.render_messages_async。
4. `renderer_hf.py:680–708` 先调用 parse_chat_messages_async（其中执行 `_postprocess_messages`），之后调用 safe_apply_chat_template。
5. safe_apply_chat_template 在实际 kwargs 过滤后调用 tokenizer.apply_chat_template。因此模板接收的是已转换的参数，而不是 ADK 发出的原 JSON-string。

六个公共源码文件均由 GitHub 单文件只读 GET 取得，使用完整 commit，保存了 URL、API 返回的 Git blob SHA、SHA256 和长度。本地内容重新计算 Git blob identity 全部相同，见 `upstream-b1388b1/provenance.json`。这些是**上游 commit 参照**，没有冒充未恢复的官方 wheel 成员。

本次对原研究同三例执行了未改写的实际 `_postprocess_messages`、`safe_apply_chat_template` 及其 kwargs resolver。真实 ADK `_safe_json_serialize` 先把原观察参数变为 JSON-string，真实 reasoning patch 补 reasoning 字段；postprocessor 将参数转回 dict，语义逐项等于真实工具事件。随后用实际缓存 CT tokenizer 和显式模板／kwargs 渲染：

| 固定100索引／真实动作0 | postprocess 参数 | 实际 helper 输出 token | 与旧显式 mapping 分支 |
| --- | --- | ---: | --- |
| 25 Tenacity wait_combine | str→dict | 2317 | 字节及 IDs 完全相同 |
| 10 Moto6159 | str→dict | 1825 | 字节及 IDs 完全相同 |
| 6 Bottle overlay | str→dict | 1435 | 字节及 IDs 完全相同 |

实际 upstream resolver 还会过滤本 CT 模板未声明、tokenizer 函数也不显式支持的 `preserve_thinking`，留下 add_generation_prompt、enable_thinking 和 return_dict。这与旧研究中省略／True／False 输出相同一致；三例 reasoning 字段均保留，不能据此声称官方丢弃 thought。

执行边界明确：AST 只选择原函数／class body 和 decorator；没有修改 preprocessing 函数体。`_postprocess_messages` 直接接收普通 dict 数据，无替代协议类。未实例化完整 parse_chat_messages_async、multimodal tracker、请求 Pydantic、LiteLLM 网络或 ADK Parts。safe_apply_chat_template 传入显式真实模板，因此其 resolve_chat_template 第一分支立即返回，`model_config=None` 不被读取；这不是伪造服务配置。`static-render-link` 只是静态 call/response 配对键。reasoning 分离沿用 cached Gemma4 纯方法加同 tag base 方法的既有披露，不复制思维文本。

该缺口现在在同 commit 的源码与纯函数层闭合：旧 mapping 候选有了实际 preprocessing 支撑。官方 setup 原脚本、真实部署 effective template／kwargs、完整 wheel 字节和端到端服务行为仍未取证；没有下一次模型采样或终局收益结论。精确源码行、来源和观察在 `core-execution.json`。
