# P016 编码准备：真实请求边界与终局 reasoning

日期：2026-10-22，北京时间。只使用 CPU 和现有依赖；未启动模型、GPU、教师调用或比赛提交。只读取 P015 已标为 train 的 boltons-298 轨迹和其训练窗口，没有读取 dev、select、heldout 正文。没有新增测试文件或测试套件。

## 已确认的模板行为

固定来源是 `google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2`，本地文件在 `research/015/tokenizer-assets/official-31b-ct-v2-tokenizer/`。

- `chat_template.jinja` SHA256：`94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413`。
- 编码器 tokenizer 指纹：`d001718145f4980b65e70b5ffa1dfce93b3eca5a5d8de2446b8c483f2b4b8927`。
- reasoning 只有在 assistant 携带 `tool_calls` 且位于最后一条 user 之后时才渲染；普通 final 的独立 reasoning 字段被模板删除。
- 这个固定模板没有读取 `preserve_thinking` 变量。传入该参数不能改变上述行为。
- assistant 同时具有 tool call 和可见 content 时，模板先前向扫描并输出随后 tool observation，再输出该 assistant 的 content。不能把这个 content 当成未见 observation 时生成的原始目标。
- no-thinking 的独立 generation prompt 会附带空 thought channel；完整历史序列化和生成请求并非在所有情况下都逐字前缀一致。

这些是固定模板本身的行为；没有换成最新版模板，也没有修改模板文件。

## 已实现的功能

`training/encoding/gemma_tool_chat.py` 更新为 `p016-causal-transaction-v4`。

1. 原有 `encode_transaction` 继续支持多个 assistant 目标、只选择部分成功目标以及普通不带 reasoning 的 final。新增逐目标 generation-prefix 核对；拒绝会使 assistant content 见到未来 observation 的 tool-call/content 混合消息。没有丢弃已有 reasoning-preservation 检查。
2. 新增独立函数 `encode_captured_completion`：每条记录以一次真实请求作为全部前缀，监督这次请求的原始生成流。工具失败、其他历史 assistant 和工具反馈都在 prompt 里，标签全部为 `-100`。final 的原始 thought channel 可保留或屏蔽其正文，已实际生成的通道分隔符与结束 token 保持原样。
3. 入口不把 parsed `reasoning_content`、`content` 和 `tool_calls` 拼成所谓原始流；不插 EOS、通道标记、工具观察或摘要。普通终局要求捕获本来存在的 `<turn|>`，工具调用要求实际 `<tool_call|>`。取消、length 截断、缺来源、前缀漂移均拒绝。
4. 输入支持原始 token IDs 或原始文本。token IDs 必须可无损解码再编码；文本模式明确标为 `raw-text-retokenized`，不宣称恢复了原始采样 token IDs。prompt/completion 分别编码后按边界追加，避免把边界重新合并成另一个 token。
5. 每次真实请求单独产生记录，尤其适用于实际 compaction 后的状态。不自动把不同请求状态合并为多目标序列。原有完整多目标入口仍要求其全部可见历史可由同一模板因果一致地表示。

调用接口示意（这是字段说明，不是可训练样本）：

```python
record = encode_captured_completion(
    {
        "sample_id": "...",
        "protocol_id": "...",
        "group": {"issue": "...", "repo": "...", "commit": "...", "split": "train"},
        "source": {"uri": "...", "sha256": "...", "format": "raw-generation-capture"},
        "messages": complete_actual_request_messages,
        "capture": {
            "format": "raw-model-stream-v1",
            "request_id": actual_request_id,
            "source_sha256": immutable_capture_file_sha256,
            "complete": True,
            "finish_reason": "stop",  # 或 tool_calls
            "prompt_token_ids": original_prompt_ids,
            "completion_token_ids": original_completion_ids_including_stop,
            # 没有 IDs 时可提供 prompt_text / completion_text；必须是原始流。
        },
    }, tokenizer, tools, max_seq_len=32768,
    chat_template=fixed_ct_v2_template,
    enable_thinking=True, thought_loss="mask",
)
```

来源 SHA 是调用者提供的溯源声明，编码器本身不会打开任意路径来认证来源。数据整理者须核对它对应的实际不可变记录、完成状态和 train 用途；不能为手工重建文本填一个 SHA 就视为真实捕获。若同时提供文本和 IDs，编码器检查二者一致。新增函数没有改现有 CLI 或数据整理程序；接入方必须明确选择实际捕获模式。

## 一次性 CPU 核对结果

使用现有 `/private/tmp/p009-replay-env/bin/python`，未安装依赖。通过临时 heredoc 执行；没有保存专用检查脚本。

| 核对材料 | 实际结果 |
|---|---|
| P015 boltons-298 六条已筛 train 窗口 | 长度仍为 3434 / 3925 / 6355 / 6741 / 7354 / 8165；监督 token 仍合计 246 |
| 同一真实 train 轨迹截至 submit 的 22 个动作同时监督 | 11,560 input tokens、948 supervised tokens；没有把它升级为包含最终回答的完整任务轨迹 |
| 改变第一动作之后的工具观察 | 第一动作及其全部因果前缀、标签完全不变 |
| 原有普通 final（无独立 reasoning） | 正常编码，包含模板实际结束边界 |
| 临时合成 thought + final，原始 IDs 入口 | 12 completion tokens；thought-mask 监督 7，include 监督 12 |
| 临时合成 no-thinking + final，原始文本入口 | 3 completion tokens，全部监督；固定模板实际 generation prompt 含空 thought channel |
| 临时合成 thought + tool call | 17 completion tokens；thought-mask 监督 12，include 监督 17 |
| 临时合成失败动作和错误反馈后接下一动作 | 769 prompt tokens 全部 mask，只有当前成功目标受监督 |
| 漂移 prompt、只有 parsed 字段、缺 final stop token | 均拒绝；没有自动补全 |
| 后来 user 导致早先 reasoning 被删除 | 拒绝，不把删除后的序列当完整多目标轨迹 |
| assistant content 被模板放到未来工具反馈之后 | 拒绝，要求原始请求/生成记录 |

所有手写诊断都显式 `synthetic=True`，`training_eligible=False`；没有写入正式训练池。`git diff --check` 通过。这里的“核对通过”只表示编码行为符合所列边界，不是模型能力结果。

## 仍然存在的明确限制

P015 boltons-298 的真实最终回答仍只有 parsed reasoning/content，没有原始 completion IDs 或原始生成文本。此前按已知通道语法手工拼接恰好得到 302 completion tokens，并不能证明实际边界逐字一致。因此本轮不转成可训练 final，原拒绝继续保留；其 source capture 原标志与单独 consumer-close 审计也没有被改写。

将来捕获原始流时，应同时记录实际 prompt、实际 completion（包括 stop token）、tokenizer/template/工具身份、request ID、完成原因与源文件哈希。如果服务省略 stop token，不能由这个入口猜补；需在采集端保留。P015 的 server prompt 用量与本地模板编码有固定 1-token 差异，既有记录没有原始 prompt IDs，因而本轮也没有把它宣称为逐 token 一致。

现成公开教师轨迹仍可作为清楚注明来源与序列化方法的离线示范准备；它们不是 Gemma 的实际 raw wire capture。两类来源不能混淆。对普通完整历史/no-thinking 情形，若 generation prefix 与历史渲染不一致，旧完整模板入口现在明确拒绝，原始捕获入口能忠实保留其实际请求。训练端、数据池和正式实验尚未因本改动自动重跑。
