# R3真实格式失败与nullable模板兼容

2026-10-28。只读取固定train R3的41份actions文件／501个原始raw_text，完整保留原文本，不执行模型命令、不读heldout或预留题、不GPU、不写测试文件。source文件SHA逐项等于原download-manifest记录。

## R3结论：未发现合法调用被parser误拒

从carrier-session-r3/run.py的SOURCE_BUNDLE提取真实collector源码，SHA `ccaa9e2f1910dd6b05031047bb5965f5ad7628ca67cb91d916860450d75ab667` 与其carrier-manifest一致；其中NativeArguments／parse_call的AST与本轮修复前实际源文件相同。按原raw_text重放，483条得到call，10条明确语法错误，8条没有visible native call。后者包括5条正常stop终止文本与3条只有tool-close／旧JSON碎片的tool_calls结束，缺失的opening/native body不能被补造成可执行调用。

10个真实错误的episode均记录对应的malformed_action原错误，没有发现harness-feedback.jsonl，属于R3当时的历史终止路径。具体为：4个裸参数名后夹杂单／双引号，3个native字符串错误地用反引号结束，2个完整参数对象后多出关闭符，1个缺参数对象的左花括号。多行代码或反斜杠本身不是这些错误原因；483条实际call中有87条包含多行字符串、20条含反斜杠，它们已按原样解析。该范围内没有实际nullable／boolean／nested-object参数实例，不能将未观察情况说成R3通过。

| 固定train ID | source位置（1-based） | 原始语法问题 | raw_text SHA前缀 |
| --- | --- | --- | --- |
| getmoto__moto-6159 | worker-0/original:29 | extra_closing_token_after_argument_object | `abd3ea2ea6a8` |
| hukkin__tomli.443a0c1b.func_basic__qlgy79aj | worker-0/original:15 | stray_quote_after_bare_argument_key | `92c516b81817` |
| jd__tenacity.0d40e76f.func_basic__rlm5c89z | worker-0/original:4 | stray_quote_after_bare_argument_key | `d06e40681b0f` |
| prettytable__prettytable.ca90b055.lm_rewrite__r791b725 | worker-0/original:10 | backtick_used_instead_of_native_string_close | `58f57b7cc722` |
| bottlepy__bottle.a8dfef30.func_basic__r3tamcap | worker-1/branch-0:6 | missing_opening_argument_object | `d6e02b7e7196` |
| getmoto__moto-5348 | worker-1/original:25 | backtick_used_instead_of_native_string_close | `6f8e7a334712` |
| iterative__dvc.1d6ea681.pr_7628 | worker-1/original:3 | stray_quote_after_bare_argument_key | `7a1cabb189c5` |
| jd__tenacity.0d40e76f.func_basic__pvngsy7s | worker-1/branch-0:5 | stray_quote_after_bare_argument_key | `7ce061898937` |
| jd__tenacity.0d40e76f.func_basic__pvngsy7s | worker-1/branch-1:16 | extra_closing_token_after_argument_object | `55280569b049` |
| jd__tenacity.0d40e76f.func_basic__pvngsy7s | worker-1/original:11 | backtick_used_instead_of_native_string_close | `17cb01dd89fa` |

完整来源路径、原始JSONL line SHA、raw_text SHA、原文和episode SHA见 [parser-observations.json](parser-observations.json)。这些计数限定已读取范围，不代表新模型修复收益。

## 独立确证并最小修复：精确None字面量

真实pinned template SHA为 `94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413`，与R3 policy一致；six-tools SHA `f0ef05bc2ddef082703d0dd872014df98d6a69f50f08f589d963ffc9fcb9a47c` 与carrier记录一致。read_file的start_line/end_line schema允许integer或null，默认null。实际tokenizer.apply_chat_template对该合法nullable输入生成：

```text
<|tool_call>call:read_file{end_line:None,filepath:<|"|>example.py<|"|>,start_line:None}<tool_call|>
```

该action片段SHA为 `5de40d4528d8fe1acbc08b5ffb6e6ad411e1d9681cd613e9ef23891b38fc0f03`。原parser只接受null／none，确实报 `Invalid native argument literal`。这是一例受控模板roundtrip兼容缺口；R3真实501动作中没有此现场，不将其归因为上述10次坏语法。修复前精确输出、schema、source与SHA先冻结于 [before-fix-evidence.json](before-fix-evidence.json)，文件SHA `b6fcb64dfb24b70397eb1b7c7bfb9a362413658ad779a30546664989e74d23c7`。

只改competition/vfarl/l4x4/run.py的NativeArguments.value：在已有常量值位置增加精确None，并映射成JSON null。保留原结构边界、closed call要求和字典工具dispatch；没有eval、模糊大小写匹配、补引号／括号、修复损坏输出或执行未知tool。模板中的bool已明确输出小写true／false并精确roundtrip，因此未增加Python True／False。

[pinned-template-controls.json](pinned-template-controls.json) 还保存合法多行、普通引号、反斜杠、heredoc、bool以及thought内引用tool控制串的模板对照，实际返回值逐字等于输入参数。nested-object对照只说明native grammar能解析；六工具当前没有接受该object值的对应字段，所以明确schema-invalid，不将它算作可执行合法工具样本。

## 修复后一次性观察

用同一501条原文本比较frozen parser和修复后parser，逐条结果、错误文本及所有参数完全相同，仍为483call／10error／8none。合法模板None现在精确返回None；NoneX、None()、None.foo、None+1、NonE、NONE和未由模板输出的Python True／False仍拒绝。六个损坏／未闭合边界对照也保持原拒绝，没有恢复虚构调用。

[after-fix-observations.json](after-fix-observations.json) 保存逐动作结果绑定及这些受控观察；SHA `f0d09dc767a6ee089ad5f98cb1880f7bb53ecb98b9c22670255a55d257e5fd51`。只修兼容性，未启动新rollout、环境工具执行或训练，不宣称真实修复收益。

## 后续官方容错语义边界

本页“10条真实坏语法／未误拒”的结论限定于严格原生语法，不代表比赛server会对它们全部终止。后续[同版官方组件链对照](../r3-official-parser-semantics/README.md)实际发现10条中8条仍被提取为结构化call、2条普通text；483合法call参数仍完全一致，8原无call不变。提取不等于可执行或修复，坏edit缺必需old_string，旧奖励与原记录没有重标。[官方请求与nudge路径](../r3-official-request-path-r1/README.md)又区分了源码／文档和完整服务未执行边界。没有为修复CRLF坏标签而放宽parser。
