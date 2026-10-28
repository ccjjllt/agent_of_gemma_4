**导出器后续缺陷：** 此处真实工具／原oracle记录保留，但旧per-action SFT中的CRLF片段有错位，不能直接训练；修正行已另存[当前监督r2](../../../question-revision-r2/supervision-r2/README.md)。不能只按文件hash／token自一致就认定completion正确。

# 其余六题Moto：真实源API、六工具及官方验收

2026-10-28。固定100原索引9/10/11/13/14/15；旧题身份、source/review pin、题面和奖励不变。新示范用真实归档源模块及真实boto3调用，全部仅在Moto离线mock context中运行，没有stub方法、AST替身或真实AWS请求。所有实际run_command均打印来源绑定并断言行为。

- 5348：空字符串DynamoDB key由被接受恢复为ValidationException。
- 6159：API Gateway冒号path由BadRequest恢复为可创建的pathPart。
- 6492：Route53创建结果由缺Location恢复为对应hosted-zone URL。
- 6755：已验证邮箱身份的MAIL-FROM从InvalidParameter恢复成功，真实读取配置确认域名。
- 7061：AutoScaling的key/value过滤从两条tag恢复为正确单条。
- 7111：Lambda创建时省略PackageType，从None恢复Zip；仅创建metadata，没有执行函数或Docker。

六题都完成新的实际recipe/call/event/messages绑定以及原完整官方文件范围0→1：5348为1fail+6pass→7pass；6159为1fail+68pass→69pass；6492为1fail+32pass→33pass；6755为1fail+37pass→38pass；7061为1fail+7pass→8pass；7111为3fail+65pass→68pass。通过数只是验收范围，模型未训练或采样，未声称模型修复收益。

原snapshot是在harness.setup后保存。此次只按旧source-inventory恢复原文件字节、symlink和mode，额外生成文件不带入；每题tree SHA匹配原pin。显式Namespace保留原budget数值类型：6755为int60min/verification300，其余float30.0min/verification240。所有实际initial_prompt SHA与旧pin一致，也与当前question-revision-r2对应task版本一致。源/预算证据在各source-binding及input-bindings.json，实际工具/官方结果保存在其实际运行目录。

5题用实际Python3.12.13。6159首个actual-run因缺jose而官方收集失败，完整失败保留；从同一既有缓存补真实jose及dist-info到独立profile-r3后，在actual-run-r2重新执行并通过。其他4题使用原冻结pure-profile-r2。

6755旧review已明确Python3.12安全回补后的strict邮件解析与此历史源版本不相容。复用同原SHA的Python3.11.8二进制；先前actual-run-py311由于clean task venv缺dateutil，在6个动作后终止，未记录为成功或行为基线。后续actual-run-py311-r2使用独立清洁依赖profile：原缓存纯库、现有uv的CFFI2.1.1 cp311、实际导入图所需的既有Mac纯依赖及cryptography50.0.2 abi3 Rust。真实clean3.11导入、RSA操作、源API及全部38官方节点均完成。没有改email.utils、测试或官方标志；profile和失败没有追认为相同环境。

本次全部profile只从既有本地缓存复制，未下载、覆盖原环境或启动GPU。真实依赖来源/文件hash/metadata已冻结在对应source-manifest.json副本；这是本机macOS CPU证据，不声称Kaggle chroot/比赛Docker等价。

独立严格canonical与SFT在independent-sft-r1。完整51条动作有13条超过4096（四个任务），没有截观察或冒称4kready。旧S600/F600及所有旧轨迹原样保留，新数据是Codex off-policy教师材料，非rollout/无logprob/无生成thinking。下一步消费须按实际task/prompt SHA和上下文长度显式连接。
