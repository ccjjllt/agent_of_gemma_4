# 固定100题当前监督导出r2

历史输入版本：当前推荐已移至[监督r3](../../question-revision-r3/supervision-r3/README.md)，只替换Bottle4／PyPika40的16行，其他789行与本版本完全相同。本页及trainer-input-r2继续绑定旧r2题面与真实来源，不追标新40题面；它与已撤回的r1跨度缺陷版本不同。

2026-10-28。使用修复后的真实模板导出器，替代已撤回的[监督r1](../supervision-r1/README.md)。[sft-actions.jsonl](sft-actions.jsonl)含**100题、805条决策、77,187个监督tokens**，29,143,157字节，SHA256 `a416cda43d8733d5b90f19e6ff744b9e3e007de932922c8c178d5b510c00869d`。67共同＋33专科、100个repair family、43个snapshot及题目身份保持不变；没有启动训练。

**训练器输入：** 使用[trainer-input-r2/encoded.jsonl](trainer-input-r2/encoded.jsonl)，SHA `ffcd3e5d366ab9100849ba8fd74d805b664743024b31646702f6651a7eb14cbf`，并采用其[协议与必需配置](trainer-input-r2/README.md)。本页sft-actions仍是源action-view，不能直接传train_sft/train_fsdp。新转换精确保留805条原IDs/mask，补attention_mask／labels／真实部署指纹；实际读取器、collator shift和工具协议均已消费，独立复核一致。转换初稿r1的无参工具metadata与源不符，已弃用且未训练；r2逐字匹配原formatter默认schema，未改动作或训练器接受规则。

## 实际修正与来源

r1中CRLF被read_text转换成LF后仍使用原字符跨度，截坏58条工具调用与8条终止文本。已在format_replays.action_examples中改为核对原bytes SHA后直接UTF8解码。按8个episode的原messages与原span整体重导66行，只改变completion、IDs、mask及相关token计数；739行／92个episode逐字节保留。没有改原prompt、工具日志、任务、oracle或parser接受范围。[缺陷与原始片段证据](../../optimization-overnight/sft-parser-roundtrip-review/README.md)和[重导来源绑定](../../semantic-audit/rendering-correction-20261028/manifest.json)保留失败数据及修正路径。

47题的381条决策来自本轮真实重放：38条新版题面、8个moto真实API行为验证增强及OSA空串距离修正。MySQL modifier隔离增强属于38题之一，不重复计题。其余53题424决策沿用历史真实teacher来源，其中Hydra 8行只修正渲染，剩余52题416行仍为原S600字节。重导不计为新的工具执行或修复成功。[consumption-index.json](consumption-index.json)逐题绑定当前task／publisher源hash／prompt、实际messages、canonical及实际选用的源行；旧批次清单只表示可审计的原始来源，不表示把旧错误行拼入现行文件。

## 手动核对与边界

805条generation-prefix与completion全部从实际messages重新按固定模板构建；706条工具调用与真实name／解码arguments精确解析一致；99条终止文本与完整真实模板片段一致，而非仅parse为None。805条CT IDs、prefix IDs与loss mask均逐项一致。独立复核再次确认全805条实际模板片段、工具／终止动作以及66行修正／739行原字节保留，见[独立复核](../../optimization-overnight/sft-parser-roundtrip-review/supervision-r2-independent-review.json)。单纯hash或token自一致曾漏掉跨度错误，因此不把它们单独当作语义正确证据。

完整上下文最大**4649**，14条动作／5题超过4096。保留工具观察与真实上下文，不截断；消费需至少4649上下文。8K／16K反向容量仍需实际硬件证据。

这是Codex指定并执行的六工具off-policy示范，thinking0；不是Gemma样本、生成思考或VFARL焦点对，没有行为logprob。新重放是macOS CPU原完整oracle baseline0／repair1证据，不能替代当前Linux/chroot适用性或模型学习收益。OSA85、qrcode SVG84及iniconfig95正确SFT保留，弱oracle仅暂停未来RL；[课程r4](../curriculum-r4.json)待采62题为32共同＋30专科。原S600/F600及历史记录保留，正式GPU训练关闭。
