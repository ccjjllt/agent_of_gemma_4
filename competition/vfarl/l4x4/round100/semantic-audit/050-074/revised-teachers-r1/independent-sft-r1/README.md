# 十题r2新版prompt：独立Gemma SFT

2026-10-28。仅固定100原索引51/52/53/54/55/66/68/69/70/74，共10个train ID。所有新teacher实际在当前r2新prompt下执行，严格canonical准入无缺项。原F2P/P2P及源身份不改，旧数据不重写。

推荐 [gemma-sft/sft-actions.jsonl](gemma-sft/sft-actions.jsonl)：10题、82条助手决策（72次真实工具调用+10条实际终止回答）、7168个助手监督tokens，最大动作上下文3813，全部在4096内。action SHA `9b3b8209ca7e5bc3f155a48c7875fca697d33df4635c796922c09f97c3cdd25b`；manifest SHA `8ee4937560dc11d991208b1961eb9edb7a9e96cdca1b0d2e443b7f75a9371cf7`。

这是Codex off-policy指定并真实执行的teacher，thinking0，空thought控制为prompt且不进loss；无生成thinking、行为logprob、Gemma rollout或训练更新。真实actions/工具观察没有改写或从旧轨迹补齐。source/review使用原真实pin，新版curator披露只保存在本目录的source-catalog，未改全局catalog。

[export-bindings.json](export-bindings.json) 逐episode给出canonical路径/hash、实际run、recipe/events/rawmessages SHA、rawmessages首prompt SHA及当前r2 expected prompt SHA。十题首prompt与agent-prompt原bytes和官方builder逐字相同。51/54/55只来自actual-run-causal；其他早期导入检查运行未拼入。7个其他recipe动作原样重放，3个causal recipe仅加强首个API复现。所有生产patch字节与旧示范相同，原task/recipe/canonical SHA手核未变。

编码复用已验证31B CT研究006 tokenizer与模板，实际transformers5.13.1/tokenizers0.22.2/jinja2 3.1.6版本留在manifest。一次性手工核对82条的ID/train标记、token/mask长度、监督数、thinking0、无logprob、逐event/call/observation配对和真实终止回答。没有截断观察，未把检查或节点数量当作模型收益。运行profile与失败原因见上一级README。
