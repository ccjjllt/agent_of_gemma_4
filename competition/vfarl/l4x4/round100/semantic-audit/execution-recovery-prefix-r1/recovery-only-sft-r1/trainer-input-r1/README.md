# Controlled-prefix recovery-only trainer input r1

独立辅助导出：固定库存的Moto5200/Tenacity26两条真实恢复轨迹，12决策（10工具＋2实际终止文本）、1060个shift后监督token。最大上下文5250，9条超过4K；8192仅数据读取上限，未测模型显存/反向容量。输入原IDs/mask逐位置映射attention_mask=1、labels=原token或-100，没有截断、丢行或替换输出token。

数据SHA256 `350e5cf41b3381bc090a028479d901d874354a785c2666788aa15b15817ec1ef`，原action-view SHA256 `e6e40c6758f2b493f33253bef3417660d5f09ed68758bda3f09b3105863da294`。只接受source action indices5–10、raw assistant indices11/13/15/17/19/21；failure event4/rawmsg10以前的错误prefix、所有工具观察及每条样本的既有历史均为零loss。实际messages/recipe/events/失败边界/原scope baseline-final pins已重核，全部12个prefix/completion由真实CT模板重建，Moto原CRLF保留。没有构造full-episode IDs/mask或绕过完整canonical准入。

这是Codex控制错误prefix＋实际Codex恢复的离策略示范，不是Gemma行为/同状态RL pair/人为终端标注。meta.synthetic=false沿用现有编码器“真实执行trace而非synthetic tokenizer/fixture”语义；一道人为变异task与一个PR task的来源独立记录。未执行的intervened-state官方result保持null；原baseline/final pins只是历史真实scope证据，不新增reward字段或模型收益。

必须使用独立protocol `vfarl-controlled-prefix-recovery-v1`及本目录deployment-tools.json，不与main100或旧protocol混用。实际配置字段如下，需合入另行明确的model/training配置后，把--data指向本目录encoded.jsonl：

```json
{
  "protocol_id": "vfarl-controlled-prefix-recovery-v1",
  "tokenizer_name_or_path": "/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/research/006/data-replay-20261013/tokenizer",
  "chat_template_path": "/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/research/006/data-replay-20261013/tokenizer/chat_template.jinja",
  "tool_schema_path": "/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/competition/vfarl/l4x4/round100/semantic-audit/execution-recovery-prefix-r1/recovery-only-sft-r1/trainer-input-r1/deployment-tools.json",
  "max_sequence_length": 8192
}
```

Tokenizer采用训练器真实GemmaTokenizer语义fingerprint `d001718145f4980b65e70b5ffa1dfce93b3eca5a5d8de2446b8c483f2b4b8927`；CT raw SHA与tools hash分别为`94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413`、`43118e0d2729405fe1613b5ee22930cde39d701c19fb754a6f618fbc0d3f6199`。它们沿用encoding.gemma_tool_chat helper，没有拿tokenizer.json rawSHA代替语义fingerprint。protocol_hash为`028d5c8d0bb792c8446b7251fa903792fce6bf5e064a4cc90dc90480641c5688`。

真实load_encoded_jsonl(max8192)、SelectedLabelCollator混合长度CPU右padding/causal shift、validate_tokenizer_protocol已消费，监督目标仍1060，提示/padding无新增loss，12条原IDs/labels映射及prefix全0已确认。原action-view首行被reader真实拒绝的错误保留在manual-consumption.json。未加载模型、优化器、GPU、新工具环境或重跑oracle。

生产入口：competition.vfarl.prepare_trainer_input的export_recovery_only或CLI --recovery-manifest＋显式--selection；main100 export函数AST保持原样。selection明确钉question-revision-r2 SHA `e665c576d29541fd65f1610157986ab4e21fa959cddbec60a1f79d8c75b80764`，只是这两题版本证据，不自动latest lookup，也不为后续r3改动其他题而改源trace。原aux manifest冻结于source-recovery-manifest.json（SHA `0f07fdce9cb87b2533e65950ef330f7916774167198c586a3f0b6ed8f0d05718`）；原路径是可更新推荐入口，冻结副本承担输入hash绑定以避免循环。
