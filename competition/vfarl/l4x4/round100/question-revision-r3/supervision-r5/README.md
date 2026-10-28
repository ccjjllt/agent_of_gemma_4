# 固定100题监督 r5

当前数据保留100题／100个family／43个snapshot，67共同＋33专科：805决策（706工具、99完整终止）、78,423监督tokens。最大完整上下文4649；19行动／7题超过4096，未截断。

源 [action-view](sft-actions.jsonl) SHA `ed99e56fc669134945766e563b604f678df91d4aef3c9037bed86acc26578bc1`。现有训练器应消费 [trainer-input-r5/encoded.jsonl](trainer-input-r5/encoded.jsonl)，SHA `a5dea1aecd85bf8478d29cd1ebf2588df498a3baa8c0ee4b7d20eaa43f20d704`，同时采用 [匹配协议配置](trainer-input-r5/README.md)。这证明数据表示与CPU消费，不代表31B模型加载、反向容量或学习效果已实跑。

相对 [r4](../supervision-r4/README.md) 只替换 index81 Hamming 的8行；其他99题797行源字节及顺序保持。新的 [真实 teacher](../../semantic-audit/075-099/hamming-truncate-oracle-r1/truncate-aware-teacher-r1/README.md) 保留默认 abc/ax=2，直接断言 truncate=True 时为1、empty/same 为严格 int0。其 [独立消费复核](../../semantic-audit/075-099/hamming-truncate-oracle-r1/truncate-aware-teacher-r1/independent-consumption-review.json) 核定实际工具观察、66原文件字节／模式、原patch、float30.0预算和当前prompt；原427 required nodes／430 cases完整baseline0→reference1。题目、source、oracle和旧奖励都未改，wrong probe未进入正样本。

51题413决策来自本轮真实teacher重放；49题392决策沿历史来源，其中原S600的48题384行保持原字节，Hydra8行保留此前CRLF修正。所有数据都是Codex规定并真实执行的离策略六工具示范，无Gemma行为logprob或生成思维，不是VFARL焦点对。题目来源另记录81个synthetic mutation、11个PR复现、8个Gym来源，不把真实执行标记解释成人工原生bug。

[consumption-index.json](consumption-index.json) 绑定全部当前task／prompt／publisher、actual messages／canonical和源action原行。生产导出器实际重建805个generation prefix，再把原IDs／mask精确映射为attention1和labels；现有reader、tokenizer语义指纹／tools／protocol及一次10行混长CPU collator消费均通过，覆盖新增8行和全局最短／最长，prompt与pad无loss。父manifest先冻结进child，再增加推荐指针；[parent-supervision-r4-manifest.json](parent-supervision-r4-manifest.json) 另保留R4原件，旧R4 parent／child／报告／数据不改。

数据绑定原 [selection](../selection.json) 及 [课程r7](../curriculum-r7.json)，course SHA `f51cdba0d2a93d8d910536ba2997c46622a0b90a67efd8bd7afd65be14a916b7`。正确SFT全部保留；未来原oracle RL暂停7个weak题56／81／82／84／85／88／95及4个runtime unknown，共11题，待采59题为32共同＋27专科。课程暂停不把unknown补0，也不重算历史奖励。

全部99条terminal保留：pinned官方允许submit后的助手响应，新的live collector也允许继续工具／再次submit／final；本次没有删除终止监督或重写原观察。辅助recovery-only仍独立，没有混入这805行。未启动模型、GPU、optimizer或正式训练；[单步SFT配置草稿](../../../../../../training/configs/sft_vfarl_current100_31b_nf4_r32_draft.README.md)已由独立后续工作更新到r5，仅完成数据dry-run；micro1／accum1／maxstep1不表示训练完全部100题。
