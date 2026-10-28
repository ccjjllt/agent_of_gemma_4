# OSA 修正示范：匹配当前固定100的预算与prompt

2026-10-28。仅固定train索引85，S／S4。已按原source-inventory重建生产源码文件内容及mode，source tree SHA `7cafc82ff1b2402aabf882a9e7ceb9c96428f3ffdf083d082b27bfba8dbc4921`。重新执行真实六工具流程，恢复原30.0分钟／100工具／300秒command与verification预算；官方prompt SHA `f020d86c2be059f2f0e01355c0f06c222d4f402e5e04794435623955d95634ab`逐字等于当前v4及question-revision-r2该项。

`../corrected-six-tools-r3/`保留全新7动作、终止回答、观察、patch和完整原18节点验收：baseline真实失败，修正后真实通过，实际post-edit断言两侧空串及相邻交换距离为`[5.0, 5.0, 1]`。原F2P只打印距离的弱点不变，该题未来RL仍暂停，正确SFT不等于可靠奖励。

推荐当前prompt增量为[gemma-sft/sft-actions.jsonl](gemma-sft/sft-actions.jsonl)：8条决策、954监督tokens、最大上下文3909；SHA `38d8222fead4bafef50944db67eac501fbde1fb99b5ee4737e409ccf60e4f130`。严格normalizer准入无缺项，原S600/F600不覆盖。沿用已验证CT tokenizer/template，实际CPU编码版本见manifest；Codex指定并实际执行的off-policy示范，thinking0、无生成思考/行为logprob/模型rollout/GPU/训练。

此前corrected-six-tools-r2及corrected-sft-r2确有真实修正行为与官方0/1，但其60分钟prompt与当前固定版本不相同，source tree也包含post-setup差异。因此它们保留为独立profile证据，不能混入当前固定prompt的监督。此次不是改写前一轮prompt/动作，而是从原源码清单与原预算重新执行。

macOS CPU重放不证明Kaggle/Linux等价或模型独立修复收益。
