# split_by 新题面的真实示范增量

2026-10-28。固定100题索引86，S／S4。题面明确连续前缀与余下部分，不将所有满足predicate的元素归到前一部分。新task/prompt与question-revision-r2对应项相同，原S600/F600及旧轨迹均保留。

`teacher-r3/`是真实官方六工具重放：7次工具调用与1条终止回答，原203个验收节点全部保留，baseline真实失败、参考生产修复后真实通过。实际初始prompt SHA为`4b50474c4a88a3ca35602a8e5396b91e62e8d7023936fbe0076dbec27f321995`，逐字等于新版官方builder结果。按原source-inventory重建文件内容和mode，source tree也与原始pin完全一致。没有将旧动作复制为新版示范。

已复用原normalizer严格准入，`canonical/split-by-revised-r3-replay-result.json`无准入缺项。单题source catalog独立披露旧原文与新curator文本及其SHA；未放宽原F2P/P2P、文件范围或评测节点。

推荐增量为[gemma-sft/sft-actions.jsonl](gemma-sft/sft-actions.jsonl)：8条决策、628个监督tokens，最大上下文3046。文件SHA为`3c5f7ea0d5f1ffb24846396165ce6a06ab94304464018ceaa580839a58e7e551`。沿用已验证CT tokenizer/template，实际CPU编码运行时版本保存在manifest。thinking0，无生成思考、行为logprob、Gemma rollout或训练更新；属于Codex指定并实际执行的off-policy示范。

首个teacher-r2启动因旧临时dependency目录已清理而在动作前拒绝，简短原因留存。随后同目录的实际重放使用默认60分钟预算及post-setup snapshot文件/mode，确实取得官方0/1，但prompt与目标新版本不一致，因此仅保留为实际备用profile证据，不进入本增量。teacher-r3恢复原30.0分钟预算、原source inventory及真实缓存依赖后重新执行，未追认前一轮动作。详见`teacher-source-reconstruction.json`。

这是macOS CPU材料修复，未证明Kaggle/Linux等价或模型独立修复收益。后续消费应显式以当前task/prompt SHA连接本增量，不能把38个题面revision都标成已有新示范。
