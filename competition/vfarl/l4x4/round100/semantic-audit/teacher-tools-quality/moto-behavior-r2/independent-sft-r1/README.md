# 六题Moto真实行为示范：独立SFT增量

2026-10-28。仅固定100原索引9/10/11/13/14/15六个train ID。6条严格canonical均eligible、reasons为空，新的实际工具观察都在改前失败、改后成功；原官方完整文件范围分别7/69/33/38/8/68节点取得本次baseline失败/reference通过。没有以语法检查替代行为验证，也没有将旧成功或旧token轨迹复制为新示范。

[gemma-sft/sft-actions.jsonl](gemma-sft/sft-actions.jsonl) 为完整增量：6题、51条助手决策（45次真实工具调用+6条实际终止回答）、6521个助手监督tokens，SHA `f2d9ee06fe8b832649d6e4f044ea2de6ee72df5f73ab7b81856d71d7d6d2e027`。thinking0、无生成thinking/行为logprob/rollout/训练更新，作者是Codex off-policy。旧S600/F600、manifest、同ID旧trace及奖励均不改。

实际最大动作上下文4647tokens；13条动作超过4096，涉及5348/6755/7061/7111。这里保留完整真实观察，未截断或伪称4k可直接消费；6159/6492的完整动作在4k内，其余4题须使用足够上下文或另行真实重放精简recipe。该长度差异不改变原官方验收事实，也不代表模型收益。

新prompt由官方builder产生，逐字等于各次actual agent-prompt（保留CRLF），且这6个task SHA与question-revision-r2对应项相同。source按旧inventory逐个恢复字节/symlink/mode，剔除仅harness生成而不在原清单的文件，tree SHA等于原pin。保留原budget类型和值（6755为int60，其他为float30.0），没有复用不同预算的prompt。

每个raw report仅记录当前真实run、recipe/messages/events/JUnit及审阅信息。原task/source revision/mechanism review使用旧真实pin；缺失原blob此前按明确缓存单行逐字SHA恢复。复用未改的normalize_results.normalize、export_prompts和format_replays；没有改官方oracle、旧奖励、全局catalog或源脚本。tokenizer/template沿用已验证31B CT研究006缓存；实际CPU transformers5.13.1/tokenizers0.22.2/jinja2 3.1.6留在manifest。

运行profile与失败详情见上级README。一次性手动核对所有51条ID、train标记、token/mask长度、监督数、thinking0及无logprob；这些检查不是训练或修复收益。无GPU、网络提交或新增测试文件。
