# 固定100题监督r3

历史输入保留：当前推荐为[监督r4](../supervision-r4/README.md)，只补Bag82的8条真实频次示范，其余797行与本版本逐字相同。本页及trainer-input-r3继续描述本版本的77,982监督tokens，不追标新82示范；它不是已撤回的跨度错误版本。

当前完整库存保持100题／100个family／43个snapshot，67共同＋33专科。本版本为805决策（706工具、99完整终止）、77,982监督tokens，最大上下文4649；19条动作／7题超过4K，完整保留。

当前训练器输入是[trainer-input-r3/encoded.jsonl](trainer-input-r3/encoded.jsonl)，SHA`91e8c5de3372068381a3ca5dbeb575ae8ca5d79b48d5bb27d4b8a72ddc99fd5b`；必须同时采用其[必需协议配置](trainer-input-r3/README.md)。[sft-actions.jsonl](sft-actions.jsonl)是可追溯源action-view（SHA`fe7b8f9fd238e183efdf0688dba3193af758357a5732ec402e6ead6b45b9bdd1`），不能直接传给SFT trainer。

相对[r2](../../question-revision-r2/supervision-r2/README.md)仅替换两题的16行，其余98题789行逐字节相同：Bottle4补真实目录destination与overwrite=False双分支；PyPika40发现题面不存在的SYSTEM_TIME导入，按question-revision-r3最小修订后真实观察DELETE／UPDATE SQL。两题都通过相同原完整oracle baseline0／repair1，生产patch与历史相同；原版本及失败/漂移尝试保留，不改旧奖励。两新teacher是数据质量增强，不是新模型收益。

49题397决策来自本轮真实重放，51题408决策沿用历史来源；其中50题400行仍为原S600字节，Hydra8行保留既有CRLF渲染修正。r1错误跨度版本继续撤回，r2是历史输入版本。当前consumption-index逐题钉task/prompt/source/messages/canonical和实际行SHA。原81变异／11PR复现／8Gym来源保持，不把这些Codex离策略轨迹当作Gemma思维或VFARL pair。

[独立聚合复核](independent-assembly-review.json)覆盖805行来源与98题字节保留、两新导出14条工具参数／2终止片段和r5课程。PyPika40另有实际CT重建的独立复核。现有production exporter又真实重建全部805个generation prefix，读取器／协议／CPUcollator实际消费通过；父manifest冻结副本避免推荐指针循环。没有以hash自一致代替实际源语义。

本版本输入时绑定[r3 selection](../selection.json)与[当时课程r5](../curriculum-r5.json)，其原manifest和冻结副本不回写。之后新确认Bag82／Marshmallow88盲点，当前RL调度改用[课程r6](../curriculum-r6.json)：5弱oracle＋4 runtime unknown暂停，待采60题（32共同＋28专科）。正确SFT、历史35领取与原奖励不变。当前macOS证据不自动成为新Linux/chroot usable；未启动模型／正式GPU训练。两条recovery-only辅助数据继续独立，未混入此805行。
