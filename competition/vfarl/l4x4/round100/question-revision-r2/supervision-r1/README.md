# 已撤回消费推荐：固定100监督r1

2026-10-28。**此r1不能训练。** 后续模板／parser与实际messages交叉核对确认CRLF跨度错位：58条工具片段及受影响终止文本与真实输出不同。此前可消费判断过早；原始canonical／messages与tokenization自一致并不能证明completion正确。已从8个受影响episode整体重导为[监督r2](../supervision-r2/README.md)，保留本文件及原SHA作缺陷证据，绝不放宽parser接受坏片段。

[sft-actions.jsonl](sft-actions.jsonl)是本次固定100题的完整off-policy监督文件：**100题、805条真实助手决策、77,257个监督tokens**。文件29,143,812字节，SHA256为`191bd117025f91094f3fa00eb1d029131974b5b9857d859093bc53206746e360`。保持67共同＋33专科、100个repair family、43个snapshot，不添加或替换任务身份。

## 本次实际修正了什么

47题使用全新真实工具示范，共381条决策：38条版本化题面全部真正按新prompt重放；8条moto补真实改前失败／改后通过的API断言；OSA改正空串距离。MySQL copy的modifier隔离增强属于38条新版之一，不额外计题。51/54/55也将原“改前只导入”补为真实行为复现，最终只选择因果重放版本。所有新示范均保持原完整oracle并取得实际baseline0／repair1和严格canonical准入；不同runtime或初始化失败保留在对应实验，不改名为成功。

其他53题的424条旧监督原始行逐字节保留，源S600文件及历史messages均仍匹配冻结SHA和当前prompt。旧47题行完全被明确替换，没有把修正示范与已知错误旧标签并拼；原S600/F600及历史记录不覆盖。逐题输入版本、canonical、实际messages、每条原始源行与SHA见[consumption-index.json](consumption-index.json)，源批次及完整统计见[manifest.json](manifest.json)。其中`new_replay_in_this_optimization=false`只表示本轮保留历史已验示范。

## 消费边界

这是Codex指定并真实执行的六工具off-policy示范，thinking0。没有Gemma行为采样、生成思考、行为logprob或本轮模型更新，不能作为VFARL的on-policy焦点对。所有100个当前publisher源hash／task版本／prompt及实际messages绑定，805条token IDs均与固定CT缓存重新编码一致，实际generation-prefix IDs与监督mask也逐项一致。强制空thought控制在prompt里，工具观察未改写。

完整历史保留后，最大上下文**4647**，有14条动作／5题超过4096，不能称4K可直接消费。使用至少4647的完整上下文，不截断或静默丢掉长动作；8K／16K配置仍需各自真实硬件容量证据，本文未运行模型反向。

OSA85、qrcode SVG84及iniconfig95的正确SFT均保留；原RL oracle分别接受常数0距离、非SVG根、恒定行号1等错误输出，故[最新RL课程](../curriculum-r4.json)暂停三题未来RL评分，原oracle与历史奖励不改。监督覆盖100题，与待采RL62题的状态不同，不能混淆。

新版真实重放是macOS CPU证据，不追认当前Linux/chroot的旧usable。38条题面revision保留同一原source、role、family、完整F2P/P2P；Kaggle新CPU／短GPU代码payload仍未提交。正式GPU训练保持关闭，以上是训练材料和机制证据的改善，不是模型修复率或比赛成绩收益。
