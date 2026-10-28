# 固定100题监督r4

当前数据保留100题／100个family／43个snapshot，67共同＋33专科：805决策（706工具、99完整终止）、78,148监督tokens，完整上下文最大4649。19条动作／7题超过4K，未截断。

源[action-view](sft-actions.jsonl) SHA`bded575fae8d21527e94fb001ba690b4ada69e50e408faa6241b70d2ffcb369c`。现有SFT reader输入为[trainer-input-r4/encoded.jsonl](trainer-input-r4/encoded.jsonl)，SHA`5555b374ee474e8816249e2b93719ccecdb9a8332f07c4a4f141708755b057ec`，须同时采用[匹配协议配置](trainer-input-r4/README.md)。这里证明数据表示及CPU消费，不表示当前31B NF4模型加载路径、反向容量或学习效果已实跑。

相对[r3](../supervision-r3/README.md)仅替换Bag82的8行，其他99题797行逐字节相同。新的[真实示范](../../semantic-audit/oracle-strength-20261028/bag82/multiplicity-teacher-r2/README.md)直接观察aab／ab=1、aaab／ab=2，以区分错误set去重与正确multiset频次；题面、原68文件源码及模式、30.0预算、原生产patch和全部427required nodes不变，完整430cases官方baseline0／repair1。首次正确重放r1包含预载计时噪声，原件保留而未混入；r2改变实际预载配置后独立执行，真实工具观察不含仪表JSON，没有事后删除观察。r2最大context3996，r1为4520，这只是两次真实数据上下文的观察差异，不是GPU节省结论。

50题405决策来自本轮真实teacher重放，50题400决策沿用历史来源；49题392行仍为原S600字节，Hydra8行保留此前CRLF修正。数据始终是Codex规定并真实执行的离策略六工具示范，没有Gemma行为logprob、生成思维或VFARL焦点对。81道变异／11PR复现／8Gym来源独立记录；不将meta.synthetic=false解释成人工原生bug。

[consumption-index.json](consumption-index.json)绑定100个当前task/prompt/publisher及源messages/canonical/action原行。Bag82有[独立真实CT复核](../../semantic-audit/oracle-strength-20261028/bag82/multiplicity-teacher-r2/independent-review.json)，生产导出器另从所有805个实际prefix重建后直接映射原IDs/mask。现有reader、协议与10条混长CPUcollator样本消费通过（覆盖新增8行和全局最短／最长）；不新增prompt或padding loss。父manifest冻结副本避免推荐指针循环。

本数据绑定[r3 selection](../selection.json)及[课程r6](../curriculum-r6.json)。五个弱oracle题82／84／85／88／95的正确SFT均保留，未来原oracle RL暂停；另4 runtime unknown仍暂停，待采60题为32共同＋28专科。旧奖励与35历史领取不改，全部原数据版本保留；r1错误跨度版本继续禁止训练。辅助recovery-only输入仍是独立12行／协议，未拼入这805行。没有模型训练、GPU作业或新的独立修复率结论。
