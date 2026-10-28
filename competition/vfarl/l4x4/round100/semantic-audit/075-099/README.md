# 固定100题索引75–99的语义审阅

2026-10-28。只读固定train题面、实际已审生产补丁、目标源码片段和原验收节点；没有读取预留四题或heldout。逐题结论和来源hash见`review.json`，有界真实源码观察见`manual-observations.json`。除OSA明确记录的整份原验收外，其他观察不是全题重新验收。

## 已证实的问题

| 索引 | 任务 | 真实发现 | 处理方向 |
| --- | --- | --- | --- |
| 77 | TextFSM List顺序 | 每行都Record的原示例只产生单元素列表，两版相同；多行累计到EOF才观察到逆序 | 澄清可触发的示例，保留顺序契约 |
| 78 | Graphene GenericScalar | 原top-level导入不存在；输出字段查询两版都正确，实际故障是输入ObjectValue literal解析 | 修正API与输入路径，不能把输出序列化当literal解析 |
| 83 | QR编码模式 | MODE_ALPHA_NUM在util，不在constants；真实optimal_mode对混合字节串从1改为2 | 修正复现导入和调用 |
| 85 | OptimalStringAlignment | 已验收示范的空串距离为0；常数0也通过全部18个原节点及官方JUnit matcher | 原RL oracle过弱，暂不能作为可靠数值奖励；保留原记录，另存正确生产候选 |
| 86 | funcy split_by | 原题要求全序列按真假分组，真实契约是连续真前缀及剩余序列 | 已产生独立问题revision，见`../../question-revision-r1/` |
| 89 | XML namespace | 原无普通属性示例两版输出相同；加入根属性后才显示@xmlns映射反转 | 澄清实际观察路径 |
| 91 | pyparsing flatten | 原actual声称完全逆序，真实为`[3,2,4,1]`；修后为`[1,2,3,4]` | 修正actual，保留目标源序 |
| 99 | funcy chunks/partition | 原chunks actual含不可能的变长切片；step=1实际抛ValueError | 修正actual和有界复现，保留分块契约 |

索引94的“iniconfig顶层没有iscommentline”怀疑经真实源码运行排除：这个归档版本确实导出该函数，原题面四个输入与实际结果一致，不应因其他版本印象改题。其余条目的范围或未执行细节在逐题表中保留，没有仅凭短补丁或共享snapshot判废。

## OSA实际语义与原oracle

在同一固定源码快照中，恢复发布者原测试资产，执行全部15个原目标文件、18个原required节点；未编写新测试文件或改变原断言。结果与原始XML/日志在`osa/`：

| 生产版本 | 原验收 | 距离(empty,hello) / 距离(hello,empty) / 距离(ab,ba) |
| --- | --- | --- |
| 原始缺方法 | 1失败、17通过 | NotImplementedError |
| 历史reviewed补丁 | 18通过 | 0 / 0 / 1 |
| 故意常数0的反例 | 18通过 | 0 / 0 / 0 |
| 修正空串边界的生产候选 | 18通过 | 5 / 5 / 1 |

后三者的原`_validate_junit_xml`都接受全部required身份。OSA测试体只打印结果，没有数值断言；其他P2P属于不同算法。常数0是专门证明oracle覆盖不足的错误实现，**绝不进入训练数据**，没有保存为可用修复或奖励样本。

正确候选保存生产patch，并实际核对可应用到原snapshot；它修正空串边界，使DP边界与非空算法一致。随后已用新审阅recipe重新执行真实官方六工具：修前完整官方验收1失败17通过，修后18通过，源绑定数值观察真实得到`[5.0,5.0,1]`；过程、prompt、工具反馈和终止回答见`osa/corrected-six-tools-r2/`，总墙钟约6.58秒。首次运行因解释器路径不存在而在工具开始前拒绝，原因另行保留。

这是一条新的Codex off-policy示范，不是模型生成或模型修复收益。历史reviewed原件及原奖励不改；新示范不能使该题的原RL oracle自动获得充分语义覆盖。后续分别处理冷启动增量与RL弱oracle资格，不能靠改测试追认其可用。

全部运行是本机CPU和复用依赖cache，不代表比赛Docker/Kaggle环境。本轮没有GPU、模型采样或optimizer步骤。
