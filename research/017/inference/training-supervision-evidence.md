# P017：537题主训练的结束与验证监督证据

2026-10-24。只读取P016既有审计、冻结清单、逐题转换元数据和生产代码；没有重新打开537题教师messages、解码训练tokens、运行模型/测试或读取heldout。D/C未作为训练来源。本说明区分可由全量元数据证明的结构事实与只有局部覆盖的行为统计，不据D症状推断训练因果。

**主训练普通final缺席现在有独立的全量结构证据，不需要从6题适配种子外推；主训练验证/恢复覆盖率仍没有全量证据。** 这两个结论应分开。

## 数据身份：确实是最终训练的537题

恢复的[training_manifest.json](../artifacts/main-sft-short-v1/training_manifest.json)记录`rows=537`、数据SHA256 `b95e0bc93dca3f66fdbf00ed5f8663cc50e7b31e7a3f84e4d9ca74c5084e50a6`、监督token 1,291,819，与P016[短语料冻结清单](../../016/data/main-sft-short-v1/manifest.json)及[训练完成摘要](../../016/training/main-short-training/final-result-summary.json)一致。冻结清单中准备期的`gpu_training_executed=false`不代表后来的训练未执行。

按该清单列出的三个分片manifest，筛选`native_sft_admitted=true`且`bucket=le16k`，得到下表；本次重新聚合的是元数据，不是轨迹正文。三个manifest字节SHA均与冻结清单一致，537个instance_id唯一；输入/监督token相加为7,119,739/1,291,819，完全对应最终训练。

| 已实际纳入分片 | 独立题 | assistant工具动作 | 有被mask动作的题 | 被mask动作 |
|---|---:|---:|---:|---:|
| pilot-admitted-v2 | 21 | 305 | 0 | 0 |
| next1000-admitted-v2 | 203 | 3,244 | 15 | 15 |
| next-remaining-admitted-v2 | 313 | 4,875 | 17 | 21 |
| 合计 | **537** | **8,424** | **32** | **36** |

对应manifest分别为[pilot](../../016/data/encoded-native/pilot-admitted-v2/swe-smith-native-manifest.json)、[next1000](../../016/data/encoded-native/next1000-admitted-v2/swe-smith-native-manifest.json)、[next-remaining](../../016/data/encoded-native/next-remaining-admitted-v2/swe-smith-native-manifest.json)。不能用旧677题/≤32K统计替代本表。

## 全537题可证明的结束结构

| 问题维度 | 可确认结果 | 依据与限制 |
|---|---:|---|
| 含受监督的末尾来源`submit` | 537/537 | producer强制最后完整调用为无参数submit，且不得mask末动作 |
| 末尾submit没有来源工具回包 | 537/537 | 每题已有`terminal_tool_response_absent=true` |
| 含独立普通assistant final监督 | 0/537 | producer只接受单工具assistant并输出空content；无工具assistant必须解析成XML工具调用，否则整条转换失败 |
| 含“末submit成功回包→普通final”的训练序列 | 0/537 | 所有序列在末submit调用处结束，既未补回执，也未补普通final |
| 含至少一个被屏蔽编辑失败动作的历史 | 32/537 | 共36动作，全部mask理由为`observed editor operation error`；不等于已确认32条有效恢复链 |

以上普通final的0是**哈希匹配的生产规则与全量转换元数据共同支持的结构结论**，不是本次重新逐token数出的独立测量。三个分片保存的`producer-source.py`与当前[build_smith_native.py](../../016/encoding/build_smith_native.py)逐字同SHA：`82df32ab32275398d25584bee465d8a621e05493e63397cdd9c6d718be96447c`。关键代码为175—203行（所有assistant转成单工具、`content=''`）、224—238行（末submit及目标选择约束）；已有[native-sft-v2.md](../../016/encoding/native-sft-v2.md)也明确记录了末submit无回包的边界。

“普通final没有受监督”不等于“没有任何结束语义”：末submit本身受监督，其前置thought可能含完成说明。也不等于“模型不会停止”：基座能力仍存在，SFT数据覆盖与实际行为之间需要实证连接。这里的无回包专指**末尾submit**；现有元数据没有逐类统计任何可能的中途submit及其回包，不能把537个末尾事实改写成“全语料任何submit都无回包”。

## mask策略：主训练与6题种子不同

主训练是`thought_loss=include`。producer保留完整真实历史，选择除mask动作之外的全部assistant动作：8,424−36=8,388个动作包含在正向目标集合；36个明确编辑失败动作及其thought不受监督，但原消息和反馈仍留在上下文。32题中29题各mask 1次、2题各2次、1题3次，其余505题没有该类mask。537题中未记录`source turn_mask=true`触发的mask。

producer的213—220行仅识别明确编辑操作错误，如找不到替换串、路径或范围无效、多处匹配、文件已存在。它**不会因为bash返回非零就屏蔽命令**：故障复现、测试失败本身可能是有价值的动作。它也没有以“执行成功”筛选所有正向动作，不能把8,388称为8,388次已验证正确行为。

[gemma_tool_chat.py](../../../training/encoding/gemma_tool_chat.py)的291—304、404—449行说明：只在所选动作上监督thought正文及工具调用；system/user、工具观察、role前缀及未选动作标签保留`-100`。工具调用监督其实际`<tool_call|>`边界，不在每次调用后强加普通EOS；普通final的`<turn|>`规则虽然由通用编码器支持，但本轮producer没有产生这种消息。当前编码器SHA `bb8bc163782efe0a015112e5b493add1ea866696dbcbc9587c4d67d21e8be122`与[冻结v5构建清单](../../016/training/generated/smith-pilot-calibration-v5/build-manifest.json)记录一致。

与此不同，[6题目标协议种子](../../016/target-protocol/README.md)是精选成功动作，70个目标、6,230监督tokens，只有工具调用受监督，reasoning与历史mask；已有统计为5题有提交监督、0可靠普通final、0可确认的上游测试动作、9个带恢复标记的动作。该9是动作数，不是9题；这些数不能替代537题的主训练分布。

## 验证与恢复：当前只有局部统计，不能填成全量比例

既有[source-profile.json](../../016/data-preparation/source-profile.json)只覆盖首批160个来源候选，其中Smith为64题；“19/64含明确测试runner”不是本轮537题的比例。按instance_id与来源SHA和本轮交叉核对后，profile仅覆盖实际入训的21个pilot问题；其中：

- 6题含明确测试runner调用，共10次；分布为3题各1次、2题各2次、1题3次，其余15题没有这种语法匹配。
- 这21题的producer元数据没有被mask动作，因此这些已识别调用按本轮选择策略属于监督目标；仍只是命令语法覆盖，不能声称相应检查通过、覆盖所有需求或构成错误恢复。
- profile定义只识别实际pytest/unittest/tox等命令，排除普通Python复现脚本，也不把工具输出中的“pytest”算调用。所以未命中不等于没有任何验证；6/21不能外推为6/537的完整验证覆盖统计，更不能把另外516题填0。

三个主分片manifest没有记录“修后验证”“验证失败→再次生产编辑→复查”“按反馈成功恢复”“在预算内结束”等全任务行为计数。32题/36次编辑错误mask只证明真实错误历史存在；它不记录后续是否针对同一错误恢复、验证是否有效。全部537题都有上游`resolved=true`、`local_replay=false`，但上游成功标签也不能代替这些过程标注。

因此现阶段只能准确报告：主训练有537个来源submit终点、没有目标运行时的提交后普通final序列；已有局部证据证明并非完全没有测试runner监督；全537题的验证/恢复链覆盖未知。无需为得到一个完整百分比临时把语法扫描冒充语义审计，也不应由6题种子的0验证推广到全体主训练。

## 对D诊断的含义

D中成功提交后的循环与主训练终局格式差异具有研究相关性，但目前只是**可检查的迁移假设**，不是已确定原因。主训练语料只教到来源submit，比赛契约允许submit回包后继续调用并需要可靠结束；这个差异值得在独立训练来源中补充和对照。与此同时，错误反馈利用、重复操作和业务验证缺口不能全归给final缺席，原生来源数据的这些覆盖仍未作全量语义统计。不得将D/C题、答案或轨迹回流训练。

本次读取的重要元数据SHA：短语料manifest `068514ff4557adc1de46d2c3e97ac391bbcb6976570b34998e0b54ef55b1a38f`；最终training_manifest `9596ed34d626174c5fc6dc748a1d76770dc67d59ca802122de02c1a21ae547e8`；局部source-profile `179799b8c526148646c2da08cfa6e32e85a0d37e4d64db3737c5bbccce96c598`。逐题正文未重新读取，训练标签未改动。

## 独立CPU权重诊断：Step30与Step68

2026-10-24 15:18北京时间的一次性只读核对，复用本地NumPy 2.4.6，使用memmap读取safetensors真实F32载荷，以float64累计平方和。没有加载基座、调用模型、安装依赖、使用网络或GPU，也没有写测试文件；上方训练数据统计不变。

两个文件均212,261,592 bytes，本次完整重算SHA与原恢复证据一致：Step30 `61e6b2b03a477ce8d0143925fd346e0c8707213cee726415c04a52dedf9133ae`，Step68 `123a5a622decb6413a0cf9c4cf4c197604d7eba00cb32c3ad27a80eef979a701`。来源：[Step30恢复回执](../artifacts/step30/recovery-receipt.json)、[最终恢复回执](../artifacts/main-sft-short-v1/recovery-receipt.json)。两份config相同，r=8、alpha=16。

**两步各600个张量、53,043,200个参数全部有限，NaN/Inf计数均0；300/300个A矩阵和300/300个B矩阵都不是全零。Step30→68的600/600个张量有实际数值变化，共53,043,041个参数值改变。** 因此这不是仅修改metadata或导出同一份数值权重。

下表为每类模块跨60层拼合后的Frobenius范数；Δ表示Step68−Step30的参数矩阵差。A与B分列，避免A的初始化尺度掩盖B的变化。

| 模块族 | A30 | A68 | ΔA | B30 | B68 | ΔB |
|---|---:|---:|---:|---:|---:|---:|
| mlp.down_proj | 12.693062 | 12.718439 | 0.340307 | 0.425857 | 0.521479 | 0.166826 |
| mlp.gate_proj | 12.668386 | 12.678382 | 0.181011 | 0.750123 | 0.917365 | 0.271330 |
| mlp.up_proj | 12.675303 | 12.684809 | 0.177654 | 0.784431 | 0.956283 | 0.284246 |
| self_attn.o_proj | 12.671632 | 12.684779 | 0.231876 | 0.416372 | 0.510750 | 0.160828 |
| self_attn.q_proj | 12.664371 | 12.672120 | 0.170771 | 0.472131 | 0.581864 | 0.170502 |
| 全部300模块 | 28.341166 | 28.370595 | 0.512897 | 1.325007 | 1.620928 | 0.487015 |

还用rank8的Gram矩阵计算实际低秩权重项`ΔW=(alpha/r)BA=2BA`的范数，没有构造或加载完整基座矩阵。300/300模块在两步的ΔW均非零，而且300/300模块的ΔW从30到68均有变化；合并Frobenius范数分别为 **1.542785 → 1.897262**，两者差的范数为 **0.579211**。使用恒等式`||BA||²_F=tr((BᵀB)(AAᵀ))`及交叉内积计算，避免把仅因A/B重参数化造成的变化当作有效权重项变化。

该诊断证明有限、非零的adapter数值更新及后半程真实权重变化；**不能从范数增大推断修复能力提高、从变化大小推断过拟合，或据此选择Step30优于Step68**。没有计算与NF4/CT基座的相对范数，也没有新增Step30推理/评测；收益仍以同环境同期任务结果判断。
