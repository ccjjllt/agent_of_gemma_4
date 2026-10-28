# R3 Glom49 early termination: real CPU recovery opportunity

2026-10-28。固定100训练索引49，`mahmoud__glom.fb3c4e76.func_basic__po5i2tm3`。[最终结果](result.json)使用**旧v4题面**（task SHA `023e91bb…`），没有使用当前修订题面`fc33199a…`，也没有改旧R3结果或当前SFT。

R3真实首动作有1294个prompt IDs和59个completion IDs，raw为thought说明后接裸`<channel|><tool_call|>`，没有opening tool call；保存的finish_reason仍是tool_calls。真实pinned GemmaTokenizer、原tools/template和旧实际prompt逐ID重建完全相同，completion解码也完全等于旧raw；当前production parse_call返回None。[输入绑定](prefix-and-input-binding.json)保存旧task、R3原文件/行、prompt、worker execution、µ/policy identity和各SHA。旧action/probabilities从未重评分、改名或改写。

两条最终真实CPU episode均从新本机TaskEnvironment开始，明确以`first_action`重放该旧动作，新trace标记`original_action_replay=True`。后续只提供已审recipe的脚本化工具suffix，绝不伪装新Gemma输出：

| 当前production协议 | 实际动作/工具/nudge | 完整原178项oracle |
|---|---|---|
| historical-three-branch对照 | 1个forced动作，0工具，0nudge；suffix未被请求 | 1fail /177pass，reward0 |
| live-original-pair，raw-preserved | 同一forced动作后真实nudge1，再执行7个官方工具和terminal，共9动作 | 178pass，reward1 |

live首个suffix请求的prefix完整保留原1294+59 IDs，然后由真实episode加入33个nudge tokens，成为1386 tokens。实际工具先观察running minima `[3,1,2]`，修复后得到`[3,1,1]`；最终提交patch逐字等于旧正确teacher。没有修改原测试、奖励、required IDs或source；完整原JUnit/official verification、原始actions、tool events及harness feedback都在[最终r2 trace](actual-episodes-r2/)中。

预算严格来自R3 worker execution：context32768、32 turns、32 tool calls、response1536、episode output8192、thought256；time_minutes为None，没有伪造旧host elapsed time或手动回填计数。新context的tool/LLM counters由真实工具和episode推进。67个原source文件bytes/modes精确匹配旧inventory；原v4 snapshot与旧teacher snapshot逐字同SHA `98cbab62…`。原snapshot额外的官方conftest保留原bytes/mode，pytest.ini本就在原67文件内。

本机复用已成功Glom CPU依赖profile，并通过现有TaskEnvironment恢复v4原有`source_console_scripts={'glom':'glom.cli:console_main'}`（原setup.py同声明）。[最终runtime](runtime-used-r2.json)明确记录本机CP312/pytest8.3.3等；不声称恢复了R3 Kaggle/chroot runtime、内核随机流或完整Snapshot。

失败原件完整保留。首次准备在把pre-setup67-file inventory误比成整个post-setup树时停止，尚无episode/action/oracle；[说明](preexecution-inventory-note.json)记录了这个差异。[实际r1](actual-episodes-r1/result.json)已经证明nudge后工具可继续且正确Min patch生效，但我漏传原glom CLI入口，导致最终test_main找不到glom，原工具返回0→0。这个已知重建runtime失败**不是有效模型/repair失败或可训练reward0**。随后只补回原已声明入口，运行最终一对historical/live对照；没有额外重复baseline或继续补环境，没有改任何旧结果。

capture=False，没有Snapshot/focal pair/learner/optimizer或canonical/SFT导出。所有新facade动作mask0、reference/behavior=None、training_eligible=False；旧动作原mask/probabilities只作为forced历史输入保留。旧动作中的GPU memory和34.395秒等字段是历史原件，绝非此次GPU使用或计时；r1 facade的seconds0只是非测量占位值，最终r2记录实际脚本编码wall time并明确不是模型generation时间。整个研究没有GPU或新模型推理。

这是一项controller恢复机会证据：这个真实旧sample形状在当前live反馈后，可以继续执行已知正确工具流程。它不估计模型会否生成该suffix、修复率或训练收益，不能给当前修订题面重贴reward，也不是同完整state的采样对照。
