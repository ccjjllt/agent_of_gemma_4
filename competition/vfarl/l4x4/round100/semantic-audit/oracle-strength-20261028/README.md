# 固定100原oracle的本轮候选审查

**后续当前结论：** 在相同100个train ID的question-revision-r3上，已再确认[Bag82频次缺口](bag82/README.md)及[Marshmallow88 bytes解码缺口](../075-099/marshmallow-bytes-oracle-r1/README.md)。连同原84／85／95，最新[课程r6](../../question-revision-r3/curriculum-r6.json)暂停5个弱oracle题的未来RL，正确SFT与旧奖励保留；待采60题为32共同＋28专科，另4个runtime unknown仍暂停。下文的“本轮新增两项”“其余97”是初始筛查阶段，不能当作当前弱项总数。

Bag82自然set去重修复通过原完整29文件／427要求节点展开430cases，实际官方verify_task结果baseline0、correct1、wrong1，但aab／ab应1却为0、aaab／ab应2却为0。原teacher的abb／abc=1不足以区分该错误，正在单独补频次行为示范。此结论来自显式依赖预载CPU profile：只import缓存依赖、不调用目标，不改deadline／seed／测试；首次冷启动Hypothesis超时的失败控制保留，未当有效wrong结果。新profile额外启动成本约8–9秒／进程，不冒充原profile或Docker完全一致；[独立来源复核](bag82/preload-r1/independent-review.md)核对全部原target及节点。

88自然`str(bytes)[2:-1]`修复通过原完整12文件／1231cases及官方JUnit validator，却仍保留换行、引号和普通UTF8的repr转义，违背proper text主契约。原source/mode相同的baseline1失败、correct/wrong全过均已保留；[独立复核](../075-099/marshmallow-bytes-oracle-r1/independent-review.json)从原snapshot＋test_patch重建12target字节。这里是direct pytest＋actual validator，不是新verify_task／模型／比赛奖励。现有teacher已验证中文bytes及数字，能排除该错误，不需替换正确SFT。

候选也有明确关闭：[Bottle5 no-op reset](../000-024/bottle-reset-oracle-audit-r1/README.md)被原DEBUG模式预备callback及后续插件响应断言抓住，365scope为baseline0、wrong0、correct1；没有因此扩大暂停范围。诊断位置93、注释94、单词边界75也被原明确值断言覆盖，见[源码级范围](diagnostic-position-candidate-review.json)，不冒充全量真实oracle实跑。

---

2026-10-28。范围仅当前question-revision-r2固定100个train任务，未读heldout。先读原source-task的F2P身份与固定adapted test_patch中的真实原测试函数，再追helper／异常／fixture／动态生成契约；缺assert仅用于定位候选，不直接判弱。

本轮新增两项题目主契约内的真实反例：

- [84 SVG输出](qrcode84/README.md)：原8个完整目标文件／81要求节点／81实际case。原mutant为6失败，返回可序列化not-svg根的明显错误实现仍81通过；源绑定保存／解析真实XML证实它不是SVG，正确reference也81通过。super-only的namespace／白底遗漏单列相邻源契约边界，不单靠该项判弱。
- [95 ParseError行号](iniconfig95/README.md)：原1个完整目标文件／28要求节点，参数展开42实际case。原mutant为1失败；保留path/msg但显示行号恒1的明显错误实现42通过；真实lineno7/13分别应显示8/14而实际1。正确reference也42通过。首轮缺pytest自带py.py导致3态0 JUnit，保留为环境失败，未作为奖励0或弱oracle证据。

原OSA85已有print-only实证，本轮未重新制造错误实现。84／95的错误patch只放在/private/tmp，研究结果明确禁止训练消费；没有teacher、SFT或canonical导出，未改题面、原oracle、奖励、旧课程或历史结果。

其余候选跟进中，furl的_set/load helper、Tenacity数值范围与日志helper、Bottle chunked-body helper及iniconfig错误helper都发现真实值／异常契约。距离81／82有多输入exact expected；Markdown表格54／55有多输入完整输出断言。DVC一个walk smoke与pyparsing一个example入口仍只是局部smoke现象，没有证明其完整任务oracle可被明显错语义通过，不建议因此暂停。

本轮没有其他实证确认的弱oracle项。结构筛查覆盖929个adapted F2P节点，789个可直接在完整新增文件AST定位，140个涉及partial PR patch或继承／动态生成，未静默标为强。原source通常不含test_patch，实际测试函数来源与每题源／task／test_patch hash保存在[oracle-screen-and-followups.json](oracle-screen-and-followups.json)。这是一轮有范围的漏洞查找，不是对其余97题可靠性的证明。

建议root将84、95与既有85一起暂停未来基于原oracle的RL奖励消费，保留正确SFT及真实历史结果；本代理没有更改eligibility。没有GPU、训练、网络下载、新测试文件或错误示范入库。
