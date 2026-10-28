# 固定100题索引25–49：reference边界审阅

2026-10-28。索引按 v4/selection.json 原tasks顺序，未读取heldout或预留评测正文。每项读取旧题与现r2题、真实source-snapshot及submitted.patch。一次性CPU只在临时生产源码应用该项原patch；没有新增专用测试文件、修改原oracle、奖励、历史数据或训练代码。

除42的独立三态外，24项有145个初次有界观察，40／46另各6个纠正API后的观察。真实生产模块路径及内容SHA逐项核对；旧／新task SHA、snapshot／reference patch SHA、完整内联命令、stdout／stderr以及初次失败均在 [reference-boundary-observations.json](reference-boundary-observations.json)。glom使用已有缓存的45个face／boltons纯Python源文件，没有下载；原归档成员和SHA见 [glom-dependency-source-manifest.json](glom-dependency-source-manifest.json)。这些边界观察不是100题CPU验收，也不是模型修复收益或完备正确性证明。

唯一确认的新增生产改进是42的modifier-list复制隔离，见 [mysql-copy/README.md](mysql-copy/README.md)。它恢复同public copy契约的相邻边界，原mutation／oracle与历史reference的有效性保持；没有把它当OSA型评分失效，也没有暂停RL。新teacher、canonical和SFT分别绑定当前r2 task／prompt，未重写原示范。

| 原索引 | 审阅边界 | 观察项 | 已建立结论／限制 |
| --- | --- | --- | --- |
| 25 | wait_combine：每项调用一次，空组合、timedelta、sum／+ | 4 | 返回分量和，空组合0；未见新的契约缺口 |
| 26 | wait_fixed：0、小数、timedelta微秒、负数配置 | 6 | 按配置秒数返回；未扩写负等待策略要求 |
| 27 | retry_if_not_result：相等／不等／None／False、异常、未设置outcome | 7 | predicate取反；异常不触发该结果条件；缺outcome报错 |
| 28 | stop_after_delay：阈值前／相等／后、0、timedelta、未设置时间 | 15 | 精确相等触发stop；未设置时间报错 |
| 29 | exception regex／literal：前缀、后缀、不匹配、成功结果、互斥参数 | 12 | regex match与literal exact各守本义；未见反转残留 |
| 30 | Retrying.wraps：positional／keyword defaults、signature、实际调用 | 2 | 默认值、wrapped身份和实际行为均保留 |
| 31 | random exponential：min=5首／后期、max0、小数min、矛盾／负配置 | 18 | 正常配置区间守min；min<0、min>max仅记录，不宣称已定义无歧义契约 |
| 32 | wrap_constant：空／单／嵌套list，空／单／双tuple，tuple IN，自定义wrapper | 8 | Array／Tuple分类与标量wrapper正确；未执行SQL数据库语义评测 |
| 33 | DistinctOptionFunction：copy隔离、COUNT／SUM DISTINCT | 2 | 原表达式保持无DISTINCT，副本有DISTINCT；COUNT(DISTINCT *)按题面记录，不发明SQL合法性结论 |
| 34 | EmptyCriterion invert：identity、复合逻辑 | 1 | 反转返回同一空criterion，复合SQL保持实际条件 |
| 35 | Term.eq：字符串、0、False、None、tuple及Tuple.eq | 6 | 与同库==逐字一致；未扩写NULL SQL语义要求 |
| 36 | rollup：mysql单／多项、默认vendor | 3 | mysql WITH ROLLUP；默认GROUP BY ROLLUP(...) |
| 37 | Query.distinct：基础／MySQL／MSSQL，旧query不变与重复调用 | 3 | 三个dialect均有DISTINCT且原query不受影响 |
| 38 | EmptyCriterion左AND：identity和两边空criterion | 1 | 保留实际criterion；未见新的契约缺口 |
| 39 | MarkdownRenderer：首／尾／唯一空item、有序、嵌套、松列表 | 6 | 无IndexError；一个嵌套输入的parser输出未建立额外CommonMark义务，未列缺陷 |
| 40 | Table.for_portion：副本隔离、delete／update、重复／互斥for状态 | 6（补跑） | 实际pypika.SYSTEM_TIME路径正确；首次照原题从terms导入失败保留，非生产patch缺陷 |
| 41 | TinyDB.close：显式关闭+context exit、仅exit、两次显式close | 3 | 题目要求只关闭一次；两个显式close仍两次，未强加通用幂等义务 |
| 42 | MySQL __copy__：baseline／reference／candidate，modifier分支与duplicates对照 | 独立三态 | 确认同copy契约相邻边界增强；已真实六工具+原1061节点0/1+新SFT；不属OSA型oracle失效 |
| 43 | BaseRequest.body：空／单字节／一般字节／多行，重复read与readline | 4 | 每次属性访问归零；tell的再次body访问也归零，符合源docstring |
| 44 | FixedOffset：0、正负半小时、-05:30、+05:45、±23:59、dt=None及UTC日期跨界 | 7 | offset与格式符号、UTC转换一致；未见新的契约缺口 |
| 45 | M：T本体／路径／表达式／S，invalid值和子spec比较 | 11 | 合法T类型保留，非法值TypeError；比较成功返回原target符合matching语义 |
| 46 | Switch：空dict／list含default，实际Match(int)匹配／不匹配 | 6（补跑） | 空集合都ValueError；合法Match分支与default正确。首次误用int转换保留 |
| 47 | Pipe：空、单None、left-to-right路径、class-name | 4 | 空Pipe保留target；顺序与None结果正确 |
| 48 | Val：None、0、False、空容器身份、递归json替换 | 7 | 精确保留值及对象identity，递归0→None正确 |
| 49 | Min：空、单值、升降序、重复minimum、字符串、嵌套分组、不同比较类型 | 9 | 非空minimum正确；空Group默认None来自明确初始化，未扩写min(empty)报错要求 |

没有为凑数量把未证实的直觉列为缺陷：负min／min>max、空SQL tuple、COUNT(DISTINCT *)的真实数据库合法性、一个嵌套Markdown的parser含义、双次显式close的一般幂等性、空Group是否应抛min(empty)异常，都没有足够契约证据升级为修复结论。对这些范围的观察已如实保留，后续若有新公开契约证据可单独推进。
