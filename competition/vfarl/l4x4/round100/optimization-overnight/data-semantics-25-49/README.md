# 固定100题语义审阅：索引25–49

2026-10-28。仅审阅v4原tasks数组[25:50]的25个train任务；没有改名单、题面、旧示范、oracle或奖励，没有GPU/远端运行，也没有读取预留4题或heldout。

本表逐题对照当前task_input、实际submitted.patch、source-snapshot.tgz中的目标函数以及原FAIL_TO_PASS函数/动态fixture。来源节选及哈希见[source-evidence.json](source-evidence.json)，有界CPU手动观察见[manual-observations.json](manual-observations.json)，结构化结论见[review.json](review.json)。未重跑完整官方oracle；部分glom全包因本机缺face未导入，相关结论明确限于原类AST及实际断言审阅。

重点问题为32（SQL dialect及tuple API）、39（HTML/Markdown renderer作用域）、44（+01时区却要求+04）。另外25/26/41的复现API、29的无界retry示例、31/46的具体expected或实际值有问题。没有确认参考submitted.patch本身语义错误。最初怀疑的Pypika Query.Table已实测有效，已排除。

| 原索引 | 任务 | 判定 | 机制、冲突与证据边界 |
| --- | --- | --- | --- |
| 25 | `jd__tenacity.0d40e76f.func_basic__pvngsy7s` | confirmed_repro_api_error | wait_combine去除各分量的×2，4个F2P检查sum/+组合的[5,8)或[6,9)范围，机制与参考补丁一致。 题面RetryCallState(outcome=...,next_action=...)不符合归档签名；seconds_since_start是只读property。两个错误在参考补丁后仍出现，不能照抄其复现。 没有确认参考修复错误；建议仅在新题面版本修复构造方式，不改旧reward。 |
| 26 | `jd__tenacity.0d40e76f.func_basic__rlm5c89z` | confirmed_repro_api_error | wait_fixed返回配置seconds而非0.1倍；13个F2P含数字/timedelta、链和实际sleep/log语义。 题面RetryCallState(1,0)缺少args/kwargs，原样复现先抛TypeError而非打印0.5。 参考修复与oracle一致；旧题面保留，未来版本修正state构造。 |
| 27 | `jd__tenacity.0d40e76f.func_basic__ttznt9oc` | aligned_with_coverage_limit | 结果predicate应取反，异常outcome不应由result-predicate决定retry；参考补丁两分支均恢复。 未发现题面与主要结果极性冲突；唯一F2P仅检查成功outcome(result=2/1)，未直接覆盖参考补丁同时修复的exception分支。 可继续研究；把exception分支覆盖不足作为机制证据边界，不能把F2P通过等同完整语义恢复。 |
| 28 | `jd__tenacity.0d40e76f.func_basic__wcqpfsdn` | aligned_with_coverage_limit | stop_after_delay使用>=，seconds_since_start未设置时抛RuntimeError；F2P精确检查0.999/1/1.001及timedelta。 题面循环在达到时限后转成功，不是精确边界的稳定复现；唯一F2P没有直接检查None状态的RuntimeError分支。 主边界要求一致；未设置状态分支不能仅凭此F2P宣称已学会。 |
| 29 | `jd__tenacity.0d40e76f.func_basic__x6b7414f` | confirmed_unbounded_reproduction | regex使用prog.match的正极性；参考补丁not bool→bool，F2P用可终止的NoCustomErrorAfterCount(3)约束。 题面两个always-raise函数都没有stop条件；默认stop_never，retry的一侧不会到达所写RetryError打印分支，可能一直忙重试。 机制和参考修复一致；未来题面版本应加入有限stop_after_attempt，不修改旧oracle或奖励。 |
| 30 | `jd__tenacity.0d40e76f.pr_406` | aligned | functools.wraps额外复制__defaults__/__kwdefaults__；题面和F2P均比较两类函数的实际默认值。 未发现expected/actual/API或作用域冲突。 未执行新完整suite；目前参考修复语义一致。 |
| 31 | `jd__tenacity.0d40e76f.pr_425` | confirmed_numeric_reproduction_conflict | uniform下界由0改为self.min；super wait_exponential先clamp到min。F2P的min=5首轮要求正好5。 题面min=5,multiplier=1却声称old范围0..1、expected 5..1；实际high已clamp至5，old应0..5，修后5..5。snippet里的retry_state也未定义。 核心min要求正确，参考补丁正确；具体复现数字冲突应单列，不把wrong range当训练目标。 |
| 32 | `kayak__pypika.1c9646f0.func_basic__4zv0lzne` | confirmed_expected_output_and_api_conflict | 列表→Array、tuple→Tuple，并保留传入wrapper_cls；F2P同时覆盖普通/MySQL方括号、PostgreSQL ARRAY与SQLite bool wrapper。 题面普通Query.insert期望ARRAY[...]，而oracle和参考补丁的普通Query为[...]；纯Python tuple==tuple得到bool，不是SQL criterion，where调用报bool.fields_。Query.Table在此版本存在，已排除这个最初候选。 参考补丁符合oracle；题面应在独立新版本指定PostgreSQLQuery，并用pypika.Tuple构造比较，当前不改。 |
| 33 | `kayak__pypika.1c9646f0.func_basic__b54t4fio` | aligned | DistinctOptionFunction.distinct置True；Count和Sum的4个F2P明确要求DISTINCT SQL。 未发现冲突；Query.Table在归档中是有效API，不应误报。COUNT(DISTINCT *)是此库选定oracle契约，未将数据库执行合法性另行推断成数据错误。 参考修复与库级SQL渲染目标一致，未执行数据库。 |
| 34 | `kayak__pypika.1c9646f0.func_basic__bsd9kp1c` | aligned_with_coverage_limit | EmptyCriterion.__invert__返回self，F2P两种组合均消去空条件得到同一field SQL。 题面与实现一致；oracle检查组合结果而非Python对象is身份，因此不应把它解读为验证了所有对象身份性质。 未确认坏示范；主组合语义一致。 |
| 35 | `kayak__pypika.1c9646f0.func_basic__wqdn07ef` | aligned | Term.eq委托==，不是!=；23个F2P覆盖bool/date/datetime/decimal/number/string以及不同SQL位置。 未发现冲突；继承的PreWhereTests断言已解析到父类实际函数，没有把静态查找失败当oracle缺失。 参考补丁语义与oracle一致。 |
| 36 | `kayak__pypika.1c9646f0.func_basic__x1i7ttk7` | minor_scope_overstatement | _rollup_sql输出WITH ROLLUP；2个F2P针对vendor=mysql的单/双groupby，P2P仍覆盖Vertica/Oracle式ROLLUP表达式。 题面末句称不论vendor均受影响；被变更函数属于MySQL WITH后缀路径，不能推广为所有vendor的rollup语义都坏。给出的mysql复现与修复一致。 小范围描述泛化，不足以判定参考修复错误。 |
| 37 | `kayak__pypika.1c9646f0.func_basic__xv03w5cm` | aligned | QueryBuilder.distinct置True；MSSQL含TOP次序、MySQL和普通单/多列SQL均由F2P覆盖。 未发现expected/actual/API冲突。 参考修复符合oracle。 |
| 38 | `kayak__pypika.1c9646f0.func_basic__yjy0eoaz` | aligned | EmptyCriterion左AND返回other，保留真实condition；8个F2P覆盖空组合、Criterion.all、generator及聚合filter。 未发现主要冲突；题面打印表达式不是identity assert，未据其措辞额外推断缺陷。 参考修复符合oracle。 |
| 39 | `lepture__mistune.bf54ef67.pr_337` | confirmed_renderer_scope_mismatch | 目标_list._render_list_item空splitlines时需要安全首行；唯一F2P是动态生成MarkdownRenderer的list_006 fixture。 题面mistune.markdown默认HTML renderer，baseline原样输入已经成功；实际故障和补丁针对MarkdownRenderer，题面未指定。 参考修复正确，题面作用域错误；不应根据默认HTML复现成功否定任务本身或篡改奖励。 |
| 40 | `kayak__pypika.1c9646f0.func_basic__w1s47eoh` | aligned_with_coverage_limit | for_portion检查已有_for_portion/_for，赋值到_for_portion；F2P测试DELETE/UPDATE及SYSTEM_TIME/自定义period。 给出的正常调用与oracle一致；重复设置和for_互斥guard未被本题两条F2P直接逐一断言。 参考修复符合期望；保留guard覆盖边界，不宣称全面证明状态机。 |
| 41 | `msiemens__tinydb.10644a0e.func_basic__qel5fui7` | confirmed_repro_api_error | TinyDB.close设置_opened=False避免__exit__再close；F2P的storage参数传Storage类。 题面传TestStorage()实例，但TinyDB.__init__执行storage(*args,**kwargs)，会先报对象不可调用，达不到双close复现。 参考修复正确；未来题面需用storage=TestStorage，当前不改旧题。 |
| 42 | `kayak__pypika.1c9646f0.func_basic__x7p5x68a` | confirmed_repro_timing_and_copy_coverage_limit | MySQL __copy__应返回newone并隔离_duplicate_updates list；参考补丁恢复两者。 原题query构造已在from_时因None报错，不会如文字所述先构造成功再print copied_query=None。44个F2P主要SQL/handler行为，未直接断言MySQL派生query的list所有权。 参考示范没有确认错误；copy ownership是需单列的弱覆盖机制，不可只以返回对象/SQL恢复宣称全部修复。 |
| 43 | `bottlepy__bottle.a8dfef30.func_basic__9aw6dtvi` | aligned | BaseRequest.body seek(1)→seek(0)，保持同一可重读owned stream；14个F2P含read/readline/size、大body和解析下游。 未发现API或expected冲突；文字说first read fine不精确，seek(1)也使首次访问漏首字节，但题面核心要求从头重读与oracle一致。 参考修复符合流所有权语义。 |
| 44 | `gweis__isodate.17cb25eb.func_basic__ydtogylj` | confirmed_expected_output_conflict | FixedOffset.utcoffset返回正向offset，包含dt=None；F2P参数表涵盖正负整小时及带分钟的time/datetime格式。 题面以FixedOffset(+1h)创建datetime，却写expected +04；参考补丁真实输出+01，符合所设时区和oracle的各自offset数据。 核心符号修复和参考补丁正确；具体expected数值错误，不应把+04硬编码成目标。 |
| 45 | `mahmoud__glom.fb3c4e76.func_basic__0xzrqz4z` | aligned | M(spec)仅接受T式spec并保留实际spec；F2P验证invalid TypeError、repr、target比较和scope。 未发现冲突；题面含invalid与valid两例，覆盖两处mutant（吞验证/传None），与参考补丁一致。 静态语义一致，未执行完整新oracle。 |
| 46 | `mahmoud__glom.fb3c4e76.func_basic__46drzyho` | confirmed_exact_error_message_conflict | Switch空cases应ValueError；dict先转换成list，再构造消息。oracle只with raises(ValueError)。 题面指定got: {}，参考补丁实际got: []；题面称默认case为None，实际未传default时为_MISSING sentinel。主异常类型要求一致。 参考修复符合oracle类型但与题面精确字符串不一致；未修改旧题或奖励。 |
| 47 | `mahmoud__glom.fb3c4e76.func_basic__a6mxeq22` | aligned | Pipe存入原steps tuple而非reversed list，保留左至右及repr；F2P含直接INT例和streaming filter/all。 未发现冲突。 静态语义一致；未加载缺依赖glom全包重新跑。 |
| 48 | `mahmoud__glom.fb3c4e76.func_basic__pkf875zo` | aligned | Val.glomit应原样返回self.value，None不能映射0；F2P递归Ref JSON实际期待[None,1]。 未发现冲突；题面具体递归表达式与oracle结构一致。 静态语义一致；未执行完整新oracle。 |
| 49 | `mahmoud__glom.fb3c4e76.func_basic__po5i2tm3` | minor_description_and_tie_coverage_limit | Min.agg需返回tree中的累计min；参考还把<=改<以不替换相等已有最小对象。F2P含[1,0,1]→0。 标题称duplicate minimum，但示例只有一个0；实际主要错误是每步return target导致最后非最小值覆盖输出。F2P数字例不证明相等对象的稳定身份。 主修复和oracle一致，描述过窄，tie身份仍是覆盖边界。 |

所有需改题面的建议仅作为后续新版本选择，未覆盖原始publisher文字或伪造旧运行证据。弱覆盖案例不被记作已证实可作弊拿分；尤其MySQL只修return仍有list共享的实际反例，但本次没有宣称它通过全部F2P/P2P。
