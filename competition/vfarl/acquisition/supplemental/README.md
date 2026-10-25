# 明确补充的20条训练来源

2026-10-25。原6条补充之外，本轮新增14条，共20条，来自10个canonical仓库；全部已通过真实官方六工具和完整修后验收，并进入当前准入清单。当前可用总数为 **81＝原96中61条＋补充20条**。原96的任务身份、原始来源和负结果保持不变，没有用补充任务静默替换原失败记录。

权威来源、完整commit、任务/审阅SHA、源码与数据集许可、原验证要求和实际证据见[catalog.json](catalog.json)。当前正例资格见[准入清单](../../replay/canonical-results-manifest.json)，对白与各adapter文件以[Gemma导出清单](../../replay/gemma-sft-manifest.json)和[课程清单](../../replay/adapter-curricula-manifest.json)为准；共享课程重复使用相同任务，不另算独立样本。

20条均来自 `SWE-bench/SWE-smith-py` 的明确 `train` split，固定revision为 `77cab9055d42ab4a5c25c89a8f937096db13558e`。它们的source patch为缺陷引入，不直接当作修复答案。每题另固定完整40位执行commit和上游snapshot，保存原task字节/哈希及独立源码、数据集许可证据。宽页train检索缓存只作筛选，未计为逐题深入审阅的正例。未摄入比赛四仓/129公共开发题或heldout正文。

## 前一批6条

| 主机制 | 来源仓库 / 精确任务ID | 实际修复契约 | 正例情况 |
| --- | --- | --- | --- |
| R5 | PyCQA/flake8<br>`PyCQA__flake8.cf1542ce.lm_rewrite__k4azrl8j` | 插件发现时去重重复发行版 | 完整验收通过；原题面 |
| R5 | PyCQA/flake8<br>`PyCQA__flake8.cf1542ce.func_pm_ctrl_invert_if__zlbk1yct` | 插件 entry point 的提供方分派 | 完整验收通过；原题面 |
| S1 | Instagram/MonkeyType<br>`Instagram__MonkeyType.70c3acf6.pr_148` | stub 注解字符串的 AST 表示；归S1 | 完整验收通过；原题面 |
| R1 | Instagram/MonkeyType<br>`Instagram__MonkeyType.70c3acf6.pr_164` | None 默认值引出的 Optional 导入闭包 | 完整验收通过；原题面 |
| R1 | Instagram/MonkeyType<br>`Instagram__MonkeyType.70c3acf6.func_pm_remove_wrapper__cz3cmljw` | 类型反序列化的模块/限定名查找 | 完整验收通过；原题面；显式基线差异profile |
| R2 | jd/tenacity<br>`jd__tenacity.0d40e76f.func_pm_ctrl_invert_if__ipm7o2gf` | 同步、native async、Tornado 调用协议分派 | 完整验收通过；原题面 |

## 本轮新增14条

| 主机制 | 来源仓库 / 精确任务ID | 实际修复契约 | 正例情况 |
| --- | --- | --- | --- |
| F3 | Suor/funcy<br>`Suor__funcy.207a7810.func_basic__i2ox965l` | 缓存载荷/截止时间解码与 TTL 重算 | 完整验收通过；原题面 |
| F3 | msiemens/tinydb<br>`msiemens__tinydb.10644a0e.func_basic__6qkvhk33` | 删除最后文档后仍须失效查询缓存 | 完整验收通过；curator题面v2 |
| F1 | Suor/funcy<br>`Suor__funcy.207a7810.func_basic__c4qlmrtn` | rpartial 尾置参数与调用时关键词优先级 | 完整验收通过；原题面 |
| F1 | Suor/funcy<br>`Suor__funcy.207a7810.func_basic__tirw2bw7` | wrapper→Call→consumer 的原参数捕获 | 完整验收通过；原题面 |
| R3 | mahmoud/boltons<br>`mahmoud__boltons.3bfcfdd0.func_basic__cgc05xp5` | descriptor 类访问与实例绑定 | 完整验收通过；原题面 |
| R3 | mahmoud/boltons<br>`mahmoud__boltons.3bfcfdd0.func_basic__z2i530nb` | IndexedSet 有序/跨 set 比较协议 | 完整验收通过；curator题面v2 |
| R3 | mewwts/addict<br>`mewwts__addict.75284f95.func_basic__nz3584mt` | 普通属性写入映射、类方法名称保护 | 完整验收通过；原题面 |
| F4 | pexpect/ptyprocess<br>`pexpect__ptyprocess.1067dbda.func_basic__fnw8atqy` | 阻塞等待并回收子进程退出状态 | 完整验收通过；原题面 |
| F4 | pexpect/ptyprocess<br>`pexpect__ptyprocess.1067dbda.func_basic__z0jqkuaw` | 非阻塞活性检查与终止/回收状态 | 完整验收通过；curator题面v2 |
| F4 | pudo/dataset<br>`pudo__dataset.5c2dc8d3.func_basic__roppugd3` | 事务退出时成功提交、异常回滚与清理 | 完整验收通过；curator题面v2 |
| R4 | marshmallow-code/apispec<br>`marshmallow-code__apispec.8b421526.pr_958` | OpenAPI 版本决定引用旁属性的表示契约 | 完整验收通过；原题面 |
| R4 | marshmallow-code/apispec<br>`marshmallow-code__apispec.8b421526.pr_904` | OpenAPI 版本决定 nullable 引用契约 | 完整验收通过；原题面 |
| R1 | Instagram/MonkeyType<br>`Instagram__MonkeyType.70c3acf6.pr_237` | bare/nested Callable 的泛型注解导入闭包 | 完整验收通过；原题面 |
| R1 | Instagram/MonkeyType<br>`Instagram__MonkeyType.70c3acf6.pr_151` | bare Tuple 与零参数 Tuple[()] 的类型身份 | 完整验收通过；原题面；显式基线差异profile |

主机制合计：S1 1条；F1 2条、F3 2条、F4 3条；R1 4条、R2 1条、R3 3条、R4 2条、R5 2条。分类依据实际生产契约和前后执行观察，不按annotation、decorator、cache等关键词凑数。例如MonkeyType148归S1；Funcy两题虽使用wrapper，修的是参数流而非callable身份。相同upstream snapshot已保留分组，不能把同源family仅换task ID后拆作独立评测样本；当前全部只用于train。

## 四个空原题面的明确版本化

TinyDB缓存失效、Boltons比较协议、PtyProcess.isalive、dataset.Database.__exit__的publisher `problem_statement`原本为空。原空题面task/blob/hash及首次wrapper-only运行都保留；采集侧明确编写最小行为描述，并在实际task-v2下重新执行官方初始prompt、六工具及双轮完整验收。

这四题的 `visible_prompt_profile=curator_authored_behavioral_clarification` 保存原/新题面SHA、准确curator题面、审阅依据与实际v2运行路径。题面只描述触发与预期行为，没有插入参考补丁、required节点列表或未来验收结果；它们不冒称原publisher描述，也不是只在最终导出时补写。其余16题保留非空原题面。

## 两条显式基线证据profile

两题均保留全部原F2P/P2P和修后原官方验收，只对固定task/blob及基线XML采用逐题审阅的 `audited_source_F2P_label_noise` profile；默认“原F2P全部在基线失败”的证据规则没有整体放宽。

- MonkeyType限定名恢复题（`cz3cmljw`）：原361项F2P在本地基线实际为52失败、309已通过，0未匹配；原8项P2P通过。真实类型反序列化从错误模块恢复到正确type身份，修后原369项要求全部通过。不能宣称361项都曾失败。原题面示例还把模块对象传给需要模块名字符串的API；该来源限制保留，实际诊断使用合法编码API，未偷偷重写原题面。
- MonkeyType151：原4项F2P在Python3.11.15基线实际为3失败、1已通过，0未匹配；修后原全部要求通过。真实`Tuple[()]`往返原本退化为bare `Tuple`，修后保留零参数泛型身份。原issue提到Python3.7，当前runtime差异不证明历史来源标签错误；不能把已通过的一项说成负例或宣称复刻了历史3.7异常。

这些是基线观测与来源标签的差异，不是放宽修后oracle。已通过的原节点不被当作rejected action；两题也不被伪装成同状态策略偏好对。

## 运行与训练边界

所有材料都是Codex编排、审阅并通过真实工具执行的off-policy冷启动示范。完整保留原F2P/P2P和原文件范围，原测试只按固定上游字节恢复，不改断言、官方matcher或缩成节点选择来取得成功。脚本耗时不是Gemma推理预算成绩。

当前使用官方SubprocessManager CPU本地后端，任务Python/依赖profile与现代ADK父环境分开，**尚未证明比赛Docker环境等价**。Pty任务的Mac进程语义、SQLite配置、Python3.11 typing行为等具体边界分别记录，不能推广为未执行平台的结果。

本目录不表示Gemma当前策略轨迹、已验证关键动作对或完整Verified Focal Agent RL数据已就绪。模型on-policy rollout、同状态策略对照和GPU训练仍为0；未编造thinking、行为logprob或训练收益。
