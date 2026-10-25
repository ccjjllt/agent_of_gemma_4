# S600：S1/S2/S5 来源与实际重放

2026-10-25。本目录累计实际处理86个独立任务，82个有完整正例证据：57个S专科与25个共同课任务。另4个独立任务未准入。Furl首题在Python3.12的旧负记录与Python3.9的正记录属于同一题，只计一次。目录中的宽页缓存和未执行候选配置不等于逐题可训练数量。未执行GPU、模型采样或偏好分叉；最终课程准入由root统一交叉复核与导出确认。

第一批固定于[reviewed-candidates-v2.json](reviewed-candidates-v2.json)：Funcy13题全部原F2P基线真实失败、修后203项原文件完整通过，0skip。初版13题曾全归S，交叉复核后将`re_find`改为共同课F1、S1次机制：真实缺陷是pattern/flags/text在包装调用中的错传，底层parser未变。修订保存在[classification-amendment.json](classification-amendment.json)，[初版review](reviewed-candidates.json)、原报告、实际task/run/recipe均保留。余下12题为S1五题、S2两题、S5五题。

第二批固定于[second-batch/reviewed-candidates.json](second-batch/reviewed-candidates.json)。XML四题与INI两题有完整双轮证据：XML命名空间声明、解析名、输出名与注释分别通过60项；INI错误位置显示与comment词法分别通过42项。全部原F2P实际失败、修后0failure/0error/0skip，题面沿用非空原发布文本。第二批新增S1一题、S2一题、S5四题。因此两批可用专科合计S1六题、S2三题、S5九题。

第三批[third-batch/reviewed-candidates.json](third-batch/reviewed-candidates.json)实际处理Pyparsing的Located与Parso的token end span。Located完整1852项通过并保留原1791条要求及6文件；Parso虽修后1346项通过，原生产模块doctest未被官方发现，未准入。

Pyparsing后两批分别固定于[pyparsing-batch/reviewed-candidates.json](pyparsing-batch/reviewed-candidates.json)与[pyparsing-curated-batch/reviewed-candidates.json](pyparsing-curated-batch/reviewed-candidates.json)，各20题完整1852项通过，原F2P均实际失败，原1791条要求及6文件不变。前批18专科、2共同课：`IndentedBlock.__init__`按strict/recursive/grouped参数路由归共同F1；`And.streamline`的真实区分为返回对象与链式调用，归共同G，不能冒称本诊断隔离了skip/indent算法。后批20专科均原空题面，明确保存未执行的原空adapted task、curator文本和哈希，并实际在task-v2题面下重放；没有伪造空题面运行。两批已独立交叉核对。S专科总计S1×18、S2×17、S4×1、S5×21。

Furl共同课程22题固定于[common-furl39-batch/reviewed-candidates.json](common-furl39-batch/reviewed-candidates.json)和[common-furl-more-batch/reviewed-candidates.json](common-furl-more-batch/reviewed-candidates.json)，全部原非空题面、77条要求与2个完整文件保留，原F2P实际失败后修复77pass、0skip。机制为F1×8、S1×4、S2×6、S3×3、R2×1，全部明确进入`S600_common`。采用真实系统Python3.9.6的独立任务环境；现代官方ADK父环境保持Python3.12.13。实际二进制与192个依赖文件哈希保存在[furl39-runtime-profile.json](furl39-runtime-profile.json)，新21题prep绑定其固定哈希。`collect_furl_more.py`只有`ACTIVE_KEYS`21题执行；余20个源码候选配置未执行、未计数。

所有来源均为固定`SWE-bench/SWE-smith-py` train revision `77cab9055d42ab4a5c25c89a8f937096db13558e`。新parse/XML/INI三页保存响应`X-Revision`、原字节SHA与逐行固定train身份核对；Funcy复用已核验的固定页receipt。每题保留完整mirror commit、独立上游snapshot、原task、许可证和全部原验证资产。缺陷引入patch没有直接作为修复答案；实际编辑只恢复已逐题审阅的生产函数契约。

## 保留的实际失败与限制

- XML namespace-binding的首次reproduction错误地期待无普通属性的root也保留`@xmlns`。实际固定源在元素有属性时才附带声明。根据真实源观察加入普通`id`属性后，以新版recipe在run002重新完成基线/修复重放。原题面、原验证要求未变；旧recipe与中止run001保留。这是collector修正，不能称Gemma错误或同状态策略收益。
- INI `lineof`：工具级源位置修复和修后42项通过均真实，但坏`lineof`同时使pytest启动读取其自身配置时排序`None`而崩溃，基线未产生JUnit。不能声称原4项F2P已逐项失败，不进入正例池。实际baseline-official-verification.json和修后JUnit均保留。
- Parse `Parser.parse`：工具级typed/lazy结果修复成功，但原P2P包含`parse.py` doctest，官方以其为验证目标排除生产`parse.py`的提交内容，实际`submitted.patch`为0字节；完整修后仍71失败、1skip。另有原README/doctest发现契约，不能通过删掉原目标、修改官方保护或偷加参数求过。当前只对这一个候选实跑，不把另外89个cache任务冒充可验收题。
- Mido仅按允许train缓存作来源布局筛选：原required使用`../dev/tests/...`，需要独立证明其原布局或保守逐节点映射，当前未重放或计入本目录任务。代数等价的`second2tick`变形也未按错误题面直接准入。
- Pyparsing `And.streamline`首次诊断未区分bug/fix，collector按预期错误约束停止；旧partial保留，新run002用真实返回对象/链式解析重新完成原全范围。不是模型错误或同状态偏好对。
- Furl `strip_scheme`原Python3.12任务环境修后74pass、3项因旧upstream与较新stdlib URL/IPv6行为不兼容而失败。原负记录保留在[common-batch/reviewed-candidates.json](common-batch/reviewed-candidates.json)。改用经原native77项通过的独立Python3.9.6任务环境，实际重新执行完整官方双轮；没有修改源、断言或要求，也没有改变父ADK环境。
- Bleach `nofollow`实际修复后423pass、4个原生xfail，但官方matcher把同参数化函数的原xfail短名与required具体case合并，导致原required的`area`用例被拒绝。没有删除xfail、改matcher或硬映射求通过；保留完整负记录，停止该仓扩充。

## 执行方式

真实使用未修改的官方六工具、Context/Budget、verify_task与JUnit matcher，官方父环境与独立任务依赖分离；当前仍是本地CPU SubprocessManager，尚未证明比赛Docker等价。除明确披露的Furl独立Python3.9.6任务profile外，任务源使用Python3.12。INI使用实际`src`源码布局，Pyparsing显式安装其原required额外依赖并设置`MPLBACKEND=Agg`，运行器记录真实源码绑定及现代父NumPy/Pydantic导入。没有伪造模块、编写测试文件、修改原断言或缩成required节点执行。

`collect.py`与`collect_more.py`用于实际采集、重放和发布来源证据。已发布review不可覆盖；继续采集应使用显式新批次，分类修订应另存版本。原始大文件与完整运行保存在本地ignored artifacts；主目录没有改共享catalog或启动训练。最终课程资格由root统一交叉复核、归一与官方对白导出确认。
