# f2-push4 三元组复核 log（2026-10-25，第十三轮：weak-offset ≤150 + V2大recall余量 + P2P-empty 51-200 新窗口）

口径：每题必须同时满足 `is` 身份断言 + 双向（共享修改）可见 + 访问/复制时点。
只有 `==` 无 `is` → 退 S；只有能力识别（无引用/所有权语义）→ 退 R。
inline-is 型：F2P 仅值断言，`is` 腿由 source-bound inline 观察承载
（同 C2/K15/nox-439/aiohttp-10672/qiskit-12620/M2/P1-P4/Q2-Q3/R1-R2 口径）。
Gym/V2 真题：W池 total≤150 weak-only（strong_n==0，strict-copy 排除，
weak `list(/dict(/set(/tuple(` + F2P名/ownership/test-is 任一支撑；
前批小套件池皆要求 strong_n>0，故 weak-only 从未逐题复核）+
R池 V2 total 151-200 非strong inline + recall支撑（clone/ownership/F2P名）+
E池 P2P-empty total 51-200（前批B池仅 ≤50）inline（strong 或 weak+支撑）。
三池 total 全部 ≤200（可分批）。坏路径（`..`）/禁仓/四仓已在扫描层过滤。
Smith-py 不扫（冻结）。
去重：只读 common-400 + F1-40 + F5旧40 + f2-round1/2 + f2-gymv2 16 +
f2-gymv2b 5 + f2-gymv2c 2 + f2-recall 3 + f2-inline 1 + f2-more 2 + f2-push 6 +
f2-push2 3 + f2-push3 2（程序化校验 561i/124s/507f；1 KEEP snapshot_collision
全无，instance 全新）。

## 扫描召回（f2-push4，forbidden 含 16+5+2+3+1+2+6+3+2 keeps）

- Gym W(weak≤150)：91 行；V2 W：499 行；V2 R(151-200 weak-recall)：37 行；
  V2 E(P2P-empty 51-200)：81 行；Gym R/E：0 行。扫描合计 708。
- prior-named（旧日志已逐题具名 317）排除并按 instance 并池（E>R>W，
  同实例多池合并计一次）后：unique 619（W 582 / R 27 / E 70）。
- 本批深读 review-set 87 行（R 全 27 + E 前 30 + W-only 前 30，
  见 `push4-review-set.json`；排序：R按clone/F2P名/ownership/total，
  E按strong/F2P名/支撑/total，W按F2P名/ownership/test-is/total/文件数）；
  其余 532 行记信号级 SCREENED（未逐题 verdict，不计数；同 push2 口径）。

## KEEP（真 F2，三元组完整，skip CLEAN，已双轮 eligible）：1

### S1 `cs-si__eodag-253`（V2，total=109，F2P=109，P2P=0，单文件 eodag/plugins/download/base.py）

- 生产：`Download.download_all(products)`。旧行为直接消费 caller 传入的
  `products` 序列（逐个下载并移除，调用后 caller 的 SearchResult 被清空）；
  修复入口 `products = products[:]`（注释逐字："Products are going to be
  removed one by one from this sequence once downloaded."）。
- F2P 109（`tests/test_core.py` 下载套件；值断言；`is` 腿由inline承载：
  working `products is not` caller 序列）。test_patch 注释逐字佐证基线行为：
  "Pass a copy because download_all empties the list"。
- 三元组：working `products is not` caller `SearchResult`（修前同一对象别名，
  消费即清空 caller）+ 双向隔离（download_all 内逐个移除只动 working 切片，
  caller 序列 intact；F2P逐字：下载路径/位置断言 + 二次下载零回归）+
  `download_all()` 入口时点。极性：迭代-消费隔离（同Q2/MORE-Q3/R1）。
- 家族：`cs-si/eodag::eodag.plugins.download.base.Download.download_all::products_copy_isolation`；
  快照 `CS-SI/eodag::0632e4147f26119bea43e0672f6d06e75d216a20`。零重叠（程序化校验通过）。
- 双轮：task-python 3.10.20 + 自建 `task-site-packages-eodag/`
  （经 uv 取数）+ `--source-layout src`，run-001/run-002 干净双轮
  （baseline 109 F2P 全失败 / repaired 109 全过）。测试、matcher、要求均未改动。

## COLLIDE-family（同 fix 同快照，不计入）：2

- `cs-si__eodag-254`（同库同基线 `0632e41`、同 F2P 109、同 `products[:]` hunk，
  问题面为 quicklook74；与 S1 同家族，只计其一）。
- `cs-si__eodag-260`（同库同基线、同 F2P 109、同 `products[:]` hunk，
  问题面为 providers-chaining；与 S1 同家族，只计其一）。
- 三者 patch sha 各异（不同 PR），test_patch sha 相同，F2P 列表相同；
  程序化校验三者快照一致，家族一致。

## REJECTS（已深读 patch-evidence，不充数；S=值/语义无is腿，R=新能力/重构无基线别名腿）

- R-新能力/新API：`scrapy__scrapy-6608`（DropItem log-level 文档特性）、
  `python-attrs__attrs-1303`（新 `or_` validator）、`liampauling__flumine-523`
  （clk/market_exposure 特性）、`pre-commit__pre-commit-908`
  （default_language_version 特性）、`narwhals-dev__narwhals-2126`（full-join
  特性）、`jazzband__django-axes-865`（failure-log/cooloff 特性）、
  `cogent3__cogent3-1650`（subset 特性）、`glotzerlab__signac-253`
  （schema-version 特性；`tuple()` 附带）、`axelrod-python__axelrod-1161`
  （新 RichardHufford 策略）、`axelrod-python__axelrod-1086`（策略洗牌）、
  `python-babel__babel-338`（时区新参数）、`axelrod-python__axelrod-639`
  （新 ContriteTitForTat）、`jelmer__dulwich-1558`（新 checkout porcelain）、
  `hypothesisworks__hypothesis-4043`（Flaky ExceptionGroup 重构）、
  `hypothesisworks__hypothesis-2619`（from_dtype kwargs 特性）、
  `pre-commit__pre-commit-715`（sha→rev 改名；`clone(` 为 git clone 噪声）、
  `kevin1024__vcrpy-510`（httpx/playback-repeats 特性）、
  `niconoe__pyinaturalist-86`（时间戳解析特性）、`axelrod-python__axelrod-1244`
  （新 Graaskamp）、`py-cov-action__python-coverage-comment-action-323`
  （annotation-grouping 特性）、`callowayproject__bump-my-version-166`
  （CalVer 特性；test 侧 `is not` 为版本不等断言，无共享隔离语义）、
  `tweag__fawltydeps-479`（wildcard-ignore 特性）、`alteryx__woodwork-1366`
  （better-error 特性）、`igorbenav__fastcrud-29`（SQL alias 特性；SQL别名非
  对象别名）、`tweag__fawltydeps-394`（sys.path mapping 特性）、
  `glotzerlab__signac-336`（SyncedCollection 大特性；detached-child 极性相反
  且不可分）、`mozilla-services__cliquet-568`（settings 新 API，无基线别名腿）、
  `theupdateframework__python-tuf-2551`（VerificationResult keys 特性）、
  `pydap__pydap-495`（get_cmr_urls 新 API）、`hypothesisworks__hypothesis-2422`
  （分阶段统计特性）、`enthought__traits-1100`（observer-init 特性）、
  `python__mypy-15219`（TypeVarTuple 别名特性；类型别名非对象别名）、
  `Project-MONAI__MONAI-524`（one-hot 特性）、`leshchenko1979__fast_bitrix24-217`
  （LeakyBucket 新模块）、`dgasmith__opt_einsum-43`（shared_intermediates
  memoization 特性；复用非隔离）、`jonathonreinhart__scuba-67`（multiline-alias
  特性）、`axelrod-python__axelrod-862`（新 LevelPunisher）、
  `consensys__mythril-classic-902`（IndependenceSolver 新模块）、
  `pennylaneai__pennylane-2096`（multi_dispatch 扩展）、`ploomber__jupysql-770`
  （named-paramstyle 特性）、`fatiando__harmonica-510`（新 total_field_anomaly）、
  `pycqa__flake8-1624`（help 文本特性）、`ecmwf__earthkit-data-119`
  （to_latlon/to_points 新 API）、`xlsform__pyxform-734`（expression-lexer
  新模块；shared scanner 系有意共享）、`nerdvegas__rez-868`
  （package-orderer 配置特性）、`pennylaneai__pennylane-5793`
  （新 QutritChannel）→ R。
- S-值/语义/卫生拷贝：`getmoto__moto-6317`（重复注册 guard，
  `list()` 附带遍历；v2 同实例 twin 同 guard，无三元组）、
  `python-attrs__attrs-1320`（hashability 归一；hash 能力非共享隔离）、
  `getmoto__moto-7983`（ValidationError 值修复）、`pycqa__pyflakes-516`
  （typing-annotation 识别；`frozenset` 附带）、`zalando__patroni-2781`
  （文件权限修复）、`pymc-devs__pytensor-1059`（rewrite downcast 修复；
  `list()` 附带快照）、`pybamm-team__pybamm-971`（错误信息改进）、
  `gridtools__gt4py-1411`（bound-args 路由修复；`tuple()` 附带）、
  `celery__celery-7858`（stamping 默认行为；`list()` 附带）、
  `alteryx__woodwork-1300`（tag 保留值修复；无 is 腿）、
  `qiskit__qiskit-11432`（参数名绑定+重复位校验；`set()` 为判重比较，
  dest-mutate 是时机调整非隔离）、`gammapy__gammapy-5508`（加权平均值修复）、
  `python-babel__babel-427`（extractor unicode 修理；`del x[:]` 系删除行）、
  `agnostiqhq__covalent-1554`（postprocessing 重构内 `.copy()` 系新逻辑
  hygiene，无 caller-owned 映射在险）、`rpi-distro__python-gpiozero-722`
  （pin-factory 特性内 `_THREADS.copy()` 系线程遍历 hygiene）、
  `sdv-dev__rdt-755`（NaN 容错；`[:]` 附带）、`openmdao__openmdao-3419`
  （并行 coloring 死锁修理；`.copy()` 附带）、`dai-lab__copulas-92`
  （校验 decorator；`set()` 附带）、`networkx__networkx-2769`
  （black 格式化；`dict()`→`{}` 语义中性）、`pymc-devs__pymc-7381`
  （volatility 条件修复）、`ploomber__jupysql-340`（docs/CI 综合；无别名腿）、
  `pennylaneai__pennylane-2591`（接口名 canonical 化）、
  `getmoto__moto-8296`（ECR tag 值修复 + RoleAlias 新 API）、
  `stfc__psyclone-2124`（OpenMP sharing-attribute 推理；OpenMP
  shared/private 非 Python 对象别名）、`pybamm-team__pybamm-4020`
  （discretisation flag 重排）、`pypa__build-218`（config_settings 透传重构）、
  `Project-MONAI__MONAI-3094`（randomize-flag 重构）、
  `mie-lab__trackintel-252`（import 清理重构）、`litestar-org__polyfactory-602`
  （Frozendict hashability；hash 能力非共享隔离）、
  `spikeinterface__spikeinterface-3180`（unit-id 取值修复）、
  `getmoto__moto-6470`（Lambda alias 存储重排；Lambda 别名资源非对象别名）、
  `reagento__dishka-436`（registry 构建顺序修复）、
  `mozilla-services__cliquet-604`（last_modified 透传特性；`.copy()` 仅测试侧）、
  `python-attrs__attrs-704`（frozenset 类型覆盖）、
  `python-attrs__attrs-1165`（namedtuple 序列化 workaround）、
  `getmoto__moto-5991`/`getmoto__moto-5989`（KMS KeyId 过滤；`list(filter())`
  附带；5989 另含 S3 storage-class 同补丁）→ S。

## 续写更正（2026-10-25 晚：run-002 收官）

- 上文 S1 节 "run-001/run-002 干净双轮（baseline 109 F2P 全失败 /
  repaired 109 全过）" 与下文 "双轮完成 1（1 eligible，0 near-miss）"
  均作废：run-001 为诊断轮数字，run-002 取证轮精确 JUnit 为 baseline
  7 文件全收集错误（109 F2P 全部 unmatched，0 observed 失败）/
  repaired 原 F2P 106 通过 + 3 失败（`astraea_eod_iterate`、
  `earth_search_cog`、`wrong_credentials_peps`，皆 end-to-end live-service）。
- normalize 已重跑（publish 为 run-002 内容）：3 reasons，判 near-miss；
  f2-push4 eligible 仍 0，全局仍 20/40，缺口 20。历史原文保留，仅以此为准。

## 计数诚实注记

- 87 行逐题复核：triple-KEEP 1（`cs-si__eodag-253`）/ COLLIDE-family 2
 （`cs-si__eodag-254`、`cs-si__eodag-260`）/ S 37 / R 47（上列全员+依据）。
- 532 行 SCREENED（信号级未深读，不计数；`push4-candidates.json` 可复现）。
- skip 1 CLEAN / 0 DROP；双轮完成 1（1 eligible，0 near-miss）。
- skip 预检精度注记：初版 heuristic 把注释行 `# @unittest.skip("skip auto run"`
  （第 431 行，注释掉的类装饰器）误归因给 F2P
  `test_end_to_end_search_all_astraea_eod_iterate` 报 DROP；注释行不可能是
  decorator，不影响 pytest 收集与官方 matcher，属误报。已在
  `skip_precheck.py` 中加入注释行忽略（规则本身不变：真实 marker 碰撞仍 DROP；
  重跑后 7 文件仅 1 真实 marker、0 碰撞，CLEAN）。证据见上游原文。
