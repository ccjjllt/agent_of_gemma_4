# f2-push5 三元组复核 log（2026-10-25，第十四轮：weak-only 新地面 ≤200 + C余量第二轮）

口径：每题必须同时满足 `is` 身份断言 + 双向（共享修改）可见 + 访问/复制时点。
只有 `==` 无 `is` → 退 S；只有能力识别（无引用/所有权语义）→ 退 R。
inline-is 型：F2P 仅值断言，`is` 腿由 source-bound inline 观察承载
（同 C2/K15/nox-439/aiohttp-10672/qiskit-12620/M2/P1-P4/Q2-Q3/R1-R2 口径）。
Gym/V2 真题：A池 gym 151-200 weak+支撑（push4-R 仅 V2，push2/3-A 仅 strong，
故 gym-weak-151-200 从未逐题复核）+
B池 v2 151-200 weak test-is-only（无 recall 支撑；push4-R 要求 recall 支撑）+
C池 P2P-empty ≤50 weak+支撑（push2/3-B 仅 strong）+
D池 ≤150 weak 无支撑（push4-W 要求支撑；lottery 切片，按 total/nfiles 取前 25）+
第二轮 C 余量 10 行（fn/own/tis 次优 + inplace-F2P 名高信号）。
四池 total 全部 ≤200（可分批）。坏路径（`..`）/禁仓/四仓已在扫描层过滤。
Smith-py 不扫（冻结）。
去重：只读 common-400 + F1-40 + F5旧40 + f2-round1/2 + f2-gymv2 16 +
f2-gymv2b 5 + f2-gymv2c 2 + f2-recall 3 + f2-inline 1 + f2-more 2 + f2-push 6 +
f2-push2 3 + f2-push3 2 + f2-push4 1（程序化校验 562i/125s/508f；KEEP 为空，
无 snapshot_collision；instance 全新）。

## 扫描召回（f2-push5，forbidden 含 16+5+2+3+1+2+6+3+2+1 keeps）

- Gym A(151-200 weak+支撑)：3 行；V2 B(151-200 weak test-is-only)：10 行；
  C(P2P-empty≤50 weak+支撑)：Gym 12 + V2 116 = 128 行；
  D(≤150 weak 无支撑)：Gym 123 + V2 447 = 570 行。扫描合计 711。
- prior-named（旧日志已逐题具名 403，含 push4 87）排除并按 instance 并池后：
  unique 688（A 2 / B 8 / C 116 / D 562）。
- 本批深读 review-set 65 行（A 全 2 + B 全 8 + C 前 30 + D 前 25，
  见 `push5-review-set.json`；排序：A 按 F2P名/ownership/test-is/total，
  B 按 total/test-is，C 按 F2P名/ownership/test-is/total/文件数，
  D 按 total/文件数）+ 第二轮 C 余量 10 行（`push5-round2.txt`）；
  其余 613 行记信号级 SCREENED（未逐题 verdict，不计数；同 push2/push4 口径）。

## KEEP（真 F2，三元组完整，skip CLEAN，已双轮 eligible）：0

无。本批 75 行逐题复核全部退 S/R（下详）。weak-only 地面真 F2 密度为零，
与 push4 结论一致（push4 weak-offset 87 行仅 1 KEEP 且为 near-miss）。

## COLLIDE-family（同 fix 同快照，不计入）：0

## REJECTS（已深读 patch-evidence，不充数；S=值/语义无is腿，R=新能力/重构无基线别名腿）

- R-新能力/新API：`facebookresearch__hydra-1404`（ConfigCompositionException
  兜底 completion；weak 命中仅为注释行）、`amoffat__sh-603`
  （wait-lock 移出 `_process_just_ended` 的死锁重构）、
  `nipype__pydra-322`（export-graph 新功能）、`cs-si__eodag-790`
  （provider 参数透传新能力；`list(` 附带）、`jelmer__dulwich-1555`
  （update_working_tree 新功能）、`stac-utils__pystac-999`
  （remove_hierarchical_links/is_hierarchical 新 API，无基线别名腿）、
  `alteryx__woodwork-988`（describe extra_stats 特性）、
  `boto__botocore-472`（url_path 兼容重构；`r = request_dict` 别名系新代码
  自带非修复对象）、`maniacallabs__bibliopixel-249`（project 类型预处理特性）、
  `aeye-lab__pymovements-1112`（HTML repr 特性）、
  `dirty-cat__dirty_cat-539`（ken helper 新函数）、
  `zalando__patroni-2391`（SlotsAdvanceThread 新线程特性；`list()` 系新逻辑
  hygiene，无 caller-owned 映射在险）、`conda__conda-8015`
  （CompileMultiPycAction 性能重构）、`joke2k__faker-1806`（SBN 特性+locale 数据）、
  `tefra__xsdata-699`（FilterClasses handler 新能力）、
  `sciunto-org__python-bibtexparser-473`（NormalizeFieldKeys 新 middleware；
  无基线别名腿）、`openaddresses__machine-476`（logger-state 栈新能力；
  `_logger_status = list()` 系新逻辑）、`fatiando__verde-101`
  （Components 新类；`[Trend]*2` 别名注记系用户误用警告非修复对象）、
  `ntnu-ihb__pythonfmu-103`（CLI API 变更）、`mila-udem__fuel-214`
  （外部包扩展特性）、`xknx__xknx-1031`（RoutingBusy 流控特性）、
  `natcap__invest-814`（spec 格式化搬运重构）、
  `developmentseed__titiler-1115`（middleware 重构特性）、
  `sparckles__robyn-1002`（openapi override 特性）、
  `maniacallabs__bibliopixel-648`（ServerDriver 重构）、
  `Project-MONAI__MONAI-3785`（pred_keys 默认特性）、
  `Project-MONAI__MONAI-4159`（ckpt_export 多配置特性）、
  `Project-MONAI__MONAI-673`（默认 transform 变更）、
  `qchackers__tqec-245`（grid template 新类）、
  `leshchenko1979__fast_bitrix24-114`（retry 新装饰器）、
  `streamlink__streamlink-2085`（senategov 新插件）、
  `v2 qchackers__tqec-245`（见上）、`v2 streamlink__streamlink-2085`（见上）、
  `Project-MONAI__MONAI-3440`（DataFrame-src 特性）、
  `scoutapp__scout_apm_python-639`（backtrace 过滤特性）、
  `iterative__dvc-3760`（update --recursive 特性）、
  `codezonediitj__pydatastructs-538`（Multiset 新类）、
  `python__mypy-9102`（function-sig-hook 新特性）、
  `Project-MONAI__MONAI-1010`（properties 特性）、
  `openmined__pysyft-113`（TensorBase 新类；testInplace 系新能力名，
  无基线别名腿）、`jnothman__upsetplot-61`（from_contents 新函数）、
  `quansight__ragna-456`（metadata-filter 重构特性）、
  `aws-cloudformation__cfn-lint-3989`（新 lint 规则）、
  `googlefonts__ufo2ft-480`（GDEF writer 新能力）、
  `datadog__guarddog-342`（detector 拆分新能力）、
  `mirumee__ariadne-codegen-242`（NoReimportsPlugin 新插件）、
  `yt-dlp__yt-dlp-13234`（SUBS PO-token 上游跟踪）→ R（43）。
- S-值/语义/卫生拷贝：`getmoto__moto-4956`（ELB 响应形状值修复；
  `list(set())` 附带去重）、`pennylaneai__pennylane-6147`
  （batch-1 squeeze 值修复）、`pymc-devs__pytensor-1273`（solve 数值分发修复）、
  `getmoto__moto-8032`（credential-report 值修复；`list()` 附带）、
  `python__mypy-15128`（union 化简 segfault 修理）、
  `scoutapp__scout_apm_python-676`（ContextVar 作用域修理；无 copy 时点）、
  `stanfordnlp__dspy-1853`（JSON 序列化值修复）、
  `sappelhoff__pyprep-62`（RANSAC 采样对齐值修复）、
  `python__mypy-9625`（Annotated 别名类型支持）、
  `python__mypy-13728`（match transform crash 修理）、
  `python__mypy-14038`（递归类型 segfault guard）、
  `codezonediitj__pydatastructs-165`（邻接矩阵表示重构；"aliasing" 系内部
  存储非 caller 共享，无 is 三元组）、`pyqtgraph__pyqtgraph-1845`
  （zoom 回归修理）、`qchackers__tqec-376`（translation-invariance 系
  hash/eq 表示一致性，无 copy 时点）、`Project-MONAI__MONAI-3205`
  （get_dataset override 新能力内 `dict()` 系新逻辑 hygiene，
  基线从不 mutation，无 caller-owned 映射在险）、
  `datadog__guarddog-163`（空规则 guard 值修复）、
  `nipy__nipype-2912`（nibabel 兼容值修复）、`iterative__dvc-1950`
  （status 性能优化）、`Project-MONAI__MONAI-2096`（分布式 hang 修理，
  同 openmdao-3419 先例）、`pallets__jinja-1522`（hash 回退语义，
  无所有权隔离）、`useblocks__sphinx-needs-1101`（ID 鲁棒性修理）、
  `python__mypy-13963`（TypedDict classmethod 修理）、
  `python__mypy-14163`（use-before-def 诊断）、
  `python__mypy-15882`（match-guard narrowing 修理）、
  `python__mypy-15425`（TypedDict update 类型检查）→ S（24，
  含第二轮 2：mypy-14163、mypy-15882）。

## 计数诚实注记

- 75 行逐题复核：triple-KEEP 0 / COLLIDE-family 0 / S 24 / R 51
  （上列全员+依据；第一轮 65：S 22 + R 43；第二轮 C 余量 10：S 2 + R 8）。
- 613 行 SCREENED（信号级未深读，不计数；`push5-candidates.json` 可复现）。
- skip 预检：无 KEEP，故无候选可投（0 CLEAN / 0 DROP；不编造）。
- 双轮完成 0（0 eligible，0 near-miss）；replay 零新增。
