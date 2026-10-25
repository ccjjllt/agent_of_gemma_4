# f2-gymv2c 三元组复核 log（2026-10-25，第五轮：F2 补 32 的第一批 2 KEEP）

口径：每题必须同时满足 `is` 身份断言 + 双向（共享修改）可见 + 访问/复制时点。
只有 `==` 无 `is` → 退 S；只有能力识别（无引用/所有权语义）→ 退 R。
套件 ≤150（Gym/V2 真题放宽口径）；坏路径（`..`）/禁仓/四仓已在扫描层过滤。
P2P=0 的任务标 P2P-EMPTY（JUnit 双条件风险，replay 时如实判定）。
pools：SWE-Gym/SWE-Gym rev `bb94ed9e…`（train 2438）+ nebius/SWE-rebench-V2
rev `10483de0…`（train python 子集 7243）。Smith-py 不再扫。
去重：只读 common-400 + F1-40 + F5旧40 + f2-round1/2 + f2-gymv2 16 keeps +
f2-gymv2b 5 keeps；本批 2 keeps 与它们零重叠（程序化校验见 build_reviewed.py +
dedup-evidence.json：542i/105s/488f）。旧目录文件未改动。

## 扫描召回（f2-gymv2c，forbidden 含 f2-gymv2 16 + f2-gymv2b 5 keeps）

- Gym strict changed-line copy 扫描：2438 行 → 25 hits（≤150）+ 7 个 >150 排除。
- V2-python strict 扫描：7243 行 → 95 hits（同条件）+ 28 个 >150 排除。
  strict 剩余 120 hits 前两轮已代表性复核为 S/R（见 f2-gymv2/b triple-review-log），本轮不重做。
- 扩召回（新）：gym-recall 64 + v2-recall 233（clone-changed / ownership-changed /
  test-side-is 三信号）；v2-isnew-screen 173（test_patch 含 `is` + patch 含 copy 词、
  未被 strict/recall 覆盖）；v2-f2pname-screen 148 + gym-f2pname-screen 24
  （F2P 名含 copy/clone/pickle/deepcopy）；v2-statescreen 64
  （np.copy/copy=True/getstate 等 changed-line 信号）。
- 合计 screening 约 800 行；逐题读 problem_statement + patch + test_patch 后，
  本批复核出 2 KEEP（下）。其余代表性 REJECT 见下（S/R 按口径）。

## KEEP（真 F2，三元组完整，skip 预检 CLEAN，双轮 eligible）

### C1 `pallets__jinja-2029`（V2，total=27，F2P=7，P2P=20）
- 内部 `missing` 单例原来是动态 `type("MissingType", ...)` 实例，无 `__reduce__`：
  copy/deepcopy 产生不同对象、pickle 直接失败（`Can't pickle ... MissingType`）。
  修复定义 `_MissingType`（`__reduce__` 返回 `"missing"`），copy/deepcopy/pickle
  均保身份。
- 三元组：`copy.copy(missing) is missing` + `deepcopy is missing` +
  `pickle.loads(pickle.dumps(missing)) is missing`（F2P `test_copy_missing`、
  `test_pickle_missing[0-5]` 逐字断言）+ 单例全局可见（双向）+ copy/pickle 时点。
  极性：单例保身份（同 K1 openmdao-2810）。
- 家族：`pallets/jinja::jinja2.utils.missing/_MissingType::singleton_copy_pickle_identity`；
  快照 `pallets/jinja::ba8847a4…`（与 2026/1613 快照均不同，生产符号不同）。零重叠。
- 双轮：baseline 7 F2P 全 F / repaired 全 T；normalize eligible（无理由）。
  src 布局，sandbox 用 `--source-layout src` 绑定（.venv 自带 jinja2 3.1.6 会 shadow，
  binding 断言已验）。

### C2 `pallets__jinja-1613`（V2，total=1，F2P=1，P2P=0 ⚠P2P-EMPTY）
- `@async_variant` 装饰器给 wrapper 错误的 `__name__/__qualname__`，pickle 无法按引用
  解析（`PicklingError: ... it's not the same object as jinja2.filters.sync_do_first`，
  即题面原文）。修复拆分 `@wraps` 赋值（名/模块/qualname 取 async 侧，其余取 sync 侧）。
- 三元组：`pickle.loads(pickle.dumps(do_first)) is jinja2.filters.do_first`
  （pickle-by-reference 身份；F2P `test_environment` 为功能覆盖，`is` 腿由
  source-bound inline 观察承载：另在原对象置属性，经 unpickle 引用可见）+
  模块级共享可见（双向）+ pickle 时点。极性：保身份（同 K1）。
- 家族：`pallets/jinja::jinja2.async_utils.async_variant::pickle_by_reference_identity`；
  快照 `pallets/jinja::09907ef3…`。零重叠。
- 双轮：baseline F2P F / repaired T；normalize eligible（无理由；P2P=0 按 K1 先例接受）。
  同 src 布局 + `--source-layout src`。markupsafe 3.0.4 与 base（jinja 3.0.x，
  `soft_str` 系）兼容已验。

## REJECT → S（仅值语义/附带 copy，无差异 `is` 腿，代表性；本批复核确认）

- tatsu-276（AST pickle 数据保真，`assert a == b`，roundtrip 天然新对象，无 `is` 腿）。
- fuel-186（IndexableDataset pickle 崩溃修，无断言；虽机制似 jinja-2026 但无 `is` 腿）。
- pyjanitor-1066（count_cumulative_unique 改输入列；可观察仅为值不变，
  `result is not df` 修前后恒成立，无差异 `is` 腿）。
- pydantic-4920（copy 后 `__fields_set__` 含 excluded；`in/not in` 成员语义，
  fields_set 本就新建，无身份差异）。
- click-2312（decorator 复用 pop cls；F2P 仅 isinstance，无差异 `is` 腿；
  option 侧 926 先例本就是 `attrs.copy()` 值修）。
- laspy-195（header bbox 最小值，值 bug）。
- jinja-1613 曾候选但 F2P（test_environment 新文件）仅 `==` 渲染断言——
  注意：C2 的 KEEP 依据是 filter 函数 pickle-by-reference 的 `is` 三元组
  （inline 观察），非该 F2P 文本；口径与 K15（ariadne，F2P 无 `is`）一致。
- moto-6308/8316/6884（S3/param 值 bug）、cfn-lint 全系（规则值语义）、
  pandas-59807（CoW replace 真机制但 pandas 需 C 编译，环境同 K2 provisional，
  未纳入）、hydra-2543（异常 pickle ctor 缺省，功能修）、hydra-772（frozen 只读）、
  dask-8792（clone_key 前缀命名）、openmdao-3315（Aitken 数值）、
  openmdao-3295（新 setup_residuals 能力）、slack-1638（clone 丢 custom_values，
  `assertEqual` 值语义）、tqec-376（`is not None` 仅空检查）、bravado-372 见 R、
  todoman-197 见 R、pyomo-2231 见 R、envisage-354 见 R、pystac-1006 见 R、
  traits-1384 见 R、jinja-2033 见 R、polyfactory-602 见 R。

## REJECT → R（纯能力/特性/重构，代表性；本批复核确认）

- yelp__bravado-core-372：新增 `__getstate__/__setstate__` + pickleable 表示
  （之前完全不可 pickle；与 jinja-2026 的“默认协议被 dunder bug 破坏”不同，
  属新能力）。
- pimutils__todoman-197：新增 `clone()` 方法（之前无；同 pyfar-58 先例）+
  edit 保 ID（值语义）。
- pyomo__pyomo-2231：`evaluate_fgh` API 对齐（题面逐字 Add … API）。
- envisage-354：observe 兼容（题面逐字 Support for observe …）。
- pystac-1006：deprecation warning 新能力（test_full_copy 随新 extension 文件附带，
  非共享修复）。
- traits-1384：导出 NoDefaultSpecified 单例（test 仅 assertEqual）。
- jinja-2033：sandbox 新增 clear/pop 禁止（安全能力）。
- polyfactory-602：SQLAlchemy nullable 修（值语义，兼 S）。
- gidgethub-35（test_router_copy 但 test_patch 无 `is`）、pyfar-58 系不赘述；
  polyfactory/jupysql/dishka/scuba/holidays 系命名 alias 噪声；
  celery-9795/trio-2392/pypa-build 系特性；pyccel-2019 系需 Fortran；
  keras-20550 系需 TF；MONAI 系需 torch；pandas 系需 C 编译（K2 同因）。

## 双轮与口径注记

- C1/C2 均为纯 Python 小套件（27/1），双轮一次通过，无 task-site-packages 需求
  （markupsafe 3.0.4 era 兼容已验）；六工具、测试/matcher/要求未改、保护路径未碰。
- 全局：6（f2-gymv2 eligible）+ 2（f2-gymv2b eligible）+ 本批 2 = 10/40，缺口 30。
- 缺口清零前不启动训练、不改 S600 与旧文件。
- 后续 continuation 提示：strict 池已枯（真 F2 密度 ~15%→已取尽）；
  recall/isnew/f2pname/statescreen 约 800 行已筛，剩余多为命名噪声与重依赖任务；
  >150 套件 35 行（Gym 7 + V2 28）仍被 ≤150 口径排除，需分批依据另议；
  pandas/MONAI/torch 系需 task-python 老解释器或重依赖方案。
