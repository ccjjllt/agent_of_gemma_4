# f2-gymv2b 三元组复核 log（2026-10-25，第四轮：F2 补 34 的第一批 5 KEEP）

口径：每题必须同时满足 `is` 身份断言 + 双向（共享修改）可见 + 访问/复制时点。
只有 `==` 无 `is` → 退 S；只有能力识别（无引用/所有权语义）→ 退 R。
套件 ≤150（Gym/V2 真题放宽口径）；坏路径（`..`）/禁仓/四仓已在扫描层过滤。
P2P=0 的任务标 P2P-EMPTY（JUnit 双条件风险，replay 时如实判定）。
pools：SWE-Gym/SWE-Gym rev `bb94ed9e…`（train 2438）+ nebius/SWE-rebench-V2
rev `10483de0…`（train python 子集 7243）。Smith-py 不再扫。
去重：只读 common-400 + F1-40 + F5旧40 + f2-round1/2 + f2-gymv2 16 keeps；
本批 5 keeps 与它们零重叠（程序化校验见 dedup-evidence.json + keep 检查）；
与 f2-gymv2 的 5 待双轮（pandas-55314、MONAI-2104/2010、nicegui-3046、earthkit-671）
及备份（moto-6716、meltano-2144、sacred-765）只读去重，未重做。

## 扫描召回（f2-gymv2b，forbidden 含 f2-gymv2 16 keeps）

- Gym strict changed-line copy 扫描：2438 行 → 25 hits（≤150）+ 7 个 >150 排除（分批依据待补）。
- V2-python strict 扫描：7243 行 → 100 hits（同条件）+ 28 个 >150 排除。
- 合计 125 剩余 strict hits（141 − 16 keeps）。本批从中复核出 5 KEEP（下）。
  其余 120 hits 的代表性 REJECT 见下（S/R 按口径）。

## KEEP（真 F2，三元组完整，skip 预检 CLEAN）

### K13 `altair-viz__altair-518`（V2，total=7，F2P=7，P2P=0 ⚠P2P-EMPTY）
- 新增 `update_nested(original, update, copy=False)`：`copy=True` 时
  `original = deepcopy(original)` 后再原地更新并返回。
  `to_dict` 改为 `copy = self.copy()` + `update_nested(..., copy=True)`，
  不再原地改 `self._kwds`。
- 三元组：`output is not original`（copy=True，新对象）+ `output2 is original`
  （copy=False，共享 by design）+ `update_nested`/`to_dict` 调用时点。
  F2P `test_update_nested` 逐字断言 `assert output is not original` /
  `assert output2 is original`；`test_basic_chart_to_dict/from_dict` 覆盖
  `to_dict` 不污染原对象。
- 家族：`altair-viz/altair::altair.utils.core.update_nested+TopLevelMixin.to_dict::copy_isolation`；
  快照 `altair-viz/altair::e37000c8…`。零重叠。
- replay 风险：base（2018）`collections.Mapping` 在 venv Python 3.12 已移除；
  双轮如实判定，环境漂移不计入三元组（见 README 缺口说明）。

### K14 `pallets__jinja-2026`（V2，total=52，F2P=12，P2P=40）
- `Undefined.__getattr__` 对 dunder 名称改抛 `AttributeError`（`__`…`__`），
  使默认 `copy/deepcopy/pickle` 协议探测正常工作；删除错误的 `__slots__`
  删除块（恢复 slots 元数据）。
- 三元组：`copied is not undef`（copy 新对象）+ 分量级共享/隔离
  （copy：`_undefined_hint/_obj/_name/_exception is` 原分量；
  deepcopy：`_undefined_obj is not` 但 `==`，其余 `is`；
  pickle：各分量 `is not` 但 `==`，异常 `is`）+ copy/deepcopy/pickle 调用时点。
  F2P `test_undefined_copy/deepcopy/pickle`（×4 类型）逐字断言上述 `is/is not`。
  题面逐字：`it's not possible to copy/deepcopy/pickle Undefined objects`。
- 家族：`pallets/jinja::jinja2.runtime.Undefined::__getattr__+slots::copy_pickle_protocol`；
  快照 `pallets/jinja::39d9ffff…`。零重叠。

### K15 `mirumee__ariadne-codegen-159`（V2，total=3，F2P=1，P2P=2）
- 新增 `_get_operation_definition_without_mixin_directive()`：
  `operation_def = deepcopy(self.operation_definition)` 后 visit 移除 mixin
  指令；`get_operation_as_str` 对工作拷贝操作，原定义不动。
- 三元组：`operation_def is not self.operation_definition`
  （deepcopy 新对象）+ 移除工作拷贝指令后原定义仍含 mixin
  （F2P 断言 `field_query2.directives[0].name.value == MIXIN_NAME` 等三处，
  而结果串 `MIXIN_NAME not in result`）+ `get_operation_as_str` 调用时点。
- 家族：`mirumee/ariadne-codegen::ResultTypesGenerator._get_operation_definition_without_mixin_directive::deepcopy_isolation`；
  快照 `mirumee/ariadne-codegen::b09a5326…`。零重叠。

### K16 `astropy__specutils-1119`（V2，total=29，F2P=1，P2P=28）
- `with_spectral_unit` 改为 `with_spectral_axis_unit` 并真正工作：
  `meta = deepcopy(self._meta)` + `meta['original_wcs'] = self.wcs.deepcopy()`，
  新增 `copy()/deepcopy()`；新谱的 meta/wcs 为独立对象。
- 三元组：`new_spec.meta is not spec.meta`（deepcopy；
  `original_wcs` deepcopy）+ 改动新谱后原谱不动
  （F2P：`new_spec.meta['original_wcs'].wcs.crval = …` /
  `new_spec.meta['header']['CRVAL1'] = …` 后断言
  `wcs_spec.wcs.wcs.crval[0] == 3.622e-7` /
  `wcs_spec.meta['header']['CRVAL1'] == 3622.`）+
  `with_spectral_axis_unit` 调用时点。
- 家族：`astropy/specutils::OneDSpectrumMixin.with_spectral_axis_unit::meta_wcs_deepcopy_isolation`；
  快照 `astropy/specutils::30575fc5…`。零重叠。
- replay 风险：astropy/gwcs 原生栈较重；CPU 沙箱如实判定。

### K17 `openmdao__openmdao-3522`（V2，total=110，F2P=2，P2P=108）
- `Problem.get_val` / `System.get_val` 新增 `copy=False` 参数：
  `copy=True` 时 `return deepcopy(val)`，否则返回内部引用（view）。
- 三元组：数组/对象 `id(foo) != id(foo_copy)`（`is not`，隔离）+
  标量 `id(mul) == id(mul_copy)`（共享 by design）+
  改动 view 污染内部（`obj[2][0] = 10` 后 `p.get_val('obj')[2][0] == 10`）而
  改动 copy 不污染（`obj_copy[2][0] == 2`）+ `get_val(..., copy=True)` 时点。
  F2P `test_get_val`（problem + system）逐字断言上述三点。
  题面逐字：`get_val may currently return a view into an internal OpenMDAO vector.
  This can have some unexpected behavior`。
- 家族：`OpenMDAO/OpenMDAO::Problem.get_val+System.get_val::copy_return_isolation`；
  快照 `OpenMDAO/OpenMDAO::97f89e58…`（与 2810/3125/3461 快照均不同，
  生产符号不同）。零重叠。
- 注：与 K11（3461，同 test_problem.py）同文件但家族/快照/F2P 均不同；
  不视为重叠（程序化校验通过）。

## REJECT → S（仅值语义/附带 copy，无 is，代表性；本批复核确认）

- cfn-lint-3342/2031/4045：`.copy()/deepcopy()` 为工作拷贝隔离，但 F2P 只测
  功能（有效/无效用例计数），无 `is`、无改动隔离断言，题面未提共享 →
  S（弱，不纳入 KEEP；如放宽即得 3，但违反 is 口径，诚实记 S）。
- structlog-331：`get_threadlocal()` 返回拷贝，但 F2P 只断言 `==`，P2P=0，
  题面为特性请求 → S。
- pystac-885/1049：`deepcopy(geometry)`，F2P 只 `assertEqual`，无 `is`/改动 →
  S。
- glyphslib-563：`deepcopy(defaults)`，F2P 为 roundtrip，无 `is`/改动 → S。
- momepy-113：`].copy()` 消 SettingWithCopyWarning，F2P 无 `is`/改动 → S。
- pymovements-1108：`deepcopy(dictionary)` 后 `pop`，F2P 只 `==`，题面为特性 →
  S（且 R 成分）。
- moto-8342/8423/7085/7434/5960/7960/7961：值/行为测试，无 `is` → S（弱备份）。
- 其余 V2 strict 小套件（trestle-1695、skypilot-5577、scikit-build-329、
  pytask-200、pypika-306、bibliopixel×4、trackintel-384/148、pydeseq2-268、
  qiskit-8790、pyccel-2044、planet-588、pip-tools-837、scikit-build-703、
  vyper-4677、pynwb-557、dynaconf-375、feature_engine-500、fastmcp-1214、
  openfermion-95、dspy-1556、pandas-ai-1048、hypothesis-3813、meltano-732、
  tqdm-270、spikeinterface-1999、libcst-162、isort-222、pygame_gui-453、
  scikit-build-514、networkx-2936/4378、scikit-activeml-504、pystac-client-696、
  pyccel-1797、tatsu-121/265/268、weasyprint-1016、pynwb-994、keras-nlp-715、
  slack-1022、smart_open-735、openmdao-3256、pandas-58245/58250、pynwb-539、
  pydantic-1752、qiskit-8134、cogent3-1343、pandas-57570、pendulum-775/776、
  litellm-11964/12003/6563、cloudpathlib-437、jsonargparse-190、woodwork-685、
  hypothesis-3824、vyper-3030、tablib-584（同 579 家族，排除）、altair-3118、
  isort-911、qiskit-7408、xonsh-5744）：逐题看过 patch/problem/F2P，
  无 `is` 三元组或仅附带 copy → S。
- Gym 剩余 25 hits（pydantic-5861、dvc-2275、mypy×3、MONAI×6、pandas-50714、
  moto×5、hydra×2、modin-6775、bokeh-13041 等）：pandas-50714
 （`base is None` + view 隔离，机制真 F2）与 modin-6775（单 df `.copy()`，
  机制真 F2）因 C 编译/重依赖环境受限，本批不纳入 KEEP（诚实记环境待定，
  不计入 eligible）；其余无 `is` 三元组 → S；hydra-989、moto-7418 等为 R。

## REJECT → R（纯能力/特性/重构，代表性；本批复核确认）

- pyfar-58：新增 `copy()` 方法（之前无），F2P 只 `id !=` + 值相等，
  无共享改动语义 → R。
- gpytorch-782：新增 `__getitem__`（deepcopy 后切片），F2P 只值相等 → R。
- pynwb-994：新增 NWBFile `copy`（软拷贝链接），题面特性 → R。
- qiskit-8134（`clear`/`copy_empty_like`）、pyccel-2044/1797（dict/list
  `.copy()` 能力）、openmdao-3522 以外的 get_val？不，3522 为 KEEP（见上；
  虽题面含“Add ability”，但 base 返回 view 共享、修复隔离，属 repair）。
- spikeinterface-2775、skypilot-5577、pytask-200、bravado-212 等：重构/特性，
  copy 附带 → R。

## 双轮与口径注记

- K14（jinja）与 K15（ariadne）为本批最易双轮（纯 Python，小套件），优先重放。
- K17（openmdao-3522，110）用 task-site-packages-openmdao 复用 profile。
- K13（altair，P2P-EMPTY + `collections.Mapping` 时代漂移）与 K16（specutils，
  astropy 栈）如实判定，环境漂移不计三元组。
- 缺口：6（旧 eligible）+ 本批 eligible（待双轮确认）→ 仍缺 34−eligible；
  不启动训练、不改 S600 与旧文件。
