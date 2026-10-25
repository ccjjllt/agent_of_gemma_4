# f2-push3 三元组复核 log（2026-10-25，第十二轮：mid-suite 151-400 + P2P-empty inline-is，push2 余量）

口径：每题必须同时满足 `is` 身份断言 + 双向（共享修改）可见 + 访问/复制时点。
只有 `==` 无 `is` → 退 S；只有能力识别（无引用/所有权语义）→ 退 R。
inline-is 型：F2P 仅值断言，`is` 腿由 source-bound inline 观察承载
（同 C2/K15/nox-439/aiohttp-10672/qiskit-12620/M2/P1-P4/Q2-Q3 口径）。
Gym/V2 真题：A池 total 151-400（含两端）BROAD-inline strong（strict-copy 排除，
f2-inline 200 cap 之外）+ B池 P2P-empty total≤50 inline-strong
（push要求total≥2之外、C2 total=1先例）。
坏路径（`..`）/禁仓/四仓已在扫描层过滤。Smith-py 不扫（冻结）。
去重：只读 common-400 + F1-40 + F5旧40 + f2-round1/2 + f2-gymv2 16 +
f2-gymv2b 5 + f2-gymv2c 2 + f2-recall 3 + f2-inline 1 + f2-more 2 + f2-push 6 +
f2-push2 3（程序化校验 559i/122s/505f；2 KEEP snapshot_collision 全无，
instance 全新）。

## 扫描召回（f2-push3，forbidden 含 16+5+2+3+1+2+6+3 keeps）

- Gym mid-inline A(151-400 strong)：11 行；V2-python mid-inline A：36 行；
  P2P-empty B(≤50 strong)：gym 7 + v2 61 = 68。扫描合计 115。
- prior-named（旧日志已逐题具名 280，含 push2）排除后：A 38 + B 36 = 74 行
  （`push3-candidates.json`；复核工作文件 `push3-review-dump.txt` 74 题
  problem+patch+testpatch、`push3-triage.txt` 紧凑版）。
- push2 已 verdict 35 行（KEEP 3/COLLIDE 1/深读 REJECT-HELD-抽查 31）不再重审；
  本批深读剩余 fresh 39 行全部（逐题 verdict 下列），无 SCREENED 余量。

## KEEP（真 F2，三元组完整，skip CLEAN，已双轮 eligible）：2

### R1 `aeye-lab__pymovements-486`（V2，total=370，F2P=25，P2P=345，双文件 dataset_files.py+gaze_dataframe.py）

- 生产：`save_preprocessed()` csv 路径。旧行为循环变量 `gaze_df` 直接别名
  caller 的 `GazeDataFrame`，随后 `gaze_df.unnest('pixel'/'position'/'velocity'/
  'acceleration')` + `gaze_df.frame.drop(column)` 全部就地改 caller frame；
  修复循环顶 `gaze_df = gaze_df.copy()`（另 `GazeDataFrame.copy()` 补
  `gaze.n_components = self.n_components`，同属本修复，unnest 依赖它）。
- F2P `test_save_preprocessed_has_no_side_effect[...]` ×25：题面逐字——save 后
  `assert_frame_equal(old_frame, new_frame)`（值断言；`is` 腿由inline承载：
  working copy `is not` caller 对象）。
- 三元组：working `gaze_df is not` caller `GazeDataFrame`（修前同一对象别名）
  + 双向隔离（改 working 侧——unnest/drop——caller frame 不动；F2P逐字）+
  `save_preprocessed()` per-file 循环时点。极性：迭代-修改隔离（同Q2/MORE-Q3）。
- 家族：`aeye-lab/pymovements::pymovements.dataset.dataset_files.
  save_preprocessed::gaze_copy_isolation`；快照 `aeye-lab/pymovements::
  eed48eb3cd91…`。零重叠（程序化校验通过）。
- 双轮：task-python 3.10.20 + 自建 `task-site-packages-pymovements/`
  （polars 0.18.15/numpy 1.25.2/pandas 2.0.3/pyarrow 11.0.0/scipy/matplotlib/
  tqdm/pytest 8.3.3，经 uv 取数）+ `--source-layout src`，run-001 干净双轮
  （baseline 25 F2P 全失败 / repaired 370 全过）。

### R2 `jupyter__kernel_gateway-118`（V2，total=15，F2P=15，P2P=0，四文件 gatewayapp+activity handlers/manager+kernels handlers）

- 生产：`ActivityManager.get_map_for_kernel()`。旧行为未知 kernel 返回单例
  `self.dummy_map`（同一对象！）；修复未知 kernel 返回
  `DEFAULT_ACTIVITY_VALUES.copy()`、已跟踪 kernel 各持自己的
  `DEFAULT_ACTIVITY_VALUES.copy()`（另 gatewayapp 单例装配 + handlers 经
  settings 取 manager + kernels handlers 同步改 import，同属本修复；
  P2 click-2312 先例：重构内嵌隔离仍计 F2）。
- F2P `TestActivityManager` ×15（`test_remove_should_not_allow_new_activities_
  for_a_kernel` 等；值断言；`is` 腿由inline承载：per-kernel map
  `is not` 他 kernel map/`DEFAULT_ACTIVITY_VALUES`）。
- 三元组：per-kernel map `is not` 他 map/全局 defaults（修前未知 kernel 同一
  `dummy_map` 别名）+ 双向隔离（publish 到一 kernel，defaults/他 kernel 不动；
  F2P逐字）+ `get_map_for_kernel()` 访问时点。极性：隔离。
  基线 inline 失败模式诚实注记：`DEFAULT_ACTIVITY_VALUES` 符号不存在
  （ImportError）+ 旧构造器不收 kernel_manager（TypeError）——隔离 API 基线缺席，
  非策略错误。
- 家族：`jupyter/kernel_gateway::kernel_gateway.services.activity.manager.
  ActivityManager.get_map_for_kernel::per_kernel_copy_isolation`；快照
  `jupyter/kernel_gateway::a5379b43ebe6…`。零重叠（程序化校验通过）。
- 双轮：task-python 3.10.20 + 自建 `task-site-packages-kernelgw/`（traitlets/
  pyzmq 26.1.0/nbformat/jupyter_core/jupyter_client 7.4.9/notebook 6.5.6
  --no-deps/ipython_genutils/tornado 6.4.1/pytest 8.3.3 等真实包，经 uv 取数；
  pyzmq 24 sdist 本机编译失败故钉 26.1.0，notebook 6.5.6 供 `notebook.utils`，
  无 shim）run-002 干净双轮（baseline 15 全失败 / repaired 15 全过）；
  run-001 为诊断轮（prepare 漏 `kernels/handlers.py` 第四个生产文件，
  repaired inline ImportError，已保留为证据；prepare 已修，复核断言已加固）。

## REJECTS（已深读 patch-evidence，不充数）

- R-新能力：`pandas-dev__pandas-56059`（全新 `Series.case_when` API，无基线
  别名腿）、`pybamm-team__pybamm-4621`（lazy summary-variables 特性 11 文件）、
  `pennylaneai__pennylane-1884`（全新 Hessian transform）、`joke2k__faker-1887`
  （新 elector_code provider）、`fatiando__harmonica-316`（新 tesseroid-layer
  特性）、`modflowpy__flopy-2271`（网格顶点值修复 `top[:]=`）、`qiskit__qiskit-
  12370`（docstring）、`qiskit__qiskit-10786`（GateDirection 特性 9 文件）、
  `ethereum__web3.py-3622`（新 eth_simulateV1 端点）、`audreyr__cookiecutter-792`
  （log-format 特性）、`nerdvegas__rez-1061`（pkg-move 特性）、`oasis-open__cti-
  stix-validator-57`（CLI 重构）、`huggingface__huggingface_hub-1040`
  （push_to_hub HTTP 特性）、`neurodatawithoutborders__pynwb-1865`
  （LinkBuilder 能力）、`Project-MONAI__MONAI-3532`（新 copy_cache/progress
  参数 + torch）→ R。
- R/S-新逻辑 hygiene：`robotframework__seleniumlibrary-1039`（新
  WebDriverCreator 内 capabilities 传递）、`sappelhoff__pyprep-34`（保留非 EEG
  通道特性内 `raw.copy()`）、`ericmjl__pyjanitor-770`（新 pivot 特性内
  `df.copy()`）、`openmdao__openmdao-2780`（linter 新 AST 逻辑）、
  `pre-commit__pre-commit-1256`（integrity-check 特性内 config.copy）、
  `agnostiqhq__covalent-1620`（CloudResourceManager 特性内 environ.copy）、
  `zigpy__zha-device-handlers-2629`（新设备 quirk attributes.copy）、
  `pymc-devs__pytensor-1396`（新图优化内 fgraph clone hygiene）→ S/R。
- S-值/语义：`python__mypy-10382`（`__all__` 值传播，无身份腿）、
  `natcap__invest-1791`（计算公式值修复）、`geopandas__geopandas-3395`
  （overlay make_valid 值修复）、`qiskit__qiskit-11206`（barrier 去重重构，
  copy 前后皆新 list）、`qiskit__qiskit-12369`（plot 颜色值修复）、
  `pymc-devs__pytensor-1414`（BLAS NaN 初始化语义 13 文件）、
  `fatiando__harmonica-419`（内存优化，copy 行无变化）、
  `narwhals-dev__narwhals-2453`（repr 分发名录）、`aio-libs__aiohttp-9071`
  （yarl extend_query 性能重构，copy 在被删分支）→ S。
- R-文档/噪声：`enthought__traits-1058`（新异常栈文件 + 文档）、
  `lektor__lektor-1143`（werkzeug 兼容重构）、`stfc__fparser-408`
  （DO CONCURRENT 新语言特性）、`rmarkello__abagen-140`（新 RNAseq fetchers +
  需网络）、`tomwhite__cubed-346`（新 fusion 特性 127+ 行）、
  `streamlink/keras-nlp/benchopt` 系文档 clone 噪声（push2 已 Decisions，
  本批未重列）→ R/噪声。

## 家族/快照状态（push3 eligible：2）

- 快照（全在122禁外+prior新）：pymovements eed48eb3cd91…、kernel_gateway
  a5379b43ebe6…。One per canon::sym；同家族复本无。
- 缺口：全局18/40 + push3 2 eligible = 20/40，缺口 20。缺口清零前不启动训练、
  不改S600与f1/f3/f4文件；全部新文件在`f2-push3/` +
  `replay/artifacts/f2-push3/` + `replay/f2-push3-*.json`。
