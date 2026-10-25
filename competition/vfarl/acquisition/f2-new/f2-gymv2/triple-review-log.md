# f2-gymv2 三元组复核 log（2026-10-25，第三轮）

口径：每题必须同时满足 `is` 身份断言 + 双向（共享修改）可见 + 访问/复制时点。
只有 `==` 无 `is` → 退 S；只有能力识别（无引用/所有权语义）→ 退 R。
套件 ≤150（Gym/V2 真题放宽口径）；坏路径（`..`）/禁仓/四仓已在扫描层过滤。
P2P=0 的任务标 P2P-EMPTY（JUnit 双条件风险，replay 时如实判定）。
 pools：SWE-Gym/SWE-Gym rev `bb94ed9e…`（train 2438）+ nebius/SWE-rebench-V2
rev `10483de0…`（train python 子集 7243）。Smith-py 不再扫。

## 扫描召回

- Gym strict changed-line copy 扫描：2438 行 → 29 hits（≤150，F2P 非空，白名单题面，非四仓，无坏路径）。
- V2-python strict 扫描：7243 行 → 112 hits（同条件）。
- 补充召回（test 侧 `is` + 身份相关 F2P 名，无 patch copy 信号）：Gym 34、V2 121，几乎全为噪声（详见下）。

## KEEP（真 F2，三元组完整，skip 预检 CLEAN）

### K1 `openmdao__openmdao-2810`（V2，total=2，F2P=2，P2P=0 ⚠P2P-EMPTY）
- `_ReprClass.__copy__/__deepcopy__` 返回 self。题面逐字：`'is' and 'is not' comparisons
  to them shouldn't break after a deepcopy`；测试文件为 test_patch 新建，
  `assertTrue(cp is _UNDEFINED)` / `assertTrue(cpf.bar is _UNDEFINED)`。
- 三元组：copy 后 `is` 单例（身份）+ 单例语义全局可见（双向）+ copy/deepcopy 调用时点。
  方向：copy 保持身份（identity-preserving polarity）。
- 家族：`OpenMDAO/OpenMDAO::openmdao.core.constants._ReprClass::copy_preserves_singleton_identity`；
  快照 `OpenMDAO/OpenMDAO::0bfb0eef…`。与禁用集零重叠（程序化校验见 dedup-evidence.json）。

### K2 `pandas-dev__pandas-55314`（Gym，total=34，F2P=1，P2P=33）
- `__finalize__` 中 `self.attrs[name] = other.attrs[name]`（浅拷贝共享嵌套对象）
  → `self.attrs = deepcopy(other.attrs)`。题面含 `assert df.attrs['transform'] is not
  df2.attrs['transform']`；F2P `test_attrs_deepcopy` 含同断言
  （`result.attrs["tags"] is not df.attrs["tags"]`）。
- 三元组：`is not`（隔离）+ clone 侧改动不污染原侧（双向）+ finalize/复制时点。
  方向：repair 建立隔离。replay 风险：workspace pandas 需 C 编译（见 README）。

### K3 `Project-MONAI__MONAI-2104`（Gym，total=12，F2P=4，P2P=8）
- 题面逐字：`it uses shallow copy for meta_data and other information, then if we change
  some value in a sample, all the samples are changed. Need to use deepcopy`。
  修复：`results[i][key] = deepcopy(data[key])`（RandWeightedCropd + RandCropByPosNegLabeld）。
- 三元组：`results[i][key] is not data[key]` 且样本间互不共享（身份）+
  改一个样本不污染其他/原侧（双向）+ `__call__` 时点。

### K4 `Project-MONAI__MONAI-2010`（Gym，total=3，F2P=1，P2P=2）
- 同文件 `croppad/dictionary.py` 不同类（RandSpatialCropSamplesd.__call__）：
  `d = dict(data)` 移入循环 + 未改动 key 逐样本 `deepcopy`。F2P `test_deep_copy`。
- 三元组同 K3（`sample[i][k] is not sample[j][k]` + 改动隔离 + `__call__` 时点）。
- 与 K3 家族不同（生产符号不同），快照不同（5cbcaf6… vs 41488339…）。

### K5 `zauberzeug__nicegui-3046`（V2，total=23，F2P=2，P2P=21）
- 为 ObservableCollection 新增 `__copy__`/`__deepcopy__`：默认 deepcopy 绕过
  `_observe` 接线导致数据丢失。F2P `test_copy` + `test_deepcopy`。
- 三元组：copy 为新对象但保留 `_parent` 接线（`copied._parent is orig._parent`，身份）+
  嵌套 deepcopy 隔离 / shallow 共享按 copy 种类（双向）+ copy 调用时点。

### K6 `stac-utils__pystac-826`（V2，total=29，F2P=1，P2P=28）
- `Asset.clone()` 浅拷贝：`mask.extra_fields["abc"] = True` 污染原 asset。
  修复 `extra_fields=deepcopy(self.extra_fields)`。F2P `test_clone`。
- 三元组：`mask.extra_fields is not asset.extra_fields`（身份）+
  clone 侧改动不污染原侧（双向）+ `clone()` 时点。

### K7 `stac-utils__pystac-794`（V2，total=80，F2P=2，P2P=78）
- `Collection.full_copy/clone` 不 deepcopy：`b.extra_fields.pop("test")` 污染 `a`。
  修复 catalog/collection/summaries 多处 copy/deepcopy + 新增 `Summaries.clone()`。
  F2P `test_clone_cant_mutate_original` + `test_clone_summary`。
- 三元组同 K6（`is not` + 改动隔离 + clone 时点）。与 K6 生产符号不同
  （Catalog/Collection/Summaries vs Asset），快照不同（a6c8927d… vs 0f308cdd…）。

### K8 `ecmwf__earthkit-data-671`（V2，total=30，F2P=2，P2P=28）
- 题面逐字：`earthkit.data modifies engine kwargs in place, causing unexpected side
  effects`；测试 `test_xr_engine_kwargs_unchanged` 断言调用者字典不变。
  修复：`copy.deepcopy(user_xarray_open_dataset_kwargs)`。
- 三元组：内部工作字典 `is not` 调用者字典（身份）+ 内部改动调用者不可见（双向）+
  `to_xarray` 调用时点。

### K9 `jazzband__tablib-579`（V2，total=136，F2P=2，P2P=134）
- 题面逐字：`Output formatter corrupts original data`（重复 export 结果递减）。
  根因 `_data = list(self._data)` 浅拷贝共享 Row；修复新增 `Row.copy()` +
  `row = row.copy()  # To not mutate internal data structure`。
- 三元组：格式化用行 `is not` 内部行（身份）+ export 不改动内部（双向）+ export 时点。
- 同家族 `tablib-584`（`Row.copy` 同一新增）只取其一：取 579
  （题面即 formatter 腐蚀，F2P 含 test_formatters），584 排除（同家族）。

### K10 `openmdao__openmdao-3125`（V2，total=2，F2P=1，P2P=1）
- 派生类注册 hook 会 wipe 基类已注册 hooks。修复新增 `_HookMeta.copy()`：
  copy any hooks with inst_id of None so we can modify them without modifying
  the original。F2P `test_inherited_class_hooks`（test_patch 新增）。
- 三元组：派生 hook `is not` 基 hook（同 stamp 保序，身份）+
  派生侧 deactivate 不影响基侧（双向）+ 继承/装饰时点。

### K11 `openmdao__openmdao-3461`（V2，total=89，F2P=1，P2P=88）
- 题面逐字：If list_driver_vars is called twice with driver_scaling=False, the upper
  and lower bounds are changed。读路径原地改动内部向量；修复
  desvars = deepcopy(desvars); cons = deepcopy(cons)。
  F2P `test_list_driver_vars_bounds_bug`。
- 三元组：工作拷贝 `is not` 内部向量（身份）+ 重复读取不漂移内部 bounds（双向）+
  list_driver_vars 调用时点。

### K12 `django__asgiref-523`（V2，total=15，F2P=1，P2P=14）
- `_CVar` 改写：逐 key ContextVar 改为单 dict ContextVar + 写/删 copy-on-write。
  F2P `test_deletion`（子 task 内 del 不得泄漏到父）。
- 三元组：每次写安装新 dict（`is not` 上一个，身份）+ 旧快照不受后写影响（双向）+
  set/del 时点。

## 备份与降级

- B1 `getmoto__moto-6716`（Gym，total=149）：reset/初始化隔离；测试只断言值。备份。
- B2 `meltano__sdk-2144`（V2，total=4）：子 context 隔离；测试为行为。备份。
- B3 `idsia__sacred-765`（V2，total=21）：`is` 腿弱。备份-弱。
- 其余备份-弱：`moto-7960/7961`、`weasyprint-1016`、`specutils-1119`、
  `mypy-12023`、`modin-6775`（环境重）。
- `cookiecutter-967` 复核后降级：`deepcopy(default)` 非因果，F2P 只测值
  相等 → REJECT（S）。

## REJECT → S（仅值语义/附带 copy，无 is，代表性）

- Gym：dvc-2275、mypy-11345、mypy-12199、MONAI-4729/3001/3373/7180/4689/4709、
  moto-4886/5725/7085/7153、hydra-1450/1401、bokeh-13041、pandas-50714、
  pydantic-5861、pandas-47900（_id 令牌无共享改动语义）、dask-10211、
  MONAI-4972/5950/6429。
- V2：cfn-lint-3342/2031/4045、trestle-1695、skypilot-5577、altair-518、
  dvc-10174、pymc-5634、glyphslib-563、moto-8342/8423/5960/7434、
  bravado-212、pypika-306、bibliopixel x5、trackintel-384/148、pydeseq2-268、
  qiskit-8790、planet-588、pip-tools-837、scikit-build-703/514、vyper-4677、
  pynwb-557/539、polyfactory-468、dynaconf-375、feature_engine-500、
  fastmcp-1214、openfermion-95、pandas-ai-1048、hypothesis-3813、meltano-732、
  tqdm-270、spikeinterface-1999、libcst-162、isort-222/911、pygame_gui-453、
  networkx-2936/4378、skactiveml-504、pystac-client-696、pystac-885/1049/1522、
  openmdao-3256、pandas-58245/58250/57570、litellm-11964/12003、slack-1022、
  pendulum-775、tablib-584（同 579 家族）、structlog-331（P2P=0+特性）。

## REJECT → R（纯能力/特性/重构，代表性）

- Gym：moto-7418、hydra-989。
- V2：pyfar-58、pyccel-2044/1797、gpytorch-782、pymovements-1108、pytask-200、
  jinja-2026、dspy-1556、qiskit-8134、pynwb-994、tatsu-121/265/268、qiskit-7408、
  xonsh-5744、altair-3118、cogent3-1343、openmdao-3522、hypothesis-3824。
- 补充召回 Gym-34/V2-121 多数为命名噪声；仅 pandas-47900、bokeh-13443、
  dask-10211 值得一看，复核后皆非 F2。

## 双轮与口径注记

- K10/K7/K9/K12/K1 均 eligible（双轮 F/T，normalize 无理由）。
- K11（openmdao-3461）near-miss：F2P 正确翻转，但 P2P
  test_duplicate_problem_name 在双相以同一 reports-warning 未触发而失败
  （scipy 1.11.4/1.14.1 两套 profile 一致，环境漂移，与修复无关）。
- K6 run-001 曾因 jsonschema 4.26（venv）与 pystac@2023 的 validation 行为漂移
  导致 2 个 P2P 在双相失败；换 era-appropriate jsonschema==4.17.3 task profile
  后 run-002 双相干净（测试未改、matcher 未动，属环境供给）。
- P2P=0（K1）：normalize 接受空 P2P（eligible true）。已如实记录。
- pandas-55314、MONAI-2104/2010、earthkit-671、nicegui-3046、openmdao-3461
  待重放（环境/编译成本见 README 缺口说明）。
