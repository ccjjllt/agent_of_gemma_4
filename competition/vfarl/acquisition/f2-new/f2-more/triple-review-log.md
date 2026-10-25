# f2-more 三元组复核 log（2026-10-25，第九轮：小套件 ≤150 inline-is 补筛）

口径：每题必须同时满足 `is` 身份断言 + 双向（共享修改）可见 + 访问/复制时点。
只有 `==` 无 `is` → 退 S；只有能力识别（无引用/所有权语义）→ 退 R。
inline-is 型：F2P 仅值断言，`is` 腿由 source-bound inline 观察承载
（同 C2/K15/nox-439/aiohttp-10672/qiskit-12620 口径）。
Gym/V2 真题 total≤150（含 150；151–200 已由 f2-recall/f2-inline 覆盖）；
strict COPY_SIG changed-line 行已由 f2-gymv2/b/c 覆盖，本批只收 BROAD-inline
（slice/`list()/dict()/set()`/`clone`/CIMultiDict/clear_cache/copyInto/`[:]`）
changed-line 信号且排除 strict-copy 行（新地面）。
坏路径（`..`）/禁仓/四仓已在扫描层过滤。Smith-py 不扫（冻结）。
去重：只读 common-400 + F1-40 + F5旧40 + f2-round1/2 + f2-gymv2 16 +
f2-gymv2b 5 + f2-gymv2c 2 + f2-recall 3 + f2-inline 1
（程序化校验 548i/111s/494f；2 KEEP snapshot_collision 全无，
instance 全新；同家族复本 1 记 COLLIDE-family 不计入）。

## 扫描召回（f2-more，forbidden 含 23+3+1 keeps）

- Gym small-inline（total≤150，inline-changed 且非 strict-copy）：245 行；
  V2-python small-inline：1159 行；合计 1404 行。
- 分层：tier0（strict-is ∧ F2P名）gym 16 + v2 88 = 104；
  tier1（strict-is 仅）gym 59 + v2 431 = 490；
  tier2（F2P名 仅）gym 7 + v2 70 = 77；tier3（inline 仅）gym 163 + v2 570 = 733。
- 强 inline 子集（clone/CIMultiDict/clear_cache/copyInto/`[:]`/`__copy__`/
  deepcopy/`.copy(`，排除纯 `list()/dict()/set()/tuple()` 构造噪声）：
  gym 43 + v2 约 120。逐题深读策略：tier0 强 inline 全读 +
  tier0 其余抽读（inline 噪声判定）+ tier1 强 inline（pluggy/pydap/werkzeug/
  qiskit-scheduling 等）全读 + gym(0,1) F2P-copy 名全读（MONAI-3662 等）；
  tier3（733 行，无 is 无 F2P 名）按“无 is 不投”记 SCREENED-OUT，
  清单在 `*-more-scan.json` 内可复现（`inline_n/test_is_n/f2p_name_hits` 分层）。

## KEEP（真 F2，三元组完整，待 skip 预检+双轮）：2

### M1 `stfc__psyclone-3029`（V2，total=41，F2P=10，P2P=31）

- 生产：`RegionTrans.get_node_list([Schedule])`（单 Schedule 的 list）
  旧行为走通用 list 分支返回 `[sched]`（包着 Schedule 本身）；
  修复与单 Schedule 同处理，返回 `nodes[0].children[:]`
  （子节点 list 的切片拷贝；`src/psyclone/psyir/transformations/region_trans.py`）。
- F2P `test_get_node_list` 新断言：`len(node_list3)==1` +
  `node_list3[0] is sched.children[0]`（元素身份）；
  旧有 `assert node_list2 is not node_list`（list 本身为拷贝）守卫复制契约。
  基线：旧代码返回 `[sched]`，`node_list3[0] is sched.children[0]` 失败。
- 三元组：`lst is not sched.children`（新 list；修前为包 Schedule 的错 list，
  修后为 children 切片拷贝）+ 双向隔离（重排/增删返回 list 不扰动树；
  元素浅共享 by design：`lst[0] is sched.children[0]`）+
  `get_node_list()` 调用时点。极性：隔离（新 list）+ 元素保身份
  （同 tablib-579 Row.copy / pystac-clone 族：容器隔离、元素共享）。
- 家族：`stfc/PSyclone::psyclone.psyir.transformations.region_trans.RegionTrans.get_node_list::single_schedule_children_copy`；
  快照 `stfc/PSyclone::70a0a9c3f0b039f5f2dc8c0c77c4f7209937772f`。零重叠
  （程序化校验通过；psyclone 既有 R/S 均为他家族）。
- P2P 跨 2 文件（region_trans_test + transformations_test 系），file-scope
  按实际 targets 执行；`fparser` 等期依赖经 task-site-packages 供给。

### M2 `qiskit__qiskit-terra-5778`（V2，total=19，F2P=2，P2P=17）

- 生产：ALAP/ASAP 调度 `new_node.op.duration = duration` 直接改共享
  `node.op` 实例（同门 CX 复用同一 op，后写的 duration 覆盖先写的；
  题面：CX_{0,1} 被标成 CX_{1,2} 的 2560dt 而非 1728dt）。
  修复每节点 `new_op = node.op.copy()`（`# need different op instance
  to store duration`），再写 duration（`qiskit/transpiler/passes/scheduling/
  alap.py + asap.py`）。
- F2P `test_duration_on_same_instruction_instance_{1_asap,2_alap}` 为值断言
  （各门 duration 正确）；`is` 腿由 source-bound inline 观察承载：
  修前 `new_node.op is node.op`（同一实例，写 duration 互相覆盖），
  修后 `new_op is not node.op` 且改 scheduled 侧 duration 原电路侧不动。
- 三元组：`new_op is not node.op`（每指令独立副本）+
  双向隔离（scheduled 电路写 duration 不污染原电路 op；同 op 多节点互不覆盖）+
  调度 pass 写时点。极性：隔离（同 K13/nox-439/aiohttp-10672/qiskit-12620 族）。
- 家族：`Qiskit/qiskit-terra::qiskit.transpiler.passes.scheduling.ASAPSchedule+ALAPSchedule::per_instruction_op_copy_duration_isolation`；
  快照 `Qiskit/qiskit-terra::62b521316e62…`（repo 键 `qiskit-terra` 与
  f2-inline `Qiskit/qiskit::2b2be030…` 不同键，快照/家族均全新）。
  零重叠（程序化校验通过）。
- 风险：qiskit-terra 该年代版（base 62b52131）双轮依赖（mock FakeMontreal 等）
  按 task-python + period profile 实际供给；目指 run-002 式干净双轮，
  按实际记 eligible 或 pending/blocked，不放宽。

## COLLIDE-family（同家族复本，不计入）：1

- `openmdao__openmdao-3406`（V2，88，F2P=1）：`Problem.list_driver_vars`
  内 `desvars = desvars.copy(); cons = cons.copy()` 后重算 bounds 显示，
  与 K11（openmdao-3461，同函数、同 `desvars/cons` 拷贝隔离、
  同 `read_path_copy_isolation` 家族 `OpenMDAO/OpenMDAO::list_driver_vars::
  read_path_copy_isolation`）逐字同家族。F2P
  `test_list_driver_vars_driver_scaling_bounds` vs K11
  `test_list_driver_vars_bounds_bug`（同文件 `test_problem.py`）。
  K11 系 triple-KEEP（虽 normalize near-miss，仍占家族），按协议只计其一，
  记 COLLIDE-family，不计入。

## REJECT → S（仅值语义/附带 copy 或 `is` 为空检查/散文/更名一致性，无差异 `is` 腿）

- `pycqa__flake8-1576`（V2，56，F55）：decision-engine 默认值 None 化重构；
  `is` 为 enum/None 身份（`decision_for(...) is expected`、`ignore is None`），
  非 copy 隔离；`.copy()` 删除系共享-by-design。→ S。
- `matchms__matchms-686`（V2，129，F129）：None 谱 export 跳过；
  `spectrum is not spectrum_in` 系 filter 纯度断言（恒新对象），
  因果为 `filter_empty_spectra` + losses 存取重构，非别名隔离修复。→ S。
- `stfc__psyclone-3029` 之外 psyclone-2124/2362（V2）：`is not` 为子句身份/
  SAVE 名集合，inline 为类型注解/`set()` 噪声，无 copy 因果。→ S。
- `qiskit__qiskit-terra-8381`（V2，38，F1）：optimizer options 默认值覆盖值 bug；
  `is` 为 docstring 散文；`.copy()` 为值合并附带。→ S。
- `jupyter__nbconvert-1981`（V2，66，F66）：exporter default_config 合并顺序值 bug；
  `c2 = super().default_config.copy()` 系防污染附带，无 `is` 腿（散文）。→ S。
- `pydap__pydap-481`（V2，69，F3）：dims/dimensions 更名一致性；
  copy 构造 `dims[:]/attributes.copy()` 系更名前已存（非修复因果），
  F2P 因新 `dims=` 签名失败；`is` 仅散文误报。→ S。
- `pytest-dev__pluggy-348`（V2，57，F2）：subset-hook-caller 内存泄漏，
  修复改快照为 live proxy（去隔离 polarity），`is` 仅 `is not None` 空检查。→ S。
- `pytest-dev__pluggy-473`（V2，37，F2）：hook 调用期注册迭代快照
  `self._hookimpls.copy()`，无 `is` 腿（docstring 散文）。→ S（机制真但无腿）。
- `pyomo__pyomo-3394`（V2，119，F5）：ConfigDict 默认值延迟特性；
  NOTSET 单例移动系附带。→ R（兼特性，记 R；本节 S/R 按主因）。
  更正：pyomo-3394 题面为 Enhancement → R。
- `hdmf-dev__hdmf-694`（V2，94，F11）：LinkSpec 无名 AttributeError；
  `assertIs` 系 spec 身份，inline `dict()` 噪声。→ S。
- `iterative__dvc-4246`（Gym，8，F2）：dvc get 浅克隆优化（git 域 clone），
  非引用隔离。→ S。
- `getmoto__moto-7270/7271`（V2，26，F11）：secretsmanager 新 key-filter 特性；
  `.copy()` 系遍历安全附带。→ R（兼特性）。
- `getmoto__moto-4787/5576/7177`（Gym/V2）：dynamodb/lambda/identitystore 值 bug；
  `set()/dict()` 构造噪声。→ S。
- `dask__dask-8316`（Gym，70，F2）：`is HighLevelGraph` 类型身份，
  `cull(set(...))` 噪声。→ S。
- `bokeh__bokeh-13443`（Gym，113，F2）：`clone()` 仅签名改名（HasProps→Self），
  `is` 系错误文案。→ S。
- `conan-io__conan-11330`（Gym，11，F2）：copy_symlink_folders 参数特性。→ R。
- 其余 bulk S（约 700 行 tier3 + tier1 构造噪声）：patch 无 OWN 拥有信号
  （`list()/dict()/set()/tuple()` 构造、`set` 差集、`sorted` 等），
  problem 无共享语义，`is` 仅散文/`is None`/`isinstance`/`is not None`，
  按“无 is 不投”记 SCREENED-OUT，清单在 `*-more-scan.json` 内可复现。

## REJECT → R（纯能力/特性/重构，无引用语义；代表性全列）

- `pyomo__pyomo-2231`（V2，16，F8）：`evaluate_fgh` API 新增（题面逐字 Feature），
  `test_clone`/`assertIsNot` 系附带。→ R。
- `maniacallabs__bibliopixel-759`（V2，19）：Limiter 动画新类（题面 Feature），
  `.clone()/[:]` 系新功能内部。→ R。
- `fatiando__verde-44`（V2，7，F7）：Chain 类新增（题面 Feature）。→ R。
- `pyjanitor-devs__pyjanitor-1451`（V2，3，F3）：alias 新函数（题面 Feature）。→ R。
- `Project-MONAI__MONAI-3662`（Gym，8，F2）：CopyItemsd 默认 times 特性。→ R。
- `pallets__werkzeug-2979`（V2，119，F9）：`|`/`|=` 新算子（题面逐字 Implement）。→ R。
- `pypa__pip-12991`（V2，110，F110）：断点续传特性。→ R。
- `pre-commit__pre-commit-616`（V2，118，F118）：fail-fast 特性。→ R。
- `enthought__traits-1384`（V2，3，F3）：NoDefaultSpecified 公开 API（docs+import）。→ R。
- `dask__dask-8253`（Gym，72，F42）：ufunc `where` 参数特性（题面逐字 Feature）。→ R。
- `conan-io__conan-16038/15514`（Gym，30/28）：git clone hide-url 特性。→ R。
- 其余 bulk R（约 60 行）：题面逐字 Feature/Add support/Provide a way +
  patch 新能力（api/参数/重构），无引用/所有权语义，清单同上可复现。

## 双轮与口径注记

- skip 预检（`skip-precheck.json`）：2 triple-KEEP 中 CLEAN 才投双轮
  （JUnit 双条件、六工具、白名单题面、不改测试、保护路径、`replay/.venv`）。
- M1/M2 双轮按 `run_official.py`（baseline 失败 + repaired 通过）执行；
  file-scope（M1：region_trans_test + transformations fingers；M2：scheduled_circuit），
  原 matcher 不动，不断言缩水。重依赖按实际记录，不自行放宽
  （nox-editable、aiohttp-flaky 两 near-miss 维持严格口径，只记录不放宽）。
- 全局：11/40 eligible（6+2+2+1；M1/M2 双轮前不计数）；缺口 29。
  （M2 eligible 后全局 12/40，缺口 28，见 `normalize-summary.json`。）
- 缺口清零前不启动训练、不改 S600 与旧文件。
