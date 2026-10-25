# f2-push 三元组复核 log（2026-10-25，第十轮：小套件 tier2/3-strong 补筛）

口径：每题必须同时满足 `is` 身份断言 + 双向（共享修改）可见 + 访问/复制时点。
只有 `==` 无 `is` → 退 S；只有能力识别（无引用/所有权语义）→ 退 R。
inline-is 型：F2P 仅值断言，`is` 腿由 source-bound inline 观察承载
（同 C2/K15/nox-439/aiohttp-10672/qiskit-12620/M2 口径）。
Gym/V2 真题 total≤150（含 150）；strict-copy 行已由 f2-gymv2/b/c 覆盖，
本批只收 tier2（F2P名+强inline，无test-is）/ tier3（仅强inline）且
strong_n>0 的行。坏路径（`..`）/禁仓/四仓已在扫描层过滤。Smith-py 不扫（冻结）。
去重：只读 common-400 + F1-40 + F5旧40 + f2-round1/2 + f2-gymv2 16 +
f2-gymv2b 5 + f2-gymv2c 2 + f2-recall 3 + f2-inline 1 + f2-more 2
（程序化校验 550i/113s/496f；6 KEEP snapshot_collision 全无，
instance 全新；同家族复本 2 记 COLLIDE-family 不计入；
pandas-56523 gym/v2 跨池同实例复本皆 S，记注不计入）。

## 方法注记（相对 f2-more 的协议修正）

- f2-more 将 tier3（733 行，无 test-is、无 F2P 名）按“无 is 不投”整批
  SCREENED-OUT。但 M2（qiskit-terra-5778，tier2，无 test-is）本身证明
  “无 TEST-is”≠“无 is 三元组”：`is` 腿可由 source-bound inline 观察承载。
  本批复核 tier2/3-strong 中此前未被逐题 verdict 的 128 行
  （f2-more 日志已逐题具名的 S/R/KEEP/COLLIDE 行排除，不重审）。
- 强信号定义：`.copy()/deepcopy/__copy__/clone/CIMultiDict/clear_cache/
  copyInto/[:]` 等真实代码信号；纯 `list()/dict()/set()/tuple()` 构造、
  `git clone` 文档、类型注释 `[:]` 计噪声（逐题人工剔除）。
- 严格性：修复对象必须是引用/所有权隔离缺陷。防御性 `.copy()` 若只为
  新逻辑 hygiene（无 caller-owned/shared 映射在险）→ S；
  新 API 本身（无 baseline 别名腿）→ S/R。本批据此降级 axes-1033、
  pip-tools-2082（皆 S）、todoman-197（S）。

## 扫描召回（f2-push，forbidden 含 23+3+1+2 keeps）

- Gym small-inline（total≤150，inline-changed 且非 strict-copy）：245 行；
  V2-python small-inline：1157 行；合计 1402 行。
- tier2（F2P名 仅）：gym 7 + v2 69 = 76；tier3（inline 仅）：gym 163 + v2 570 = 733。
- tier2/3-strong 且未被旧日志逐题具名：128 行（T2 13 + T3 115），即本批
  逐题复核全集（`push-candidates.json`；复核工作文件 `push-review-dump.txt`
  128 题 problem+patch+testpatch、`push-triage.txt` 紧凑版）。

## KEEP（真 F2，三元组完整，待 skip 预检+双轮）：6

### P1 `pyfar__pyfar-123`（V2，total=38，F2P=1，P2P=37，单文件 test_coordinates.py）

- 生产：`Coordinates.__getitem__` 切片。旧行为 `new = self`（别名！），
  切片直接改原对象 `_points`；修复 `new = self.copy()`
  （`pyfar/coordinates.py` 单 hunk；docstring 同步 mutable→copied）。
- F2P `test_getitem` 新增断言：切片后 `set_cart`，原 `cshape==(2,)` 且
  值全零（值断言；`is` 腿由 inline 承载）。
- 三元组：`new is not self`（修前 `is` 别名）+ 双向隔离（改切片侧原侧
  不动；F2P 逐字）+ `__getitem__` 调用时点。极性：隔离。
- 家族：`pyfar/pyfar::pyfar.coordinates.Coordinates.__getitem__::
  slice_copy_isolation`；快照 `pyfar/pyfar::1a2767c6fa60…`。零重叠
  （程序化校验通过）。

### P2 `pallets__click-2312`（V2，total=35，F2P=1，P2P=34，单文件 test_arguments.py）

- 生产：`argument()/option()` 装饰器工厂。旧行为内层 `decorator` 对共享
  闭包 `attrs` 做 `pop("cls")`（argument 直接改共享 dict；option 靠
  `attrs.copy()` 隔离）；复用同一装饰器第二次即错类。修复 `cls` 提升为
  显式形参，内层零 mutation（`src/click/decorators.py`；option 的旧
  `attrs.copy()` 随之移除）。
- F2P `test_when_argument_decorator_is_used_multiple_times_cls_is_preserved`
  为值断言（两命令皆 CustomArgument；`is` 腿由 inline 承载）。
- 三元组：两命令 params 互异（`foo.params[0] is not bar.params[0]`）且
  同为 CustomArgument + 共享 `attrs` 映射复用前后不变（修前第二次应用
  可见第一次的 pop 副作用）+ 装饰器应用时点。极性：共享-mutation 隔离。
- 家族：`pallets/click::click.decorators.argument+option::
  reusable_decorator_shared_attrs_isolation`；快照
  `pallets/click::c77966e092ed…`（round1 禁 `fde47b4b` 外新快照；
  click-1679 同仓不同家族/快照，见 S）。零重叠（程序化校验通过）。

### P3 `aio-libs__aiohttp-8997`（V2，total=95，F2P=2，P2P=93，单文件 test_web_request.py）

- 生产：`BaseRequest.clone(scheme/host)`。旧行为只刷 `_cache` 不重建
  `url`，克隆体 `url` 沿用原 scheme/host；修复按新 scheme/host
  `with_scheme/with_host` 重建 url 并同步 `_cache`
  （`aiohttp/web_request.py` 主 hunk；其余为 ClientSession 文档/签名
  噪声，非生产）。
- F2P `test_clone_absolute_scheme/host` 为值断言（`req2.url.scheme/host`
  正确；`is` 腿由 inline 承载）。
- 三元组：`req2 is not req` 且 `req2.url is not req.url` + 双向隔离
  （改克隆侧 scheme/host，原请求 url 不动；同 R2 headers 族思路，
  不同生产符号）+ `clone()` 调用时点。极性：隔离。
- 家族：`aio-libs/aiohttp::aiohttp.web_request.BaseRequest.__init__+clone::
  absolute_url_cache_isolation`；快照 `aio-libs/aiohttp::ed5576dba118…`
  （R2 为 `web_response.Response.headers` 家族，不同符号，不碰撞）。
  8996/8990 同 F2P 同 patch 系同家族复本，记 COLLIDE-family（见下）。
  零重叠（程序化校验通过）。

### P4 `hypothesisworks__hypothesis-2435`（V2，total=20，F2P=1，P2P=19，单文件 test_cache_implementation.py）

- 生产：`cached_strategy` 全局 `STRATEGY_CACHE` 多线程竞态；修复
  `threading.local()` 每线程独立 `LRUReusedCache` + `get_cache/
  clear_cache`（`hypothesis/strategies/_internal/core.py` 主 hunk；
  同 asgiref-523 上下文 copy-on-write 族思路）。
- F2P `test_cache_is_threadsafe_issue_2433_regression`（4 线程×1000 builds
  零 errors；`is` 腿由 inline 承载）。
- 三元组：线程 A 缓存 `is not` 线程 B 缓存（修前全局同一对象 `is` 共享）
  + 跨线程条目互不可见 + 策略构造（访问）时点。极性：按 owner 隔离。
- 家族：`hypothesisworks/hypothesis::hypothesis.strategies._internal.core.
  cached_strategy+get_cache::thread_local_cache_isolation`；快照
  `hypothesisworks/hypothesis::021d2a694469…`。零重叠（程序化校验通过）。
- 风险：F2P 为概率性竞态回归测试，baseline 失败需实际验证；若本地不
  触发则记 near-miss（K11 先例），不放宽。

### P5 `Project-MONAI__MONAI-2446`（Gym，total=3，F2P=1，P2P=2，单文件 test_smartcachedataset.py）

- 生产：`SmartCacheDataset.__init__` 在 `shuffle` 时原地 `randomize(data)`，
  改调用方传入的 datalist；修复 `data = copy(data)` 后再 shuffle
  （`monai/data/dataset.py` 单 hunk + docstring）。
- 三元组：`data is not datalist` + 构造后调用方 list 不变 + 构造时点。
  极性：隔离。教科书 F2（同 K2 pandas-attrs 族思路）。
- 家族：`Project-MONAI/MONAI::SmartCacheDataset.__init__::
  input_datalist_copy_isolation`；快照 `Project-MONAI/MONAI::05b2da61d7…`。
  零重叠（程序化校验通过）。
- 环境：torch 全栈（K3/K4 先例），本批 pending，不投双轮，不计数。

### P6 `Project-MONAI__MONAI-6127`（Gym，total=64，F2P=16，P2P=48）

- 生产：`RandCropByLabelClasses` 将 `self.ratios` 别名写入 `ratios_` 后
  缺类置零直接改内部变量，训练越久采样越偏；修复
  `ratios_ = list(ensure_tuple(...))` 工作拷贝（`monai/transforms/utils.py`
  主 hunk；其余为 pad-width tuple 化噪声）。
- 三元组：工作 list `is not self.ratios` + 缺类置零不污染内部变量 +
  transform 调用时点。极性：隔离。
- 家族：`Project-MONAI/MONAI::RandCropByLabelClasses::
  ratios_working_copy_isolation`；快照 `Project-MONAI/MONAI::f754928bb22…`。
  零重叠（程序化校验通过）。
- 环境：torch 全栈（K3/K4 先例），本批 pending，不投双轮，不计数。

## COLLIDE-family（同家族复本，不计入）：2

- `aio-libs__aiohttp-8996`（V2，95，F2P=2）：同 P3 同 F2P
  （`test_clone_absolute_scheme/host`）、同 patch 系，基线 `7c3ff649…`
  vs P3 `ed5576db…`。家族与 P3 逐字同，记 COLLIDE-family，不计入。
- `aio-libs__aiohttp-8990`（V2，98，F2P=2）：同上，基线 `5c3d50f9…`。
  记 COLLIDE-family，不计入。

## REJECT → S（仅值语义/附带 copy 或 `is` 为空检查/散文/新API hygiene，无差异 `is` 腿）

- [3] `openmdao__openmdao-3259`（V2，34，F34）：lambda pickle 并行组 bug；
  `list()/set()` 噪声；MPI 倾向。→ S。
- [12] `pallets__click-1679`（V2，30，F20）：shell-completion ctx_args
  透传；`args.copy()` 为既有附带。→ S（与 P2 同仓不同家族）。
- [13] `pydantic__pydantic-6288`（Gym，35，F1）：discriminated-union 递归
  import 失败；`s.copy()` 附带。→ S。
- [16] `zauberzeug__nicegui-4692`（V2，59，F12）：mermaid/codemirror 值 bug；
  浏览器倾向（K5 先例）。→ S。
- [18] `pennylaneai__pennylane-4831`（V2，6，F2）：queuing 合并值 bug；
  工作 list 附带。→ S。
- [19] `pandas-dev__pandas-58360`（Gym，46，F6）：cummin/cummax 值 bug；
  `codes.copy()` 附带；pandas-C。→ S。
- [20] `yt-dlp__yt-dlp-4704`（V2，28，F1）：extractor/rename 值 bug。→ S。
- [21] `pandas-dev__pandas-56757`（Gym，77，F20）：interpolate 新实现；
  pandas-C。→ S。
- [22] `jazzband__pip-tools-2082`（V2，30，F1）：no/only-binary 输出顺序值 bug；
  `.copy()` 为新展示逻辑 hygiene（旧代码无 shared-mutation 缺陷，无
  baseline 别名腿）。→ S（严格口径）。
- [23] `audreyr__cookiecutter-825`（V2，4，F3）：本地路径解析值 bug；
  `clone()` 为 git 域改名附带（cookiecutter-967 先例 S）。→ S。
- [27] `Project-MONAI__MONAI-7000`（Gym，46，F10）：orientation 值 bug；
  `.clone()` 防御附带；torch。→ S。
- [29] `Project-MONAI__MONAI-5916`（Gym，5，F5）：question 非 bug；torch。→ S。
- [31] `pymc-devs__pymc-5759`（V2，6，F6）：采样 shape 值 bug；aesara 栈。→ S。
- [34] `qiskit__qiskit-terra-7789`（V2，58，F1）：并行 transpile 性能重构；
  非正确性隔离；qiskit 重依赖。→ S。
- [36] `twisted__treq-400`（V2，22，F22）：multipart  vendoring 打包重构；
  alias 为名别名噪声。→ S。
- [37] `tobikodata__sqlmesh-1817`（V2，8，F2）：format 可读性值改；
  `query.copy()` 两侧皆存，无隔离行为变化。→ S。
- [39] `robotframework__seleniumlibrary-1248`（V2，13，F9）：capabilities 值 bug；
  浏览器倾向。→ S。
- [40] `automl__configspace-382`（V2，93，F1）：print 值 bug。→ S。
- [42]/[43] `pandas-dev__pandas-56523`（Gym/V2 跨池同实例，71，F1）：PERF；
  pandas-C。→ S（跨池复本注记）。
- [44] `pyqtgraph__pyqtgraph-1725`（V2，7，F1）：除零值 bug；Qt。→ S。
- [46] `pandas-dev__pandas-50857`（Gym，21，F3）：categorical replace 值 bug；
  `set()` 噪声；pandas-C。→ S。
- [48] `robotframework__seleniumlibrary-1366`（V2，11，F3）：capabilities 值 bug；
  浏览器倾向。→ S。
- [49] `pandas-dev__pandas-55678`（Gym，142，F2）：merge_asof dtype+perf；
  Cython；`[:]` 类型注释噪声。→ S。
- [52] `jupyter__jupyter_client-994`（V2，2，F2）：缺 kernel 崩溃 UX；
  `[:]` 切片赋值噪声。→ S。
- [53] `Project-MONAI__MONAI-1030`（Gym，25，F14）：ImageReader 多线程重构；
  torch。→ S（env-blocked 注记）。
- [54] `Project-MONAI__MONAI-6781`（Gym，86，F1）：metatensor 索引值 bug；
  torch。→ S。
- [55] `pandas-dev__pandas-50506`（Gym，2，F2）：to_numpy 双拷贝 PERF。→ S。
- [58] `sopel-irc__sopel-2646`（V2，9，F3）：URL sanitize 异常值 bug；
  `[:]` 噪声。→ S。
- [59] `sebrut__pygrocy-72`（V2，27，F8）：API 返回值重构；`headers.copy()`
  为新 helper 附带。→ S。
- [60] `neurodatawithoutborders__pynwb-814`（V2，30，F1）：to_dataframe
  shape 值 bug；`col[:]` 噪声。→ S。
- [62]/[64] `pandas-dev__pandas-57169`（Gym 65 / V2 107）：stable sort 值 bug；
  pandas-C。→ S。
- [63] `pandas-dev__pandas-50331`（Gym，77，F1）：to_numpy na_value 值 bug；
  pandas-C。→ S。
- [67] `iterative__dvc-1638`（Gym，3，F1）：ssh 缓存检查；远端域 copy。→ S。
- [68] `pandas-dev__pandas-50606`（Gym，3，F1）：to_numpy 值 bug；pandas-C。→ S。
- [70] `openmdao__dymos-1165`（V2，5，F1）：default val 值 bug。→ S。
- [71] `core-gatech-group__serpent-tools-21`（V2，9，F2）：版本+API 重构。→ S。
- [72] `martinfleis__momepy-94`（V2，11，F1）：None-height 值 bug；
  `right.copy()` 既有防御。→ S。
- [73]/[75] `vaexio__vaex-388/389`（V2，12/14）：join 语义值 bug；
  copy 重排两侧皆存。→ S。
- [80] `natcap__invest-1159`（V2，24，F8）：decay 公式值 bug；gdal 原生。→ S。
- [81] `alteryx__evalml-1321`（V2，24，F1）：random_state 透传值 bug；
  `clone()` 附带。→ S。
- [82] `ross-rotordynamics__ross-822`（V2，25，F1）：save/load 序列化值 bug；
  `__dict__.copy()` 附带。→ S。
- [83] `nipy__nibabel-753`（V2，41，F1）：warnings filter 值 bug；
  `.copy()` 未动。→ S。
- [84] `pycqa__pyflakes-754`（V2，42，F1）：scope 栈值 bug；`[:]` 未动。→ S。
- [85] `pypa__setuptools-1921`（V2，51，F3）：find_links 切分值 bug；
  `[:]` 附带。→ S。
- [87]/[88] `Project-MONAI__MONAI-4716/5845`（Gym，83/86，F1）：metatensor
  切片值 bug；torch。→ S。
- [91] `tobikodata__sqlmesh-3446`（V2，132，F1）：migration 值 bug；
  copy 两侧皆存。→ S。
- [92] `aws-cloudformation__cfn-lint-2691`（V2，2，F1）：规则放宽值 bug；
  `path[:]` 附带。→ S。
- [95] `pybamm-team__pybamm-587`（V2，4，F1）：参数传播值 bug；重栈。→ S。
- [97] `keras-team__keras-20079`（V2，6，F1）：序列化值 bug；TF 重依赖。→ S。
- [98] `pyqtgraph__pyqtgraph-1764`（V2，6，F1）：maxTraces 反逻辑值 bug；
  Qt。→ S。
- [100]/[102] `aws-cloudformation__cfn-python-lint-1301/1305`（V2，7/8）：
  规则值 bug；`set()` 噪声。→ S。
- [104] `pydantic__pydantic-5272`（Gym，12，F3）：serializer 签名分发值 bug。→ S。
- [105] `pyccel__pyccel-264`（V2，12，F2）：类型转换 crash；Fortran 链。→ S。
- [106] `maniacallabs__bibliopixel-658`（V2，12，F2）：websocket 重构；
  `clients.copy()` 未动。→ S。
- [108] `softlayer__softlayer-python-650`（V2，16，F4）：transport 重构。→ S。
- [111] `cloudtools__troposphere-2064`（V2，26，F1）：校验值 bug。→ S。
- [116] `yt-project__unyt-109`（V2，41，F1）：ufunc 比较值 bug；
  `out[:]` 附带。→ S。
- [119] `softlayer__softlayer-python-648`（V2，77，F5）：CLI 特性混杂。→ S。
- [120] `jazzband__django-axes-1033`（V2，79，F1）：sensitive 参数未脱敏值 bug；
  `.copy()` 为本地构建 dict 的 hygiene（无 caller-owned 映射在险，
  无 baseline 别名腿）。→ S（严格口径）。
- [121] `google__yapf-59`（V2，86，F86）：换行格式化值 bug。→ S。
- [122] `stingraysoftware__stingray-289`（V2，93，F3）：rms 公式值 bug；
  `powers.copy()` 附带。→ S。
- [123] `qiskit__qiskit-11272`（V2，95，F1）：barrier set 操作值 bug；
  qiskit 重依赖。→ S。
- [126] `Project-MONAI__MONAI-5075`（Gym，107，F8）：Spacing 值 bug；torch。→ S。
- [127] `kozea__weasyprint-2301`（V2，111，F1）：margin 布局值 bug；
  `style.copy()` 附带；系统库倾向。→ S。
- [11] `pimutils__todoman-197`（V2，16，F2）：edit-save 大重构，
  `clone()` 为新增 API（无 baseline 别名腿；F2P 为新方法值断言）。
  → S（严格口径；is 腿非 discriminating）。

## REJECT → R（纯能力/特性/重构，无引用语义；代表性全列）

- [1] `pyjanitor-devs__pyjanitor-1287`（V2，22，F3）：[ENH] 新值清洗能力；
  `df.copy()/df[:]` 为新路径附带。→ R。
- [2] `jcrist__msgspec-181`（V2，86，F86）：json-schema 生成特性；
  P2P-EMPTY。→ R。
- [4] `eventlet__eventlet-1047`（V2，51，F51）：fork 文档+datetime；
  clone 仅文档散文。→ R。
- [8] `scrapy__scrapy-1267`（V2，23，F23，P2P-EMPTY）：路径兼容特性；
  `base.copy()` 被替换附带。→ R。
- [9] `pypa__setuptools_scm-247`（V2，24，F24，P2P-EMPTY）：symlink 跟随特性；
  git 域。→ R。
- [10] `pre-commit__pre-commit-904`（V2，43，F43，P2P-EMPTY）：“Teach try-repo…”
  特性；git clone 域。→ R。
- [15] `pymc-devs__pymc-7746`（V2，2，F2，P2P-EMPTY）：“Implement model
  transform…”；`clone_replace` 新能力。→ R。
- [17] `Project-MONAI__MONAI-2321`（Gym，3，F3，P2P-EMPTY）：key 保序特性请求；
  torch。→ R。
- [24] `openmdao__dymos-1146`（V2，5，F5，P2P-EMPTY）：“ControlGroup cleanup”
  任务重构。→ R。
- [26] `pyfar__pyfar-123` 之外 `pyfar__pyfar-716`（V2，10，F10，P2P-EMPTY）：
  正则化重构。→ R。
- [28] `rochacbruno__dynaconf-191`（V2，109，F2）：“Missing an isolated
  namespace [RFC]” 特性请求；`store.copy()` 为新 `as_dict` 附带。→ R。
- [30] `pyccel__pyccel-2277`（V2，64，F2）：“[DICTS] Support…” 特性；
  Fortran 链。→ R。
- [32] `pygfx__pygfx-1044`（V2，29，F3）：geometry Store 重构。→ R。
- [33] `pyccel__pyccel-2158`（V2，42，F2）：“[DICTS] Support…” 特性。→ R。
- [35] `pyccel__pyccel-2031`（V2，119，F2）：“[SETS] Support…” 特性。→ R。
- [38] `pyg-team__pytorch_geometric-6928`（V2，7，F7，P2P-EMPTY）：“🚀 Feature”。→ R。
- [41] `enthought__traits-889`（V2，57，F2）：“Allow Enum…” 特性；C 扩展。→ R。
- [45] `jazzband__pip-tools-2060`（V2，4，F2）：API 文档页特性。→ R。
- [47] `Project-MONAI__MONAI-3945`（Gym，22，F10）：import 特性请求；torch。→ R。
- [50] `ecmwf__earthkit-data-283`（V2，127，F1）：“Is your feature request…”；
  eccodes 原生栈（K8 先例）。→ R。
- [51] `aws-cloudformation__cfn-python-lint-1483`（V2，2，F2，P2P-EMPTY）：
  新 lint 检查能力。→ R。
- [56] `aws-cloudformation__cfn-lint-2786`（V2，3，F1）：split 新检查能力。→ R。
- [57] `spikeinterface__spikeinterface-3347`（V2，8，F1）：run_info 追踪能力。→ R。
- [61] `benchopt__benchopt-362`（V2，32，F32，P2P-EMPTY）：“ENH: be able to…”；
  git-clone 文档噪声。→ R。
- [65] `samuelcolvin__pydantic-1814`（V2，122，F9）：时区解析新能力。→ R。
- [66] `aws-cloudformation__cfn-python-lint-2006`（V2，2，F1）：新 key 检查。→ R。
- [74] `codezonediitj__pydatastructs-221`（V2，14，F14，P2P-EMPTY）：排序算法
  特性请求；git-clone 文档噪声。→ R。
- [76] `Project-MONAI__MONAI-1107`（Gym，15，F1）：多通道支持特性；torch。→ R。
- [77] `mozilla-services__cliquet-680`（V2，19，F5）：key-prefix 新能力。→ R。
- [78] `matchms__matchms-265`（V2，23，F23，P2P-EMPTY）：“Add retention data
  filters”。→ R。
- [79] `tdameritrade__stumpy-817`（V2，23，F2）：“Allow user to use
  pre-calculated mp”；`mp.copy()` 为新参数附带。→ R。
- [86] `cekit__cekit-932`（V2，64，F2）：“refactor template.jinja”。→ R。
- [89] `jelmer__dulwich-1171`（V2，92，F2）：“Allow custom symlink handling…”；
  git 域。→ R。
- [90] `pandas-dev__pandas-53122`（Gym，117，F22）：“ENH/PERF: add validate
  parameter”；pandas-C。→ R。
- [93] `pyccel__pyccel-1829`（V2，3，F3，P2P-EMPTY）：“Add a pyccel.lambdify
  function”。→ R。
- [94]/[101] `streamlink__streamlink-2654/2653`（V2，3/7）：“Plugin Request”；
  git-clone 文档噪声。→ R。
- [96] `aws-cloudformation__cfn-python-lint-1640`（V2，4，F2）：新校验表能力。→ R。
- [99] `conan-io__conan-14727`（Gym，7，F1）：“[feature] Same host and build
  profile option?”；`items[:]` 附带。→ R。
- [103] `codezonediitj__pydatastructs-315`（V2，10，F10，P2P-EMPTY）：LCS 新算法。→ R。
- [107] `getmoto__moto-5084`（Gym，15，F4）：launch-template spot 特性请求；
  moto 域。→ R。
- [109] `iamconsortium__pyam-202`（V2，18，F2）：`check_duplicates` 新校验选项；
  `.copy()` 为新校验附带。→ R。
- [112] `aeon-toolkit__aeon-2565`（V2，29，F29，P2P-EMPTY）：“[ENH] wrap/implement
  Global Alignment DTW”；numba。→ R。
- [113] `pytest-dev__pytest-xdist-862`（V2，30，F30，P2P-EMPTY）：“Implement
  work-stealing scheduler”。→ R。
- [114] `iterative__dvc-1855`（Gym，31，F4）：`remote://` 特性；dvc 域。→ R。
- [117] `getmoto__moto-5841`（Gym，45，F1）：“Implement Describe Security Group
  Rules”；`x.copy()` 为新 API 附带。→ R。
- [118] `iterative__dvc-1926`（Gym，50，F1）：“Add an option -d|--downstream”；
  dvc 域。→ R。
- [124] `wtforms__wtforms-592`（V2，95，F2）：“A different approach for filters”
  新能力。→ R。
- [125] `Project-MONAI__MONAI-4174`（Gym，103，F26）：reduction 参数特性请求；
  torch。→ R。

## 双轮与口径注记

- skip 预检（`skip-precheck.json`）：6 triple-KEEP 逐快照必做；只有
  skip-CLEAN ∩ 三元组-KEEP 才投双轮（P5/P6 torch-pending 不投双轮）。
- 4 双轮（P1–P4）按 `run_official.py`（baseline 失败 + repaired 通过）执行；
  file-scope（单文件闭包：pyfar test_coordinates.py / click test_arguments.py /
  aiohttp test_web_request.py / hypothesis test_cache_implementation.py），
  原 matcher 不动，不断言缩水。重依赖按实际记录，不自行放宽
  （near-miss 只记录不放宽）。
- 全局计数以各批 `normalize-summary.json` 为准；本批双轮前不计数。
- 缺口清零前不启动训练、不改 S600 与旧文件。
