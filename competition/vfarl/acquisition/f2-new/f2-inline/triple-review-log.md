# f2-inline 三元组复核 log（2026-10-25，第八轮：417 SCREENED-OUT 补筛 inline-is）

口径：每题必须同时满足 `is` 身份断言 + 双向（共享修改）可见 + 访问/复制时点。
只有 `==` 无 `is` → 退 S；只有能力识别（无引用/所有权语义）→ 退 R。
inline-is 型：F2P 仅值断言，`is` 腿由 source-bound inline 观察承载
（同 C2/K15/nox-439/aiohttp-10672 口径）。
Gym/V2 真题 ≤200（含 200）；>200 不在本批。
坏路径（`..`）/禁仓/四仓已在扫描层过滤。Smith-py 不扫（冻结）。
P2P=0 标 P2P-EMPTY。skip 碰撞 DROP 快照不投双轮。
pools：SWE-Gym rev `bb94ed9e…`（train 2438）+ SWE-rebench-V2 rev `10483de0…`
（train python 子集 7243）。去重：只读 common-400 + F1-40 + F5旧40 +
f2-round1/2 + f2-gymv2 16 + f2-gymv2b 5 + f2-gymv2c 2 + f2-recall 3
（程序化校验 547i/110s/493f；1 KEEP snapshot_collision 全无，instance 全新；
同家族复本 2 记 COLLIDE-family 不计入）。

## 扫描召回（f2-inline，forbidden 含 23+3 keeps）

- Gym le200-forbidden-excl：112 行；V2-python le200-forbidden-excl：350 行；
  合计 462 行（含跨池同实例复本 11，唯一 451）。
  与任务称 417 差 45，为 forbidden 含 recall-3 + 跨池复本 + REAL_IS 散文口径抖动，如实记录。
- 其中 strict-copy（COPY_SIG）13 行已在 f2-large 35 行内复核（S/R/DROP），本批不重做：
  gym 3（moto-5960 S、moto-5012 S、MONAI-5107 R）+ v2 10（unleash-285 S、
  isort-972 S、structlog-283 DROP、rdt-670 S、traits-1026 S、universal_pathlib-372 R、
  werkzeug-2975 S、isort-1029 S、copier-2145 DROP、pandas-57314 S）。
- 2 快照碰撞排除（零重叠）：gym moto-7167、v2 click-2271（撞 round1 禁快照）、
  v2 moto-7167（撞 `getmoto/moto::e5f96219…`）。注：v2 moto-7167 与 gym moto-7167
  同实例跨池复本，计 2 行。
- REAL_IS 48 行已在 f2-recall 逐题复核（KEEP 3 / S-R 45），本批不重做。
- 剩余 399 行（含跨池复本 11，唯一 388）逐题三元组复核（下）。
  方法：全量 OWN 拥有信号（`.copy()/deepcopy/__copy__/clone/CIMultiDict/clear_cache/[:]/copy_into`
  全 patch 搜索）+ F2P 名信号 + problem 共享语义 + 全 patch/题面深读
  （OWN>0 全读 + F2P≤10 & f2pn≥1 全读 24 行 + cache 模式全读，其余 bulk 按结构退 S/R）。

## KEEP（真 F2，三元组完整，待 skip 预检+双轮）：1

### I1 `qiskit__qiskit-12620`（V2，total=169，F2P=1，P2P=168）

- 生产：`QuantumCircuit.copy_empty_like()` 应返回空电路，但内部 `_parameters`
  缓存被拷贝带过，导致空电路 `parameters` 非空；修复 `cpy._parameters = None`
  失效缓存（`qiskit/circuit/quantumcircuit.py` 单 hunk，另 1 个 releasenotes 新文件非生产）。
- patch 全文（2 文件）：
  `cpy._calibrations = copy.deepcopy(...)` 上下文 + `cpy._parameters = None`
  （invalidate parameters caching）+ releasenotes。F2P
  `test_parameters_property_independent_after_copy` 为功能覆盖：
  `set(qc2.parameters)==set()` + `qc3.rz(b)` 后 `qc1` 不动 + `qc1.rz(c)` 后 `qc3` 不动
 （`assertEqual` 值断言，`is` 腿由 inline 承载，同 C2/nox/aiohttp）。
- 三元组：`cpy._parameters is not qc._parameters`（修前同列表 `is` 别名，
  修后 `None vs list` 为 `is not` 新对象；F2P 值断言 + inline `is` 观察）+
  双向隔离（改 copy 侧原侧不动，反之亦然，F2P 逐字）+
  `copy_empty_like()/copy()` 复制时点。极性：隔离（同 K13/nox/aiohttp/hdmf 族）。
- 家族：`Qiskit/qiskit::qiskit.circuit.quantumcircuit.QuantumCircuit.copy_empty_like::parameter_cache_invalidation_isolation`；
  快照 `Qiskit/qiskit::2b2be030…`。零重叠（程序化校验通过）。
  单文件闭包（F2P+P2P 全在 `test/python/circuit/test_parameters.py`），≤200 可分批。
- 风险：qiskit 1.x Rust `_accelerate` 编译（类 K2 pandas-C 先例），双轮按实际记
  eligible 或 pending/blocked，不放宽。

## COLLIDE-family（同家族复本，不计入）：2

- `qiskit__qiskit-12621`（V2，176，F2P=1）：同 issue #12617、同 patch
  `cpy._parameters = None`、同 F2P 名，基线 `9092ee78…` vs I1 `2b2be030…`。
  家族与 I1 逐字同，不视为独立，按协议只计其一（取小套件 I1）。
- `aio-libs__aiohttp-10674`（V2，160，F2P=1）：同 f2-recall R2（aiohttp-10672）
  同 patch `real_headers = CIMultiDict(headers)`、同 F2P
  `test_passing_cimultidict_to_web_response_not_mutated[CIMultiDict]`、
  同 problem，基线 `a60d447a…` vs R2 `3d11b551…`。
  家族 `aio-libs/aiohttp::aiohttp.web_response.Response.headers::copy_isolation`
  与 R2 逐字同，记 COLLIDE-family，不计入。

## LARGE-DONE（13，f2-large 已复核 S/R/DROP，不重做）

- gym moto-5960 S、moto-5012 S、MONAI-5107 R；v2 unleash-285 S、isort-972 S、
  structlog-283 DROP、rdt-670 S、traits-1026 S、universal_pathlib-372 R、
  werkzeug-2975 S、isort-1029 S、copier-2145 DROP、pandas-57314 S。
  详见 `f2-large/triple-review-log.md`。

## COLLIDE-snapshot（2，不投）

- `getmoto__moto-7167`（Gym/V2 同实例复本，194）：快照碰撞
  `getmoto/moto::e5f96219…`（forbidden），零重叠不投。
- `pallets__click-2271`（V2，192）：快照碰撞 round1 禁
  `pallets/click::fde47b4b`，不投。

## REALIS48（48，f2-recall 已复核，不重做）

- KEEP 3（nox-439、aiohttp-10672、hdmf-848）；S-R 45（hydra-2085 R、
  psyclone-2380 R、aiohttp-9671 S、hypothesis-3412 S、sabnzbd-2679 S、
  aiohttp-9029 S、pyomo-3288 S、psyclone-2637 R、pre-commit-906 R、
  pydantic-5322 S、hypothesis-2660 S、aiohttp-7829 S、bigquery-2218 S、
  psyclone-2301 R、aiohttp-9032 S、modin-6608 S、aiohttp-9047 S、
  pydicom-1048 R、hypothesis-3540 R、anta-1021 S、pystac-1512 S、
  psyclone-2508 R、mkosi-2954 R、anta-969 S、nox-432 R、attrs-763 S、
  moto-7850 S、hikari-785 S、aiohttp-7907/7908 S、aiohttp-8038 S、
  dask-10485 S、traits-1335 S、aiohttp-11289 S、hdmf-751 R、
  tornado-2162 S、psyclone-2147 S、nox-935 R、aiohttp-11291/11292 S、
  celery-8143 S、pandas-49933 S、moto-4890 S、repobee-166 R、aiohttp-8632 S）。
  详见 `f2-recall/triple-review-log.md` + `realis-48-dump.txt`。

## REJECT → S（仅值语义/附带 copy 或 `is` 为空检查/散文，无差异 `is` 腿；深读代表性全列）

- `iterative__dvc-1892`（Gym 157，F2P 16）：dvc add 链接类型值 bug；
  patch 大重构，`.copy()`（path_info）附带；F2P 功能测试无 `is`；题面无共享语义。→ S。
- `modin-project__modin-6751`（Gym/V2 同实例复本，197，F2P 3）：reset_index dtype
  值 bug；patch dtype 管道，`.copy()`（_known_dtypes 等）附带；F2P 值断言无 `is`。→ S（兼重依赖）。
- `modin-project__modin-6547`（Gym 170，F2P 1）：setitem dtype 值 bug；
  patch dtype 管道，`dtypes.copy()` 附带；F2P 值断言无 `is`。→ S（兼重依赖）。
- `modin-project__modin-5949`（Gym 156，F2P 1）：binary op dtype 值 bug；patch 同系；无 `is`。→ S。
- `qiskit__qiskit-11695`（V2 182，F2P 8）：approximation_degree 默认值 bug；
  patch transpile/QASM；F2P `test_measure_all_after_deepcopy` 为附带 copy 名，
  因果为默认值，无 `is` 腿。→ S（兼重依赖）。
- `pandas-dev__pandas-56769`（V2 172，F2P 2）：replace/groupby 值 bug；
  patch `__finalize__` 元数据值传递（非 `is` 隔离）；F2P `==` 无 `is`。→ S（兼 C 编译）。
- `geofileops__geofileops-649`（V2 194，F2P 10）：copy_layer 文件拷贝默认层命名值 bug；
  域拷贝非引用隔离；无 `is`。→ S。
- `elastic__curator-748`（V2 181，F2P 7）：alias 过滤（ES 别名名）值 bug；
  非引用别名；无 `is`。→ S。
- `stfc__psyclone-1914`（V2 165，F2P 4）：Fortran 隐式参数错误处理值 bug；无 copy。→ S。
- `getmoto__moto-6469`（V2 157，F2P 6）：secrets 校验值 bug；F2P alias 为 AWS 别名名巧合；
  patch 无 copy。→ S。
- `getmoto__moto-8094`（V2 152，F2P 3）：kms alias 值 bug；同上。→ S。
- `getmoto__moto-4793`（Gym 174，F2P 2）：S3 copy_object 域拷贝值 bug；非引用隔离。→ S。
- `getmoto__moto-5767`（Gym 180，F2P 3）：dynamodb executor 值 bug；无 `is`。→ S。
- `pandas-dev__pandas-55754`（Gym 165，F2P 1）：repr 值 bug；无 copy。→ S。
- `aio-libs__aiohttp-10534/10551`（V2 167/168，F2P 155/156）：socket_factory 特性束 +
  connector 值 bug；`.copy()`（content_mapping_proxy）附带；F2P 海量失败非隔离。→ S。
- 其余 bulk S（约 340 行）：patch 无 OWN 拥有信号、problem 无共享语义、F2P 大套件值 bug
  或散文 `is`（`is None/isinstance/is part`），按“无 is 不投”记 SCREENED-OUT，
  清单在 `*-inline-scan.json` 内可复现（`inline_n/f2p_name_hits/test_is_n` 分层）。

## REJECT → R（纯能力/特性/重构，无引用语义；深读代表性全列）

- `hdmf-dev__hdmf-1172`（V2 179，F2P 2）：题面逐字 Feature（backend 配置覆盖）；
  patch 新增参数；F2P 类型断言无 `is`。→ R。
- `python-attrs__attrs-430`（V2 159，F2P 1）：aliasing 为属性命名遮蔽（property/itemgetter），
  非引用别名；patch 模板改名；无 `is`。→ R。
- `dwavesystems__dwave-cloud-client-632`（V2 183，F2P 183，P2P-EMPTY）：
  entry_points 兼容重构；F2P 海量（含 isolation 名但体为 smoke），patch 无 copy 因果。→ R。
- `huggingface__huggingface_hub-1580`（V2 197，F2P 3）：alias 为 API 方法别名（hf_hub_download），
  非引用；patch endpoint 参数；无 `is`。→ R。
- `stfc__psyclone-2567`（V2 177，F2P 2）：Fortran 一元括号输出值 bug；无 copy、无 `is`。→ S。
- `jelmer__dulwich-1069`（V2 180，F2P 2）：identity 为 git 作者身份（name/email），
  非引用身份；patch 校验规则；无 `is` 腿。→ R。
- 其余 bulk R（约 45 行）：题面逐字 Feature/Add support/Provide a way +
  patch 新能力（api/参数/重构），无引用/所有权语义，清单同上可复现。

## 双轮与口径注记

- skip 预检（`skip-precheck.json`）：1 triple-KEEP 中 CLEAN 才投双轮
  （JUnit 双条件、六工具、白名单题面、不改测试、保护路径、`replay/.venv`）。
- I1 双轮按 `run_official.py`（baseline 失败 + repaired 通过）执行；
  file-scope 单文件（`test/python/circuit/test_parameters.py`），原 matcher 不动，
  不断言缩水（F2P/P2P 原样保留）。重依赖（Rust `_accelerate`）按实际记录，
  不自行放宽（nox-editable、aiohttp-flaky 两 near-miss 待协议裁决，本批不碰）。
- 全局：10/40 eligible 不变（本批双轮前不计数；I1 按实际定级）。
- 缺口清零前不启动训练、不改 S600 与旧文件。
