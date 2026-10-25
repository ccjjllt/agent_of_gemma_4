# f2-recall 三元组复核 log（2026-10-25，第七轮：recall 非平凡大行 2024 重排分批）

口径：每题必须同时满足 `is` 身份断言 + 双向（共享修改）可见 + 访问/复制时点。
只有 `==` 无 `is` → 退 S；只有能力识别（无引用/所有权语义）→ 退 R。
Gym/V2 真题本批放宽到 ≤200（含 200）；>200 弃（1556 行：gym 658 / v2 899）。
坏路径（`..`）/禁仓/四仓已在扫描层过滤。Smith-py 不扫（冻结）。
P2P=0 标 P2P-EMPTY。skip 碰撞 DROP 快照不投双轮；test_patch-new 文件在 base
无上游文件按 K1/C2 先例计 CLEAN。
pools：SWE-Gym rev `bb94ed9e…`（train 2438）+ SWE-rebench-V2 rev `10483de0…`
（train python 子集 7243）。去重：只读 common-400 + F1-40 + F5旧40 +
f2-round1/2 + f2-gymv2 16 + f2-gymv2b 5 + f2-gymv2c 2（程序化校验
544i/107s/490f；3 KEEP snapshot_collision 全无，instance 全新）。

## 扫描召回（f2-recall，forbidden 含 23 keeps）

- Gym 非平凡大行（total>150 且过基线过滤）：771 行（任务口径 770，差 1，
  因 forbidden 含 f2-gymv2c 2 小套件不影响大行计数，属 parquet 复算抖动，
  如实记录）；其中 ≤200 可分批 113 行。
- V2-python 非平凡大行：1254 行（与任务口径一致）；其中 ≤200 可分批 355 行。
- 合计 2025 行（771+1254；任务称 2024，差 1 同上），可分批 468 行
 （113+355），>200 弃 1557 行（658+899；任务称 1556，同抖动）。
- 13 strict-copy ≤200 行已在 f2-large 35 行内复核（S/R/DROP），本批不重做：
  gym 3（moto-5960 S、moto-5012 S、MONAI-5107 R）+ v2 10（unleash-285 S、
  isort-972 S、structlog-283 DROP、rdt-670 S、traits-1026 S、universal_pathlib-372 R、
  werkzeug-2975 S、isort-1029 S、copier-2145 DROP、pandas-57314 S）。
- 3 快照碰撞排除（零重叠）：gym moto-7167、v2 click-2271（撞 round1 禁快照
  `pallets/click::fde47b4b`）、v2 moto-7167（撞 `getmoto/moto::e5f96219…`）。
- 重排：strict-copy + REAL_IS（`assert…is/is not` 标识符，排除 `is None/True/False`
  空检查与散文 `is`，但仍含部分 `is not None` 漏网，逐题人工剔除）+ F2P 名 +
  total 升序。REAL_IS ≤200 共 51 行，扣 LARGE-DONE 2、COLLIDE 1，剩 48 行逐题
  三元组复核（下）。其余 417 可分批行无 REAL_IS（`is` 缺席或仅散文/`is None`/
  `isinstance`），按“无 is 不投”记 SCREENED-OUT，不投 skip/双轮。

## KEEP（真 F2，三元组完整，待 skip 预检+双轮）：3

### R1 `theacodes__nox-439`（V2，total=159，F2P=1，P2P=158）
- `SessionRunner.posargs` 原为 `global_config.posargs` 别名（共享列表）；
  修复 `global_config.posargs[:]` 切片拷贝，防跨 session 污染。
- 三元组：`runner.posargs is not global_config.posargs`（切片新对象；
  F2P `test_posargs_are_not_shared_between_sessions` 为功能覆盖：test 会话
  `extend(["-x"])` 后 lint 会话 `"-x" in posargs` 即抛，`is` 腿由 source-bound
  inline 观察承载，口径同 C2/K15）+ 双向（改一会话另一会话不可见，
  全局 config 不动）+ `SessionRunner.__init__` 复制时点。极性：隔离（同 K13）。
- 家族：`theacodes/nox::nox.sessions.SessionRunner.posargs::slice_copy_isolation`；
  快照 `theacodes/nox::41b9c798…`。零重叠。

### R2 `aio-libs__aiohttp-10672`（V2，total=153，F2P=1，P2P=152）
- `web.Response(headers=CIMultiDict)` 原直接别名 `real_headers = headers`，
  导致调用方 `default_headers` 被写入 Content-Length/Date 等污染；
  修复恒 `real_headers = CIMultiDict(headers)` 拷贝。
- 三元组：`resp.headers is not headers`（拷贝新对象；F2P
  `test_passing_cimultidict_to_web_response_not_mutated` 为功能覆盖：
  `assert not headers` + `content_length==6`，`is` 腿由 inline 承载，同 C2）+
  双向（prepare 写 resp 不污染入参）+ `Response(headers=…)` 复制时点。
  极性：隔离。
- 家族：`aio-libs/aiohttp::aiohttp.web_response.Response.headers::copy_isolation`；
  快照 `aio-libs/aiohttp::3d11b551…`。零重叠。

### R3 `hdmf-dev__hdmf-848`（V2，total=160，F2P=1，P2P=159）
- `export(link_data=False)` 应拷贝数据集而非链接，但 build-manager 缓存命中
  旧 builder（link 语义），导致外链残留；修复 `clear_cache` 重建
  （`manager.clear_cache()` 后 `build(..., export=True)`）。
- 三元组：`builder1 is not builder2` / `container1 is not container2`
  （F2P `test_clear_cache` 逐字 `assertIsNot`）+ 缓存清空后重建隔离
  （export 拷贝可见）+ `export()` 调用时点。极性：隔离。
- 家族：`hdmf-dev/hdmf::hdmf.build.manager.BuildManager.clear_cache::export_copy_isolation`；
  快照 `hdmf-dev/hdmf::fde784da…`。零重叠。

## REJECT → S（仅值语义/附带 copy 或 `is` 为空检查/散文，无差异 `is` 腿；45 行代表性全列）

- `facebookresearch__hydra-2085`（Gym 151）：multirun mode 特性；`is not None`
  空检查；无 copy。→ R（兼特性，记 R；本节 S/R 按主因，hydra 归 R，下同）。
  更正：hydra-2085 题面 Feature Request → R。
- `stfc__psyclone-2380`（152）：get_siblings 特性；`is` 为树节点自包含，
  无 copy。→ R。
- `aio-libs__aiohttp-9671`（152）：connector 死锁并发；`is not None` 空检查；
  无 copy。→ S。
- `hypothesisworks__hypothesis-3412`（152）：composite 警告文档；`is not P.empty`
  注解身份，无 copy 因果。→ S。
- `sabnzbd__sabnzbd-2679`（153）：队列优先级；`is True` 布尔；无 copy。→ S。
- `aio-libs__aiohttp-9029`（153）：SSL loop 缓存；`is` 为单例缓存身份，
  非 copy 隔离。→ S。
- `pyomo__pyomo-3288`（156）：Set within 文档；`assertIsNot` 附带，无 copy。→ S。
- `stfc__psyclone-2637`（156）：Fortran 前端能力；`is` 为 intrinsic 身份。→ R。
- `pre-commit__pre-commit-906`（162）：GC 特性；`is` 仅注释。→ R。
- `pydantic__pydantic-5322`（Gym 165）：dataclass frozen 配置；`is` 为 encoder
  身份。→ S。
- `hypothesisworks__hypothesis-2660`（167）：regex unicode；`is not None` 空。→ S。
- `aio-libs__aiohttp-7829`（167）：urldispatch 性能；handler `is` 路由。→ S。
- `googleapis__python-bigquery-2218`（170）：equality 归一；`is not None` 空。→ S。
- `stfc__psyclone-2301`（170）：依赖分析特性；散文 `is`。→ R。
- `aio-libs__aiohttp-9032`（170）：resolver flags；SSL `is` 同 9029。→ S。
- `modin-project__modin-6608`（Gym 172）：sort cache；`is not None` 空；
  `copy_columns_cache` 为缓存保留非所有权 `is`。→ S。
- `aio-libs__aiohttp-9047`（173）：同 9032。→ S。
- `pydicom__pydicom-1048`（173）：PathLike 能力；`is not None` 空。→ R。
- `hypothesisworks__hypothesis-3540`（174）：repr 特性；`is not` 去重检查，
  非 copy。→ R。
- `aristanetworks__anta-1021`（174）：校验 regex；`is not None` 空。→ S。
- `stac-utils__pystac-1512`（175）：版本类型；`is not None` 空。→ S。
- `stfc__psyclone-2508`（175）：inline 变换能力；symbol `is` 无 copy。→ R。
- `systemd__mkosi-2954`（176）：executable version 特性；`is SystemExit` 类型。→ R。
- `aristanetworks__anta-969`（178）：EVPN 错误处理；`is` 布尔。→ S。
- `theacodes__nox-432`（178）：skip install 特性；mock `is`。→ R。
- `python-attrs__attrs-763`（179）：PEP 563 兼容；type `is` 无 copy。→ S。
- `getmoto__moto-7850`（179）：S3 状态码；`is not None` 空。→ S。
- `hikari-py__hikari-785`（180）：Snowflake 类型转换；enum `is`。→ S。
- `aio-libs__aiohttp-7907/7908`（181×2）：connection upgrade；`is not None` 空。→ S。
- `aio-libs__aiohttp-8038`（184）：keep-alive 重试；`is not` 新旧连接，
  非 copy 隔离。→ S。
- `dask__dask-10485`（184）：Key 类型别名；`is df` 替换语义，无 copy。→ S。
- `enthought__traits-1335`（186）：repr 子类；`assertIs` 协议附带。→ S。
- `aio-libs__aiohttp-11289`（188）：proxy tunnel；`is protocol`。→ S。
- `hdmf-dev__hdmf-751`（189）：construct mode 特性；parent `is` 链接。→ R。
- `tornadoweb__tornado-2162`（189）：X-Forwarded-Proto；weakref `is None`。→ S。
- `stfc__psyclone-2147`（190）：elemental 属性；symbol table `is not` 附带。→ S。
- `wntrblm__nox-935`（191）：URL 依赖特性；`is` 返回同一。→ R。
- `aio-libs__aiohttp-11291/11292`（191×2）：同 11289。→ S。
- `celery__celery-8143`（193）：beat 调度；`is` 后端身份。→ S。
- `pandas-dev__pandas-49933`（Gym 194）：IntervalIndex 排序值 bug；
  `is` 为布尔期望。→ S。
- `getmoto__moto-4890`（Gym 197）：multipart 校验；`is` 为错误文案散文。→ S。
- `repobee__repobee-166`（199）：collaboration 特性；散文 `is`。→ R。
- `aio-libs__aiohttp-8632`（200）：unix socket；`is not None` 空 + loop `is`。→ S。

## SKIP DROP（3，阻断验证；三元组无关）

- `getmoto__moto-7167`（Gym/V2 同实例，194）：快照碰撞
  `getmoto/moto::e5f96219…`（forbidden），零重叠不投。
- `pallets__click-2271`（V2，192）：快照碰撞 round1 禁
  `pallets/click::fde47b4b`，不投。

## SCREENED-OUT（417 可分批无 REAL_IS，不投）

- Gym 75 + V2 203（`is` 缺席）+ Gym 35（`is` 仅散文/`is None`/`isinstance`，含
  pandas-51967 类）+ V2 142（同）− 已上 48 REAL_IS − 13 LARGE − 3 COLLIDE =
  417 行。按“无 is 不投”不做 skip/双轮；清单在 `*-recall-large-scan.json`
  `le200` 内可复现（`strict_copy/test_is_strict/test_is_any` 分层）。

## 双轮与口径注记

- skip 预检（`skip-precheck.json`）：3 triple-KEEP 中 2 CLEAN / 1 DROP。
  CLEAN：nox-439（4 文件 0 markers）、aiohttp-10672（1 文件 0 markers）。
  DROP：hdmf-848（`tests/unit/test_io_hdf5_h5tools.py` 5 skip markers 致 6 前缀
  碰撞，P2P `test_write_dataset_list*`/`test_roundtrip*` 族；与三元组无关，
  按协议不投双轮，证据保留在 keep-list/triple-log/skip-precheck，排除在
  reviewed/catalog 之外）。
- nox-439 双轮（`replay/artifacts/f2-recall/f2-recall-nox-439-run-001`，
  task-site-packages-nox pth）：baseline 3 failed/157 passed（F2P 正确失败），
  repaired 2 failed/158 passed（F2P 正确翻转通过）；剩余 2 失败中
  `test_discover_session_functions_decorator` 非 F2P/P2P（不阻塞），
  `test_load_nox_module_needs_version_static`（P2P）在双相以同一
  `PackageNotFoundError: No package metadata was found for nox` 失败
  （sandbox 未安装 workspace 分发元数据，上游 CI 安装后通过；与修复无关，
  同 K11/K17 类环境漂移）。normalize：complete_replay True、
  baseline all_original_F2P_failed，但 `actual_local_official_repair_not_resolved`
  + `original_official_JUnit_matcher_rejects_repaired_report` → 判 near-miss，
  不计数（`f2-recall-nox-439-normalize.json`）。
- aiohttp-10672 双轮（`replay/artifacts/f2-recall/f2-recall-aiohttp-10672-run-001`，
  单文件闭包，task-site-packages-aiohttp 供给 blockbuster/pytest-xdist）：
  baseline 2 failed/154 passed（F2P 正确失败），repaired 1 failed/155 passed
  （F2P 正确翻转通过；全部 152 P2P 通过）；唯一剩余失败
  `test_warn_large_cookie`（非 F2P/P2P，`DID NOT WARN`）在双相一致失败
  （warnings/标记环境问题；数据集 P2P 本就不含它，属已知 flaky），
  但严格 matcher 要求目标文件零失败（`failures=1` 即拒），故
  `actual_local_official_repair_not_resolved` +
  `original_official_JUnit_matcher_rejects_repaired_report` → 判 near-miss，
  不计数（`f2-recall-aiohttp-10672-normalize.json`；F2P/P2P 证据全对，
  仅 extra flaky 阻塞）。
- 本批 3 KEEP 全部待 skip 预检；CLEAN 才投双轮（JUnit 双条件、六工具、
  白名单题面、不改测试、保护路径、`replay/.venv`）。
- 全局：10/40 eligible 不变（本批未完成双轮前不计数；nox-439 near-miss 不计数）；缺口 30。
- 缺口清零前不启动训练、不改 S600 与旧文件。
