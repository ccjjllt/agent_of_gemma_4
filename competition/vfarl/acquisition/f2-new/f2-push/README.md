# F600-F2 f2-push 批次记录（2026-10-25，第十轮：小套件 tier2/3-strong 补筛）

只做题目与 CPU 真实工具数据，不启动训练。
只读使用 `f2-replace/`、`candidate-list-40.json`、`reviewed-candidates.json`、
`catalog.json`、`dedup-baseline.json`（下去重用）与 `f2-gymv2/`、`f2-gymv2b/`、
`f2-gymv2c/`、`f2-large/`、`f2-recall/`、`f2-inline/`、`f2-more/`
（27 keeps 只读去重，未改动其任何文件）；未碰 `f1-new` / `f3-new` /
`f4-new` / `f5-new` 与 S600 文件。本批全部产物在全新目录 `f2-push/`；
replay 产物为新 `replay/f2-push-*` 与 `replay/artifacts/f2-push/`（新）。
Smith-py 冻结未扫。

## 约束口径

- 只从 SWE-Gym（`SWE-Gym/SWE-Gym` train rev `bb94ed9e…`，2438 题）与
  SWE-rebench-V2（`nebius/SWE-rebench-V2` train rev `10483de0…`，
  python 子集 7243 题）找 F2；Smith-py 不再扫。
- 仅 train；排除 `fastapi/fastapi、textualize/rich、psf/requests、encode/httpx` 四仓。
- 与 common-400 + F1-40 + F5旧40 + F2前九轮（含 f2-gymv2 16 keeps、
  f2-gymv2b 5 keeps、f2-gymv2c 2 keeps、f2-recall 3 keeps、f2-inline 1 keep、
  f2-more 2 keeps 的 family / snapshot / instance）零重叠
  （`dedup-evidence.json`：550i/113s/496f，程序化校验；6 triple-KEEP
  `snapshot_collision` 全无，instance 全新；同家族复本 2 个记
  COLLIDE-family，不计入；pandas-56523 gym/v2 跨池同实例复本皆 S）。
- 三元组：`is` 身份 + 双向（共享修改）可见 + 访问/复制时点；无 `is` 退 S，
  纯能力退 R；坏路径（`..`）过滤。inline-is 型指 F2P 仅值断言、
  `is` 腿由 source-bound inline 观察承载（同 C2/K15/nox-439/aiohttp-10672/
  qiskit-12620/M2 口径）。
- 本批专属：小套件 total≤150 的 tier2/3-strong 补筛（strict-copy 行已由
  f2-gymv2/b/c 覆盖，本批排除）。f2-more 将 tier3 整批 SCREENED-OUT
  （“无 test-is 不投”），但 M2 本身无 test-is，证明“无 TEST-is”≠
  “无 is 三元组”；本批复核 128 行 tier2/3-strong 中此前未被逐题 verdict
  的全部（旧日志已具名行排除，不重审）。
- 严格性：修复对象必须是引用/所有权隔离缺陷。防御性 `.copy()` 若只为
  新逻辑 hygiene（无 caller-owned/shared 映射在险）→ S；新 API 本身
  （无 baseline 别名腿）→ S/R。本批据此降级 axes-1033、pip-tools-2082、
  todoman-197（皆 S，详见 triple log）。
- skip 前缀碰撞预检逐快照必做；只有 skip-CLEAN ∩ 三元组-KEEP 才投双轮。
- 白名单题面（verbatim problem_statement）、JUnit 双条件（baseline 全部原 F2P
  失败、repaired 原 matcher 通过）、不改测试、六工具唯一、保护路径不可碰。
  用 `replay/.venv` 执行；period-correct 第三方依赖经 `--task-site-packages` /
  `--task-python` 供给（测试/matcher 不动）。

## 计数（诚实）

- **f2-push normalize eligible：4；全局 16/40（12+4），缺口 24。**
- 128 逐题复核：triple-KEEP 6 / COLLIDE-family 2 / 其余 S-R；
  skip 6 CLEAN / 0 DROP；双轮完成 4（4 eligible，0 near-miss）；
  2 torch-pending（MONAI-2446/6127，K3/K4 先例，不计数）。
- reviewed 6；catalog 6；replay 产物见下。

| # | instance | pool | total | 双轮 | normalize |
|---|---|---|---|---|---|
| P1 | pyfar__pyfar-123（Coordinates 切片拷贝隔离） | V2 | 38 | baseline 仅 F2P 失败 / repaired 38 全过（run-002；run-001 见下） | eligible |
| P2 | pallets__click-2312（decorator 共享 attrs 隔离） | V2 | 35 | baseline F / repaired T（run-001，src-layout） | eligible |
| P3 | aio-libs__aiohttp-8997（Request.clone url 隔离） | V2 | 95 | baseline 2 F2P 失败 / repaired 95 全过（run-001） | eligible |
| P4 | hypothesisworks__hypothesis-2435（策略缓存线程隔离） | V2 | 20 | baseline F2P 失败 / repaired 全过（run-002；run-001 见下） | eligible |
| P5 | Project-MONAI__MONAI-2446（datalist 拷贝隔离） | Gym | 3 | 未投（torch-pending） | pending |
| P6 | Project-MONAI__MONAI-6127（ratios 工作拷贝隔离） | Gym | 64 | 未投（torch-pending） | pending |

- COLLIDE-family：aiohttp-8996/8990（同 P3 家族，不计入）。

## 本批产物

- `scan_push.py` + `gym-push-scan.json`（245）/ `v2-push-scan.json`（1157）：
  小套件 BROAD-inline 扫描（strict-copy 行排除；tier/strong/OWN 分层；
  parquet sha 见文件内）+ `push-candidates.json`（128 tier2/3-strong
  未复核行）+ `push-review-dump.txt`（128 题 problem+patch+testpatch）/
  `push-triage.txt`（紧凑版）。
- `dedup-evidence.json`（550i/113s/496f）+ `keep-list.json`（6 triple-KEEP，
  含完整 F2P/P2P lists）+ `keep-families.json`（6）+ `pages-v2/`（4）/
  `pages-gym/`（2，逐题 source row + `.receipt.json`，pinned revision API
  receipts，row_idx/F2P/P2P 校验）+ `base-resolutions.json`。
- `pyfar/click/aiohttp/hypothesis/monai-2446/monai-6127.LICENSE`（上游快照
  逐字节：MIT/BSD-3/Apache-2.0/MPL-2.0/Apache-2.0×2）+
  `reviewed-candidates.json`（6 条完整审阅）+ `catalog.json`（6 条，
  含 `source_review_sha256`）。
- `triple-review-log.md`：128 行逐题 verdict（KEEP 6 / COLLIDE-family 2 /
  S-R 全列 + 协议修正注记 + 降级依据）。
- `skip_precheck.py` + `skip-precheck.json`（6 快照全量：上游抓取证据 +
  前缀碰撞结论；6 CLEAN / 0 DROP）。
- `prepare_p_{pyfar123,click2312,aiohttp8997,hypothesis2435}.py`：单题 prepare
  （上游 tarball + forward-edit 重建验证 + recipe + source-preparation）。
- `record_one.py` + `f2-push-*-normalize.json`（4 eligible）+ `summarize.py` +
  `normalize-summary.json`（4 eligible / 0 near-miss / 2 pending；全局 16/40）。
- replay（未动 S600）：`replay/f2-push-*-{reviewed-recipe,source-preparation,
  replay-result}.json`，`replay/artifacts/f2-push/{*-source{,.tgz},*-task.json,
  *-run-001(-002),*.log,task-site-packages-{pyfar,hypothesis,aiohttp}}`。

## 双轮环境注记（均只在 `artifacts/f2-push/` 内新增，未改测试/matcher/要求）

- P2 click：parent `.venv`（3.12）+ `--source-layout src`（workspace click
  8.1.4.dev0 绑定断言；`.venv` 自带 click 8.5.0 会 shadow，recipe 断言已验）。
- P4 hypothesis：run-001 为诊断轮（`.venv` 残留 hypothesis 6.x pytest 插件
  与 workspace 5.14.0 冲突，`is_hypothesis_test` ImportError，双相 exit 4，
  已保留为证据）；run-002 改 task-python 3.10.20 + 自建
  `task-site-packages-hypothesis/`（attrs/sortedcontainers/pytest 8.3.3，
  经 uv 取数）干净双轮（baseline F2P 竞态失败 / repaired 全过）。
- P1 pyfar：task-python 3.10.20 + `task-site-packages-pyfar/`（numpy 1.26.4 /
  scipy / deepdiff / matplotlib / python-sofa / urllib3 / pytest 8.3.3，经 uv
  取数；pyfftw 缺席走 numpy 回退，上游 try/except 允许）。
  run-001 为诊断轮（matplotlib 3.10 移除了 `gca(projection=)`，2 P2P 双相
  一致失败，F2P 正确翻转，已保留为证据）；按 period 钉到 matplotlib==3.5.3
  （`gca(projection=)` 仅 DeprecationWarning）后 run-002 干净双轮
  （baseline 仅 F2P 失败 / repaired 38 全过）。测试、matcher、要求均未改动。
- P3 aiohttp：parent `.venv`（workspace aiohttp 3.11.0.dev0 直接导入，
  C 扩展缺席走纯 Python 回退）+ 自 `f2-recall` 只读复制的
  `task-site-packages-aiohttp/`（blockbuster/xdist 插件；旧目录未动）。
  run-001 干净双轮（baseline 恰 2 F2P 失败 / repaired 95 全过）。
- P5/P6 MONAI：torch 全栈（K3/K4 先例），本批 pending，不投双轮，不计数。

## 缺口说明与下一步

1. 小套件 BROAD-inline 强信号池已枯：1402 行中 tier2/3-strong 未复核 128
   行本批全部逐题 verdict（KEEP 6 + COLLIDE 2 + S-R 120），tier0/1 及
   已具名行此前已覆盖。剩余未看仅 torch-pending 2（环境）与 >200 大行
   （本批口径外）。
2. 4 eligible 全部单文件闭包（38/35/95/20 total），family/snapshot/instance
   全新（程序化校验），F2P 正确翻转，P2P 零回归。
3. 缺口清零前不启动训练、不改 S600 文件。

## 附：对旧文件的非触碰声明

- 本轮只读旧文件（去重用）；旧文件未运行、未覆盖。仅 `f2-push/` 与
  `replay/f2-push-*`、`replay/artifacts/f2-push/` 为新（`task-site-packages-
  aiohttp/` 系从 `f2-recall` 只读复制，旧目录未动）。
- 未动 `f1-new` / `f3-new` / `f4-new` / `f5-new`、S600、`f2-replace/`、
  `f2-gymv2*/`、`f2-large/`、`f2-recall/`、`f2-inline/`、`f2-more/` 内任何文件。
