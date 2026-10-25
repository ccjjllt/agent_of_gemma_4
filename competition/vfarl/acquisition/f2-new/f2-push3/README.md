# F600-F2 f2-push3 批次记录（2026-10-25，第十二轮：mid-suite 151-400 + P2P-empty，push2 余量）

只做题目与 CPU 真实工具数据，不启动训练。
只读使用 `f2-replace/`、`candidate-list-40.json`、`reviewed-candidates.json`、
`catalog.json`、`dedup-baseline.json`（下去重用）与 `f2-gymv2/`、`f2-gymv2b/`、
`f2-gymv2c/`、`f2-large/`、`f2-recall/`、`f2-inline/`、`f2-more/`、`f2-push/`、
`f2-push2/`（38 keeps 只读去重，未改动其任何文件）；未碰 `f1-new` / `f3-new` /
`f4-new` / `f5-new` 与 S600 文件。本批全部产物在全新目录 `f2-push3/`；
replay 产物为新 `replay/f2-push3-*` 与 `replay/artifacts/f2-push3/`（新）。
Smith-py 冻结未扫。

## 约束口径

- 只从 SWE-Gym（`SWE-Gym/SWE-Gym` train rev `bb94ed9e…`，2438 题）与
  SWE-rebench-V2（`nebius/SWE-rebench-V2` train rev `10483de0…`，
  python 子集 7243 题）找 F2；Smith-py 不再扫。
- 仅 train；排除 `fastapi/fastapi、textualize/rich、psf/requests、encode/httpx` 四仓。
- 与 common-400 + F1-40 + F5旧40 + F2前十一轮（含 f2-gymv2 16 keeps、
  f2-gymv2b 5 keeps、f2-gymv2c 2 keeps、f2-recall 3 keeps、f2-inline 1 keep、
  f2-more 2 keeps、f2-push 6 keeps、f2-push2 3 keeps 的 family / snapshot /
  instance）零重叠（`dedup-evidence.json`：559i/122s/505f，程序化校验；
  2 triple-KEEP `snapshot_collision` 全无，instance 全新）。
- 三元组：`is` 身份 + 双向（共享修改）可见 + 访问/复制时点；无 `is` 退 S，
  纯能力退 R；坏路径（`..`）过滤。inline-is 型指 F2P 仅值断言、
  `is` 腿由 source-bound inline 观察承载（同 C2/K15/nox-439/aiohttp-10672/
  qiskit-12620/M2/P1-P4/Q2-Q3 口径）。
- 本批专属新地面（前十一轮均未逐题复核）：push2 候选 85 行中 push2 已 verdict
  35 行之外、剩余 fresh 39 行全部逐题复核（A池 mid-suite 151-400 11 行 +
  B池 P2P-empty 28 行；push2 大套件抽查/REJECT 行不再重审）。
- 严格性：修复对象必须是引用/所有权隔离缺陷。防御性 `.copy()` 若只为
  新逻辑 hygiene（无 caller-owned/shared 映射在险）→ S；新 API 本身
  （无 baseline 别名腿）→ S/R；全文件重写伴随的附带 copy → R。
- skip 前缀碰撞预检逐快照必做；只有 skip-CLEAN ∩ 三元组-KEEP 才投双轮。
- 白名单题面（verbatim problem_statement）、JUnit 双条件（baseline 全部原 F2P
  失败、repaired 原 matcher 通过）、不改测试、六工具唯一、保护路径不可碰。
  用 `replay/.venv` 执行；src 布局仓库经 `--source-layout src` 绑定；
  period-correct 第三方依赖经 `--task-site-packages` / `--task-python`
  供给（测试/matcher 不动）。

## 计数（诚实）

- **f2-push3 normalize eligible：2；全局 20/40（18+2），缺口 20。**
- 39 行逐题复核：triple-KEEP 2 / 其余 S-R（详见 triple log，无 SCREENED 余量）；
  skip 2 CLEAN / 0 DROP；双轮完成 2（2 eligible，0 near-miss）。
- reviewed 2；catalog 2；replay 产物见下。

| # | instance | pool | total | 双轮 | normalize |
|---|---|---|---|---|---|
| R1 | aeye-lab__pymovements-486（save_preprocessed gaze 拷贝隔离） | V2 | 370 | baseline 25 F2P 全失败 / repaired 370 全过（run-001） | eligible |
| R2 | jupyter__kernel_gateway-118（per-kernel activity-map 拷贝隔离） | V2 | 15 | baseline 15 全失败 / repaired 15 全过（run-002；run-001 见下） | eligible |

## 本批产物

- `scan_push3.py` + `gym-push3-scan.json`（A11/B7）/ `v2-push3-scan.json`
  （A36/B61）+ `push3-candidates.json`（A38/B36，prior-named 280 排除）+
  `dump_fresh.py` + `push3-review-dump.txt`（fresh 39 题 problem+patch+
  testpatch）/ `push3-triage.txt`（紧凑版）。
- `mk_keeps.py` + `dedup-evidence.json`（559i/122s/505f）+ `keep-list.json`
  （2 triple-KEEP，含完整 F2P/P2P lists）+ `keep-families.json`（2）+
  `pages-v2/`（2，逐题 source row + `.receipt.json`，pinned revision API
  receipts，row_idx/F2P/P2P 校验）+ `base-resolutions.json`（2，全 40 位免解析）。
- `pymovements/kernel_gateway.LICENSE`（上游快照逐字节：MIT/BSD-3-Clause）+
  `reviewed-candidates.json`（2 条完整审阅）+ `catalog.json`（2 条，
  含 `source_review_sha256`）。
- `triple-review-log.md`：39 行逐题 verdict（KEEP 2 / S-R 37 全列依据）。
- `skip_precheck.py` + `skip-precheck.json`（2 快照：2 CLEAN / 0 DROP）。
- `prepare_r_{pymovements486,kernel118}.py`：单题 prepare（上游 tarball +
  forward-edit 重建验证 + recipe + source-preparation；kernel118 含第 4 个
  生产文件 `kernels/handlers.py`，断言已加固防漏）。
- `record_one.py` + `f2-push3-*-normalize.json`（2 eligible）+ `summarize.py` +
  `normalize-summary.json`（2 eligible / 0 near-miss / 0 pending；全局 20/40）。
- replay（未动 S600）：`replay/f2-push3-*-{reviewed-recipe,source-preparation,
  replay-result}.json`，`replay/artifacts/f2-push3/{*-source{,.tgz},*-task.json,
  *-run-001(-002),*.log,task-site-packages-{pymovements,kernelgw}}`。

## 双轮环境注记（均只在 `artifacts/f2-push3/` 内新增，未改测试/matcher/要求）

- R1 pymovements：task-python 3.10.20（`artifacts/sfr/task-python` 现货复用）
  + `--source-layout src` + 自建 `task-site-packages-pymovements/`
  （polars 0.18.15/numpy 1.25.2/pandas 2.0.3/pyarrow 11.0.0/scipy 1.11.2/
  matplotlib 3.7.2/tqdm/pytest 8.3.3，经 uv 取数）run-001 干净双轮
  （baseline 恰 25 F2P 失败 / repaired 370 全过）。测试、matcher、要求均未改动。
- R2 kernel_gateway：task-python 3.10.20 + 自建 `task-site-packages-kernelgw/`
  （traitlets/pyzmq 26.1.0/nbformat/jupyter_core/jupyter_client 7.4.9/
  notebook 6.5.6 --no-deps/ipython_genutils/tornado 6.4.1/pytest 8.3.3 等真实包，
  经 uv 取数；pyzmq 24 sdist 本机编译失败故钉有轮子的 26.1.0，
  `zmq.eventloop` 仍可用；notebook 6.5.6 供 `notebook.utils`，无 shim）。
  run-001 为诊断轮（prepare 漏第 4 生产文件致 repaired inline ImportError，
  已保留为证据）；补文件后 run-002 干净双轮（baseline 15 全失败 /
  repaired 15 全过）。测试、matcher、要求均未改动。
- R1/R2 基线 inline 失败均为预期任务复现（side-effect 断言失败 / 隔离 API
  缺席），非策略错误；recipe 同命令双相，行为差完全来自生产修复。

## 缺口说明与下一步

1. mid-suite 151-400 + P2P-empty 新地面已枯：push2 候选 85 行中 35 已 verdict，
   剩余 39 行本批全部逐题 verdict（KEEP 2 + S-R 37），无信号级 SCREENED 余量。
   后续需开新地面（新 total 窗口 / 新信号族 / Gym 未扫区），另行扫描。
2. 2 eligible 中 R1 双文件闭包（370 total，P2P 345 零回归）、R2 四文件闭包
   （15 total，P2P-empty，C2 先例），family/snapshot/instance 全新（程序化校验），
   F2P 正确翻转。
3. 缺口清零前不启动训练、不改 S600 文件。

## 附：对旧文件的非触碰声明

- 本轮只读旧文件（去重用）；旧文件未运行、未覆盖。仅 `f2-push3/` 与
  `replay/f2-push3-*`、`replay/artifacts/f2-push3/` 为新。
- 未动 `f1-new` / `f3-new` / `f4-new` / `f5-new`、S600、`f2-replace/`、
  `f2-gymv2*/`、`f2-large/`、`f2-recall/`、`f2-inline/`、`f2-more/`、
  `f2-push/`、`f2-push2/` 内任何文件。
