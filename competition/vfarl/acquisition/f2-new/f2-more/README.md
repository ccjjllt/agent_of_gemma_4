# F600-F2 f2-more 批次记录（2026-10-25，第九轮：小套件 ≤150 inline-is 补筛）

只做题目与 CPU 真实工具数据，不启动训练。
只读使用 `f2-replace/`、`candidate-list-40.json`、`reviewed-candidates.json`、
`catalog.json`、`dedup-baseline.json`（下去重用）与
`f2-gymv2/`、`f2-gymv2b/`、`f2-gymv2c/`、`f2-large/`、`f2-recall/`、`f2-inline/`
（23+3+1 keeps 只读去重，未改动其任何文件）；未碰 `f1-new` /
`f3-new` / `f4-new` / `f5-new` 与 S600 文件。本批全部产物在全新目录
`f2-more/`；replay 产物为新 `replay/f2-more-*` 与
`replay/artifacts/f2-more/`（新）。Smith-py 冻结未扫。

## 约束口径

- 只从 SWE-Gym（`SWE-Gym/SWE-Gym` train rev `bb94ed9e…`，2438 题）与
  SWE-rebench-V2（`nebius/SWE-rebench-V2` train rev `10483de0…`，
  python 子集 7243 题）找 F2；Smith-py 不再扫。
- 仅 train；排除 `fastapi/fastapi、textualize/rich、psf/requests、encode/httpx` 四仓。
- 与 common-400 + F1-40 + F5旧40 + F2前八轮（含 f2-gymv2 16 keeps、
  f2-gymv2b 5 keeps、f2-gymv2c 2 keeps、f2-recall 3 keeps、f2-inline 1 keep 的
  family / snapshot / instance）零重叠（`dedup-evidence.json`：548i/111s/494f，
  程序化校验；2 triple-KEEP `snapshot_collision` 全无，instance 全新；
  同家族复本 1 个记 COLLIDE-family，不计入）。
- 三元组：`is` 身份 + 双向（共享修改）可见 + 访问/复制时点；无 `is` 退 S，
  纯能力退 R。inline-is 型指 F2P 仅值断言、
  `is` 腿由 source-bound inline 观察承载（同 C2/K15/nox-439/aiohttp-10672/
  qiskit-12620 口径）。
- 本批专属：小套件 total≤150 的 BROAD-inline changed-line 补筛
  （strict COPY_SIG 行已由 f2-gymv2/b/c 覆盖，本批排除）；
  151–200 已由 f2-recall/f2-inline 覆盖，不在本批。
- skip 前缀碰撞预检逐快照必做；只有 skip-CLEAN ∩ 三元组-KEEP 才投双轮。
- 白名单题面（verbatim problem_statement）、JUnit 双条件（baseline 全部原 F2P
  失败、repaired 原 matcher 通过）、不改测试、六工具唯一、保护路径不可碰。
  用 `replay/.venv` 执行；period-correct 第三方依赖经 `--task-site-packages`
  供给（测试/matcher 不动）。

## 计数（诚实）

- **f2-more normalize eligible：1（qiskit-terra-5778）；near-miss 1
  （psyclone-3029，上游后修漂移，见下）；全局 12/40，缺口 28。**
- 1404 小套件 inline 行（gym 245 + v2 1159）：triple-KEEP 2
 （psyclone-3029、qiskit-terra-5778）/ COLLIDE-family 1（openmdao-3406，
  同 K11 家族）/ 其余 S-R（含 733 tier3 SCREENED-OUT）；
  skip 2 CLEAN / 0 DROP；双轮完成 2（1 eligible + 1 near-miss）。
- reviewed 2；catalog 2；replay 产物见下。

| # | instance | pool | total | 双轮 | normalize |
|---|---|---|---|---|---|
| M1 | stfc__psyclone-3029（get_node_list 单 Schedule 展平拷贝隔离） | V2 | 41 | baseline 10 F2P 全失败 / repaired 10 F2P 全翻转，30/31 P2P 通过（1 P2P 上游后修漂移，见下） | near-miss（严格口径，不放宽） |
| M2 | qiskit__qiskit-terra-5778（调度逐指令 op.copy 时长隔离） | V2 | 19 | baseline 2 F2P 失败 + 17 P2P 通过 / repaired 19 全通过（run-001 干净） | eligible |

- M1 与 K11-近亲 openmdao-3406 同家族
  `OpenMDAO/OpenMDAO::list_driver_vars::read_path_copy_isolation`，
  只计其一（K11 先占），记 COLLIDE-family 不计入。

## 本批产物

- `scan_more.py` + `gym-more-scan.json`（245）/ `v2-more-scan.json`（1159）：
  小套件 BROAD-inline 扫描（strict-copy 行排除；inline/strict/F2P名分层；
  parquet sha 见文件内）。
- `dedup-evidence.json`（548i/111s/494f）+ `keep-list.json`（2 triple-KEEP，
  含完整 F2P/P2P lists）+ `keep-families.json`（2）+ `pages-v2/`（逐题 source
  row + `.receipt.json`，pinned revision API receipts，row_idx/F2P/P2P 校验）+
  `base-resolutions.json`。
- `psyclone.LICENSE`（BSD-3-Clause）+ `qiskit-terra.LICENSE`（Apache-2.0，
  均为上游快照逐字节）+ `reviewed-candidates.json`（2 条完整审阅）+
  `catalog.json`（2 条，含 `source_review_sha256`）。
- `triple-review-log.md`：逐题 verdict（KEEP 2 / COLLIDE-family 1 /
  S-R 代表性全列 + 结构性结论）。
- `skip_precheck.py` + `skip-precheck.json`（2 快照全量：上游抓取证据 +
  前缀碰撞结论；2 CLEAN）。
- `prepare_m_psyclone3029.py` / `prepare_m_qiskit5778.py`：单题 prepare
  （上游 tarball + forward-edit 重建验证 + recipe + source-preparation）。
- `record_one.py` + `f2-more-*-normalize.json` + `summarize.py` +
  `normalize-summary.json`（0 eligible / 1 near-miss / 1 pending；
  全局 11/40，缺口 29）。
- replay（未动 S600）：`replay/f2-more-*-{reviewed-recipe,source-preparation,
  replay-result}.json`，`replay/artifacts/f2-more/`（task.json/source*/run-*/logs，
  `task-site-packages-psyclone/`，`task-site-packages-qiskit-terra/`，
  `psyclone-3029-case-collisions.json`）。
- 双轮环境（均只在 `artifacts/f2-more/` 内新增，未改 Python 源码与测试）：
  parent `replay/.venv`（3.12）+ task-python `artifacts/sfr/task-python/
  cpython-3.10.20` + `task-site-packages-psyclone/`（fparser 0.2.5 /
  pytest 8.3.3 等，经 uv 取数）+ `task-site-packages-qiskit-terra/`
  （retworkx 0.11.0 / numpy 1.21.6 / scipy 1.7.3 / pytest 6.2.5 / ddt 等
  period 钉死版）+ workspace 内本地编译的
  `stochastic_swap/utils/swap_trial.cpython-310-darwin.so`（Cython 0.29.37，
  免上游 Rust；sha 见下）。

## 缺口说明与下一步

1. M1（psyclone-3029）判 near-miss 不放宽：repaired 41 passed 中仅
   `transformations_test.py::test_omploop_trans_new_options`（P2P）失败，
   baseline 双相一致失败（11 failed = 10 F2P + 同一 P2P）。根因经查为上游后修：
   `psyGen.py::stringify_annotation` 在本快照为 `str(annotation)`（plain type
   渲染 `<class 'bool'>`），上游 master 已改为
   `annotation.__name__`（渲染 `'bool'`，测试期望）。该 P2P 只能随上游后修通过，
   与本次 F2 修复（region_trans get_node_list）无关，按 K11/aiohttp-10672 先例
   判 near-miss。F2P 10/10 正确翻转证据完整保留。
2. M2（qiskit-terra-5778）normalize eligible（run-001 干净双轮）：
   baseline 2 F2P 失败 + 17 P2P 通过 / repaired 19 全通过。Cython 双扩展本地
   编译通过（`utils…so 673b0ff9…` / `swap_trial…so ac6e26d2…`），inline 观察
   双相 discriminating（基线共享 op 实例 + 双 1728；修后独立实例 + 时长各异）。
   全局 12/40，缺口 28。
3. 小套件 inline-is 密度：1404 行中真 F2 仅 2（+1 家族复本），其余多为
   `list()/dict()/set()` 构造噪声、`is None` 空检查、散文 `is` 与纯特性；
   口径从严，无 `is` 不投。
4. 缺口清零前不启动训练、不改 S600 文件。

## 附：对旧文件的非触碰声明

- 本轮只读旧文件（去重用）；旧文件未运行、未覆盖。仅 `f2-more/` 与
  `replay/f2-more-*`、`replay/artifacts/f2-more/` 为新。
- 未动 `f1-new` / `f3-new` / `f4-new` / `f5-new`、S600、`f2-replace/`、
  `f2-gymv2*/`、`f2-large/`、`f2-recall/`、`f2-inline/` 内任何文件。
- f2-inline（qiskit-12620）双轮已完成：run-002 baseline 1F/168P +
  repaired 169P，normalize eligible（重验 True/[]）；nox-439 与
  aiohttp-10672 两 near-miss 重验维持原判（严格口径，不放宽，只记录）。
