# F600-F2 f2-push4 批次记录（2026-10-25，第十三轮：weak-offset ≤150 + V2大recall余量 + P2P-empty 51-200 新窗口）

只做题目与 CPU 真实工具数据，不启动训练。
只读使用 `f2-replace/`、`candidate-list-40.json`、`reviewed-candidates.json`、
`catalog.json`、`dedup-baseline.json`（下去重用）与 `f2-gymv2/`、`f2-gymv2b/`、
`f2-gymv2c/`、`f2-large/`、`f2-recall/`、`f2-inline/`、`f2-more/`、`f2-push/`、
`f2-push2/`、`f2-push3/`（40 keeps 只读去重，未改动其任何文件）；未碰 `f1-new` /
`f3-new` / `f4-new` / `f5-new` 与 S600 文件。本批全部产物在全新目录 `f2-push4/`；
replay 产物为新 `replay/f2-push4-*` 与 `replay/artifacts/f2-push4/`（新）。
Smith-py 冻结未扫。

## 约束口径

- 只从 SWE-Gym（`SWE-Gym/SWE-Gym` train rev `bb94ed9e…`，2438 题）与
  SWE-rebench-V2（`nebius/SWE-rebench-V2` train rev `10483de0…`，
  python 子集 7243 题）找 F2；Smith-py 不再扫。
- 仅 train；排除 `fastapi/fastapi、textualize/rich、psf/requests、encode/httpx` 四仓。
- 与 common-400 + F1-40 + F5旧40 + F2前十二轮（含 f2-gymv2 16 keeps、
  f2-gymv2b 5 keeps、f2-gymv2c 2 keeps、f2-recall 3 keeps、f2-inline 1 keep、
  f2-more 2 keeps、f2-push 6 keeps、f2-push2 3 keeps、f2-push3 2 keeps 的
  family / snapshot / instance）零重叠（`dedup-evidence.json`：561i/124s/507f，
  程序化校验；1 triple-KEEP `snapshot_collision` 全无，instance 全新；
  同库同基线同 fix 复本 2 个记 COLLIDE-family，不计入；
  跨池同实例 1 个（moto-6317 gym+v2）合并计一次，判 S）。
- 三元组：`is` 身份 + 双向（共享修改）可见 + 访问/复制时点；无 `is` 退 S，
  纯能力退 R；坏路径（`..`）过滤。inline-is 型指 F2P 仅值断言、
  `is` 腿由 source-bound inline 观察承载（同 C2/K15/nox-439/aiohttp-10672/
  qiskit-12620/M2/P1-P4/Q2-Q3/R1-R2 口径）。
- 本批专属新地面（前十二轮均未逐题复核；total 全部 ≤200 可分批）：
  W池 weak-offset（total≤150，strict-copy 排除，strong_n==0，
  weak `list(/dict(/set(/tuple(` + F2P名/ownership/test-is 任一支撑；
  前批小套件池皆要求 strong_n>0）+
  R池 V2大recall余量（V2 total 151-200，非strong inline + recall支撑）+
  E池 P2P-empty 新窗口（P2P==0，total 51-200；前批B池仅 ≤50）。
  扫描 708 行（gym W91/R0/E0 + v2 W499/R37/E81），prior-named 317 排除并
  跨池并集后 unique 619（W582/R27/E70），深读 review-set 87 行
  （R全27 + E前30 + W-only前30），余 532 行记信号级 SCREENED（不计数，
  同 push2 口径）。
- 严格性：修复对象必须是引用/所有权隔离缺陷。防御性 `.copy()` 若只为
  新逻辑 hygiene（无 caller-owned/shared 映射在险）→ S；新 API 本身
  （无 baseline 别名腿）→ S/R；全文件重写伴随的附带 copy → R。
- skip 前缀碰撞预检逐快照必做；只有 skip-CLEAN ∩ 三元组-KEEP 才投双轮。
  精度注记：初版 heuristic 把注释行 `# @unittest.skip` 误归因报 DROP；
  注释行不可能是 decorator，不影响收集/matcher，属误报，已修
  （规则不变：真实碰撞仍 DROP；详见 triple log）。
- 白名单题面（verbatim problem_statement）、JUnit 双条件（baseline 全部原 F2P
  失败、repaired 原 matcher 通过）、不改测试、六工具唯一、保护路径不可碰。
  用 `replay/.venv` 执行；period-correct 第三方依赖经 `--task-site-packages` /
  `--task-python` 供给（测试/matcher 不动；eodag 另有 metadata-only
  dist-info + entry_points 转录，见下）。

## 计数（诚实）

- **f2-push4 normalize eligible：0；全局仍 20/40（20+0），缺口 20。**
- 87 逐题复核：triple-KEEP 1（`cs-si__eodag-253`）/ COLLIDE-family 2
  （`cs-si__eodag-254`、`cs-si__eodag-260`，同库同基线同 `products[:]` fix、
  同 F2P，不计入）/ S 37 / R 47（详见 triple log；532 行 SCREENED 不计数）；
  skip 1 CLEAN / 0 DROP；双轮完成 1（0 eligible，1 near-miss，见下）。
- reviewed 1；catalog 1；replay 产物见下。

| # | instance | pool | total | 双轮 | normalize |
|---|---|---|---|---|---|
| S1 | cs-si__eodag-253（download_all products 切片拷贝隔离） | V2 | 109 | baseline 全文件收集错误（test_patch 引用新生产符号，见下）/ repaired 106/109（余 3 live-service 环境失败，见下） | near-miss |

- S1 双轮详况（run-001 诊断 + run-002 取证，均保留）：
  baseline 7 文件全部收集错误（`tests/context.py` 经 test_patch 引用新生产符号
  `path_to_uri`，基线生产无此符号；属“测试补丁依赖修复本身”的收集失败，
  非逐题 F2P 执行失败，normalize 计
  `original_F2P_failure_not_observed_in_baseline_JUnit`）；
  repaired run-002：在 entry_points 转录 + shapely 1.8.5 + click 8.0.4
  period 供给后，106/109 F2P 通过；余 3 个 end-to-end F2P 失败皆为外部
  live 服务漂移（astraea 回传 `'Not Available'` 非 WKB 几何致解析错误、
  earth_search_cog 下载 0 字节、peps 搜空；2021 年后服务端行为变更，
  与本修复及本环境无关），属环境失败（nox-439 先例），不计模型负例。
  判 near-miss，不计数。

## 本批产物

- `scan_push4.py` + `gym-push4-scan.json`（W91/R0/E0）/ `v2-push4-scan.json`
  （W499/R37/E81）+ `push4-candidates.json`（unique 619，W582/R27/E70，
  prior-named 317 排除，跨池并集）+ `dump_push4.py` +
  `push4-review-dump.txt`（87 题 problem+patch+testpatch）/
  `push4-triage.txt`（紧凑版）+ `push4-review-set.json`（87 review /
  532 screened）。
- `dedup-evidence.json`（561i/124s/507f）+ `mk_keeps.py` + `keep-list.json`
  （1 triple-KEEP，含完整 F2P/P2P lists）+ `keep-families.json`（1）+
  `pages-v2/`（1，逐题 source row + `.receipt.json`，pinned revision API
  receipts，row_idx/F2P/P2P 校验）+ `base-resolutions.json`。
- `eodag.LICENSE`（Apache-2.0，上游快照逐字节）+
  `reviewed-candidates.json`（1 条完整审阅）+ `catalog.json`（1 条，
  含 `source_review_sha256`）。
- `triple-review-log.md`：87 行逐题 verdict（KEEP 1 / COLLIDE-family 2 /
  S 37 / R 47 全列依据 + skip 精度注记）。
- `skip_precheck.py` + `skip-precheck.json`（1 快照：7 文件全量 +
  注释行修正；1 CLEAN / 0 DROP）。
- `prepare_s_eodag253.py`：单题 prepare（上游 tarball + forward-edit 重建验证
  35 hunks/10 文件 + recipe + source-preparation）。
- `record_one.py` + `f2-push4-eodag-253-normalize.json`（near-miss）+
  `summarize.py` + `normalize-summary.json`（0 eligible / 1 near-miss /
  0 pending；全局 20/40，缺口 20）。
- replay（未动 S600）：`replay/f2-push4-*-{reviewed-recipe,source-preparation,
  replay-result}.json`，`replay/artifacts/f2-push4/{eodag-253-source{,.tgz},
  *-task.json,*-run-001(-002),*.log,task-site-packages-eodag}`。

## 双轮环境注记（均只在 `artifacts/f2-push4/` 内新增，未改测试/matcher/要求）

- task-python 3.10.20（`artifacts/sfr/task-python` 现货复用）+ 自建
  `task-site-packages-eodag/`（click 8.0.4/shapely 1.8.5/flasgger 0.9.5/
  owslib/geojson/pyproj/usgs/boto3/flask/whoosh/pystac/faker/pytest 等，
  经 uv 取数；setuptools 75 供 pkg_resources；boto3/botocore 先被 usgs
  钉到 2015 版后升级回现代版；eodag 本体仅 metadata-only dist-info
  2.2.0（含逐字转录 setup.py entry_points 的 entry_points.txt，
  无代码 shim，workspace 源码仍是唯一 import 源）。
- run-001 为诊断轮（缺 entry_points 致全插件 PluginNotFound + 现代
  shapely/click 行为漂移；baseline 全文件收集错误已定位为
  test_patch 新符号依赖，已保留为证据）。
- run-002 为取证轮（entry_points/period 依赖补齐；baseline 仍全文件收集错误
  ——test_patch 引用 `path_to_uri` 而基线生产无此符号，属任务固有结构，
  非策略错误；repaired 106/109 F2P 通过，3 个原 F2P 失败皆为 end-to-end
  live-service（astraea/earth_search_cog/peps，见续写）。测试、matcher、
  要求均未改动。
- S1 基线 inline 失败为预期任务复现（caller 序列 3→1 被消费，side-effect
  断言失败），非策略错误；recipe 同命令双相（leaf 网络下载 stub，
  真实生产 retry/consume 循环双相执行），行为差完全来自生产修复。

## 缺口说明与下一步

1. weak-offset 池经验证几乎全为附带构造（`list()/dict()/set()/tuple()` 值语义）
  或新能力（R 47），真 F2 密度极低（KEEP 1/87）；P2P-empty 51-200 新窗口多为
  大 PR 新特性（R 占优）；V2 151-200 weak-recall 余量多为值修复（S 占优）。
2. 唯一 KEEP（eodag-253）因 JUnit 双条件不可达判 near-miss：
  baseline 腿（test_patch 新符号致全文件收集错误，逐题 F2P 不可观察）+
  repaired 腿（7 end-to-end F2P 需已下线/不可达 live 服务）。
  family/snapshot/instance 全新（程序化校验），机制本身成立但严格口径下不计数。
3. 缺口清零前不启动训练、不改 S600 文件；后续需开更新地面（另行扫描授权）。

## 续写（2026-10-25 晚：run-002 收官，publish+normalize 完成）

- run-002 已完成（`replay-output run-002`，phase_b 311s，EXIT=0）。
  `replay/f2-push4-eodag-253-replay-result.json` 已 publish（patch_sha
  `7fbe4646…` 与 run-002 `submitted.patch` 一致，tool_events 41），
  `record_one.py` normalize 已重跑：仍 3 reasons
  （`original_F2P_failure_not_observed_in_baseline_JUnit` +
  `actual_local_official_repair_not_resolved` +
  `original_official_JUnit_matcher_rejects_repaired_report`），判 near-miss。
- 精确 JUnit（normalize 同口径复算）：baseline 7 文件全收集错误，
  109 F2P 全部 unmatched（0 observed 失败）；repaired 139 collected /
  24F+1E+2S，其中原 F2P 106 通过、3 失败，失败皆为 end-to-end live-service：
  `test_end_to_end_search_all_astraea_eod_iterate`、
  `test_end_to_end_search_download_earth_search_cog`、
  `test_end_to_end_wrong_credentials_peps`。
- 更正本文 `$S1 双轮详况` 与 `triple-review-log.md` S1 节旧句
  （"repaired 102/109，7 end-to-end" / "baseline 109 全失败/repaired 109 全过"/
  "双轮完成 1（1 eligible）"）：run-001 诊断轮数字已作废，以 run-002 取证为准；
  **f2-push4 normalize eligible 仍 0，全局仍 20/40，缺口 20。**
  后续 20 个缺口由全新目录 `f2-push5/` 承接（新地面，见其 README）。

## 附：对旧文件的非触碰声明

- 本轮只读旧文件（去重用）；旧文件未运行、未覆盖。仅 `f2-push4/` 与
  `replay/f2-push4-*`、`replay/artifacts/f2-push4/` 为新。
- 未动 `f1-new` / `f3-new` / `f4-new` / `f5-new`、S600、`f2-replace/`、
  `f2-gymv2*/`、`f2-large/`、`f2-recall/`、`f2-inline/`、`f2-more/`、
  `f2-push/`、`f2-push2/`、`f2-push3/` 内任何文件。
