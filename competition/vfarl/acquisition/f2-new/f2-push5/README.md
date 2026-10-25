# F600-F2 f2-push5 批次记录（2026-10-25，第十四轮：weak-only 新地面 ≤150/151-200 + P2P-empty≤50-weak + C余量第二轮）

只做题目与 CPU 真实工具数据，不启动训练。
只读使用 `f2-replace/`、`candidate-list-40.json`、`reviewed-candidates.json`、
`catalog.json`、`dedup-baseline.json`（下去重用）与 `f2-gymv2/`、`f2-gymv2b/`、
`f2-gymv2c/`、`f2-large/`、`f2-recall/`、`f2-inline/`、`f2-more/`、`f2-push/`、
`f2-push2/`、`f2-push3/`、`f2-push4/`（40 keeps 只读去重，未改动其任何文件）；
未碰 `f1-new` / `f3-new` / `f4-new` / `f5-new` 与 S600 文件。本批全部产物在
全新目录 `f2-push5/`；replay 零新增（无 KEEP，无双轮）。
Smith-py 冻结未扫。

## 约束口径

- 只从 SWE-Gym（`SWE-Gym/SWE-Gym` train rev `bb94ed9e…`，2438 题）与
  SWE-rebench-V2（`nebius/SWE-rebench-V2` train rev `10483de0…`，
  python 子集 7243 题）找 F2；Smith-py 不再扫。
- 仅 train；排除 `fastapi/fastapi、textualize/rich、psf/requests、encode/httpx` 四仓。
- 与 common-400 + F1-40 + F5旧40 + F2前十三轮（含 f2-gymv2 16 keeps、
  f2-gymv2b 5 keeps、f2-gymv2c 2 keeps、f2-recall 3 keeps、f2-inline 1 keep、
  f2-more 2 keeps、f2-push 6 keeps、f2-push2 3 keeps、f2-push3 2 keeps、
  f2-push4 1 keep 的 family / snapshot / instance）零重叠
  （`dedup-evidence.json`：562i/125s/508f，程序化校验；
  KEEP 为空，无 snapshot_collision，instance 全新）。
- 三元组：`is` 身份 + 双向（共享修改）可见 + 访问/复制时点；无 `is` 退 S，
  纯能力退 R；坏路径（`..`）过滤。inline-is 型指 F2P 仅值断言、
  `is` 腿由 source-bound inline 观察承载（同 C2/K15/nox-439/aiohttp-10672/
  qiskit-12620/M2/P1-P4/Q2-Q3/R1-R2 口径）。
- 本批专属新地面（前十三轮均未逐题复核；total 全部 ≤200 可分批）：
  A池 gym-weak-151-200（total 151-200，strict-copy 排除，strong_n==0，
  weak `list(/dict(/set(/tuple(` + F2P名/ownership/test-is 任一支撑；
  push4-R 仅 V2、push2/3-A 仅 strong，故 gym-weak-151-200 从未复核）+
  B池 V2-weak-151-200-testis-only（V2 total 151-200，非strong inline，
  无 recall 支撑，支撑仅 test-is；push4-R 要求 recall 支撑）+
  C池 P2P-empty≤50-weak（P2P==0，total≤50，strict 排除，strong_n==0，
  weak + 支撑；push2/3-B 仅 strong）+
  D池 weak-nosupport-≤150（total≤150，strict 排除，strong_n==0，
  weak 但无任何支撑；push4-W 要求支撑；lottery 切片）+
  第二轮 C余量 10 行（fn/own/tis 次优 + inplace-F2P 名）。
  扫描 711 行（gym A3/C12/D123 + v2 B10/C116/D447），prior-named 403 排除并
  跨池并集后 unique 688（A2/B8/C116/D562），深读 review-set 65 行
  （A全2 + B全8 + C前30 + D前25）+ 第二轮 10 行，余 613 行记信号级
  SCREENED（不计数，同 push2/push4 口径）。
- 严格性：修复对象必须是引用/所有权隔离缺陷。防御性 `.copy()`/`dict()`
  若只为新逻辑 hygiene（无 caller-owned/shared 映射在险）→ S；
  新 API 本身（无 baseline 别名腿）→ S/R；全文件重写伴随的附带 copy → R。
  （本批实例：MONAI-3205 `dict(self.dataset_params)` 系新 override 能力内
  hygiene，基线从不 mutation → S；patroni-2391 `list()` 系新线程特性内
  hygiene → R；botocore-472 `r = request_dict` 别名系新代码自带非修复对象 → R。）
- skip 预检：无 KEEP，无候选可投（0 CLEAN / 0 DROP，不编造）。
- 白名单题面（verbatim problem_statement）、JUnit 双条件（baseline 全部原 F2P
  失败、repaired 原 matcher 通过）、不改测试、六工具唯一、保护路径不可碰。
  用 `replay/.venv` 执行（本轮零双轮，故只用于扫描/复核；双轮协议见下）。

## 计数（诚实）

- **f2-push5 normalize eligible：0；全局仍 20/40（20+0），缺口 20。**
- 75 逐题复核：triple-KEEP 0 / COLLIDE-family 0 / S 24 / R 51
  （第一轮 65：S 22 + R 43；第二轮 C 余量 10：S 2 + R 8；详见 triple log；
  613 行 SCREENED 不计数）；skip 0 CLEAN / 0 DROP；双轮完成 0
  （0 eligible，0 near-miss）。
- reviewed 0；catalog 0；replay 产物 0。

## 本批产物

- `scan_push5.py` + `gym-push5-scan.json`（A3/C12/D123）/
  `v2-push5-scan.json`（B10/C116/D447）+ `push5-candidates.json`
  （unique 688，A2/B8/C116/D562，prior-named 403 排除）+
  `dump_push5.py` + `push5-review-dump.txt`（65 题 problem+patch+testpatch）/
  `push5-triage.txt`（紧凑版）+ `push5-patches.txt`（65 题 patch 压缩版）/
  `push5-round2.txt`（第二轮 10 题）+ `push5-review-set.json`（75 review /
  613 screened）。
- `dedup-evidence.json`（562i/125s/508f）+ `mk_keeps.py` +
  `keep-list.json`（空）+ `keep-families.json`（空）+
  `reviewed-candidates.json`（0 条）+ `catalog.json`（0 条）。
- `triple-review-log.md`：75 行逐题 verdict（KEEP 0 / COLLIDE 0 /
  S 24 / R 51 全列依据）。
- `skip-precheck.json`（无候选：0 snapshots）+
  `summarize.py` + `normalize-summary.json`（0 eligible / 0 near-miss /
  0 pending；全局 20/40，缺口 20）。
- replay 零新增（无 KEEP，无 `replay/f2-push5-*`，
  无 `replay/artifacts/f2-push5/`）。

## 缺口说明与下一步

1. weak-only 地面真 F2 密度为零（本批 75 行 0 KEEP；连同 push4-W，
   weak 切片累计 162 行仅 1 near-miss KEEP）。Gym+V2、total≤200、
   非 strict/strong 的可复核新地面至此基本扫尽：
   A/B 池已全复核（2+8），C 池已复核 40/116，D 池已复核 25/562
   （余量皆为更低信号同池行）。
2. 缺口 20 清零的可选项（需授权，不在本轮内擅动）：
   a. C 池余量 76 行 + D 池余量 537 行继续深读（同池低信号，期望 ≈0）；
   b. 解冻 Smith-py（`77cab905…`，F2 初始主来源，之后十三轮冻结；
      全新地面，但需取数授权与去重基线扩展）；
   c. 放宽 total≤200（大套件分批协议已在 f2-large 确立，但 push2/3-A
      151-400 strong 已枯；>400 strong 从未扫描）。
3. 缺口清零前不启动训练、不改 S600 文件；后续地面另行扫描授权。

## 附：对旧文件的非触碰声明

- 本轮只读旧文件（去重用）；旧文件未运行、未覆盖。仅 `f2-push5/` 为新，
  无 replay 新增。
- 未动 `f1-new` / `f3-new` / `f4-new` / `f5-new`、S600、`f2-replace/`、
  `f2-gymv2*/`、`f2-large/`、`f2-recall/`、`f2-inline/`、`f2-more/`、
  `f2-push/`、`f2-push2/`、`f2-push3/`、`f2-push4/` 内任何文件。
  （`f2-push4/` 仅由本任务前半段续写 run-002 收官：publish 确认 +
  normalize 重跑为 near-miss + README/triple-log 数字更正，见其续写节。）
