# F600-F2 f2-large 批次记录（2026-10-25，第六轮：大套件 35 行分批依据）

只做题目与 CPU 真实工具数据，不启动训练。
只读使用 `f2-replace/`、`f2-replace2/`、`candidate-list-40.json`、
`reviewed-candidates.json`、`catalog.json`、`dedup-baseline.json`（下去重用）与
`f2-gymv2/`、`f2-gymv2b/`、`f2-gymv2c/`（23 keeps 只读去重，未改动其任何文件）；
未碰 `f1-new` / `f3-new` / `f4-new` / `f5-new` 与 S600 文件。
本批全部产物在全新目录 `f2-large/`；replay 产物为零（验证集为空，故无新增
`replay/f2-large-*` 与 `replay/artifacts/f2-large/`）。
Smith-py 冻结未扫（`scan_large.py` 无 Smith 分支；gpxpy-143 参照见下）。

## 约束口径

- 只从 SWE-Gym（`SWE-Gym/SWE-Gym` train rev `bb94ed9e…`，2438 题）与
  SWE-rebench-V2（`nebius/SWE-rebench-V2` train rev `10483de0…`，
  python 子集 7243 题）找 F2；Smith-py 不再扫。
- 仅 train；排除 `fastapi/fastapi、textualize/rich、psf/requests、encode/httpx` 四仓。
- 与 common-400 + F1-40 + F5旧40 + F2前五轮（含 f2-gymv2 16 keeps、
  f2-gymv2b 5 keeps、f2-gymv2c 2 keeps 的 family / snapshot / instance）零重叠
  （`dedup-evidence.json`：544i/107s/490f，程序化校验；35 行
  `snapshot_collision` 全无，instance 全新）。
- 三元组：`is` 身份 + 双向（共享修改）可见 + 访问/复制时点；无 `is` 退 S，
  纯能力退 R；坏路径（`..`）过滤。
- 大套件专属：本批只处理此前因 `total>150` 被排除的 strict-copy 35 行
  （Gym 7 + V2 28）；`≤150` 严格池已枯，不重做。
- skip 前缀碰撞预检逐快照必做（`skip-precheck.json`，25 CLEAN / 10 DROP；
  3 个 test_patch-new 文件 404 按 K1/C2 先例重分类为 CLEAN，见下）。
- 白名单题面（verbatim problem_statement）、JUnit 双条件（baseline 全部原 F2P
  失败、repaired 原 matcher 通过）、不改测试、六工具唯一、保护路径不可碰。
  用 `replay/.venv` 执行（本轮零双轮，故只用于扫描/预检；双轮协议见下）。

## 大套件分批协议（本批确立的依据）

- 顺序：skip 预检 → 三元组复核 → 单文件 file-scope 分批验证。
  只有 skip-CLEAN ∩ 三元组-KEEP 才投双轮。
- file-scope：只跑目标文件（`verification_target_files=[单文件]`），原 matcher
  不动，不断言缩水（F2P/P2P 原样保留）。normalize 要求 `file_scope` 为真且
  `missing_files` 为空，故单文件闭包（全部 F2P+P2P 同一文件）可直接 file-scope；
  多文件需一次执行全部 required 文件（仍 file-scope，仍不动 matcher），超大
  套件另议 verification timeout 与 task-site-packages。
- 本批单文件闭包 10/35（CLEAN 8 + DROP 2）；多文件 25/35。验证集为空
  （CLEAN∩KEEP=∅），故零双轮——协议本身是本批交付物。

## 计数（诚实）

- **f2-large normalize eligible：0；全局仍 10/40（6+2+2），缺口 30。**
- 35 行：skip 25 CLEAN / 10 DROP；三元组 KEEP 0（CLEAN 中 S 16 + R 9；
  DROP 10 阻断，兼 S/R）；双轮完成 0。
- reviewed 0；catalog 0；replay 产物 0。

## 本批产物

- `scan_large.py` + `gym-large-scan.json`（7）/ `v2-large-scan.json`（28）：
  strict changed-line copy 且 `total>150` 专属扫描（forbidden 含 23 keeps；
  单文件闭包标记 `single_file_required_closure`；parquet sha 见文件内）。
- `dedup-evidence.json`（544i/107s/490f）+ `skip_precheck.py` +
  `skip-precheck.json`（35 快照全量：upstream 抓取证据 + 前缀碰撞结论；
  25 CLEAN / 10 DROP；3 重分类注记）。
- `triple-review-log.md`：35 行逐题 verdict（KEEP 0 / S / R / DROP + 结构性结论
  + 单文件表 + Smith-gpxpy 参照）。
- `keep-list.json`（[]）+ `keep-families.json`（{}）+
  `reviewed-candidates.json`（0 条）+ `catalog.json`（0 条）+
  `normalize-summary.json`（0 eligible，全局 10/40，缺口 30）。
- 无 `pages-*/`（KEEP 为空故无逐题 source row 落盘；35 行清单在
  `*-large-scan.json` 内含 F2P 全量 + P2P 样例 + hits；全量 F2P/P2P 在
  pinned parquet 中可复现，`scan_large.py show <pool> <instance_id>` 可调阅）；
  无 `*.LICENSE`（无 KEEP 快照）；无 prepare/record/normalize 单题文件。

## 缺口说明与下一步

1. 大套件 strict-copy 命中几乎全为附带 copy：值 bug 防御拷贝（moto/isort/RDT/
   music21/babel 系）、文档/签名（music21-791、pandas-57518）、新能力
   （fsspec/dulwich/stingray/hydra/sceptre/MONAI 系）。`is` 腿缺席是唯一共性
   （`is` 命中皆散文/`is None`/`isinstance`/非 F2P 测试）；pennylane-4540 的
   `is not`/`is` 在非 F2P 测试，F2P 仅 `==`，按口径退 S。
2. 10 DROP 与三元组无关（matcher 前缀规则；pennylane/pandas/dask/moto 系
   skip 密集）。即使三元组过也需先解 DROP。
3. Smith 参照：`tkrajina__gpxpy.09fc46b3.func_basic__fwnb1y0h`（总数 143，
   单文件 `gpxpy/gpx.py`，`GPX.clone: deepcopy→copy` 教科书 F2，skip CLEAN）
   仍是最近的真 F2 大套件例子，但 Smith 冻结，本批不动；若未来放宽 Smith
   尺寸门槛，它是单文件 file-scope 首选。
4. 后续大套件方向（需新授权）：V2/Gym recall 信号（clone-changed 等）中
   `total>150` 的非 trivial 行（gym 770 / v2 1254，按 `scan_c.py recall` 口径）
   尚未分批；多文件 file-scope 执行超时与重依赖（torch/C 编译/pandas 全量）
   另议。缺口清零前不启动训练、不改 S600 文件。

## 附：对旧文件的非触碰声明

- 本轮只读旧文件（去重用）；旧文件未运行、未覆盖。`git status` 可验
  （仅 `f2-large/` 为新；`replay/` 无新增）。
- 未动 `f1-new` / `f3-new` / `f4-new` / `f5-new`、S600、`f2-replace/`、
  `f2-replace2/`、`f2-gymv2*/` 内任何文件。
