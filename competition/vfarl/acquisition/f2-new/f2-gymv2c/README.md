# F600-F2 f2-gymv2c 批次记录（2026-10-25，第五轮：补 32 的第一批）

只做题目与 CPU 真实工具数据，不启动训练。
只读使用 `f2-replace/`、`f2-replace2/`、`candidate-list-40.json`、
`reviewed-candidates.json`、`catalog.json`（下去重用）与 `f2-gymv2/`、
`f2-gymv2b/`（21 keeps 只读去重，未改动其任何文件）；未碰 `f1-new` / `f3-new` /
`f4-new` / `f5-new` 与 S600 文件。本批全部产物在全新目录 `f2-gymv2c/`；
replay 产物在 `replay/artifacts/f2-gymv2c/`（新）与
`replay/f2-gymv2c-*-reviewed-recipe/source-preparation/replay-result.json`（新）。

## 约束口径

- 只从 SWE-Gym（`SWE-Gym/SWE-Gym` train rev `bb94ed9e…`，2438 题）与
  SWE-rebench-V2（`nebius/SWE-rebench-V2` train rev `10483de0…`，
  python 子集 7243 题）找 F2；Smith-py 不再扫。
- 仅 train；排除 `fastapi/fastapi、textualize/rich、psf/requests、encode/httpx` 四仓。
- 与 common-400 + F1-40 + F5旧40 + F2前四轮（含 f2-gymv2 16 keeps 与 f2-gymv2b 5 keeps 的
  family / snapshot / instance）零重叠（`dedup-evidence.json`：542i/105s/488f，
  程序化校验 + `build_reviewed.py` 逐题断言）。
- 三元组：`is` 身份 + 双向（共享修改）可见 + 访问/复制时点；无 `is` 退 S，
  纯能力退 R；坏路径（`..`）过滤。
- 小套件优先；Gym/V2 真题放宽到 150 个测试（含 150）；>150 的 35 行
  （Gym 7 + V2 28）仍排除，未纳入本批。
- skip 前缀碰撞预检逐快照必做（`skip-precheck.json`，2/2 CLEAN，0 DROP；
  1613 的目标测试文件为 test_patch 新建，base 无上游文件，天然无碰撞，
  同 f2-gymv2 K1 先例）。
- 白名单题面（verbatim problem_statement）、JUnit 双条件（baseline 全部原 F2P
  失败、repaired 原 matcher 通过）、不改测试、六工具唯一、保护路径不可碰。
  用 `replay/.venv` 执行；src 布局仓库经 `--source-layout src` 绑定
  （`.venv` 自带 jinja2 3.1.6 会 shadow workspace，recipe 内 binding 断言已验）。

## 计数（诚实）

- **f2-gymv2 eligible 6 + f2-gymv2b eligible 2 + f2-gymv2c eligible 2 = 全局 10/40，缺口 30。**
- 本批 normalize eligible：2（jinja-2029、jinja-1613）；双轮完成 2（含 0 near-miss）；
  reviewed 2；缺口 32→30（本批 +2）。

| # | instance | pool | total | 双轮 | normalize |
|---|---|---|---|---|---|
| C1 | pallets__jinja-2029（missing 单例 copy/pickle 保身份） | V2 | 27 | baseline 7 F2P 全 F / repaired 全 T | eligible |
| C2 | pallets__jinja-1613（async_variant pickle 按引用保身份，P2P-EMPTY） | V2 | 1 | baseline F2P F / repaired T | eligible |

- C1/C2 与 K14（jinja-2026，同仓）快照/生产符号/家族/F2P 均不同，不视为重叠
  （程序化校验通过；同仓多快照先例：openmdao×4、pystac×2）。

## 本批产物

- `scan_c.py` + `gym-strict-scan.json`（25 hits）/ `v2-strict-scan.json`（95 hits）+
  `gym-recall-scan.json`（64）/ `v2-recall-scan.json`（233）+ `v2-isnew-screen.json`
  （173）/ `gym-isnew-screen.json`（21）+ `v2-f2pname-screen.json`（148）/
  `gym-f2pname-screen.json`（24）+ `v2-statescreen.json`（64）+ triage 工作文件。
- `dedup-evidence.json`（542i/105s/488f）+ `keep-list.json`（2）+
  `keep-families.json`（2）+ `pages-v2/`（逐题 source row + `.receipt.json`，
  pinned revision API receipts，row_idx 校验）+ `base-resolutions.json`（见下）。
- `jinja.LICENSE`（BSD-3-Clause，两快照逐字节一致，sha 与 f2-gymv2b 一致
  `3b49dcee…`）+ `reviewed-candidates.json`（2 条完整审阅）
  + `catalog.json`（2 条，含 `source_review_sha256`）。
- `triple-review-log.md`：全部逐题 verdict（KEEP/S/R + 结构性结论）。
- `skip_precheck.py` + `skip-precheck.json`：2 快照 CLEAN。
- `prepare_c_jinja2029.py` / `prepare_c_jinja1613.py`：单题 prepare
  （上游 tarball + forward-edit 重建验证 + recipe + source-preparation）。
- `record_one.py` + `f2-gymv2c-*-normalize.json`（2）+ `summarize.py` +
  `normalize-summary.json`（2 eligible / 0 near-miss / 0 pending）。
- replay（未动 S600）：`replay/f2-gymv2c-*-{reviewed-recipe,source-preparation,
  replay-result}.json`，`replay/artifacts/f2-gymv2c/{*-source{,.tgz},*-task.json,
  *-run-001}`。无 task-site-packages 需求（markupsafe 3.0.4 era 兼容）。

## 缺口说明与下一步

1. 本批 2 KEEP 全部双轮 eligible。全局 10/40，缺口 30。
2. strict 池已枯（141 hits → 23 keeps，前后三轮取尽，剩余多为已复核 S/R）；
   本轮 6 新信号约 800 行已筛，剩余多为命名噪声（alias/shared-libs/frozen）、
   值语义 bug、无 `is` 腿的 pickle 保真修与重依赖任务（pandas-C 编译/MONAI-torch/
   TF/Fortran/MPI/浏览器）。
3. 缺口清零前不启动训练、不改 S600 文件。

## 附：对旧文件的非触碰声明

- 本轮只读 `f2-replace/`、`f2-replace2/`、`candidate-list-40.json`、
  `reviewed-candidates.json`、`catalog.json`、`dedup-baseline.json` 与 `f2-gymv2/`、
  `f2-gymv2b/`（去重用）；旧文件未运行、未覆盖。`git status` 可验（仅 `f2-gymv2c/` 与
  `replay/f2-gymv2c-*`、`replay/artifacts/f2-gymv2c/` 为新）。
