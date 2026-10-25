# F600-F2 f2-gymv2b 批次记录（2026-10-25，第四轮：补 34 的第一批）

只做题目与 CPU 真实工具数据，不启动训练。
只读使用 `f2-replace/`、`f2-replace2/`、`candidate-list-40.json`、
`reviewed-candidates.json`、`catalog.json`（下去重用）与 `f2-gymv2/`
（16 keeps 只读去重，未改动其任何文件）；未碰 `f1-new` / `f3-new` /
`f4-new` / `f5-new` 与 S600 文件。本批全部产物在全新目录 `f2-gymv2b/`；
replay 产物在 `replay/artifacts/f2-gymv2b/`（新）与
`replay/f2-gymv2b-*-reviewed-recipe/source-preparation/replay-result.json`（新）。

## 约束口径

- 只从 SWE-Gym（`SWE-Gym/SWE-Gym` train rev `bb94ed9e…`，2438 题）与
  SWE-rebench-V2（`nebius/SWE-rebench-V2` train rev `10483de0…`，
  python 子集 7243 题）找 F2；Smith-py 不再扫。
- 仅 train；排除 `fastapi/fastapi、textualize/rich、psf/requests、encode/httpx` 四仓。
- 与 common-400 + F1-40 + F5旧40 + F2前三轮（含 f2-gymv2 16 keeps 的
  family / snapshot / instance）零重叠（`dedup-evidence.json`：537i/100s/483f，
  程序化校验；f2-gymv2 的 5 待双轮与 3 备份只读去重，未重做）。
- 三元组：`is` 身份 + 双向（共享修改）可见 + 访问/复制时点；无 `is` 退 S，
  纯能力退 R；坏路径（`..`）过滤。
- 小套件优先；Gym/V2 真题放宽到 150 个测试（含 150）；>150 的 35 行
  （Gym 7 + V2 28）已计数排除，分批依据待补，未纳入本批。
- skip 前缀碰撞预检逐快照必做（`skip-precheck.json`，5/5 CLEAN，0 DROP）。
- 白名单题面（verbatim problem_statement）、JUnit 双条件（baseline 全部原 F2P
  失败、repaired 原 matcher 通过）、不改测试、六工具唯一、保护路径不可碰。
  用 `replay/.venv` 执行；period-correct 第三方依赖经 `--task-site-packages`
  供给（测试/matcher 不动）。

## 计数（诚实）

- **f2-gymv2 eligible 6/40 + f2-gymv2b eligible 2 = 全局 8/40，缺口 32。**
- 本批 normalize eligible：2（jinja-2026、ariadne-159）；双轮完成 3（含 1 near-miss）；
  reviewed 5；缺口 34→32（本批 +2）。
- 剩余 120 strict hits（25 Gym + 95 V2 未复核为 KEEP 的）+ 35 个 >150 待分批，
  见 `triple-review-log.md` 代表性 REJECT；Gym pandas-50714 / modin-6775 机制真 F2
  但因 C 编译/重依赖环境受限未纳入 KEEP（诚实记环境待定）。

| # | instance | pool | total | 双轮 | normalize |
|---|---|---|---|---|---|
| K13 | altair-518（update_nested copy=True 隔离） | V2 | 7（P2P=0） | 待（`collections.Mapping` 时代漂移，见下） | pending |
| K14 | jinja-2026（Undefined copy/pickle 协议） | V2 | 52 | baseline 12 F2P 全 F / repaired 全 T | eligible |
| K15 | ariadne-159（operation deepcopy 后去 mixin） | V2 | 3 | baseline F2P F / repaired T（task-site-packages-ariadne） | eligible |
| K16 | specutils-1119（with_spectral_axis_unit meta/WCS 隔离） | V2 | 29 | 待（astropy/gwcs 原生栈，见下） | pending |
| K17 | openmdao-3522（get_val copy=True 隔离） | V2 | 110 | F2P 正确翻转；P2P test_duplicate_problem_name 双相环境失败（同 K11） | near-miss |

- K17 near-miss：F2P 2 个 `test_get_val` 在双轮中正确翻转
  （baseline TypeError/值漂移 / repaired 通过），但 P2P
  `test_duplicate_problem_name` 在双相以同一 reports-warning 未触发而失败
  （与 K11 同因：`OpenMDAOWarning: 'scaling' already active` 环境漂移，
  task-site-packages-openmdao profile；与修复无关）。判 near-miss，不计数。
- K13 pending：三元组完整（`is not`/`is` 在 F2P 逐字），workspace+recipe 已备；
  base（2018）`collections.Mapping` 在 venv Python 3.12 已移除，
  repaired 在 .venv 同样失败（时代漂移），待 `task-python` 老解释器方案，未投双轮。
- K16 pending：三元组完整（新谱改动后原谱不动，F2P 逐字），workspace+recipe 已备；
  需 astropy/gwcs 原生栈（venv 无 astropy），未投双轮。
- cfn-lint-3342/2031/4045 诚实记 S：虽为工作拷贝隔离，但 F2P 无 `is`、无改动隔离断言、
  题面未提共享；放宽即得 3，但违反 is 口径，不纳入。

## 本批产物

- `scan_b.py` + `gym-strict-scan.json`（25 hits）/ `v2-strict-scan.json`（100 hits）：
  strict changed-line copy 扫描（forbidden 含 f2-gymv2 16 keeps；>150 排除 35 行计数）。
- `dedup-evidence.json`（537i/100s/483f）+ `keep-list.json`（5）+
  `keep-families.json`（5）+ `pages-v2/`（逐题 source row + `.receipt.json`，
  pinned revision API receipts，row_idx 校验）+ `base-resolutions.json`。
- `*.LICENSE`（5 上游快照 LICENSE：altair/jinja BSD-3，ariadne BSD-3，
  specutils BSD-3，openmdao Apache-2.0）+ `reviewed-candidates.json`（5 条完整审阅）
  + `catalog.json`（5 条，含 `source_review_sha256`）。
- `triple-review-log.md`：全部逐题 verdict（KEEP/S/R/备份 + 结构性结论）。
- `skip_precheck.py` + `skip-precheck.json`：5 快照上游测试文件抓取证据与
  前缀碰撞结论（5 CLEAN，0 DROP）。
- `prepare_b_{jinja,ariadne,openmdao3522,altair,specutils}.py`：单题 prepare
  （上游 tarball + forward-edit 重建验证 + recipe + source-preparation）。
- `record_one.py` + `f2-gymv2b-*-normalize.json`（3）+ `summarize.py` +
  `normalize-summary.json`（2 eligible / 1 near-miss / 2 pending）。
- replay（未动 S600）：`replay/f2-gymv2b-*-{reviewed-recipe,source-preparation,
  replay-result}.json`，`replay/artifacts/f2-gymv2b/{*-source{,.tgz},*-task.json,
  *-run-001,task-site-packages-ariadne}`。
- K15 的 graphql-core==3.2.3/pydantic==1.10.13/isort/autoflake/black/click/httpx/toml
  与 K17 的 scipy 等均为 task-site-packages 环境供给；测试、matcher、要求均未改动。

## 缺口说明与下一步

1. 本批 5 KEEP 中 3 已双轮（2 eligible + 1 near-miss），2 待重放（K13/K16，环境原因，
   非三元组原因）。全局 8/40，缺口 32。
2. 继续找 32 个：剩余 120 strict hits 需逐题复核（本批方法：F2P `is` 信号 +
   patch copy + 题面共享语义三方对齐，弱候选诚实记 S/R）；35 个 >150 需分批依据；
   Gym pandas-50714 / modin-6775 与 K13/K16 需老解释器/重依赖方案。
3. 缺口清零前不启动训练、不改 S600 文件。

## 附：对旧文件的非触碰声明

- 本轮只读 `f2-replace/`、`f2-replace2/`、`candidate-list-40.json`、
  `reviewed-candidates.json`、`catalog.json`、`dedup-baseline.json` 与 `f2-gymv2/`
 （去重用）；旧文件未运行、未覆盖。`git status` 可验（仅 `f2-gymv2b/` 与
  `replay/f2-gymv2b-*`、`replay/artifacts/f2-gymv2b/` 为新）。
