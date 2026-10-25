# F600-F2 f2-gymv2 批次记录（2026-10-25，第三轮）

只做题目与 CPU 真实工具数据，不启动训练。
只读使用 `f2-replace/`、`f2-replace2/`、`candidate-list-40.json`、
`reviewed-candidates.json`、`catalog.json`（下去重用），未改动它们；
未碰 `f1-new` / `f3-new` / `f4-new` / `f5-new` 与 S600 文件。

## 约束口径

- 只从 SWE-Gym（`SWE-Gym/SWE-Gym` train rev `bb94ed9e…`，2438 题）与
  SWE-rebench-V2（`nebius/SWE-rebench-V2` train rev `10483de0…`，
  python 子集 7243 题）找 F2；Smith-py 不再扫。
- 仅 train；排除 `fastapi/fastapi、textualize/rich、psf/requests、encode/httpx` 四仓。
- 与 common-400 + F1-40 + F5旧40（快照仍禁）+ F2前两轮已占用的
  family / snapshot / instance 零重叠（`dedup-evidence.json`：521i/84s/467f，
  程序化校验；Gym 与 common 重叠的 13 个 Moto 实例均排除）。
- 三元组：`is` 身份 + 双向（共享修改）可见 + 访问/复制时点；无 `is` 退 S，
  纯能力退 R；坏路径（`..`）过滤。
- 小套件优先；Gym/V2 真题放宽到 150 个测试（含 150）。
- skip 前缀碰撞预检逐快照必做（`skip-precheck.json`，16/16 CLEAN，0 DROP；
  openmdao-2810 的测试文件为 test_patch 新建，base 无上游文件，天然无碰撞）。
- 白名单题面（verbatim problem_statement）、JUnit 双条件（baseline 全部原 F2P
  失败、repaired 原 matcher 通过）、不改测试、六工具唯一、保护路径不可碰。
  用 `replay/.venv` 执行；period-correct 第三方依赖经 `--task-site-packages`
  供给（测试/matcher 不动）。

## 计数（诚实）

- **normalize eligible：6 / 40，缺口 34。**
- 双轮完成 7 题（6 eligible + 1 near-miss 待定，见下）。
- reviewed-candidates 12 条（K1–K12）；catalog 12 条。

| # | instance | pool | total | 双轮 | normalize |
|---|---|---|---|---|---|
| K1 | openmdao-2810（_ReprClass copy 保身份） | V2 | 2（P2P=0） | baseline F / repaired T | eligible |
| K12 | asgiref-523（_CVar copy-on-write） | V2 | 15 | F / T | eligible |
| K9 | tablib-579（export Row.copy） | V2 | 136 | F / T | eligible |
| K6 | pystac-826（Asset.clone） | V2 | 29 | F / T（run-002，jsonschema 4.17.3 profile） | eligible |
| K7 | pystac-794（Collection/Catalog/Summaries clone） | V2 | 80 | F / T（同 profile） | eligible |
| K10 | openmdao-3125（hook copy-on-inherit） | V2 | 2 | F / T（scipy profile） | eligible |
| K11 | openmdao-3461（list_driver_vars 读路径） | V2 | 89 | F / T（F2P 通过；1 个 P2P 双相环境失败，见下） | near-miss |

- K11 near-miss：F2P `test_list_driver_vars_bounds_bug` 在双轮中正确翻转
  （baseline ValueError 漂移 / repaired 通过），但 P2P
  `test_duplicate_problem_name` 在双相以同一 `OpenMDAOWarning: 'scaling'
  already active` 未触发而失败（scipy 1.11.4 与 1.14.1 两套 profile 下完全一致，
  与修复无关的 reports 子系统环境漂移）。判 near-miss，不计数。
- K2（pandas-55314，Gym）：教科书 F2（测试含 `is not`），但 workspace pandas
  需 C 编译且 base（2.1-dev）不支持 venv 的 Python 3.12；待 `task-python` 老解释器
  方案，未投双轮。
- K3/K4（MONAI-2104/2010，Gym）：真 F2（题面逐字 aliasing），但需 torch 全栈；
  老 MONAI（2022）与新 torch 兼容风险高，未投双轮。
- K5（nicegui-3046，V2）：真 F2（`__copy__/__deepcopy__`），但 F2P+P2P 中 16/23
  需 Playwright `screen` fixture（浏览器），CPU 沙箱不可行，未投双轮。
- K8（earthkit-671，V2）：真 F2（kwargs 原地改动），但 F2P 需下载远端 GRIB +
  eccodes/cfgrib 原生栈，未投双轮。
- 备份：moto-6716（Gym，149）、meltano-2144、sacred-765（弱）、moto-7960/7961
  （弱）、weasyprint-1016（弱）、specutils-1119（弱）、mypy-12023（弱）、
  modin-6775（环境重）；cookiecutter-967 复核降级为 S。

## 本批产物

- `scan_gym.py` / `scan_v2.py`：strict changed-line copy 扫描（Gym 29 hits、
  V2-python 112 hits）+ `show` 复核输出；parquet 源
  `/tmp/opencode/{gym,v2}-train.parquet`（pinned revision URL 下载，
  sha 见 `*-strict-scan.json`）。
- `skip_precheck.py` + `skip-precheck.json`：16 快照上游测试文件抓取证据与
  前缀碰撞结论（15 直接 CLEAN + 1 新文件 CLEAN，0 DROP）。
- `dedup-evidence.json`：521i/84s/467f 禁用集 + 16 keeps 程序化零重叠校验口径
  （`keep-families.json`；Gym↔V2 236 跨池同实例已 note，keeps 均单池单用；
  与 common 重叠的 13 Gym-Moto 排除）。
- `keep-list.json`（16：12 KEEP + 4 备份）+ `pages-gym/`、`pages-v2/`（逐题
  source row + `.receipt.json`，pinned revision API receipts，row_idx 校验）。
- `*.LICENSE`（8 上游快照 LICENSE）+ `reviewed-candidates.json`（12 条完整审阅）
  + `catalog.json`（12 条，含 `source_review_sha256`）。
- `triple-review-log.md`：全部逐题 verdict（KEEP/S/R/备份 + 结构性结论）。
- `prepare_k{1,6,7,9,10,11,12}.py`：单题 prepare（上游 tarball + forward-edit
  重建验证 + recipe + source-preparation）。
- `record_one.py` + `f2-gymv2-*-normalize.json`：逐题 normalize 结论。
- replay（未动 S600）：`replay/f2-gymv2-*-{reviewed-recipe,source-preparation,
  replay-result}.json`，`replay/artifacts/f2-gymv2/{*-source{,.tgz},*-task.json,
  *-run-00*,task-site-packages-{tablib,pystac,openmdao}}`。
- K6 的 jsonschema==4.17.3 与 K9 的 openpyxl/xlrd/xlwt/odfpy/tabulate、
  K10/K11 的 scipy==1.11.4/networkx/pyDOE2 均为 task-site-packages 环境供给；
  测试、matcher、要求均未改动。

## 缺口说明

1. 真 F2 基准率：Gym strict 29 命中 → KEEP 3（pandas/MONAI×2）+ 备份 2；
   V2 strict 112 命中 → KEEP 9 + 备份 2；补充召回（Gym-34/V2-121）几乎全噪声。
   真题池 F2 密度远高于 Smith-py（<0.5%），但 40 独立家族仍需继续掃 V2 剩余行
   （>150 套件禁入；P2P=0、浏览器/编译/原生栈任务受限）。
2. 12 KEEP 中 7 已双轮（6 eligible），5 待重放（K2/K3/K4/K5/K8，环境原因，
   非三元组原因）。缺口 34 清零前不启动训练、不改 S600 文件。

## 附：对旧文件的非触碰声明

- 本轮只读 `f2-replace/`、`f2-replace2/`、`candidate-list-40.json`、
  `reviewed-candidates.json`、`catalog.json`（去重用）；`scan-hits.json` 等旧文件
  未运行、未覆盖。`dedup-baseline.json` 只读。
