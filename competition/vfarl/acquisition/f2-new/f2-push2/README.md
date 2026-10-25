# F600-F2 f2-push2 批次记录（2026-10-25，第十一轮：mid-suite 151-400 + P2P-empty inline-is）

只做题目与 CPU 真实工具数据，不启动训练。
只读使用 `f2-replace/`、`candidate-list-40.json`、`reviewed-candidates.json`、
`catalog.json`、`dedup-baseline.json`（下去重用）与 `f2-gymv2/`、`f2-gymv2b/`、
`f2-gymv2c/`、`f2-large/`、`f2-recall/`、`f2-inline/`、`f2-more/`、`f2-push/`
（33 keeps 只读去重，未改动其任何文件）；未碰 `f1-new` / `f3-new` /
`f4-new` / `f5-new` 与 S600 文件。本批全部产物在全新目录 `f2-push2/`；
replay 产物为新 `replay/f2-push2-*` 与 `replay/artifacts/f2-push2/`（新）。
Smith-py 冻结未扫。

## 约束口径

- 只从 SWE-Gym（`SWE-Gym/SWE-Gym` train rev `bb94ed9e…`，2438 题）与
  SWE-rebench-V2（`nebius/SWE-rebench-V2` train rev `10483de0…`，
  python 子集 7243 题）找 F2；Smith-py 不再扫。
- 仅 train；排除 `fastapi/fastapi、textualize/rich、psf/requests、encode/httpx` 四仓。
- 与 common-400 + F1-40 + F5旧40 + F2前十轮（含 f2-gymv2 16 keeps、
  f2-gymv2b 5 keeps、f2-gymv2c 2 keeps、f2-recall 3 keeps、f2-inline 1 keep、
  f2-more 2 keeps、f2-push 6 keeps 的 family / snapshot / instance）零重叠
  （`dedup-evidence.json`：556i/119s/502f，程序化校验；3 triple-KEEP
  `snapshot_collision` 全无，instance 全新；同家族复本 1 个记
  COLLIDE-family，不计入）。
- 三元组：`is` 身份 + 双向（共享修改）可见 + 访问/复制时点；无 `is` 退 S，
  纯能力退 R；坏路径（`..`）过滤。inline-is 型指 F2P 仅值断言、
  `is` 腿由 source-bound inline 观察承载（同 C2/K15/nox-439/aiohttp-10672/
  qiskit-12620/M2/P1-P4 口径）。
- 本批专属新地面（前十轮均未逐题复核）：A池 mid-suite total 151-400
  BROAD-inline strong（strict-copy 排除，f2-inline 200 cap 之外）+
  B池 P2P-empty total≤50 inline-strong（push要求total≥2之外、C2 total=1
  先例）。扫描 118 行（A 49 + B 69），prior-named 269 排除后候选 85 行
  （A 41 + B 44），深读 ~15 行，KEEP 3。
- 严格性：修复对象必须是引用/所有权隔离缺陷。防御性 `.copy()` 若只为
  新逻辑 hygiene（无 caller-owned/shared 映射在险）→ S；新 API 本身
  （无 baseline 别名腿）→ S/R；全文件重写伴随的附带 copy → R。
- skip 前缀碰撞预检逐快照必做；只有 skip-CLEAN ∩ 三元组-KEEP 才投双轮
  （cfn-426 系 test_patch-new 文件，base 无上游文件，天然无碰撞，
  同 f2-gymv2-K1 / f2-gymv2c-1613 先例，记 CLEAN）。
- 白名单题面（verbatim problem_statement）、JUnit 双条件（baseline 全部原 F2P
  失败、repaired 原 matcher 通过）、不改测试、六工具唯一、保护路径不可碰。
  用 `replay/.venv` 执行；src 布局仓库经 `--source-layout src` 绑定；
  period-correct 第三方依赖经 `--task-site-packages` / `--task-python`
  供给（测试/matcher 不动）。

## 计数（诚实）

- **f2-push2 normalize eligible：2；全局 18/40（16+2），缺口 22。**
- 85 候选中深读 ~15 行：triple-KEEP 3 / COLLIDE-family 1 / REJECT/HELD/SCREENED
  余下（详见 triple log；未逐题深读者记 SCREENED，不计数）；
  skip 3 CLEAN / 0 DROP；双轮完成 2（2 eligible，0 near-miss）；
  1 Rust-pending（qiskit-terra-8741，见下，不计数）。
- reviewed 3；catalog 3；replay 产物见下。

| # | instance | pool | total | 双轮 | normalize |
|---|---|---|---|---|---|
| Q1 | Qiskit__qiskit-terra-8741（idle-wires qubits/clbits 拷贝隔离） | V2 | 209 | 未投（Rust _accelerate-pending） | pending |
| Q2 | awslabs__cfn-python-lint-426（convert_dict 迭代快照拷贝隔离） | V2 | 1 | baseline F2P 失败 / repaired 全过（run-002；run-001 见下） | eligible |
| Q3 | more-itertools__more-itertools-412（spy head 拷贝隔离） | V2 | 323 | baseline 仅 F2P 失败 / repaired 323 全过（run-001） | eligible |

- COLLIDE-family：qiskit-terra-8764（同 M2 调度 op-copy 家族，不计入）。

## 本批产物

- `scan_push2.py` + `gym-push2-scan.json`（A11/B7）/ `v2-push2-scan.json`
  （A38/B62）+ `push2-candidates.json`（A41/B44，prior-named 269 排除）+
  `push2-review-dump.txt`（85 题 problem+patch+testpatch）/
  `push2-triage.txt`（紧凑版）。
- `dedup-evidence.json`（556i/119s/502f）+ `keep-list.json`（3 triple-KEEP，
  含完整 F2P/P2P lists）+ `keep-families.json`（3）+ `pages-v2/`（3，
  逐题 source row + `.receipt.json`，pinned revision API receipts，
  row_idx/F2P/P2P 校验）+ `base-resolutions.json`（3，全 40 位免解析）。
- `qiskit-terra/cfn-python-lint/more-itertools.LICENSE`（上游快照逐字节：
  Apache-2.0/MIT No Attribution/MIT）+ `reviewed-candidates.json`（3 条
  完整审阅）+ `catalog.json`（3 条，含 `source_review_sha256`）。
- `triple-review-log.md`：KEEP 3 全列三元组 + COLLIDE 1 + REJECTS 深读
  依据 + SCREENED 说明。
- `skip_precheck.py` + `skip-precheck.json`（3 快照：2 直取 CLEAN +
  cfn-426 test_patch-new CLEAN 共 3 CLEAN / 0 DROP）。
- `prepare_q_{more412,cfn426,qiskit8741}.py`：单题 prepare（上游 tarball +
  forward-edit 重建验证 + recipe + source-preparation）。
- `record_one.py` + `f2-push2-*-normalize.json`（2 eligible）+ `summarize.py` +
  `normalize-summary.json`（2 eligible / 0 near-miss / 1 pending；全局 18/40）。
- replay（未动 S600）：`replay/f2-push2-*-{reviewed-recipe,source-preparation,
  replay-result}.json`，`replay/artifacts/f2-push2/{*-source{,.tgz},*-task.json,
  *-run-001(-002),task-site-packages-cfn426}`。

## 双轮环境注记（均只在 `artifacts/f2-push2/` 内新增，未改测试/matcher/要求）

- Q3 more-itertools：parent `.venv`（3.12，纯 python 无额外依赖）。
  run-001 干净双轮（baseline 323 tests/1 failure 恰为 F2P `test_immutable` /
  repaired 323 全过）。
- Q2 cfn-python-lint：run-001 为诊断轮（缺 `--source-layout src`，
  `cfnlint` 模组未找到，双相 error，已保留为证据）；run-002 加
  `--source-layout src` + task-python 3.10.20 + 自建
  `task-site-packages-cfn426/`（six/pyyaml/requests/aws-sam-translator/
  jsonpatch/pytest 8.3.3/setuptools 68，经 uv 取数）干净双轮
  （baseline 1 test/F2P 失败 / repaired 1 全过）。测试、matcher、要求均未改动。
- Q1 qiskit-terra：Rust 全栈（`qiskit._accelerate` 需 Rust 构建，0.22.0
  源码 tarball 无预编译 `.so`；I1 0.46 workspace 自带 `.so` 先例不适用；
  预编译 0.46.2 `_accelerate` 与 0.22 源码不兼容，workspace 优先 shadow
  致 import 失败，已用 /tmp 探针验证）。本批 pending，不投双轮，不计数。

## 缺口说明与下一步

1. mid-suite 151-400 + P2P-empty 新地面首批 3 KEEP 已投 2（2 eligible）。
   85 候选中剩余 ~70 行信号级 SCREENED（多文件/重依赖/大套件为主），
   可按同口径继续深读（单文件/小F2P/ownership 优先）。
2. 2 eligible 全部单文件闭包（1/323 total），family/snapshot/instance
   全新（程序化校验），F2P 正确翻转，P2P 零回归。
3. 缺口清零前不启动训练、不改 S600 文件。

## 附：对旧文件的非触碰声明

- 本轮只读旧文件（去重用）；旧文件未运行、未覆盖。仅 `f2-push2/` 与
  `replay/f2-push2-*`、`replay/artifacts/f2-push2/` 为新。
- 未动 `f1-new` / `f3-new` / `f4-new` / `f5-new`、S600、`f2-replace/`、
  `f2-gymv2*/`、`f2-large/`、`f2-recall/`、`f2-inline/`、`f2-more/`、
  `f2-push/` 内任何文件。
