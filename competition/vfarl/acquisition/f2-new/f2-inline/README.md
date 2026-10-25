# F600-F2 f2-inline 批次记录（2026-10-25，第八轮：417 SCREENED-OUT 中补筛 inline-is 型）

只做题目与 CPU 真实工具数据，不启动训练。
只读使用 `f2-replace/`、`f2-replace2/`、`candidate-list-40.json`、
`reviewed-candidates.json`、`catalog.json`、`dedup-baseline.json`（下去重用）与
`f2-gymv2/`、`f2-gymv2b/`、`f2-gymv2c/`、`f2-large/`、`f2-recall/`
（23 keeps + 3 recall keeps 只读去重，未改动其任何文件）；未碰 `f1-new` /
`f3-new` / `f4-new` / `f5-new` 与 S600 文件。本批全部产物在全新目录
`f2-inline/`；replay 产物为新 `replay/f2-inline-*` 与
`replay/artifacts/f2-inline/`（新）。Smith-py 冻结未扫。

## 约束口径

- 只从 SWE-Gym（`SWE-Gym/SWE-Gym` train rev `bb94ed9e…`，2438 题）与
  SWE-rebench-V2（`nebius/SWE-rebench-V2` train rev `10483de0…`，
  python 子集 7243 题）找 F2；Smith-py 不再扫。
- 仅 train；排除 `fastapi/fastapi、textualize/rich、psf/requests、encode/httpx` 四仓。
- 与 common-400 + F1-40 + F5旧40 + F2前七轮（含 f2-gymv2 16 keeps、
  f2-gymv2b 5 keeps、f2-gymv2c 2 keeps、f2-recall 3 keeps 的 family /
  snapshot / instance）零重叠（`dedup-evidence.json`：547i/110s/493f，
  程序化校验；1 triple-KEEP `snapshot_collision` 全无，instance 全新；
  同家族复本 2 个记 COLLIDE-family，不计入）。
- 三元组：`is` 身份 + 双向（共享修改）可见 + 访问/复制时点；无 `is` 退 S，
  纯能力退 R；坏路径（`..`）过滤。inline-is 型指 F2P 仅值断言、
  `is` 腿由 source-bound inline 观察承载（同 C2/K15/nox-439/aiohttp-10672 口径）。
- 本批专属：补筛 recall 417 SCREENED-OUT（le200 无 REAL_IS 行）。
  本地重算 le200-forbidden-excl 462 行（112 gym + 350 v2；与任务称 417 差
  45，为 forbidden 含 recall-3 + 跨池复本 11 + REAL_IS 口径（散文 `is`）抖动，
  如实记录；去重后唯一实例 451）。
  Gym/V2 真题放宽到 ≤200（含 200）；>200 不在本批。
- skip 前缀碰撞预检逐快照必做；只有 skip-CLEAN ∩ 三元组-KEEP 才投双轮。
- 白名单题面（verbatim problem_statement）、JUnit 双条件（baseline 全部原 F2P
  失败、repaired 原 matcher 通过）、不改测试、六工具唯一、保护路径不可碰。
  用 `replay/.venv` 执行；period-correct 第三方依赖经 `--task-site-packages`
  供给（测试/matcher 不动）。

## 计数（诚实）

- **f2-inline normalize eligible：1（qiskit-12620）；全局 11/40（6+2+2+1），缺口 29。**
- 451 唯一实例：triple-KEEP 1（qiskit-12620，dual eligible）/ COLLIDE-family 2
  （qiskit-12621 同家族、aiohttp-10674 同家族）/ LARGE-DONE 13 / COLLIDE-snapshot 2
  / REALIS48 48（已在 f2-recall 复核：KEEP 3 / S-R 45）/ 其余 S-R 385；
  skip 1 CLEAN / 0 DROP；双轮完成 1（eligible，见下）。
- reviewed 1；catalog 1；replay 产物见下。

| # | instance | pool | total | 双轮 | normalize |
|---|---|---|---|---|---|
| I1 | Qiskit__qiskit-12620（copy_empty_like 参数缓存失效隔离） | V2 | 169 | baseline 168 pass + F2P 1 fail / repaired 169 pass（run-002） | eligible |

- I1 与 I-复本 qiskit-12621 同家族
  `Qiskit/qiskit::qiskit.circuit.quantumcircuit.QuantumCircuit.copy_empty_like::parameter_cache_invalidation_isolation`，
  只计其一；aiohttp-10674 与 f2-recall R2 同家族
  `aio-libs/aiohttp::aiohttp.web_response.Response.headers::copy_isolation`，
  记 COLLIDE-family 不计入。

## 本批产物

- `scan_inline.py` + `gym-inline-scan.json`（112）/ `v2-inline-scan.json`（350）：
  le200-forbidden-excl 重排扫描（forbidden 含 23+3 keeps；inline/strict/F2P名分层；
  parquet sha 见文件内）。
- `dedup-evidence.json`（547i/110s/493f）+ `keep-list.json`（1 triple-KEEP）+
  `keep-families.json`（1）+ `pages-v2/`（逐题 source row + `.receipt.json`，
  pinned revision API receipts，row_idx 校验）+ `base-resolutions.json`。
- `qiskit.LICENSE`（Apache-2.0，上游快照逐字节）+
  `reviewed-candidates.json`（1 条完整审阅）+ `catalog.json`（1 条，
  含 `source_review_sha256`）。
- `triple-review-log.md`：全部逐题 verdict（KEEP 1 / COLLIDE-family 2 /
  LARGE-DONE 13 / COLLIDE-snapshot 2 / REALIS48 48 / S-R 385 + 结构性结论）。
- `skip_precheck.py` + `skip-precheck.json`（1 快照全量：上游抓取证据 +
  前缀碰撞结论）。
- `prepare_i_qiskit12620.py`：单题 prepare（上游 tarball + forward-edit 重建验证
  + recipe + source-preparation）。
- `record_one.py` + `f2-inline-qiskit-12620-normalize.json` + `summarize.py` +
  `normalize-summary.json`（1 eligible / 0 near-miss / 0 pending；全局 11/40，缺口 29）。
- replay（未动 S600）：`replay/f2-inline-*-{reviewed-recipe,source-preparation,
  replay-result}.json`，`replay/artifacts/f2-inline/{*-source{,.tgz},*-task.json,
  *-run-001,*-run-002,task-site-packages-qiskit}`。
- 双轮环境（均只在 `artifacts/f2-inline/` 内新增/补齐，未改 Python 源码与测试）：
  parent `replay/.venv`（3.12）+ task-python `artifacts/sfr/task-python/
  cpython-3.10.20`（与 install_config `python_base_310` 一致）+
  `task-site-packages-qiskit/`（numpy 1.26.4 / scipy 1.13.1 / rustworkx 0.15.1 /
  sympy 1.12.1 / dill / stevedore 5.2.0 / symengine / ddt / pytest 8.3.3 等 period 钉死版，
  另含官方 `qiskit-terra==0.46.2` wheel 的 dist-info 以供给 transpile 插件 entry points）+
  workspace 内补官方同版本 wheel 的预编译 `qiskit/_accelerate.abi3.so` /
  `qiskit/_qasm2.abi3.so`（免 Rust 重编；sha 见下）。
  run-001 为诊断轮（缺 entry points 时 13 个 transpile 相关 P2P 双相失败，
  F2P 正确翻转，已保留为证据）；补 dist-info 后 run-002 干净双轮。

## 缺口说明与下一步

1. 417 中 inline-is 密度极低：全量 OWN 拥有信号仅 10 行（含复本），
   真 F2 仅 qiskit 对（F2P 值断言 + `cpy._parameters=None` 缓存失效，
   同 hdmf-848 `clear_cache` 族）。其余 OWN 行多为附带 copy
   （dvc path_info、modin dtypes、moto/qiskit 注释级）或大家族值 bug
   （aiohttp connector 特性束），按口径退 S/R。
2. qiskit-12620 为单文件闭包（F2P+P2P 全在
   `test/python/circuit/test_parameters.py`，169 total），patch 单 hunk，
   家族/快照/instance 全新，是理想 file-scope 候选；重依赖（qiskit 0.46 Rust
   `_accelerate` 编译，类 K2 pandas-C 编译先例）已按 task-python 方案解决：
   task-python 3.10.20 + period 依赖 profile + 官方同版本 wheel 预编译 `.so` +
   dist-info entry points，run-002 双轮干净（baseline 仅 F2P 失败 /
   repaired 169 全过），normalize eligible。 workspace `.so` sha256：
   `_accelerate.abi3.so e2a47c9913288d8f…65bc0a9` / `_qasm2.abi3.so ae7cfa05f0cfa440…e514855`
   （出自 `qiskit_terra-0.46.2-cp38-abi3-macosx_11_0_arm64.whl`）。
3. aiohttp-10674 为 R2 同家族复本（同 patch、同 F2P、不同基线
   `a60d447a` vs `3d11b551`），记 COLLIDE-family；qiskit-12621 为 I1 同家族
   复本（同 patch、同 F2P、不同基线 `9092ee78` vs `2b2be030`），记 COLLIDE-family。
   两者均不计入 40。
4. 缺口清零前不启动训练、不改 S600 文件。

## 附：对旧文件的非触碰声明

- 本轮只读旧文件（去重用）；旧文件未运行、未覆盖。仅 `f2-inline/` 与
  `replay/f2-inline-*`、`replay/artifacts/f2-inline/` 为新。
- 未动 `f1-new` / `f3-new` / `f4-new` / `f5-new`、S600、`f2-replace/`、
  `f2-replace2/`、`f2-gymv2*/`、`f2-large/`、`f2-recall/` 内任何文件。
