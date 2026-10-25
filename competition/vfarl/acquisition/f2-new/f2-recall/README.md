# F600-F2 f2-recall 批次记录（2026-10-25，第七轮：recall 非平凡大行 2024 重排分批）

只做题目与 CPU 真实工具数据，不启动训练。
只读使用 `f2-replace/`、`f2-replace2/`、`candidate-list-40.json`、
`reviewed-candidates.json`、`catalog.json`、`dedup-baseline.json`（下去重用）与
`f2-gymv2/`、`f2-gymv2b/`、`f2-gymv2c/`、`f2-large/`（23 keeps + 35 大套件依据，
只读去重，未改动其任何文件）；未碰 `f1-new` / `f3-new` / `f4-new` / `f5-new`
与 S600 文件。本批全部产物在全新目录 `f2-recall/`；replay 产物为新
`replay/f2-recall-*` 与 `replay/artifacts/f2-recall/`（新）。
Smith-py 冻结未扫。

## 约束口径

- 只从 SWE-Gym（`SWE-Gym/SWE-Gym` train rev `bb94ed9e…`，2438 题）与
  SWE-rebench-V2（`nebius/SWE-rebench-V2` train rev `10483de0…`，
  python 子集 7243 题）找 F2；Smith-py 不再扫。
- 仅 train；排除 `fastapi/fastapi、textualize/rich、psf/requests、encode/httpx` 四仓。
- 与 common-400 + F1-40 + F5旧40 + F2前六轮（含 f2-gymv2 16 keeps、
  f2-gymv2b 5 keeps、f2-gymv2c 2 keeps 的 family / snapshot / instance）零重叠
  （`dedup-evidence.json`：544i/107s/490f，程序化校验；3 triple-KEEP
  `snapshot_collision` 全无（2 CLEAN 进入 reviewed；hdmf-848 triple-KEEP 但
  skip-DROP 阻塞，保留在 keep-list/triple-log/skip-precheck 做去重，
  排除在 reviewed/catalog 之外），instance 全新）。
- 三元组：`is` 身份 + 双向（共享修改）可见 + 访问/复制时点；无 `is` 退 S，
  纯能力退 R；坏路径（`..`）过滤。
- Recall 大行专属：本批处理此前因 `total>150` 被排除的 recall 非平凡行
  （Gym 771 + V2 1254 = 2025 行；任务口径 2024，差 1 为 forbidden 含 f2-gymv2c
  后 parquet 复算抖动，如实记录）；`≤150` 严格池已枯（f2-gymv2c），
  strict 大行 35 已在 f2-large 分批（本批不重做）。
- 本批套件放宽到 ≤200（含 200）；>200 弃（1557 行：gym 658 / v2 899）。
  可分批 468 行（gym 113 + v2 355）。
- 重排：strict-copy + REAL_IS（`assert…is/is not` 标识符，排除 `is None/True/False`
  空检查与散文 `is`）+ F2P 名 + total 升序。REAL_IS ≤200 共 51 行，
  扣 LARGE-DONE 13（strict 已复核）中 2、COLLIDE 3 中 1，剩 48 行逐题三元组复核。
  其余 417 可分批行无 REAL_IS，按“无 is 不投”记 SCREENED-OUT。
- skip 前缀碰撞预检逐快照必做（`skip-precheck.json`，3 triple-KEEP 中
  2 CLEAN / 1 DROP）；只有 skip-CLEAN ∩ 三元组-KEEP 才投双轮。
- 白名单题面（verbatim problem_statement）、JUnit 双条件（baseline 全部原 F2P
  失败、repaired 原 matcher 通过）、不改测试、六工具唯一、保护路径不可碰。
  用 `replay/.venv` 执行；period-correct 第三方依赖经 `--task-site-packages`
  供给（测试/matcher 不动）。

## 计数（诚实）

- **f2-recall normalize eligible：0；全局仍 10/40（6+2+2），缺口 30。**
- 48 逐题复核：triple-KEEP 3（nox-439、aiohttp-10672、hdmf-848）/ S-R 45；
  skip 2 CLEAN / 1 DROP（hdmf-848：h5tools 5 skip markers 致 6 碰撞，阻塞）；
  双轮完成 2（nox-439 near-miss + aiohttp-10672 near-miss），0 eligible。
- reviewed 2；catalog 2；replay 产物见下。

| # | instance | pool | total | 双轮 | normalize |
|---|---|---|---|---|---|
| R1 | theacodes__nox-439（SessionRunner posargs 切片拷贝隔离） | V2 | 159 | baseline F2P F / repaired F2P T；1 P2P 双相环境失败（缺安装元数据） | near-miss |
| R2 | aio-libs__aiohttp-10672（Response headers 拷贝隔离） | V2 | 153 | baseline F2P F / repaired F2P T；全部 152 P2P 通过，1 extra flaky 双相失败 | near-miss |
| R3 | hdmf-dev__hdmf-848（export clear_cache 拷贝） | V2 | 160 | 未投（skip-DROP 阻塞） | blocked |

- R3 triple-KEEP 但 skip-DROP：证据保留在 keep-list（3）/triple-log/skip-precheck
  做去重，排除在 reviewed/catalog（2）之外。

## 本批产物

- `scan_recall_large.py` + `gym-recall-large-scan.json`（771 非平凡/le200 113）/
  `v2-recall-large-scan.json`（1254/le200 355）：recall 非平凡大行重排扫描
  （forbidden 含 23 keeps；strict/REAL_IS/F2P名分层；parquet sha 见文件内）。
- `show_recall.py` + `realis-48-dump.txt`：48 REAL_IS 候选逐题复核工作文件。
- `dedup-evidence.json`（544i/107s/490f）+ `keep-list.json`（3 triple-KEEP）+
  `keep-families.json`（3）+ `pages-v2/`（逐题 source row + `.receipt.json`，
  pinned parquet provenance）+ `base-resolutions.json`（2 CLEAN）。
- `nox.LICENSE` + `aiohttp.LICENSE`（Apache-2.0，上游快照逐字节）+
  `reviewed-candidates.json`（2 条完整审阅）+ `catalog.json`（2 条，
  含 `source_review_sha256`）。
- `triple-review-log.md`：48 行逐题 verdict（KEEP 3 / S-R 45 + 结构性结论 +
  LARGE-DONE 13 + COLLIDE 3 + SCREENED-OUT 417 + >200 弃 1557）。
- `skip_precheck.py` + `skip-precheck.json`（3 快照全量：上游抓取证据 +
  前缀碰撞结论；2 CLEAN / 1 DROP）。
- `prepare_r_nox439.py` / `prepare_r_aiohttp10672.py`：单题 prepare
  （上游 tarball + forward-edit 重建验证 + recipe + source-preparation）。
- `record_one.py` + `f2-recall-nox-439-normalize.json`（near-miss）+
  `f2-recall-aiohttp-10672-normalize.json`（near-miss）+ `summarize.py` +
  `normalize-summary.json`（0 eligible / 2 near-miss / 0 pending / 1 blocked）。
- replay（未动 S600）：`replay/f2-recall-*-{reviewed-recipe,source-preparation,
  replay-result}.json`，`replay/artifacts/f2-recall/{*-source{,.tgz},*-task.json,
  *-run-001,task-site-packages-{nox,aiohttp}}`。
- nox 的 argcomplete/colorlog/virtualenv/py 与 aiohttp 的 blockbuster/xdist
  均为 task-site-packages 环境供给；测试、matcher、要求均未改动。

## 缺口说明与下一步

1. Recall 大行 REAL_IS 密度低（48/468 ≈ 10%），真 F2 更稀（triple-KEEP 3/48），
   且多为 inline-`is` 型（F2P 仅值断言，`is` 腿由 source-bound 观察承载，
   同 C2/K15 口径）。strict-copy 信号在 recall 大行几乎缺席（13 strict 全为
   f2-large 已复核），新真 F2 来自非 strict 拷贝（切片 `[:]`、构造拷贝
   `CIMultiDict(h)`、缓存重建）。
2. nox-439 near-miss 根因：P2P `test_load_nox_module_needs_version_static`
   需已安装分发元数据（上游 CI 安装后通过），sandbox 未安装 workspace，
   双相一致失败；F2P 正确翻转。判 near-miss 不计数（K11/K17 先例）。
   若未来协议允许 workspace editable 安装，可转 eligible。
3. hdmf-848 skip-DROP 与三元组无关（matcher 前缀规则；h5tools skip 密集）。
   即使三元组过也需先解 DROP。
4. 后续方向（需新授权）：417 SCREENED-OUT 行（无 REAL_IS）中可能藏 inline-`is`
   型真 F2（如 F2P 值断言 + 拷贝补丁，类似 nox/aiohttp 但 test_patch 无 `is`），
   需逐题读 patch/题面补筛；>200 弃行另议分批；pandas/MONAI/torch 系重依赖另议。
   缺口清零前不启动训练、不改 S600 文件。

## 附：对旧文件的非触碰声明

- 本轮只读旧文件（去重用）；旧文件未运行、未覆盖。仅 `f2-recall/` 与
  `replay/f2-recall-*`、`replay/artifacts/f2-recall/` 为新。
- 未动 `f1-new` / `f3-new` / `f4-new` / `f5-new`、S600、`f2-replace/`、
  `f2-replace2/`、`f2-gymv2*/`、`f2-large/` 内任何文件。
