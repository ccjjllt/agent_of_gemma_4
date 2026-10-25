# F600-F2 f2-replace 批次记录（2026-10-25）

只做题目与 CPU 真实工具数据，不启动训练。不动 f1/f3/f4/f5 与 S600 文件。

## 约束口径

- 仅 train；Smith-py `77cab905…13558e`；排除
  `fastapi/fastapi、textualize/rich、psf/requests、encode/httpx` 四仓。
- 与 S600-common-400 的 family/snapshot/instance 零重叠（沿用
  `../dedup-baseline.json` 禁用集，程序化校验）。
- 三元组：`is` 身份 + 双向可见 + 访问时点；无 `is` 退 S、纯能力退 R；
  大 F2P（>30）/大套件（P2P>600）/坏路径/禁仓/mega-rewrite 换小 clean 题。
- 白名单题面、JUnit 双条件、不改测试、六工具、保护路径、仅 train、排除四仓。
- `normalize eligible` 才计数（supplemental=`f2-replace/catalog.json`）。

## 计数（诚实）

- **eligible：0 / 40，缺口 40。**
- 双轮完成 1 题（`f2-replace-deepdiff-01`），结论 near-miss（见下）。
- 三元组逐题复核约 50 项：KEEP（真 F2）1 项，其余 S/R/unclean/硬排除。

## 本批产物

- `build.py`：strict changed-line copy 扫描（全量 pages，限 28 允许仓、
  剔除 common-400 身份）+ shortlist；`strict-scan.json`（58 hits）。
- `prepare_01.py` / `record_01.py`：单题 prepare→replay→normalize 链路。
- `seperman__deepdiff.ed252022.func_pm_ctrl_invert_if__w1di5431.task.json`
  （源行 blob，同旧 40-list SHA `86053bff…`）。
- `reviewed-candidates.json`（1 条完整审阅）+ `catalog.json`（1 条）。
- `deepdiff.LICENSE`（MIT，上游 `ed252022…`）。
- `triple-review-log.md`：全部逐题 verdict（S/R/unclean/排除 + 结构性结论）。
- `normalize-01.json`：normalize 形式结论。
- replay：`replay/f2-replace-deepdiff-01-{reviewed-recipe,source-preparation,
  replay-result}.json`，`replay/artifacts/f2-replace/{f2-deepdiff-w1di5431-source{,.tgz},
  upstream-seperman__deepdiff{,.tgz},f2-replace-deepdiff-01-{task.json,run-001{,.log}}}`。

## 唯一双轮证据（w1di5431，判 near-miss 不计数）

- 三元组（repair 重建隔离方向）：`_to_delta_dict(directed=True)` 时点，
  结果侧 `is not` live store，结果侧改动不污染 live。
- baseline 959 tests/1 failure（恰 F2P
  `test_list_of_alphabet_and_its_delta`）；repaired 959/0 failure。
- `resolved=False`：上游无条件 skip
  （`test_list_difference_ignore_order_report_repetition[234]` 等）经官方
  matcher 前缀规则否决 P2P `test_list_difference`
  （另有 runslow/remapping skip 多重否决）；normalize 双原因同 click
  near-miss。禁改三件套下不可修。

## 缺口说明与下一步建议

1. 28 仓取数已全（`../pages`，含收据；scan-hits/scan-used 已合并 15 仓）。
   新仓真 F2 为零；旧 40-list 经复核仅 w1di5431 为真但结构性不可入选。
2. 结构性阻塞（两快照实证）：`pallets/click@fde47b4b`（macOS skip + xfail
   前缀）与 `seperman/deepdiff@ed252022`（无条件 skip 前缀否决）对全套件
   P2P 任务不可入选；mido 系坏路径。后续 F2 只能来自小套件 + 无上游
   skip 前缀碰撞的新快照，且须先跑单轮收集 skip 碰撞预检再投双轮。
3. 缺口清零前不启动训练、不改 S600 文件。
