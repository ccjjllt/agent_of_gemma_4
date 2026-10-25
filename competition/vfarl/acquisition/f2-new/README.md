> 2026-10-26 复核更新：下文为旧采集历史，不能作为当前完成数。已重新核对原20条真实通过证据并修正6条机制归类；新增实际通过数据、来源和失败边界以 [f2-audit/README.md](f2-audit/README.md) 为准。旧的全仓/全快照禁用、字面is三元组与100测试数限制属于旧搜索启发，不是比赛准入要求。

# F600-F2（引用/所有权隔离）40 新任务：采集与重放记录

日期 2026-10-25。只做题目与 CPU 真实工具数据，不启动训练，不动 S600 已有文件。

## 硬约束执行口径

- 仅 train；Smith-py `77cab9055d42ab4a5c25c89a8f937096db13558e` 为主，Gym/Rebench-V2 可补；排除 `fastapi/fastapi、textualize/rich、psf/requests、encode/httpx` 四仓。
- 与 S600-common-400 的 family/snapshot/instance 全部不重复。基线见 `dedup-baseline.json`（400 families、64 snapshots、400 instances、49 repo_keys；来源 `s600/course-manifest.json` + `acquisition/supplemental/catalog.json` + `pilot96/catalog.json`，三者 SHA 已固定）。
- 每题 F2 三元组：`is` 身份断言 + 共享修改可见性（改 copy 侧原侧可见/或明确隔离边界）+ 复制时点。只有 `==` 无 `is` 打回退 S；只有能力识别退 R。
- 流程同 S：`acquisition/f2-new/reviewed-candidates.json` → replay 双轮官方重放（baseline 失败 + repaired 通过）→ `normalize` eligible 才计数。白名单题面（verbatim publisher problem_statement）、JUnit 双条件（baseline 全部原 F2P 失败、repaired 原 matcher 通过）、不改测试、六工具唯一、保护路径不可碰。用 `replay/.venv` 执行。

## 当前产物（本轮）

- `dedup-baseline.json`：S600-common-400 禁用集（400/64/400）。
- `fetch_scan.py` + `pages/train-*.json`（112 文件/56 页，含 receipt；第二批 10 仓取数中因 429 停止，已缓存部分页）+ `scan-hits.json`（第一批 542 行，5 个 F2 信号；第二批未完成覆盖）。
- `reviewed-candidates.json`（1 条完整审阅：click meta shared）+ `catalog.json`（同 1 条，normalize 用 supplemental 格式）。
- `candidate-list-40.json`：40 项 provisional 清单（4 快照：dspy 6、sunpy 1、pydantic 26、deepdiff 7；F2P/P2P、strong 信号、文件、row_idx、task SHA；家族 provisional，仅 25 独立，待人工去重；replay 全部 pending，其中 click 另计为 near-miss，不占 40）。
- 现货 `.task.json` 42 个：40 清单 + `mido__mido…cyg5oo56`（路径 `../dev/tests` 坏，需退回）+ `pallets__click…82024b34`（near-miss 已重放）。
- `click.LICENSE`（BSD-3-Clause，上游 `fde47b4b…`）、数据集 README SHA 沿用 `18afda…808`。
- replay 新文件（未动 S600）：`replay/f2-new-click-meta-shared-{reviewed-recipe,source-preparation,replay-result}.json`，`replay/artifacts/f2-new/f2-new-click-meta-shared-{task.json,source/,run-001/,run-001.log,run-002/,run-002.log}`，`replay/artifacts/f2-new/{upstream/,*.tgz,*.revision.json}`。

## 首个双轮证据（click meta shared，F2 三元组完整但判不合格）

- 配方：`read src/click/core.py:523-548` + 基线 `run_command`（`m1 is m2`、`m1 is ctx.meta`、`m1['f2key']` 经 `ctx.meta`/`m2` 可见；expect error）+ `edit`（`return self._meta.copy()` → `return self._meta`，锚唯一）+ 修后同命令（expect ok）+ `submit_patch`。六工具、白名单题面、保护路径未碰。
- run-001：缺 `tests/test_context.py`（buggy 源缺失，上游补 `test_patch` 14899B 后）→ run-002 正常双轮。
- run-002：baseline 634 tests/2 failures（恰为 2 个原 F2P：`test_context_meta`、`test_make_pass_meta_decorator`，证据 profile `all_original_F2P_failed`，complete replay True）；repaired 634 tests/0 failures/611 passed/23 skipped，但 `resolved=False`。
- 不合格双原因（normalize 直接调用确认）：`actual_local_official_repair_not_resolved` + `original_official_JUnit_matcher_rejects_repaired_report`。根因一：必需 P2P `tests/test_types.py::test_path_surrogates` 在 macOS 因 FS 不支持 non-UTF-8 而 skip；根因二：必需 `tests/test_shell_completion.py::test_group` 被同前缀 xfail `tests.test_chain::test_group_chaining` 按 `_node_matches_requirement` 前缀规则污染（S 称 Dask9573 类问题）。两者均为严格口径下不可修（不改测试/matcher、不删要求），该题计 near-miss，不计入 40。
- 同快照本地仅此 1 个 copy 任务；同仓其他 copy 需等第二批扫描（click 全仓 645 unused，7 页）。

## 40 清单口径与缺口（诚实计数）

- `normalize eligible`：0/40。本轮唯一双轮为 click near-miss（catalog 外单测：eligible False，见上）。
- 40 清单：`candidate-list-40.json` 40 项，均为 Smith-py train、非四仓、快照与 common-400 无重复（4 新快照），F2P 1–119、P2P 189–5437，strong 信号含 `__copy__/__deepcopy__/model_copy/copy.copy/copy.deepcopy/deepcopy(/def copy(`，坏路径（`../`）已滤除。但：
  1. 家族 provisional 仅 25 独立（pydantic 26 项多共享 `pydantic/main.py.BaseModel`，需人工按生产函数/因果拆分或用第二批多样仓替换 15 项）；
  2. 抽查显示多项为 incidental copy（如 dspy parallelizer 取消、sunpy title 分支、pydantic 注解顺序 + 附带 copy 删除），真 F2 因果需逐题读 patch/F2P 名称判定，当前仅 deepdiff `w1di5431`（`_iterable_opcodes` 共享 vs 拷贝，F2P 1）初判真 F2，其余 39 待人工三元组复核（`==` 无 `is` 退 S、能力识别退 R）；
  3. 全部 40 待 source-preparation（test_patch 缺失文件补齐、上游 LICENSE 固定、执行 commit `git ls-remote` 固定）与双轮重放；pydantic 5400 级大套件需后台分批，deepdiff/dspy/sunpy 中套件优先。
- 第二批扫描（click/cloudpickle/pydantic/deepdiff/starlette/safety/trio/paramiko/webargs/astroid）已取 46 页，待完成；第三批（cantools/pygments/oauthlib/dspy 全量/pptx/pptx/parso/bleach 等）未启动；Gym/Rebench-V2 未动用。
- mido 5 项因 `../dev/tests` 坏路径已排除；click 因上述双原因排除；两者不占 40。

## 下一步（不超本轮授权）

1. 完成第二批 + 第三批取数，替换 40 中大 F2P（>30）、incidental 及同家族碰撞，补到 40 独立家族、多快照。
2. 逐题写 F2 三元组 inline 观察（含 `is`、双向可见性、复制时点）与唯一 edit 锚，固定 LICENSE/执行 commit/test_patch。
3. 用 `.venv` 逐题双轮，`normalize`（以本目录 `catalog.json` 为 supplemental）逐题定级，仅 eligible 计数；保留全部原始 JUnit/要求/matcher 返回，不放宽。
4. 缺口清零前不启动训练、不改 S600文件。
