# F600-F2 f2-replace2 批次记录（2026-10-25，第二轮）

只做题目与 CPU 真实工具数据，不启动训练。不动 `f1-new` / `f3-new` / `f4-new` / `f5-new` 与 S600 文件。
本轮不新增 `f2-replace2` 以外的 replay 产物（无试点双轮，见下）。

## 约束口径

- 仅 train；Smith-py `77cab9055d42ab4a5c25c89a8f937096db13558e`；排除
  `fastapi/fastapi、textualize/rich、psf/requests、encode/httpx` 四仓。
- 与 common-400 + F1-40 + F5旧40（作废但快照仍禁）的 family / snapshot / instance
  零重叠。禁用集：`520` instances、`87` snapshots、`441` families
  （common-400 `dedup-baseline.json` + F1-40 `reviewed-candidates.json` 40 条 +
  F5旧40 `acquired-40.json` 3 快照 + f2-replace `catalog.json` 1 条 +
  旧40-list 40 instances + deepdiff/click/pydantic/dspy/sunpy/mido/cantools 快照显式禁）。
- 三元组：`is` 身份 + 双向（共享修改）可见 + 访问/复制时点；无 `is` 直接退 S
  且不浪费重放；纯能力退 R；大 F2P（>30）/ 大套件（P2P>600）/ 坏路径 / 禁仓 /
  mega-rewrite 换小 clean 题。
- 只选小套件（F2P+P2P 总数<100）、无生产节点 P2P（P2P 全为测试节点）、
  test 文件路径与 workspace 对应；优先此前未用过的合规小仓；
  每个快照先单题试点，试点 eligible 才批量。
- skip 碰撞预检：拉上游测试文件，查 `@pytest.mark.skip / xfail`
  （含 `skipif / pytest.skip / unittest.skip`）与 F2P/P2P 同前缀
  （官方 matcher `_node_matches_requirement`：`short_node.startswith(req_base + '_')`
  或 `req_base + '['`，另有 `68字符` 与纯数字后缀规则），有则直接弃快照，不投双轮。
- `normalize eligible` 才计数（supplemental=`f2-replace2/catalog.json`）；
  用 `replay/.venv` 执行（本轮零试点，故未调用）。

## 计数（诚实）

- **eligible：0 / 40，缺口 40。**
- 双轮完成 0 题：9 个小套件 copy 候选逐题三元组复核后全部无 `is` 身份语义，
  按“无is直接退S不浪费重放”规则零试点。
- skip 预检 5 快照（4 小套件 + 1 尺寸超限教科书题）全干净，无前缀碰撞。

## 本批产物

- `pages2/`：5 个新取 train 页（`5600/9600/14100/17700/47800`）+ receipt
  （X-Revision 均为 `77cab905…13558e`），覆盖此前未缓存的小仓行
  （string-similarity / markdownify / mypy / gTTS / PySnooper）。
- `triple-review-log.md`：9 小套件题 + 1 教科书题逐题 verdict。
- `skip-precheck.json`：5 快照 14 上游测试文件抓取证据与前缀碰撞结论。
- `dedup-evidence.json`：禁用集计数与零重叠程序化校验口径。
- `reviewed-candidates.json`（0 条）+ `catalog.json`（0 条）+
  `normalize-summary.json`（eligible 0，无 replay-result）。
- 取数脚本与中间 sweep 输出在 `/tmp`（`f2r2_forbid.json`、
  `f2r2_smallsweep.json`、`f2r2_allcached_copy.json`、`f2r2_small100.json`），
  未写入采集目录。

## 缺口说明

1. 全缓存 28900 去重 instances（f2-new + f1-new + f5-new pages 284 唯一偏移 +
   pages2 新增 500 行）strict changed-line copy 扫描：禁用外命中仅 48，
   其中总数<100 仅 9 题 / 4 新快照
   （dvc@1d6ea681 1 题、scrapy@35212ec5 1 题、sqlglot@036601ba 6 题、
   channels@a144b4b8 1 题）。9 题三元组全无 `is` 身份（`is` 命中皆为英文散文 /
   `isinstance` / AST 遍历脚手架），8 退 S、1 退 R，零试点。
2. 唯一教科书 F2（gpxpy@09fc46b3 `clone: deepcopy→copy`，F2P=1/P2P=142）
   三元组完整但总数 143 超 `<100` 硬门槛，计尺寸超限 near-miss，不计数、不试点。
3. skip 预检：dvc / scrapy / sqlglot / channels 上游测试文件零 skip/xfail 标记；
   gpxpy 唯一 `@unittest.skipIf`（`LxmlTest::test_checklxml`，条件触发）
   与全部 143 个 F2P/P2P 无前缀碰撞。5 快照 skip 全干净——弃快照原因全部是
   三元组/尺寸，而非 skip 碰撞。
4. 小套件 `copy-anywhere` 185 题 + 松散 `clone/shallow/shar/alias` 752 题复核：
   有 copy 的小题其 `is` 皆散文噪音，无遗漏真 F2。真 F2 基准率与首轮一致（<0.5%）。

## 附：对上一批文件的误触声明

- 2026-10-25 16:49 本子智能体误运行 `acquisition/f2-new/fetch_scan.py`
  （无参数全量模式），覆盖了 `acquisition/f2-new/scan-hits.json` 与
  `scan-used.json`（mtime 16:49，`scan-hits.json` 现为 12448 行全量分，
  `f2_signal=159`）。此前首轮记录的 542 行 / 5 信号版本已丢失且无法恢复
  （目录未纳入 git）。`candidate-list-40.json`、`reviewed-candidates.json`、
  `catalog.json`、现货 `.task.json` 与 `f2-replace/` 均未触碰。
  后续取数只写本目录 `pages2/`，不再运行全量扫描。
- 缺口清零前不启动训练、不改 S600 文件。
