# f2-replace2 三元组复核 log（2026-10-25）

口径：每题必须同时满足 `is` 身份断言 + 双向（共享修改）可见 + 访问/复制时点。
只有 `==` 无 `is` → 退 S；只有能力识别（无引用/所有权语义）→ 退 R。
总数≥100 → 尺寸超限 near-miss（不试点、不计数）。
大 F2P（>30）、坏路径（`..`）、禁仓、四仓、mega-rewrite（无唯一小锚 edit）→ 换小 clean 题。
本轮 9 小套件题 + 1 教科书题，无 KEEP。

## 候选池来源

- 全缓存 28900 去重 instances（f2-new/f1-new/f5-new pages 284 唯一偏移 + pages2 500 行），
  程序化禁用（520 instances / 87 snapshots / 441 families）后 strict changed-line
  copy 扫描命中 48；总数<100 仅 9 题 / 4 新快照（下文逐题）。
- 小套件 `copy-anywhere` 185 题与松散 clone/shallow/shar/alias 752 题复核：
  有 copy 的小题其 `is` 行皆英文散文（`is not/ is broken/ is required`）或
  `isinstance`，无遗漏真 F2。

## REJECT → S（无 is 身份，值语义/附带 copy）

1. `iterative__dvc.1d6ea681.pr_9033`（F2P=2，P2P=1，总数 3，
   `dvc/render/convert.py + dvc/render/converter/vega.py`，row 33103）
   - patch 新增两处 `deepcopy`（`_group_by_rev(deepcopy(renderer.datapoints))`、
     `deepcopy(file2datapoints.get(...))`），主缺陷为 `NameError: _group_by_rev`
     未定义（F2P `test_to_json_vega*` 皆 NameError 失败）。
   - `is`：patch 零 `is`；problem 的 `is` 三行皆散文
    （`is broken / is not defined`）。无身份断言 → S。
   - copy 为重构附带防御性拷贝，非因果。→ REJECT（S），不试点。
2. `scrapy__scrapy.35212ec5.pr_5526`（F2P=13，P2P=5，总数 18，
   `scrapy/http/headers.py`，row 30094）
   - patch 仅删 `from collections.abc import Mapping` 导入 + 空行
     （`copy = __copy__` 两侧同文，只差文末换行，无功能改动）；
     13 个 F2P（`HeadersTest::test_copy` 等）失败来自导入缺失，与 copy 无关。
   - `is`：patch 零 `is`；problem 唯一 `is` 为散文
    （`the latter is not true`）。无身份断言 → S。
   - copy 行是未改动上下文（空白噪音）。→ REJECT（S），不试点。
3. `tobymao__sqlglot.036601ba.lm_rewrite__ld4dbcvw`（F2P=3，P2P=53，总数 56，
   `sqlglot/optimizer/pushdown_predicates.py`，row 20663）
   - patch 重写 `replace_aliases`（`predicate = predicate.copy()` 防御性首拷贝 +
     表限定列分支），主缺陷为别名替换条件（table qualifier 处理），
     copy 两侧保留（`aliases[x].copy()` → `alias_map[x].copy()`）。
   - `is` 行皆注释散文（`If this is a column reference…`）。无身份断言 → S。
   - lm_rewrite mega，`is` 为 `is` 散文。→ REJECT（S），不试点。
4. `tobymao__sqlglot.036601ba.lm_rewrite__w3nmrc62`（F2P=3，P2P=53，总数 56，
   同文件，row 20698）
   - patch 在 `_replace_alias` 加表限定分支（`aliases[name].copy()` 上下文），
     主缺陷同上（qualified columns 别名替换），copy 非因果。
   - `is` 皆散文（`This is used when pushing down…`）。→ REJECT（S），不试点。
5. `tobymao__sqlglot.036601ba.lm_rewrite__hqb9f41y`（F2P=7，P2P=49，总数 56，
   `sqlglot/optimizer/qualify_columns.py`，row 20654，patch 6728B mega-rewrite）
   - patch 重写 `_expand_alias_refs`（`alias_map[name].copy()` 防御性拷贝），
     主缺陷为别名展开范围/条件，copy 非因果。
   - `node is expression` 为 AST 遍历脚手架（跳过 SELECT 子句防递归），
     不是 copy 所有权身份；其余 `is` 皆散文。→ REJECT（S），不试点。
6. `tobymao__sqlglot.036601ba.func_pm_remove_loop__w8jloc4n`（F2P=9，P2P=47，
   总数 56，同文件，row 20587）
   - patch 删 `exp.column(select.args["alias"].copy())` 一行，主缺陷不在 copy；
     problem 零 `is`，patch 唯一 `is` 为注释散文。→ REJECT（S），不试点。
7. `tobymao__sqlglot.036601ba.lm_rewrite__cfjfnvhf`（F2P=12，P2P=44，总数 56，
   `sqlglot/optimizer/isolate_table_selects.py`，row 20636）
   - patch 加 `tables[node.name].copy()`，主缺陷为表选择隔离条件，
     copy 为返回值防御拷贝；`is` 皆散文。→ REJECT（S），不试点。
8. `tobymao__sqlglot.036601ba.combine_module__bzrdxdkd`（F2P=13，P2P=43，
   总数 56，两文件，row 19952）
   - 同类 `tables[node.name].copy()` 防御拷贝；patch/problem 零实质 `is`。
     → REJECT（S），不试点。

## REJECT → R（纯能力/重构，无引用语义）

9. `django__channels.a144b4b8.pr_1976`（F2P=15，P2P=62，总数 77，
   `channels/layers.py`，row 28570，patch 5461B multi-hunk）
   - `deepcopy(message)` 两侧同文（仅队列 API `put_nowait→await put` 与
     容量/校验/群组逻辑重写）；主缺陷为通道校验方法缺失
    （`AttributeError: valid_channel_name`）与容量语义，非 copy 所有权。
   - `is` 皆注释散文（`If join time is older…`）。→ REJECT（R），不试点。

## EXCLUDE（真 F2 模式但尺寸超限，不占 40、不试点）

- `tkrajina__gpxpy.09fc46b3.func_basic__fwnb1y0h`（F2P=1，P2P=142，总数 143，
  `gpxpy/gpx.py`，row 26208）
  - `GPX.clone: mod_copy.deepcopy(self) → mod_copy.copy(self)`，教科书 F2：
    `is`（浅拷贝下 `cloned.tracks is original.tracks`）、双向可见
    （clone 侧 `reduce_points/smooth` 污染原侧）、时点（`clone()` 调用）。
    problem 陈述完全对应，F2P `test_clone_and_smooth` 单题精准。
  - 但总数 143 > 100 硬门槛；同快照其余 5 题亦 143。
    → 尺寸超限 near-miss（类 click/deepdiff near-miss 口径），不试点、不计数。
  - skip 预检干净（下文），若未来放宽尺寸门槛可优先单题试点此题。

## 结构性结论

- 9 小套件 copy 候选三元组 Keeper 为 0：小仓 copy patch 全部是防御性/附带拷贝
  或未改动上下文，`is` 身份语义缺席。按规则零重放，eligible 0。
- 唯一真 F2（gpxpy）被尺寸门槛排除；skip 预检 5 快照全干净，
  故本轮弃快照原因全部是三元组/尺寸（见 README 弃快照清单），无 skip 碰撞弃快照。
