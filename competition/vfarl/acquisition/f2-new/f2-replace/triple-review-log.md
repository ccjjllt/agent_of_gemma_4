# F2-replace 三元组复核 log（2026-10-25）

口径：每题必须同时满足 is身份断言 + 双向（共享修改）可见 + 访问/复制时点。
只有 `==` 无 `is` → 退 S；只有能力识别（无引用/所有权语义）→ 退 R。
大 F2P（>30）、大套件（P2P>600）、坏路径（`..`）、禁仓（common-400 四仓外另 49 禁仓）、
mega-rewrite（无法做唯一小锚 edit）→ 换小 clean 题。

## KEEP（真 F2，三元组完整）

1. `seperman__deepdiff.ed252022.func_pm_ctrl_invert_if__w1di5431`（F2P=1，P2P=918，
   `deepdiff/serialization.py::SerializationMixin._to_delta_dict`）
   - bug 把 directed 分支的 filtered-rebuild 与 else 分支的 alias 对调。
   - is：修后内部结果 `is not` live store；bug 下 `is`（alias）。
   - 双向可见：结果侧 append，live store 在修后不变、bug 下被污染。
   - 时点：`_to_delta_dict(directed=True)` 调用时点快照。
   - 方向：repair 重建隔离（isolation polarity）。F2P 在 bug 下 `TypeError`、
     修后通过（已本地验证）。test_patch 补 `tests/test_delta.py`。→ 已 prepare + replay 中。

## REJECT → S（仅值语义、无 is 身份）

- `...ydxd6gz2`：group_by 索引/排序反转；`deepcopy(item)` 为未改动的上下文行。
- `...b7p39v9j`：Delta 构造 bidirectional/raise-ValueError 反转；`deepcopy` 上下文。
- `...ka65an9g`：json 序列化 reversed 顺序；`copy(obj)` 为附带分支改动。
- `...e47ywswk`：TypeError 提前导致 reverseiterator 分支不可达；`copy` 附带。
- `...y73hdi0i` / `...y9fltuzf`：`%`→`//` 格式化 bug，无 copy 缺陷。
- `...wuxqz4eo`：`encode`/`decode` 对调。
- `...5pe274si`：`safe_to_import` 字符串未转 set；`.copy()` 附带。
- `...8jdzrbkp`：added/removed 键对调；`.copy()` 未改动上下文。
- `...8mumbt60`：`to_flat_rows` `_parse_path` 等删除致 NameError。
- `...6vvmfyn2`：`_get_reverse_diff` 重写，无 copy 信号。
- `...idobe3er`：`__add__` 操作排序重排；`deepcopy` 两侧保留，无 alias 引入。
- `sunpy__sunpy.f8edfd5c.func_pm_ctrl_invert_if__1j7z9x4r`：title 分支反转；
  `deepcopy(plot_settings)` 上下文。
- `dspy...pr_7940`：`copy()` 方法搬家附带，主缺陷为新增 `__call__` NameError。
- `dspy...pr_7991`：display_table 类型改动，无 copy 缺陷。
- `prettytable...4vol5y63`（禁仓，另计）：slice 丢失，值语义。

## REJECT → R（纯能力/删除能力，无引用语义）

- `...hmwaxeu8`：group_by 整段删除。`...u3apb5r9`：ignore-order 循环删除。
- `...dg5ddk6g`：group_by 重写。`...72azpgh6`：pairing 算法掏空（含 cache 删除）。
- `dspy...pr_8105`：async callback 分支删除（copy 仅出现在注释）。
- `dspy...pr_8135`：async cache wrapper 删除。
- `cantools...47xnh4ai`：e2e 赋值块删除。
- `astroid...3eua4raa`：brain stub 形状数值，静态推断能力。

## REJECT → unclean（mega-rewrite，无唯一小锚 / F2P 过大）

- `cantools...8r04vbpi`（F2P=39）：`dump_string` 整函数 shuffle，
  主症状 UnboundLocalError；deepcopy 移位附带。
- `cantools...gddphasv`（F2P=127）：`_check_mux` 重写丢弃分支隔离。
- `cantools...ejujkgq4`（F2P=1 但整函数重写）：主缺陷为 XML 路径能力缺失，
  data_ids 共享为次生且 bug 下 e2e 为 None 不可达。
- `cantools...i2iqiy7a`：plot Signals 重写（argparse 解析丢失为主）。
- `dspy...pr_7914`：ParallelExecutor 整文件重写；
  usage_tracker deepcopy 丢失附带于隔离重构。

## EXCLUDE（真 F2 模式但硬约束冲突，不占 40）

- mido 全系（`ta374gzq/js7xzx5x/qfjyu11h/cyg5oo56/0y6014so/surq1kqz/pr_474/9cfvotta/k8pqdhio/pr_455`）：
  `vars(self)` alias vs `.copy()` 为教科书 F2，但 F2P 路径全部为 `../dev/tests/...` 坏路径。
- click `fde47b4b...82024b34` near-miss：题面禁入（brief 明确不可入选）。
- pydantic 40-list 26 项：同 `main.BaseModel` 家族碰撞 + P2P 5400 级套件，整类替换。
- deepdiff 40-list `ozkjjv8h`（86）、`p5hmxq60`（119）、`7i2mr9eq`（106）：大 F2P，替换。
- tenacity/prettytable/pypika 命中：禁仓（common-400 成员）。

## 待定（取数中：bleach/parso 余/typeguard/pptx/freezegun/pygments 余/oauthlib）

第二批已缓存（trio/paramiko/webargs/safety/starlette/cloudpickle/asteroid/cantools）
strict changed-line copy 扫描：仅 cantools 5 + asteroid 1（上文已判）。
`is`-identity 扫描 600+ 命中全为 enum/None 对比噪声。
等剩余 59 页取完后重扫；若无真 F2 则如实报缺口。

## 第二批复核（fetch 全量 28 仓后，strict changed-line 扫描 58 hits）

- `paramiko...b9zsi38p`（F2P=1，P2P=553）：`__reduce__` 截断 `errors[:1]`。
  单行 clean，但纯值丢失、无 is、无共享可见 → **S**。REJECT。
- `cloudpickle...9pt7uoq9`（F2P=2）：`_make_dict_items` 丢首项/反转 → **S**。REJECT。
- `cloudpickle...q1i4f88l`：dump 递归改 fallback → **R**。REJECT。
- oauthlib 8 项（P2P=667）：868k2n2u/oer23bqe 验证条件反转 → **S**；
  xbulf5rr 删除 validate 方法 → **R**。REJECT。
- `cantools...x8mo9dm9`（F2P=39）：dump_string 重写 → unclean。REJECT。
- typeguard 全系（F2P=46~131）：`deepcopy` 均为 AST transformer 上下文，
  主缺陷为条件反转 → **S** 整类 + 大 F2P。REJECT。
- bleach 2 项（P2P=36，F2P=185/36）：InBodyPhase 方法删除致 NameError、
  getPhases 掏空致 ImportError → **R** + 大 F2P。REJECT。
- `pptx...vp1fcpgx`（F2P=2，P2P=2655）：idx/order 对调 + 插入方向，
  `deepcopy(ser)` 两侧保留 → **S** + 大套件。REJECT。
- trio/parso/pygments `is` 命中：enum 身份对比翻转或注释 → **S**，
  且 P2P 全部 >600。REJECT。
- safety `format_sub is/is not` 翻转：enum 对比 → **S**。REJECT。
- freezegun 107 行：零 copy/ownership 信号。无候选。

## 结构性结论（load-bearing）

1. `w1di5431` 双轮：baseline 959 tests/1 failure（恰为 F2P），
   repaired 959 tests/0 failure，但 `resolved=False`。
   normalize：`actual_local_official_repair_not_resolved` +
   `original_official_JUnit_matcher_rejects_repaired_report`，
   与 click near-miss 同双因不同实例：上游 `@pytest.mark.skip`
  （`test_list_difference_ignore_order_report_repetition[234]`）经官方
   matcher 前缀规则（`short_node.startswith(req_base + '_')`）否决 P2P
   `test_list_difference`，另有 runslow/remapping 类 skip 构成多重否决。
   不改测试/matcher/要求前提下不可修 → **near-miss #2**。
2. 同快照 f3-new-deepdiff-01/02 抽查同样 `resolved=False` 同一 message：
   `seperman/deepdiff@ed252022` 快照对“P2P 含 test_list_difference 的全套件任务”
   结构性不可入选（禁改三件套下）。旧 40-list 中 deepdiff 7 项全军。
3. 真 F2 基准率：全量 28 仓 strict 扫描 58 hits 逐题复核后，真三元组仅
   w1di5431（40-list 内），新仓零新增。mido 系真 F2 模式但 `../dev/tests`
   坏路径；click 系题面禁入。40 独立 family 缺口暂无法以合规小 clean 题填补。
