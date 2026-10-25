# f2-push2 三元组复核 log（2026-10-25，第十一轮：mid-suite 151-400 + P2P-empty inline-is）

口径：每题必须同时满足 `is` 身份断言 + 双向（共享修改）可见 + 访问/复制时点。
只有 `==` 无 `is` → 退 S；只有能力识别（无引用/所有权语义）→ 退 R。
inline-is 型：F2P 仅值断言，`is` 腿由 source-bound inline 观察承载
（同 C2/K15/nox-439/aiohttp-10672/qiskit-12620/M2/P1-P4 口径）。
Gym/V2 真题：A池 total 151-400（含两端）BROAD-inline strong（strict-copy 排除，
f2-inline 200 cap 之外新地面）；B池 P2P-empty total≤50 inline-strong
（push要求total≥2之外、C2 total=1先例的新地面）。
坏路径（`..`）/禁仓/四仓已在扫描层过滤。Smith-py 不扫（冻结）。
去重：只读 common-400 + F1-40 + F5旧40 + f2-round1/2 + f2-gymv2 16 +
f2-gymv2b 5 + f2-gymv2c 2 + f2-recall 3 + f2-inline 1 + f2-more 2 + f2-push 6
（程序化校验 556i/119s/502f；3 KEEP snapshot_collision 全无，instance 全新；
qiskit-terra-8764 同 M2 家族记 COLLIDE-family 不计入）。

## 扫描召回（f2-push2，forbidden 含 16+5+2+3+1+2+6 keeps）

- Gym mid-inline A(151-400 strong)：11 行；V2-python mid-inline A：38 行；合计 49。
  P2P-empty B(≤50 strong)：gym 7 + v2 62 = 69。扫描合计 118。
- prior-named（旧日志已逐题具名 269）排除后：A 41 + B 44 = 85 行
 （`push2-candidates.json`；复核工作文件 `push2-review-dump.txt` 85 题
  problem+patch+testpatch、`push2-triage.txt` 紧凑版）。
- 本批深读：A池单文件/小F2P优先（qiskit-terra-8741、more-itertools-412、
  pandas-53977、copier-2189、modin/dask/pandas大套件抽查）+ B池 total≤5
  单文件优先（conan-13390、cfn-426、cfn-1409、ctapipe-1893、trio-2392、
  pyjanitor/bids/sktime抽查、qiskit-terra-8764家族核对）。其余 85-~15 行
  为信号级SCREENED（未逐题深读，不计数、不声称为REJECT）。

## KEEP（真 F2，三元组完整，待 skip 预检+双轮）：3

### Q1 `qiskit__qiskit-terra-8741`（V2，total=209，F2P=1，P2P=208，单文件 visualization/circuit/_utils.py）

- 生产：`_get_layered_instructions()`。旧行为 `qubits = circuit.qubits`
  / `clbits = circuit.clbits`（别名！），随后 `reverse_bits` 分支做
  `qubits.reverse()/clbits.reverse()`、idle-wires 分支做 `qubits.remove/
  clbits.remove`——全部直接改原电路的qubit/clbit list；修复
  `qubits = circuit.qubits.copy()` / `clbits = circuit.clbits.copy()`
  （单 hunk，两行）。
- F2P `test_does_not_mutate_circuit`（`TestTextIdleWires`）：draw
  `idle_wires=False` 后原电路仍可正常二次draw/qpy（值断言；`is` 腿由
  inline 承载，同 M2/C2/nox/aiohttp/qiskit-12620）。
- 三元组：`qubits is not circuit.qubits` 且 `clbits is not circuit.clbits`
  （修前 `is` 别名，修后拷贝）+ 双向隔离（改返回 qubits 侧——reverse/
  remove idle wires——原 `circuit.qubits` 不动；F2P逐字：draw不mutate
  circuit）+ `_get_layered_instructions()`（draw路径）调用时点。
  极性：隔离。
- 家族：`Qiskit/qiskit-terra::qiskit.visualization.circuit._utils.
  _get_layered_instructions::idle_wires_qubits_clbits_copy_isolation`；
  快照 `Qiskit/qiskit-terra::b35b18a90ff1…`（I1为`Qiskit/qiskit::2b2be030…`
  不同repo键，M2为同repo调度pass不同符号，均不碰撞）。零重叠（程序化
  校验通过）。

### Q2 `awslabs__cfn-python-lint-426`（V2，total=1，F2P=1，P2P=0，单文件 src/cfnlint/helpers.py）

- 生产：`convert_dict()`。旧行为 `for k, v in template.items()` 直接迭代
  原dict视图，随后循环体内 `del template[k]; template[new_k] = ...`
  （key的str_node包装）——迭代中改大小必RuntimeError/漏键；修复
  `for k, v in template.copy().items()`（迭代快照拷贝，改原dict；
  同单 hunk，另有str_node key包装与list分支传参，同属本修复）。
- F2P `test_success_run`（`TestConvertDict`）：convert后全key为str_node、
  值递归转换（值断言；`is` 腿由inline承载：`template.copy() is not
  template`，C2先例P2P-empty total=1）。
- 三元组：`template.copy() is not template`（迭代时点快照）+ 双向隔离
  （改原template侧——del/reinsert新key——迭代快照侧不动，无跳键/报错；
  F2P逐字类型断言）+ `convert_dict()` 调用时点。极性：迭代-修改隔离。
- 家族：`awslabs/cfn-python-lint::cfnlint.helpers.convert_dict::
  dict_iter_copy_isolation`；快照 `awslabs/cfn-python-lint::4765ff51ce92…`。
  零重叠（程序化校验通过）。

### Q3 `more-itertools__more-itertools-412`（V2，total=323，F2P=1，P2P=322，单文件 more_itertools/more.py）

- 生产：`spy(iterable, n)`。旧行为 `return head, chain(head, it)`——返回的
  head list与chain共享同一list对象（别名！）；修复 `return head.copy(),
  chain(head, it)`（单 hunk，一行）。
- F2P `test_immutable`（`SpyTests`）：题面逐字——`head[1]='bad'`后
  `list(it)` 仍为原序列（值断言；`is` 腿由inline承载：返回head
  `is not` 链内head）。
- 三元组：返回 `head_copy is not chained_head`（修前同一对象`is`别名）
  + 双向隔离（改返回head侧原迭代器侧不动；F2P逐字）+ `spy()` 返回时点。
  极性：隔离（同pyfar切片/M2/op-copy族）。
- 家族：`more-itertools/more-itertools::more_itertools.more.spy::
  head_copy_isolation`；快照 `more-itertools/more-itertools::c0465331cbc0…`。
  零重叠（程序化校验通过）。

## COLLIDE-family（同家族复本，不计入）：1

- `qiskit__qiskit-terra-8764`（V2，total=4，P2P-empty）：`new_op =
  node.op.copy()`，与M2（qiskit-terra-5778，ALAP/ASAP每指令op.copy存
  duration）同生产符号/同因果（`qiskit.transpiler.passes`调度op隔离），
  记COLLIDE-family，不计入。

## REJECTS（已深读 patch-evidence，不充数）

- R-rewrite/能力：`conan-io__conan-13390`（PremakeDeps全文件重写78+行，
  `.copy()`随旧`_get_cpp_info`删除，bug为缺`aggregate_components`能力，
  无引用隔离三元组）→ R。
- S-新文件：`aws-cloudformation__cfn-python-lint-1409`（新文件
  AuroraDBInstanceProperties规则，`path[:] + [prop]`为新逻辑hygiene，
  无baseline别名腿）→ S。
- S-性能/值：`pandas-dev__pandas-53977`（Block.convert免碎片：no-op时回
  `[self.copy()] if copy else [self]`，F2P为fragmentation值断言，
  无caller-owned共享映射在险；另pandas-C/206套件依赖墙）→ S。
- HELD-特性/依赖：`python-trio__trio-2392`（线程propagation大特性，
  `self.context.copy()`埋于Run/RunSync新类+文档+newsfragment多文件，
  trio全套件macOS墙先例）→ HELD（特性+环境）。
- S-hygiene抽查：`bids-standard__pybids-574`（`entities.copy()`本地防御，
  无caller映射在险）、`sktime__sktime-5570`（`df.copy()`同）、
  `ericmjl__pyjanitor-735`（`func(df.copy())`同）、`pyjanitor-1317/1315`
  （`df[:]`同）→ S。
- 噪声抽查：`streamlink-2654/2653`（git clone文档）、`keras-nlp-64`
  （clone文档）、`benchopt-361`（git clone）、`codezonediitj-85`
  （import deepcopy未使用）→ R/噪声。
- 大套件抽查（信号弱/多文件/重依赖，不投）：aiohttp-10534/10551
  （16-17文件，F155+P12，content_mapping代理，doc/changelog噪声）、
  pvlib-2292（docs噪声）、pymc-7637（pre-commit+注释）、modin-6759/6758
  （ray/dask墙+4文件）、copier-2189（docs+2文件，shallow parse注释）、
  stumpy-655/656（4文件，astype噪声）、psyclone-1850/2685（8文件，
  changelog+copy噪声）、pandas-51005/57084/57078/57046（C/大套件）、
  dask-8930/7033（356-370套件，ravel/copy噪声）、networkx-5616
  （18文件）、laspy-342（4文件）、pyccel-1606（6文件clone）、
  fastmcp-434（11文件docs）、aeon-1609（25文件）、pydantic-7825
  （注释）、pyam-236（4文件release notes）、ffsim-295、music21-866
  等 → S/R/SCREENED（未逐题深读者记SCREENED，不计数）。
- B池其余total≥9多文件/重依赖行（pybamm/abagen/pyprep/cubed/
  kernel_gateway/covalent/rez/hub/cookiecutter/pennylane/pyjanitor/
  traits/pre-commit/pytensor/harmonica/zha/flopy等）信号级SCREENED，
  未逐题深读，不计数。

## 家族/快照状态（push2 eligible 待双轮：3）

- 快照（全在119禁外+prior新）：qiskit-terra b35b18a90ff1…、
  cfn-python-lint 4765ff51ce92…、more-itertools c0465331cbc0…。
  One per canon::sym；同家族复本8764已排除。
- 缺口：全局16/40 + push2待定3（双轮后定级）。缺口清零前不启动训练、
  不改S600与f1/f3/f4文件；全部新文件在`f2-push2/` +
  `replay/artifacts/f2-push2/` + `replay/f2-push2-*.json`。
