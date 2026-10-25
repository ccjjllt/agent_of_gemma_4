# f2-large 三元组复核 log（2026-10-25，第六轮：大套件 35 行分批依据）

口径：每题必须同时满足 `is` 身份断言 + 双向（共享修改）可见 + 访问/复制时点。
只有 `==` 无 `is` → 退 S；只有能力识别（无引用/所有权语义）→ 退 R。
套件 >150（本批专属；Gym/V2 真题原来 ≤150 口径下被排除的 35 行）。
坏路径（`..`）/禁仓/四仓已在扫描层过滤。Smith-py 不扫（冻结）。
P2P=0 标 P2P-EMPTY。skip 碰撞 DROP 快照不投双轮（10 个）；
test_patch-new 文件在 base 无上游文件按 K1/C2 先例计 CLEAN（3 个已重分类）。
pools：SWE-Gym rev `bb94ed9e…`（train 2438）+ SWE-rebench-V2 rev `10483de0…`
（train python 子集 7243）。去重：只读 common-400 + F1-40 + F5旧40 +
f2-round1/2 + f2-gymv2 16 + f2-gymv2b 5 + f2-gymv2c 2（程序化校验
544i/107s/490f；35 行 snapshot_collision 全无，instance 全新）。

## 扫描召回（f2-large，forbidden 含 23 keeps）

- Gym strict-copy 且 total>150：7 行（单文件闭包 4 + 多文件 3）。
- V2-python strict-copy 且 total>150：28 行（单文件闭包 6 + 多文件 22）。
- 合计 35 行。本批逐题复核：KEEP 0，S/R 见下。skip 预检 25 CLEAN / 10 DROP。

## KEEP（真 F2，三元组完整）：无（0）

- 35 行逐题读 problem_statement + patch + test_patch + F2P 名后，无一题同时满足
  `is` 身份断言（F2P 内逐字 `is/is not` 且与 copy 所有权因果绑定）+
  双向共享修改可见 + 复制时点。最接近的 pennylane-4540（`is not`/`is` 在非 F2P
  测试）、traits-1026（test_copy 名但无身份语义）、werkzeug-2975（18 个 copy 名
  F2P 但 patch 仅文档）均缺至少一条腿，按口径诚实记 S/R，不放宽。
- 文件-scope 协议已就绪但验证集为空（CLEAN∩KEEP=∅），故本批零双轮、零 eligible。
  这是分批依据本身：大套件 strict-copy 命中几乎全为附带 copy（见结构性结论）。

## REJECT → S（仅值语义/附带 copy，无 `is` 身份腿；CLEAN 16）

- `getmoto__moto-5960`（Gym，157，单文件，CLEAN）：GSI projection 值 bug；
  patch 加 `results = copy.deepcopy(results)` 防御拷贝；F2P 2 个 scan 测试无 `is`；
  题面无共享语义。→ S。
- `getmoto__moto-5012`（Gym，160，单文件，CLEAN）：transact 错误格式化值 bug；
  patch 仅移动 `copy.deepcopy(get_item)` 行（空白级）；F2P 2，无 `is`。→ S。
- `Project-MONAI__MONAI-4638`（Gym，232，多 14，CLEAN）：convert_to_numpy dtype
  值 bug；patch 删 `deepcopy(dict(data))` inverse；F2P 51，无 `is`。→ S。
- `unleash__unleash-client-python-285`（V2，151，多 3，CLEAN）：get_variant 值 bug；
  patch 删 `import copy` + 两处 `deepcopy(DISABLED_VARIATION)`（去拷贝化）；
  F2P 6，无 `is`。→ S（去拷贝方向，非隔离修复）。
- `timothycrosley__isort-972`（V2，159，单文件，CLEAN）：py2/3 stdlib 值 bug；
  patch `_default.copy()` 附带；F2P `test_python_version`，无 `is`。→ S。
- `timothycrosley__isort-1029`（V2，161，单文件，CLEAN）：注释移动值 bug；
  patch 加 `deepcopy(self.comments)`；F2P `test_moving_comments_issue_726`，
  无 `is`。→ S（copy 因果但无身份腿）。
- `sdv-dev__rdt-670`（V2，163，多 5，CLEAN）：默认 transformer 配置值 bug；
  patch 改 `deepcopy` 来源；F2P 11，`is` 仅 `is None`。→ S。
- `pandas-dev__pandas-57314`（V2，165，多 2，CLEAN）：Timestamp dtype 值 bug；
  patch 删 `copy`/`pickle` 导入（无关）；F2P 2，无 `is`。→ S。
- `enthought__traits-1026`（V2，173，多 3，CLEAN-newfile）：通知语义值 bug
  （`added != removed` → `added or removed`）；copy 命中来自 TraitSet 重写附带；
  F2P 173 含 `test_copy/test_deepcopy` 名但体为 `assertEqual`（无 `is` 所有权腿），
  题面零 copy。→ S。
- `pallets__werkzeug-2975`（V2，195，多 2，CLEAN）：deprecate OrderedMultiDict；
  patch 仅文档/类型注解（`copy` 命中在 docs）；F2P 195 含 18 个 copy 名但皆
  `test_copies_are_mutable` 值语义（`==`），无 `is` 腿；文档改动无法因果修复
  195 测试。→ S。
- `cuthbertlab__music21-791`（V2，233，单文件，CLEAN）：sphinx 文档格式；
  patch 仅 docstring 加 `copy.deepcopy(s)` 用法句；无功能改动；F2P 1。→ S。
- `getmoto__moto-7993`（V2，239，多 3，CLEAN）：RDS modify 值 bug；
  patch 改 `event.copy()` 路径；F2P 3，无 `is`。→ S。
- `cuthbertlab__music21-1346`（V2，257，单文件，CLEAN）：makeTies 值 bug；
  patch 删 `copy.deepcopy(ts)` 行；F2P 2，无 `is`。→ S。
- `python-babel__babel-621`（V2，264，多 5，CLEAN）：weeknumber 值 bug；
  patch 删 `def copy`（odict 方法移除）；F2P 1，无 `is`。→ S。
- `getmoto__moto-7992`（V2，376，多 5，CLEAN）：EventBridge targets 值 bug；
  patch 改 `event.copy()` 路径；F2P 5，无 `is`。→ S。
- `pennylaneai__pennylane-4884`（V2，1153，多 3，CLEAN）：map_wires/cond 值 bug；
  patch 删 `copy.copy(self.mv)` 两行；F2P 3（map_wires），无 `is`。→ S。

## REJECT → R（纯能力/特性/重构，无引用语义；CLEAN 9）

- `Project-MONAI__MONAI-5107`（Gym，179，多 5，CLEAN）：GridPatch MetaTensor
  支持（特性）；patch 删 `deepcopy` 导入 + 重排；F2P 178 网格测试，
  test_patch 唯一 is 行为注释 `# assert is a shallow copy`。→ R。
- `facebookresearch__hydra-2014`（Gym，289，单文件，CLEAN）：instantiate 顶层
  list 支持（题面逐字 Feature Request）；patch 加 `deepcopy(config)` 附带；
  F2P 13，无 `is`。→ R。
- `fsspec__universal_pathlib-372`（V2，157，单文件，CLEAN-newfile）：新增
  ProxyUPath `copy/copy_into` API（题面逐字 Provide a way to extend）；
  F2P 157 皆新文件新测试（含 `test_copy_path`）；P2P-EMPTY。→ R（新能力）。
- `jelmer__dulwich-1108`（V2，205，多 2，CLEAN）：题面逐字 Add ability to pack
  refs；patch 加 `.copy()` 重读；F2P 3，无 `is`。→ R。
- `stingraysoftware__stingray-234`（V2，206，多 3，CLEAN）：rebin 返回类型一致性
  （特性）；patch 加 `copy.copy(self)`；F2P 18，无 `is`。→ R。
- `sdv-dev__rdt-758`（V2，208，多 3，CLEAN）：multi-column transformer 校验方法
  新增（题面逐字 Add a method）；patch 加 `deepcopy`；F2P 9，无 `is`。→ R。
- `sdv-dev__rdt-759`（V2，238，多 4，CLEAN）：OneHotEncoder category 支持
  （特性）；patch 同 758 系 `deepcopy`；F2P 10，无 `is`。→ R。
- `cloudreach__sceptre-473`（V2，300，多 17，CLEAN）：nonscalar resolver 参数
  支持（特性）；patch 加 `copy.copy(node)`；F2P 7，无 `is`。→ R。
- `Project-MONAI__MONAI-6080`（Gym，1309，多 50，CLEAN-newfile）：transform
  functionals 评审实现（题面逐字 feature request）；patch `deepcopy(meta)`；
  F2P 181，is 行皆 LICENSE 散文；需 torch 全栈。→ R（兼重依赖不可行）。

## SKIP DROP（10，阻断验证；三元组亦多为 S/R，不浪费重放）

- `hynek__structlog-283`（161）：`test_simplejson` skip + `test_pickle` skip 致
  31 前缀碰撞；patch 仅 CHANGELOG。→ DROP（兼 R/S）。
- `copier-org__copier-2145`（164）：`test_exclude_file` xfail + `test_empty_dir`
  skip 致 3 碰撞；patch 附带。→ DROP（兼 S）。
- `canonical__operator-1074`（215）：`test_send_signal_number` skip 致 2 碰撞；
  patch `deepcopy(labels)` 附带；F2P 215 全新（P2P-EMPTY，特性倾向）。→ DROP。
- `pennylaneai__pennylane-4540`（423）：46 个 markers 致 478 碰撞；F2P
  `test_hamiltonian_grouping_indices` 仅 `==`（`is not`/`is` 在非 F2P 测试），
  patch 去拷贝化；→ DROP（兼 S）。
- `dask__dask-6776`（498）：xfail/skip 致 11 碰撞；partition_info 特性；
  → DROP（兼 R）。
- `getmoto__moto-8739`（503）/`getmoto__moto-8745`（503，同基线 `32592a53`）：
  rds skip 致各 11 碰撞；→ DROP（兼 R/S）。
- `pennylaneai__pennylane-3983`（1403）：skip 致 16 碰撞；`__copy__` 新增但
  F2P 无 `is`；→ DROP（兼 S）。
- `pandas-dev__pandas-56758`（2599）：skip 致 20 碰撞；→ DROP（兼 S）。
- `pandas-dev__pandas-57518`（13652）：skip 致 40 碰撞；patch 仅 `copy` 签名
  `deep|None→deep`；→ DROP（兼 S）。

## NEW-FILE CLEAN 重分类（3，已计入 CLEAN）

- `fsspec-372`（`upath/tests/test_extensions.py` new file，404→CLEAN）、
  `traits-1026`（`traits/tests/test_trait_set_object.py` new file，404→CLEAN）、
  `MONAI-6080`（`tests/test_spatial_combine_transforms.py` new file，404→CLEAN）：
  base 无上游文件故无 markers 可碰撞；按 f2-gymv2 K1 / f2-gymv2c C2 先例计 CLEAN。
  三元组仍各退 R/S（见上），不投双轮。

## 单文件 file-scope 可行性（分批依据核心表）

- 单文件闭包（全部 F2P+P2P 同一文件，共 10/35）：moto-5960、moto-5012、
  hydra-2014、dask-6776（DROP）、fsspec-372（CLEAN-newfile，R）、isort-972、
  isort-1029、copier-2145（DROP）、music21-791、music21-1346。
  其中 CLEAN 8 个皆三元组退 S/R，故文件-scope 验证集为空——但协议已验证：
  若未来出现单文件真 F2（is 过），则 `verification_target_files=[该文件]` +
  原 matcher 不动 + 不缩水断言即可双轮（normalize `missing_files` 为空、
  `file_scope` 为真）。
- 多文件（25/35）：file-scope 需一次执行全部 required 文件（仍 file-scope，
  仍不动 matcher）；本批无 KEEP，故未执行；pandas-57518（4 文件/13652 题）、
  pennylane-3983（5/1403）等即使三元组过也需分批跑（verification timeout 与
  task-site-packages 另议）。
- Smith gpxpy-143（`GPX.clone: deepcopy→copy` 教科书 F2，单文件，总数 143）
  仍为最近的真 F2 大套件参照，但 Smith 冻结，本批不动；若未来放宽 Smith 尺寸
  门槛，它是单文件 file-scope 首选（skip 已 CLEAN，见 f2-replace2）。

## 双轮与口径注记

- 本批 25 CLEAN 全部三元组退 S/R（无 `is` 腿为唯一共性；`is` 命中皆散文/
  `is None`/`isinstance`/非 F2P 测试），按“无 is 直接退 S 不浪费重放”规则零试点。
- 10 DROP 因 skip 前缀碰撞阻断（matcher `_node_matches_requirement` 前缀规则），
  与三元组无关，不投双轮。
- 全局仍 10/40，缺口 30。本批贡献分批依据（35 行清单 + skip 全量 + 单文件表），
  不贡献 eligible。缺口清零前不启动训练、不改 S600 与旧文件。
