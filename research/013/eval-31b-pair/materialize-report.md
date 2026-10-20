# P013 step-2 物化报告（CPU）

日期：2026-10-20。 verdict: **FREEZE VALID (CPU stage)**。

纪律：只读分片标识列（instance_id/resolved/model/traj_id/行号），未读任何问题正文、参考补丁、test_patch；未读 heldout；未执行 FAIL_TO_PASS / PASS_TO_PASS（GPU V2/V3 阶段）；一次性手动命令；未建测试文件；snapshots/work 大体积产物不进 git。

## 1. 逐题结果

| task | repo | instance_id | base_commit | snapshot | 字节 | import | collect | 状态 |
|---|---|---|---|---|---|---|---|---|
| t01 | pallets/click | pallets__click.fde47b4b.pr_2604 | 01ce28d1459a | pallets__click.fde47b4b.pr_2604.tar.gz | 723937 | import_ok_snapshot | collect_partial | staged_ok |
| t02 | pallets/markupsafe | pallets__markupsafe.620c06c9.pr_402 | 654e45a0aff2 | pallets__markupsafe.620c06c9.pr_402.tar.gz | 60780 | import_failed | collect_failed | staged_ok |
| t03 | pallets/markupsafe | pallets__markupsafe.620c06c9.pr_418 | 21069fc5637b | pallets__markupsafe.620c06c9.pr_418.tar.gz | 63776 | import_ok_snapshot | collect_ok | staged_ok |
| t04 | pallets/markupsafe | pallets__markupsafe.620c06c9.pr_379 | ef1645060d91 | pallets__markupsafe.620c06c9.pr_379.tar.gz | 60584 | import_failed | collect_failed | staged_ok |
| t05 | pallets/markupsafe | pallets__markupsafe.620c06c9.pr_400 | 9d918e9f0866 | pallets__markupsafe.620c06c9.pr_400.tar.gz | 60667 | import_ok_snapshot | collect_failed | staged_ok |
| t06 | pydicom/pydicom | pydicom__pydicom.7d361b3d.pr_1938 | 927e85a9a279 | pydicom__pydicom.7d361b3d.pr_1938.tar.gz | 5969172 | import_failed | collect_failed | staged_ok |
| t07 | pydicom/pydicom | pydicom__pydicom.7d361b3d.pr_2069 | 09fded6bbe5d | pydicom__pydicom.7d361b3d.pr_2069.tar.gz | 5982650 | import_failed | collect_failed | staged_ok |
| t08 | pydicom/pydicom | pydicom__pydicom.7d361b3d.pr_1867 | b09c388cb335 | pydicom__pydicom.7d361b3d.pr_1867.tar.gz | 5982464 | import_failed | collect_failed | staged_ok |
| t09 | pydicom/pydicom | pydicom__pydicom.7d361b3d.pr_2163 | ceaea04e48fd | pydicom__pydicom.7d361b3d.pr_2163.tar.gz | 5957079 | import_failed | collect_failed | staged_ok |
| t10 | pydicom/pydicom | pydicom__pydicom.7d361b3d.pr_1859 | b2090318825c | pydicom__pydicom.7d361b3d.pr_1859.tar.gz | 5500817 | import_failed | collect_failed | staged_ok |
| t11 | pydicom/pydicom | pydicom__pydicom.7d361b3d.pr_1995 | f6f2f4a27d6a | pydicom__pydicom.7d361b3d.pr_1995.tar.gz | 5988522 | import_failed | collect_failed | staged_ok |
| t12 | pydicom/pydicom | pydicom__pydicom.7d361b3d.pr_1987 | 8efbb6a346e9 | pydicom__pydicom.7d361b3d.pr_1987.tar.gz | 5969168 | import_failed | collect_failed | staged_ok |
| s01 | pydicom/pydicom | pydicom__pydicom.7d361b3d.pr_1920 | 3d026d27c5c3 | pydicom__pydicom.7d361b3d.pr_1920.tar.gz | 5974292 | import_failed | collect_failed | staged_ok [备] |

## 2. 依赖记录（静态清单，wheel 实取延后 GPU Linux）

- t01 pallets__click.fde47b4b.pr_2604: 构建文件[pyproject.toml]；license=LICENSE.txt；import=import_ok_snapshot；collect=collect_partial。
- t02 pallets__markupsafe.620c06c9.pr_402: 构建文件[pyproject.toml, setup.py]；license=LICENSE.txt；import=import_failed；collect=collect_failed。
- t03 pallets__markupsafe.620c06c9.pr_418: 构建文件[pyproject.toml, setup.py]；license=LICENSE.txt；import=import_ok_snapshot；collect=collect_ok。
- t04 pallets__markupsafe.620c06c9.pr_379: 构建文件[pyproject.toml, setup.py]；license=LICENSE.txt；import=import_failed；collect=collect_failed。
- t05 pallets__markupsafe.620c06c9.pr_400: 构建文件[pyproject.toml, setup.py]；license=LICENSE.txt；import=import_ok_snapshot；collect=collect_failed。
- t06 pydicom__pydicom.7d361b3d.pr_1938: 构建文件[pyproject.toml]；license=LICENSE；import=import_failed；collect=collect_failed。
- t07 pydicom__pydicom.7d361b3d.pr_2069: 构建文件[pyproject.toml]；license=LICENSE；import=import_failed；collect=collect_failed。
- t08 pydicom__pydicom.7d361b3d.pr_1867: 构建文件[pyproject.toml]；license=LICENSE；import=import_failed；collect=collect_failed。
- t09 pydicom__pydicom.7d361b3d.pr_2163: 构建文件[pyproject.toml]；license=LICENSE；import=import_failed；collect=collect_failed。
- t10 pydicom__pydicom.7d361b3d.pr_1859: 构建文件[pyproject.toml]；license=LICENSE；import=import_failed；collect=collect_failed。
- t11 pydicom__pydicom.7d361b3d.pr_1995: 构建文件[pyproject.toml]；license=LICENSE；import=import_failed；collect=collect_failed。
- t12 pydicom__pydicom.7d361b3d.pr_1987: 构建文件[pyproject.toml]；license=LICENSE；import=import_failed；collect=collect_failed。
- s01 pydicom__pydicom.7d361b3d.pr_1920: 构建文件[pyproject.toml]；license=LICENSE；import=import_failed；collect=collect_failed。

## 3. 失败与替补动作（R1/R2/R3/R4）

- All 12 slots staged_ok covering 3 repos; freeze VALID pending GPU V2/V3.

## 4. 冻结有效性

- CPU 阶段 verdict: **FREEZE VALID (CPU stage)**；freeze_valid_cpu=True。
- GPU V2（修前失败）+ V3（参考通过）均实际跑通后方可进入 24 attempts（V4 门控），执行清单见 `gpu-prevalidate-checklist.md`。

## 5. CPU 环境说明（非裁判结论，仅供 GPU 建 env 对照）

- pydicom 8 题 import/collect 失败根因为 **Python 版本**：快照 `requires-python >= 3.10`（flit_core 构建），
  本机 CPU 为 Python 3.9.6，`X | Y` 注解在 def 时求值直接 TypeError，与注入 bug 无关；GPU Linux 必须用 ≥3.10 重测。
- markupsafe t02/t04 import 报 `NameError`（`re` / `_simple_escaping_wrapper` 未定义）且三题（t02/t04/t05）
  `tests/` 仅剩 `__init__.py+conftest.py`（同仓 t03 有 28 个可收集用例作对照）：属快照自身状态记录，
  是否对应 FAIL_TO_PASS 以 GPU V2 为准，CPU 不下裁判结论。
- click t01：快照源码可导入（8.1.8，`__file__` 指向快照内），collect 593 个用例、exit=2（个别收集错误），GPU 重测。

