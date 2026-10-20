# P013 GPU 预验证作业设计（V2 修前失败 / V3 参考通过）

供主智能体推送 GPU 执行。CPU 阶段未执行任何裁判；以下全部待 GPU 实机跑通，log 指针入 `dev12-freeze.json` 后方可 V4 放行。
前置：CPU 物化 verdict `VALID`；逐题执行 V1→V2→V3→V4。

## 0. 通用环境

- GPU Linux 镜像，Python（以各题 requires-python 为准），`pip install -e .` 或按 deps 记录安装；
- 官方 SWE-smith loader 按 `instance_id` 物化（不消费教师轨迹）；快照 HEAD 必须等于下表 base_commit；
- 每题作业隔离目录；V2/V3 日志落盘并记录 sha256 指针；判据区分性（A/B 拉开差距）**不是**预验证门槛。

## 1. V1 物化（官方 loader 重建 + 构建/许可记录）

```bash
# <iid> 逐题：
git clone --branch <instance_id> https://github.com/<mirror>.git task && cd task
git rev-parse HEAD  # 必须等于 base_commit
python -m venv .venv && . .venv/bin/activate && pip install -e .  # 按需加测试 extras
# 记录： snapshot 字节 / 构建系统 / 许可证（许可文件见 deps 记录）
```

## 2. V2 修前失败（buggy 环境真实失败）

```bash
# 在 buggy 快照上应用当题 criterion test_patch（GPU 阶段编写 tests/test_p013_*.py），然后：
pytest tests/test_p013_<name>.py::<F2P_node> -x -q 2>&1 | tee v2-prefail.log
# 要求： exit != 0 且为断言/行为失败（非收集错误/缺依赖）；log 指针入 dev12-freeze.json
```

## 3. V3 参考通过（同环境 F2P 通过 + P2P 通过）

```bash
# 应用参考补丁后同环境：
pytest tests/test_p013_<name>.py::<F2P_node> -q 2>&1 | tee v3-postpass.log        # 要求通过
pytest tests/test_p013_<name>.py -q 2>&1 | tee v3-p2p.log                         # 同文件 P2P 通过
# 要求： 两者 exit == 0；log 指针入 dev12-freeze.json
```

## 4. V4 门控与换题

- V2+V3 均实际跑通 → 该题放行 24 attempts；任一不满足 → 走 R1（同仓下一 rank）/R2（s01 跨仓浮动）；
- R3：click t01 无替补，失败即冻结无效（重走冻结）；R4：整仓不可物化即冻结无效；
- 永远禁止：按模型成败换题；重召 3 个触碰 issue；跳哈希顺序挑题；列 heldout/预留题 ID。

## 5. 逐题基线（CPU 物化实测，供 GPU 对照）

| task | mirror | base_commit | snapshot 字节 | import | collect | 构建文件 |
|---|---|---|---|---|---|---|
| t01 | swesmith/pallets__click.fde47b4b | 01ce28d1459a | 723937 | import_ok_snapshot | collect_partial | pyproject.toml |
| t02 | swesmith/pallets__markupsafe.620c06c9 | 654e45a0aff2 | 60780 | import_failed | collect_failed | pyproject.toml,setup.py |
| t03 | swesmith/pallets__markupsafe.620c06c9 | 21069fc5637b | 63776 | import_ok_snapshot | collect_ok | pyproject.toml,setup.py |
| t04 | swesmith/pallets__markupsafe.620c06c9 | ef1645060d91 | 60584 | import_failed | collect_failed | pyproject.toml,setup.py |
| t05 | swesmith/pallets__markupsafe.620c06c9 | 9d918e9f0866 | 60667 | import_ok_snapshot | collect_failed | pyproject.toml,setup.py |
| t06 | swesmith/pydicom__pydicom.7d361b3d | 927e85a9a279 | 5969172 | import_failed | collect_failed | pyproject.toml |
| t07 | swesmith/pydicom__pydicom.7d361b3d | 09fded6bbe5d | 5982650 | import_failed | collect_failed | pyproject.toml |
| t08 | swesmith/pydicom__pydicom.7d361b3d | b09c388cb335 | 5982464 | import_failed | collect_failed | pyproject.toml |
| t09 | swesmith/pydicom__pydicom.7d361b3d | ceaea04e48fd | 5957079 | import_failed | collect_failed | pyproject.toml |
| t10 | swesmith/pydicom__pydicom.7d361b3d | b2090318825c | 5500817 | import_failed | collect_failed | pyproject.toml |
| t11 | swesmith/pydicom__pydicom.7d361b3d | f6f2f4a27d6a | 5988522 | import_failed | collect_failed | pyproject.toml |
| t12 | swesmith/pydicom__pydicom.7d361b3d | 8efbb6a346e9 | 5969168 | import_failed | collect_failed | pyproject.toml |

注：import/collect 为 macOS CPU 抽查（PYTHONPATH 直引、collect-only），不代表 GPU Linux 结论；pydicom 若 CPU import 缺 numpy，属环境缺件记录，GPU 建全量 env 后重测。

