# P013 GPU 预验证 kernel 耗时/配额估算（CPU 阶段写入，GPU 实测校准）

日期 2026-10-20。类比 P012 倍率 2x（GPU Linux 冷 venv + 全量 env 保险系数）。

## 输入规模（CPU 实测）

- 13 快照共 ~46MB：click 0.7MB + markupsafe 4×~60KB + pydicom 8×~6MB。
- replay 共 52 个 str_replace ops（无 create；t07 最多 16 ops，t05 9 ops）。
- 13 criterion 文件 `py_compile` 全过（CPU）。

## 耗时模型（单机串行，13 题）

主导项是每题独立 venv + `pip install -e .`：

| 题组 | 题数 | 单题估计（含 2x） | 小计 |
|---|---|---|---|
| click t01 | 1 | ~6 min（小包，安装快） | ~6 min |
| markupsafe t02–t05 | 4 | ~6 min/题 | ~24 min |
| pydicom t06–t12+s01 | 8 | ~20 min/题（numpy 等全量 env + 大测试收集） | ~160 min |
| V2/V3 pytest 本体 | 13 | ~2 min/题（单节点 + 同文件全跑） | ~26 min |

合计 ≈ **3.5–4h 墙钟**（串行）；超时上限按 6h 设（`INSTALL_TIMEOUT=1200`，
`PYTEST_TIMEOUT=900`）。t12 优先跑：若最高风险题 early-fail，
R1/R2 路径 30 min 内可见。

## 配额

- GPU：1 次 kernel 运行（L4 单机即可；preverify 无模型、无 harness，
  不耗推理配额；`enable_internet=false`）。
- 数据集：私有 `p013-task-material` ~50MB（快照 46MB + criterion/replay/index/jsonl <1MB）。
- 存储：`/kaggle/working/preverify` 峰值 ≈ 13×（解包树 + venv）：pydicom 解包 ~11MB +
  venv ~300–600MB/题 ×8 + 小包 ≈ **3–5GB**，顺序复用 workdir 可控（脚本逐题重建目录）。
- 人工：主智能体执行推送 3 步（建数据集 → 推 kernel → 取回 `dev12-freeze.json` +
  `result-*.json`）；V4 裁决只认 GPU 产物，不编造。

## P012 先例对照（prepare/full 双模式思想）

- P012 `prepare`（预验编译 + 工具注册）→ `full`（24 attempts）：本 kernel 即 P013 的
  prepare 等价物——V1 建 env + V2/V3 校准先行，`dev12-freeze.json` 为 full（24 attempts
  配对评测）的唯一放行依据。
- 数据与代码分离（P012 carrier 先例）：criterion/replay/快照走版本化私有数据集挂载，
  kernel 代码文件只含 `preverify.py`（单文件推送上限友好）。
