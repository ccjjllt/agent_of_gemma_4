# Notebook 目录导航

独立运行载体按研究轮次归档，生成器集中放在 `builders/`。本目录保存的是各轮历史载体；当前研究进度、实际结果和 GPU 安排以 [P015 执行记录](../research/015/README.md)为准。

| 目录 | 内容 |
|---|---|
| [001/](001/) | 初始基线与 smoke 载体 |
| [002/](002/) | 资格、容量、preflight 与 joint explore；包含其 Kaggle 元数据 |
| [003/](003/) | 预算、上下文与 A0/B/C 联合探索 |
| [005/](005/) | single_state 草稿、单臂与 T0 双臂；[双臂说明](005/005-t0-single-state-pair.README.md) |
| [006/](006/) | 历史 exec_closure 与 Budget-Fit 参考对照；[使用说明](006/006-reference-pair.README.md) |
| [007/](007/) | Public-12 提示对照；[使用说明](007/007-public12-pair.README.md) |
| [013/](013/) | 31B repair → review 诊断载体 |
| [builders/](builders/) | 五个使用标准库的生成器，共享原有跨模块导入链 |

## 重建

从项目根目录运行对应命令。生成器本身只处理本地源码与冻结材料，不加载模型；执行生成后的 Notebook 或 single-cell 文件会按该轮配置运行真实实验。重建会改写对应载体，部分生成器也会更新 `research/<轮次>/.../kernel/`，所以日常查阅与目录核对无需重建。

```sh
PYTHONDONTWRITEBYTECODE=1 python3 notebooks/builders/build_005_single_state.py
PYTHONDONTWRITEBYTECODE=1 python3 notebooks/builders/build_005_t0_single_state_pair.py
PYTHONDONTWRITEBYTECODE=1 python3 notebooks/builders/build_006_reference_pair.py
PYTHONDONTWRITEBYTECODE=1 python3 notebooks/builders/build_007_public12_pair.py
PYTHONDONTWRITEBYTECODE=1 python3 notebooks/builders/build_013_diag_pair.py
```

历史载体依赖当时的镜像、wheelhouse、冻结材料及任务划分。尤其 P005 原六题 panel 包含后来预留的任务，不能直接用于当前调参；运行前应遵循对应说明和现有研究计划。

## 2026-10-22 路径迁移

| 旧路径 | 当前路径 |
|---|---|
| `notebooks/NNN-<文件名>` | `notebooks/NNN/NNN-<文件名>`，轮次为 001、002、003、005、006、007、013 |
| `notebooks/build_<名称>.py` | `notebooks/builders/build_<名称>.py` |
| `notebooks/kernel-metadata.json` | `notebooks/002/kernel-metadata.json`，`code_file` 仍相对该目录 |

本次仅移动历史 single-cell、ipynb 和运行元数据，保留其原始字节。它们内部的生成器路径、来源清单和 SHA，以及 `research/` 中历史记录的路径，仍描述当时的版本；查找现有文件时按上表转换。新生成器已使用当前路径和项目根位置。

根目录原 `p007-12b-cpu-nf4-cache.py` 已去重，其完全相同的原件保留在 [P007 CPU 缓存入口](../research/007/12b-qat-cache-and-train-20261017/cpu-nf4-cache/cache_12b.py)。
