# 目录整理记录

2026-10-22。保留已有研究编号、训练/推理数据、权重、真实成功与失败结果、来源指纹及配额记录；整理前 Git 工作区干净。整理没有启动训练、GPU 或官方评测，没有读取 heldout 内容，也没有新增测试文件或套件。

## 目录调整

- 根目录增加 [项目首页](../README.md)，文档、研究、Notebook 各增加导航入口。
- `notebooks/` 中编号载体归入 `001/`、`002/`、`003/`、`005/`、`006/`、`007/`、`013/`；五个生成器统一归入 `notebooks/builders/`。对应本地输入输出路径、重建命令及文档链接同步调整。
- `notebooks/kernel-metadata.json` 归入 `notebooks/002/`，与其 `002-preflight.ipynb` 同目录。
- 删除根目录重复的 `p007-12b-cpu-nf4-cache.py`，唯一维护入口保留为 `research/007/12b-qat-cache-and-train-20261017/cpu-nf4-cache/cache_12b.py`；两份原文件逐字相同，SHA256 为 `91a1ba6b1cc5fe61c7aa1909e4e71051b8bb621d27d9a30762576d78948d7718`。

旧 Notebook 路径 `notebooks/NNN-名称` 对应新路径 `notebooks/NNN/NNN-名称`；旧生成器路径 `notebooks/build_*.py` 对应 `notebooks/builders/build_*.py`。详见 [Notebook 导航](../notebooks/README.md)。生成的历史 single-cell 与 ipynb 保持原字节；研究产物内记载的旧路径、版本和 SHA 仍表示当时来源，不改写原始运行证据。

## 已清理的可重建内容

以下目录没有 Git 跟踪文件；删除前已核对来源与保留副本。字节数为被删除普通文件的逻辑大小，不等同文件系统实际回收空间。

| 清理位置 | 文件数 | 字节数 | 保留证据与恢复来源 |
| --- | ---: | ---: | --- |
| `research/015/evaluation/deps/` | 4,623 | 55,394,592 | 本机临时 venv；保留 `evaluation/material/environments/` 的依赖清单与离线 wheel，实际 Linux 结果另存于 `cpu-prevalidation/output-v2/` |
| `research/015/evaluation/material/workspaces/` | 4,248 | 182,249,603 | CPU 修前/修后执行工作树；保留 `material/snapshots/`、来源判据、参考补丁和 `evaluation/evidence/` 原始日志 |
| `research/015/evaluation/cpu-prevalidation/staging/` | 229 | 82,264,386 | 上传前暂存副本；原上传包保留在 `data/panel-material.tar.gz` 和 `data-v2/panel-material.bin`，逐文件来源在 `input-manifest.json` |
| `research/015/capacity/output-attention/p015-attention/deps/` | 119 | 1,784,237 | 算子作业安装副本；`attention_run.py` 会从原 P005 哈希核对的 wheels 安装，实际算子结果和日志保留 |
| `research/014/panel/local-cpu/snapshots/` | 4,778 | 112,999,667 | 一次性 CPU 克隆；P013 `eval-31b-pair/snapshots/` 保留 13 个原快照，`attempts/`、冻结清单与原失败日志保留 |
| `research/014/panel/local-cpu-v2/snapshots/` | 4,778 | 112,999,667 | 同上；P014 实际八题发布材料另保留在 `panel/material/` |
| `.DS_Store`、`.pyc`、`.pyo`（`.git/`、`.venv/` 除外） | 57 | 1,015,375 | OS/Python 自动生成，可重新产生 |
| **合计** | **18,832** | **548,707,527** | 约 **523.3 MiB**，另删除根目录 15,583 字节重复脚本 |

这些临时环境和工作树需要时按相应轮次的材料说明重建，再执行原实验命令；整理后不会保留可直接复用的这些展开目录。P014 原本机失败日志和两版冻结 JSON 均保留。

## 保留边界

`training/artifacts/`、`results/`、原始训练/推理输出、模型权重、离线 wheels 和 tokenizer 均保留。没有清空 Git 历史，也没有移动 `research/002`–`015`。上游源码中的 `tests/`、数据源既有判据与 test patches 是实验素材，不按目录名当垃圾删除。

## 手动核对

36 份迁移的历史载体与元数据逐字节比对一致；五个生成器完成 Python AST、标准库导入链和输入输出路径核对。相关 Markdown 本地链接与 P002 `code_file` 均可解析，`git diff --check` 无错误。清理范围外的数据/权重文件路径和大小与清理前元数据一致；原始快照、离线依赖、材料包和 P014 两版本机日志均保留。没有运行或重建历史 Notebook。
