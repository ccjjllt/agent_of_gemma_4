# P014 新开发题 CPU 准备

2026-10-20。使用已有 P013 13 题池，按预先固定的 t01..t12,s01 顺序，未观察任何 P014 模型输出。

## 已实跑结果

本地 Python 3.12 隔离环境中，对每题校验完整 git HEAD，并确认模块来自该题工作树。沿用已有 criterion 文本内联执行，不复制或新增 test 文件，不修改原判据。修前运行一次、参考生产编辑后运行同一判据一次。

最终选定 **8 题 / 3 仓**：`t01,t02,t03,t04,t05,t06,t08,t09`，均修前退出码 1、修后退出码 0。原始结果与日志在 `local-cpu-v2/`，冻结清单为 `local-cpu-v2/panel-freeze.json`。

首轮曾发现 pydicom 缺 distribution metadata；在隔离环境对原始 t06 baseline 做真实 editable 安装后恢复。旧环境错误原件保留在 `local-cpu/`，不计模型失败。`t07` 和 `s01` 因参考编辑锚点不匹配未进入有效题；其他已通过题作为模型运行前确定的剩余开发资源保留。没有声称独立 P2P 回归已经执行。

## GPU 读取接口

`material/` 已包含：

- `agent-tasks.json`：仅 task_id、instance_id、repo、完整 base_commit 与 problem_statement。原题取自 criteria-index 指定轨迹第一条用户消息中的 `<pr_description>`，不含教师 assistant 输出或旧 `/testbed` 运行指令。
- `host-criteria.json`：仅供外部裁判的 criterion_source、f2p_node、p2p_nodes（空）、module；不放入 Agent 工作区。
- `snapshots/<instance_id>.tar.gz`：原始源码直接位于 archive 根目录，保留 `.git` 与完整 HEAD；不含参考生产修改。
- `wheelhouse/`：Linux CPython 3.12 numpy/pytest/构建工具依赖，以及从未修复 t06 baseline 构建的纯 Python pydicom wheel。该 wheel 为 distribution metadata 提供真实安装来源；运行时仍强制 PYTHONPATH 指向实际任务源码。
- `material-manifest.json`：上述产物大小和哈希。

## 真正自包含的 CPU 作业

`cpu-kernel/panel.py` 是 84 KB 单 code_file，已嵌入本批全部小文本输入与必要运行代码，不依赖上传目录中的邻接文件。`cpu-kernel/kernel-metadata.json` 为 `zhuating/p014-frozen-eight-material-cpu`（2026-10-20 20:04 北京时间推送回执的实际 slug，Kaggle 按 title 创建），CPU、联网、无比赛源。

根代理已统一推送，本子任务没有推送或启动任何 GPU。作业先 clone 八题原始快照并核对 HEAD，下载 Linux wheels、构建/安装 pydicom metadata，再对冻结八题在 Linux 运行必要 F2P，输出 `/kaggle/working/p014-panel/`。该目录可经 kernel_sources 供 GPU 使用。Linux 若未重现本地八题有效性，作业会保存真实日志并失败，不静默换题。源码、判据、参考编辑仅在 CPU/host 层使用，GPU Agent 只看到 task 原题和隔离工作树。

已修复现有 `research/013/eval-31b-pair/preverify-kernel/preverify.py`，移除旧固定 12 题门槛、任意 ImportError 环境误分类、重复同文件伪 P2P，以及依赖本地邻接文件却未上传的使用方式。当前为实际 CPU 实验运行代码，无新增测试体系。

2026-10-20 20:07 北京时间：v1 在 Kaggle Debian 的 ensurepip 不可用处退出；根代理已修为 `venv.create(with_pip=False, system_site_packages=True)`，从 host 复用 pip，并统一推送 v2。当前 code_file 已含修复；没有另存含旧配置的 builder，原 preverify 运行源也不再创建 venv。未因此重推其他作业或扩充验证。
