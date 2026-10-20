# P014 唯一三题确认批（CPU 已完成）

在首批 GPU 结果返回前，整体固定剩余有效开发题 **t10、t11、t12** 为唯一一次确认批，不依据稍后的模型输赢挑选其中题目。

这三题全部来自 pydicom：HTJ2K 常量、未知 SRT 映射比较、DataElement repr 一致性。修前失败/参考后通过已在 `../local-cpu-v2/` 实跑；本地打包准备没有重复执行任何判据或回归；后续线上 CPU 仅在 Linux 复现这三题必要的修前/修后结果。

`material/agent-tasks.json` 只含任务身份、完整 40 位 base_commit 和指定轨迹第一条用户消息中的原 PR description；`material/host-criteria.json` 为外部裁判接口，不给 Agent。`material/snapshots/*.tar.gz` 保留 `.git/HEAD`，源码直接在 archive 根目录。`material/panel-freeze.json` 记录本次预先选择及已有原始结果来源。

已推送载体：`cpu-kernel/`，id/title 对应实际 slug **zhuating/p014-confirm-three-material-cpu**。单 code_file 约 37 KB，直接复用首批根代理修正后的成熟 CPU 实现，`with_pip=False, system_site_packages=True`；只替换嵌入的三题数据和数量。联网 CPU 作业会准备独立输出、Linux wheels 与 baseline pydicom metadata，无 GPU。

根代理已于 UTC 约 2026-10-20 12:16（北京时间 20:16）推送 `zhuating/p014-confirm-three-material-cpu/1`。本子任务只读查询与取回原件，没有推送作业。三题同属一个仓库，且 t10/t12 修改通常较小，不能把它们描述成新增跨仓泛化证据；单独报告确认结果。

## 线上实际结果

只读查询时间：`2026-10-20T12:49:10.629171+00:00`；API 状态 **COMPLETE**。Linux Python 3.12.13 的冻结文件写于 UTC `2026-10-20T12:16:58Z`，三题全部修前退出码 1、参考修后退出码 0，完整 HEAD 与原先固定身份一致。

| 题目 | 修前 | 参考修后 | 结果 |
|---|---:|---:|---|
| t10 HTJ2K 常量 | 1 | 0 | 有效 |
| t11 未知 SRT 映射 | 1 | 0 | 有效 |
| t12 DataElement repr | 1 | 0 | 有效 |

已取回 [实际冻结清单](remote-cpu/output/p014-panel/panel-freeze.json)、[32 项产物清单](remote-cpu/output/p014-panel/material-manifest.json)、[状态查询](remote-cpu/status.json) 和主作业日志。清单包含 agent 原题、host 判据、三份原始快照和离线 wheelhouse；日志以 `P014_MATERIAL_READY 3` 结束，没有 failure 产物。

本次仅确认 CPU 材料有效，未在三题上运行任何模型，不构成新增修复数。本批不作为正式提交 `57438326` 的门槛；主线提交与首批结果由根代理单独记录。
