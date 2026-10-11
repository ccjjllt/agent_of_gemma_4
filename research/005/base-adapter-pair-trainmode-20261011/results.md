# 尚未进行真实 SFT 或 adapter 配对

更新：2026-10-11。本轮没有真实SFT adapter，因此尚无base/adapter新增修复、回退或耗时差。原始状态文件可从提交 `d5fb6a3` 回溯，不再另维护skipped/blocked JSON。

训练态6032对照中，baseline在SDPA前向OOM；offload完成前向后反向CUDA OOM，最后观察到重算层53，具体失败算子未定位。两臂均未完成首次Adam。9,841/12,832未运行。完整证据见[实验结果](../save-on-cpu-6032-trainmode-20261011/results.md)。

预留八题仍为 `fastapi_13713`、`requests_6592`、`requests_6644`、`requests_7309`、`requests_7505`、`rich_3130`、`rich_3296`、`rich_3469`，不得进入训练数据。诊断adapter不作为修复收益臂。没有正式比赛提交。

下一步研究安排见[研究计划](../../../docs/plan/005-trainmode-save-on-cpu.md)。先争取真实修复机制信号与独立训练数据，不继续扩建阶段门控。
