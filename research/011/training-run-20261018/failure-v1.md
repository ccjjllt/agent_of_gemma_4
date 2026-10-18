# P011 V1：资源关联配置错误，训练未开始

2026-10-18用户报告失败后取回V1日志。实际运行ID为 `run-011-continuous-sft-20261018T104604Z-15b73e9f`。本次是提交配置错误，不是训练OOM、数据损坏或模型优化失败。

## 已确认事实

- CLI push确实传了 `--accelerator NvidiaL4 -t 7200`；[服务器回读metadata](remote-v1/kernel-metadata.json)也保存了 `machine_shape: NvidiaL4`，但 `competition_sources` 为空。
- 真实 `nvidia-smi` 遥测显示 **2×Tesla T4，每卡15360MiB**。三次采样利用率与显存占用均为0。
- `training/failure.txt` 明确报错：`This run requires exactly four L4 GPUs: ['Tesla T4', 'Tesla T4']`。
- 监督器实际耗时75.1508秒，退出码1，没有超时；错误发生在加载基座、启动torchrun之前，**训练步数0、checkpoint数量0**。
- ccjjllt账号累计配额29307.285→29389.877秒，同期差值82.592秒（约1分23秒）。这是共享账号累计差值，不冒称独立账单。

## 为什么GPU选错

[官方Upgraded Accelerators说明](https://www.kaggle.com/competitions/gemma-4-developer-agent/overview/upgraded-accelerators)要求L4仅供关联本比赛的Notebook使用，同时关闭互联网。原文："L4s are only available for notebooks attached to this competition."

P009/P010成功作业都配置了 `competition_sources: ["gemma-4-developer-agent"]`；P011改为纯训练载体时错误地删除了这个关联，使其不满足L4资格。比赛关联与读取比赛任务内容是不同操作：保持训练输入隔离并不要求移除关联。

服务器保存的请求型号仍为L4，而实际硬件是T4；没有后台调度日志，因此不声称看到了平台内部回退分支。但资格配置缺失与实际分配错误均有直接证据。上次报告把NvidiaL4请求直接写成4×L4执行配置，确认不足，现已纠正。

## V2的实际修复

1. 恢复比赛关联，metadata与CLI同时明确 `NvidiaL4`，互联网保持关闭。
2. id对齐现存作业 `ccjjllt/p011-12b-continuous-sft-139-issues`，提交同一作业V2。
3. runner在安装依赖前记录真实GPU数量、型号、显存和UUID；不满足四张L4立即退出。训练阶段原有硬件要求继续保留。

139个问题、两轮70步、LoRA配方及编码数据SHA保持不变。代码只读取载体内已冻结训练数据，没有读取比赛任务或保留评测内容。本次没有新增测试文件或额外GPU探针，修复后直接继续这批训练。

原始V1载体保存于 `submitted-v1/`，远端原件在 `remote-v1/`，真实日志在 `gpu-output-v1/results/011/<run>/`。V2回执独立保存为 `job-request-v2.json`；不能仅凭RUNNING或请求metadata宣称实际四张L4或优化进度已确认。
