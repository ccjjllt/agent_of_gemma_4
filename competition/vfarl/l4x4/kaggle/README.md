# Kaggle L4×4 VFARL 实验载体

账号固定为 `qingyu1232`，凭据只在调用 CLI 的进程内读取，不修改全局账号，也不写入 Notebook。用户已授权本实验私有训练材料和 Notebook 上传、运行、更新；不另加墙钟限制，仍受平台时限和本账号实际配额约束。GPU 启动由主代理统一协调。余额须重新读取：第四轮启动前的 `quota-before-gpu-v4.json` 记录累计已用 1.52h、剩余 28.48h，这不是运行中的实时余额。历史开工余额 30.00h 保留在早期 readback 中。

采样与训练复用官方 QAT-unquantized v2 派生的 NF4 缓存 `ccjjllt/p005-cpu-nf4-cache`，不是直接对 CT 打包权重做 QLoRA。CT v2 tokenizer/template 固定；本轮 r32、七个文本投影、BF16、1024 thinking，初始 LoRA B 为零。尚不据此声称 NF4 与部署 CT 权重的迁移效果相同。

## 当前执行方式

`NvidiaL4` 在本实验真实返回四张 NVIDIA L4。载体再次测量卡数和名称，只接受这个四卡 shape；实际计额看账号读数，不按 GPU 张数自行推算费用。

**采样使用两个进程，每个模型分到两张 L4**：分别 `CUDA_VISIBLE_DEVICES=0,1` 和 `2,3`，传 `--policy-gpus 2 --worker-count 2`。模型禁止 CPU/disk offload。默认最多五个 focal states，两个进程配额为 3/2；每题一次原始真实运行，每状态三条分支为原动作重放和两次新采样。重放动作不计为新生成，也不进入随机分支优势组。

所有 collector 退出后，训练仍使用 `torchrun --nproc-per-node=4` 的四 rank FSDP。只有真实 mixed-outcome 数据才可能产生 PPO 更新；全零或同奖组不伪造优势。更新先写 adapter、`optimizer-step.json`、rank update 与 training manifest，再做后置 logprob 诊断。诊断失败不能重训同一 batch 冒充恢复。若已有真实更新，两个双卡进程用 `--task-limit 4 --worker-count 2` 对总共四个固定题做 post-update 运行。

参数改变或非零梯度不独自证明有效学习。还需核对真实异动作异奖、非零分叉优势、同一执行方式下前后 advantage-weighted logprob 的方向，以及完整验收来源。

第四轮实际 Notebook 是 `qingyu1232/vfarl-l4x4-two-gpu-policy-training/1`，源码和回执在 `train-v4/`、`train-v4-push.json`，实时记录在 `train-evidence-v4/`。Kaggle 按标题规范化了初始请求的 slug；已回读确认 private、GPU、无 Internet 和源码 SHA 一致。该轮冻结时只采 99 个已验收题，jail 为 `d1cacb…072a`，不会因本地后续修复而热更新。

## 固定输入和 CPU 证据

| 输入 | 固定来源 | 字节数 / SHA256 |
| --- | --- | --- |
| 100 题材料 v4 | `qingyu1232/vfarl-l4x4-train100/4` | 124,593,514 / `4b3792470dd1dacc73e1899ba38a3d96f2054bfad916d4a9b51b7e4db4333504` |
| 已修任务环境 | `qingyu1232/vfarl-l4x4-tail2-cpu/1` | 967,806,660 / `9475d9a2617135c276d3b4ad87884b28425f871e806c07b62bfc62d4af0cce7e` |
| NF4 producer manifest | `ccjjllt/p005-cpu-nf4-cache` | `3a9661d56f0eec51474741b01d7ddb2932fb4836d7f776051d8ece7c333750cf` |

`cpu-evidence-index-v4.json` 绑定 v4 的 100 个 task 与最终 baseline=0、reviewed repair=1 的真实逐题来源。它合并五次有明确版本的 CPU 结果，不是声称重新跑过一次全量 100 题，也不是模型成绩。被否决的 CA 752 候选 `/1` 不在索引中。

材料只来自 S600 train 白名单。v1→v2 在任何策略采样前替换六个 PTY 不兼容题；v2→v3 替换七个 socket 不兼容题；v3→v4 替换最后一个 PTY 题。保留各版 selection、questions、substitutions 与来源 hash，不依据模型分数挑替补，不读 heldout。

资产包括原 task、source snapshot、runtime、run-summary 和实际 reviewed patch。`submitted.patch`、reference/test 验收要求只给监督/裁判进程，策略工具看不到这些宿主路径。

DVC 最后一个环境的实际修复是：保留 xdist 等原依赖，仅移除本任务不使用、却无条件打开 socket 的 rerunfailures 插件；复制实际公共 CA 资源；将 jail 内 `/var/tmp/dvc` 固定指向被快照覆盖的 taskroot 私有 cache。未开放宿主 `/var`、未改变网络禁令或原 34 个验收节点。`dvc-ca-evidence-v3/` 中冻结 jail `71336c…d98e9` 的结果为 baseline 31 fail + 3 pass、reviewed repair 34 pass。当前主 jail 已合入该实证修复，第四 GPU 仍保留旧冻结源。

## 载体参数

`build.py --selection-path competition/vfarl/l4x4/v4/selection.json --out <新目录>` 只打包，不上传。`build_runner.py` 只冻结脚本和 metadata，不启动 GPU；实际提交命令写在 carrier manifest 中。

| 参数 | 行为 |
| --- | --- |
| `--mode prevalidate` | CPU 官方 baseline/reviewed-repair 验收；默认四 worker，可用 `--only-task` 做真实单题复现。 |
| `--mode train` | 两个双卡 collector，随后按数据执行四 rank FSDP；硬件总量仍四 L4。 |
| `--runtime-kernel owner/slug/version --runtime-sha256 HASH` | 固定已完成 runtime 版本和实际 archive hash；GPU 载体必须明确提供。 |
| `--dataset-version 4` | 固定材料版本，不能依赖 latest。 |
| 多次 `--prevalidation-report PATH` | 冻结真实报告；按明确顺序后面的同 task 记录替换前面的，并保存来源 hash。只保留当前 selection 中的 task。 |
| `--collect-only` | 保存所请求的真实 rollout/分支，跳过 optimizer 和 post-update；不暗示更新发生。 |
| `--focal-states N` | 全作业最多 N 个状态，默认 5，按 ceil/floor 分给两 worker。0 表示只采原始运行。 |
| `--context 4096` / `8192` | 同时设置 collector 上下文与训练配置最大序列长度；不同预算的状态组不得混合。 |
| `--only-task ID` | 限定本次采样/CPU 验收；全局题序仍用于固定采样 seed。空 partition 提前返回，不装空闲模型。 |
| `--jail-source PATH` | CPU 可冻结待验证候选；GPU override 只接受已实际验收的 d1ca 或 713 两个完整 SHA，路径/hash/依据写入 manifest。 |
| `--export-runtime` | CPU 子集修环境后也导出 runtime archive；单纯验收可不重复发布大 archive。 |

构建 GPU 载体时，材料清单使用 `artifacts/material-v4/material-build.json`，runtime 使用上表固定版本。最初一题环境不通过不能冒作策略失败；当前 100 题的最终来源按 `cpu-evidence-index-v4.json` 所列顺序冻结即可。只有最终 GPU push 使用 `--accelerator NvidiaL4`。

## 后续两个独立臂

必须等第四轮完成后依次运行，不并发启动 GPU，也不热改已运行脚本。

- A：DVC 单题 4K，使用已验收 713 主 jail 和 tail2 runtime；`--collect-only --focal-states 0`，只补覆盖，不根据结果挑更新。
- B：七题 8K，保持第四轮同一 d1ca jail、tail2 runtime、初始 policy/seed、1536 总响应 tokens（含最多 1024 thinking tokens）。只从第四轮 original 的 verified=true、reward=0 且真实 context 耗尽记录中，按冻结的全局题序取前七；不看分支结果挑题。若第四轮已有有效更新，B 只做 capacity/rollout（collect-only、states 0）；若第四轮没有 mixed group，B 固定最多五个 states 后独立尝试一次 PPO。选择时核对 original summary 的 termination 与 ledger：context 余额 <16（等价 final_context_tokens≥4096−15），并且 output 余额≥16。例如 4084 tokens 的 context 终止符合，3852 tokens 的 malformed_action 不符合。不能混合 4K/8K 的同状态组。

## 保存与观察

先将 immutable NF4 文件单流复制到足够大的 `/tmp`，逐文件验证 producer size/SHA，再让各模型读本地副本；实际空间不足时明确记录并保留原挂载路径。第二轮复制 18.30GB 用时 146.44s，随后四模型的权重加载进度约 4s；首轮直接并发读挂载时约 24:36。权重和 policy identity 没有因此改变。

每 60s 保存四卡 `nvidia-smi`、worker 存活/RSS/Swap/Threads 和原始日志尾部；转发真实 policy request/action、episode result、focal group 和 collection error。未来载体的训练阶段也使用同样 telemetry。日志服务 HTTP500 不等于作业失败；需只读核 status、当前源码 hash、普通日志和输出来源，不能把接口列出的旧版本输出当作当前成功。

完成或异常后，先停本次 worker 并清本次 task UID，再清 unpacked 输入和临时 NF4 cache。强制中断的实际 workspace（含 `.git`）先保存小型 `.tgz` 和 SHA，再删 ephemeral sandbox 的 readonly runtime 镜像；动作、工具反馈、异常、验收和 state snapshots 保留。展开的 sandbox/state root 不进 Git，仍在本地和固定 Kaggle 输出中；manifest 与紧凑证据保留。

## 早期失败保留为历史

早期四个单卡 collector 的 2/1/1/1 状态配额不再是运行建议：

- GPU 1（`vfarl-l4x4-train/1`）：四模型加载成功，但 HF template 返回容器不符导致 83 次动作前失败，0 action、0 update；见 `train-core-evidence-v1/result-summary.json`。
- GPU 2（`vfarl-l4x4-train/2`）：5 个真实 action 后在 3,258/3,390-token 后续 prefill OOM，0 完整 episode、0 update；真实工具结果和 Agent 创建的复现脚本保留。
- GPU 3（`vfarl-l4x4-train-chunked/1`）：加入分块 prefill/reference 和 allocator 后仍在 3,384-token 前缀的 bitsandbytes dequant 临时分配 OOM；7 个已见真实 action，未完成 update。其失败支持改成每模型两卡，不重复同单卡布局。

首 CPU 阶段 17 profiles 中曾只有 16 个完成，profile-02 的本机 `pipdeptree==0.1.dev1+gedbd59b41` 不可下载；实际删除这个宿主开发工具有记录。bwrap/user namespace 被拒，mount/PID/net unshare 也 EPERM，Landlock ENOSYS。随后真实 capability 与 `jail-probe/2` 证明 chroot/dropuid/caps/NNP/seccomp、gold 隐藏、readonly runtime 和 task UID 进程清理可用，才进行正式同隔离的官方验收。

原 `vfarl-l4x4-cpu-stage` 输出发布曾卡住，改用相同优化源码的 `vfarl-l4x4-linux-runtime/1` 恢复，实际基础 17 profiles 成功。`task_site_packages_profile=null` 的 19 题原来继承各自父环境，并非 pytest-only，已按真实 metadata 分为 11 个 stock profiles。最初全量 CPU 36/100 可用，修复后的 R2 为 83/100，后续替补和具体环境修复形成上面的累计 100/100；这些都不是策略成绩。

所有实现核对使用一次性 AST/手工命令，没有新增维护测试套件。仓库原有官方验收是实验判据；本文不是训练完成或高分专家声明。
