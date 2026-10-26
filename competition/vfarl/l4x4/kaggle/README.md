# Kaggle L4×4 VFARL 实验载体

账号固定为 `qingyu1232`，凭据只在调用 CLI 的进程内读取，不修改全局账号，也不写入 Notebook。用户已授权本实验私有训练材料和 Notebook 上传、运行、更新；不另加墙钟限制，仍受平台时限和本账号实际配额约束。GPU 启动由主代理统一协调。新 16K 研究前的只读 `16k-before-readback.json` 确认累计已用 10.78h、剩余 19.22h；首个概率校准后为 11.63h/18.37h，第二个校准后为累计 13.39h、剩余 16.61h，见对应 `calibration-16k-evidence-v1/`、`calibration-16k-evidence-v2/completion-readback.json`。启动新的 GPU 作业前仍需读取最新余额。历史开工余额 30.00h 和各轮读数继续保留。

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
| `--context 4096` / `8192` / `16384` | 同时设置 collector 上下文与训练配置最大序列长度；不同预算的状态组不得混合。 |
| `--training-config PATH` | 明确冻结模型/RL YAML 的全文、仓库相对路径及 SHA；默认既有 4K YAML，主 16K 使用独立 16K YAML。runner 另记声明窗口与实际 `--context`，不暗中选择校正目标。 |
| `--calibration-data PATH` | 冻结真实动作引用 manifest，用现有训练入口依次执行双卡 actor 和四 rank FSDP 的零 optimizer 概率研究。 |
| 多次 `--calibration-action-file PATH` | 将 manifest 引用的原始 actions.jsonl 按仓库相对路径、原始文本冻结入源码包，文件和行哈希由消费者核对，old logprob 不重写。 |
| `--calibration-only` | 只执行上述概率研究，不运行任务 collector、optimizer 或 post-update。与 collect-only 的真实 rollout 用途分开。 |
| `--only-task ID` | 限定本次采样/CPU 验收；全局题序仍用于固定采样 seed。空 partition 提前返回，不装空闲模型。 |
| `--jail-source PATH` | CPU 可冻结待验证候选；GPU override 只接受已实际验收的 d1ca 或 713 两个完整 SHA，路径/hash/依据写入 manifest。 |
| `--export-runtime` | CPU 子集修环境后也导出 runtime archive；单纯验收可不重复发布大 archive。 |

构建 GPU 载体时，材料清单使用 `artifacts/material-v4/material-build.json`，runtime 使用上表固定版本。最初一题环境不通过不能冒作策略失败；当前 100 题的最终来源按 `cpu-evidence-index-v4.json` 所列顺序冻结即可。只有最终 GPU push 使用 `--accelerator NvidiaL4`。

## 当前授权：16K 优先的完整闭环

用户已明确将下一步改为 **16K 优先，只有实际容量不成立才下降到 8K**。低奖励、同奖组或低成功率不构成降低上下文的理由。最初的 4K/8K 两臂计划保留为历史；16K 新载体不沿用旧状态组，也不热修改任何既有作业。

优先修正焦点状态与概率计算：恢复前缀必须已经包含至少一次真实失败工具反馈，不能继续把 turn0 初始状态或纯成功 find/read 叫作失败反馈后的恢复。完整验收为 0 或 1 的原 episode 均可提供这种真实错误后的状态，保留 origin_episode_reward；不能把已恢复成功的原 episode 改称整题失败。参考概率使用独立真实前向，旧行为概率保持原件。先以不看奖励、按全局顺序和长度选择的真实动作做 actor 同路径重复评分、独立 reference 与四 rank FSDP 零更新评分，保留每条概率、ratio、执行身份、显存和参数前后哈希。这是数值研究，不产生奖励、优势或训练完成声明。

校准载体先只装一次 NF4 本地副本：actor 使用可见卡 0–1，退出后才启动四 rank FSDP；`--calibration-only` 明确禁止 optimizer。完整闭环载体使用既定两个双卡 collector、3/2 个反馈后状态和独立四卡 FSDP。若真实出现 mixed group，保存 actor theta0 两次评分，训练后重新加载 adapter 保存同路径 theta1，并做四个固定训练题的实际 post-update rollout。

主 16K 采样把独立 reference 计算集中在真正进入 focal loss 的随机分支首动作：这些动作的 reference 由同冻结基座、匹配采样计算路径的独立前向实际得到。其余 original/continuation 保留真实 behavior logprob，reference 明确为未计算，不得进入训练或 KL。这减少无训练用途的重复解码，不复制 old、不重建未知 behavior，也不把未评分动作冒称有 reference。

16K 初始任务采用已允许的 DVC 补题及旧 4K original 中按冻结全局序号排列的前七个真实 context 失败任务，选题不读取分支成功率。原 100 题材料、五份 CPU 验收来源及已验收 713 jail 全部绑定；使用新的私有 Notebook slug 保留旧输出。实际选题与来源以新 manifest 为准，本文不将未执行的后续实验计为成果。

首个零更新校准已实际完成：`qingyu1232/vfarl-l4x4-probability-calibration-16k/1`，source SHA `e0da8c31d7db300cff0f3c06f8c3e54c475b29b487fc8c487b3076c2887fa930`。三条真实动作的 actor 同路径重评分、重复评分、独立匹配 reference 均与实际 old 逐 token 一致；原分块 reference 则最大差 0.1024/0.0214/0.9345。四 rank FSDP current 对 old 的最大差为 0.2946/0.1099/2.1216，而其自身重复、grad/no-grad、reference/current 均一致。四 rank 的参数 shard hash 前后未变，optimizer 为 0；actor/FSDP 分别耗时 1155.31/114.13 秒，本次配额 0.85h。精确数组、来源及结果见 `calibration-16k-evidence-v1/`。这些实际序列仅 1551/3575/3767 tokens，不能将配置上限 16K 当作已测 16K 容量；因此没有直接沿用该概率路径启动主采样。

第二个零更新校准也已实际完成：`qingyu1232/vfarl-l4x4-probability-aligned-16k/1`，source SHA `f1ece93424001e7af7e9b07c8b44ed71b734e7f62449b3d9f463a7ae96d05e03`。它统一 bnb dequant-linear 路径、使用 BF16 冻结权重的分块 FP32 概率 head，并禁用 TF32/BF16 reduced-precision reduction；相同三条真实源、行和 mask 上，比较本轮 actor 与本轮 FSDP，最大差为 0.06146/0.10496/0.85255，MAE 为 0.005006/0.004132/0.002808。actor 重复及独立 reference、FSDP 重复及 grad/no-grad/reference 均逐 token 一致，但跨执行方式仍未全部满足原 0.08 上限，不能宣布概率完全对齐。实际 whole-action log importance 为 −0.04029/+0.03805/+1.46243，对应未截断权重 0.96051/1.03878/4.31642；这些校准动作没有奖励或更新用途。actor/FSDP 耗时 2751.53/134.21 秒，本次配额 1.76h，LoRA 和四 rank shard hash 均未改变。长 1062-token 动作每次匹配路径评分约 743 秒，后续需据此估计实际采样成本。完整数组和配对方法结果见 `calibration-16k-evidence-v2/result-summary.json`；原 native old 数组原样保留为历史诊断，不被新执行概率覆盖。作业结束已停所有 worker、清理临时输入与 NF4 缓存。

依据上述独立无奖励校准，主 16K YAML 明确启用 `rollout_correction: sequence_tis`、`rollout_is_max: 2`：保留真实 actor behavior 概率，以冻结的实际 learner no-grad 概率作为 PPO 起点，完整 focal action 的重要性权重只做预设上截断；同 learner 执行路径的 0.08 检验继续保留。它承认跨执行方式差异及截断带来的偏差，不把概率中心化后冒作零漂移。决策和校准来源绑定在 `../16k/probability-decision.json`。

主闭环作业于 2026-10-26 04:19:12 UTC 提交为 `qingyu1232/vfarl-l4x4-focal-recovery-16k/1`，source SHA `a69c8a16ec92ce135de726a193b6600d81d93165c54e9a99579b8aa56d91f22b`。`train-16k-v1/` 冻结明确的 16K YAML、原八题、五个反馈后状态上限和所有材料/CPU 来源；提交起点配额 13.39h/16.61h。回读已确认私有、GPU、无 Internet、源码一致；实际阶段和硬件以 `train-16k-evidence-v1/` 为准，不将已提交视为已产生分叉或完成更新。

## 历史待办：此前两个独立臂

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
