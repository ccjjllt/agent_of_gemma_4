# 实际执行记录

日期：2026-10-26（北京时间）。本页只记已发生的结果，不把准备或 CPU 数值验证记作模型训练。

用户续办授权：已明确要求睡觉期间自主继续研究、不等待其回答。延续既有 qingyu1232 GPU 配额、私有训练材料和实验 Notebook 上传/运行/更新授权；不切换账号或新增付费资源。

## 最新已核实状态

- **第四轮已完成 99 道真实任务：6 成功、93 失败、0 unknown。** 结论来自最终落盘的完整验收，详见 [第四轮结果与原始证据](results-v4.md)。两个采样器各跨两张 L4，已验证本轮 4K/r32/BF16 采样容量；四卡反向和 8K 容量仍未实测。
- **五个同状态分叉组均为 `[0,0,0]`，实际 optimizer step 为 0。** 状态恢复、真实分支和完整验收已跑通，但没有可用于本次奖励更新的异奖组，尚未完成 VFARL 奖励学习闭环。
- **CPU 环境：v4 清单累计 100/100 可用，DVC 尚未 GPU 采样。** 原始版本完整验收 reward 0、参考修复 reward 1 的跨批次证据保留；CPU 环境可用不计作模型成功，GPU 成功率的分母仍为 99。
- **第四轮墙钟 16669.047 秒；实际配额本轮消耗 9.26h，累计已用 10.78h、剩余 19.22h。** 这是[完成后平台实际读取](kaggle/completed-v4-readback.json)，没有按卡数推算 GPU-hours。
- **8K 对照、DVC 补题、无 optimizer 的概率校准均尚未运行。** 代码准备与已完成实验结果分开记录。

下文保留所有阶段的原始过程记录；其中“正在运行／待结果”等表述是当时事实，最终状态以本节及 [第四轮结果](results-v4.md)为准。

## 已发生

- 已从 `main` 的 `ea9fa421` 创建 `codex/vfarl-l4x4-training-validation`，S600 冻结数据未修改。
- 启动前 `qingyu1232` 的只读配额为 GPU 已用 0.00h、剩余 30.00h。用户随后取消本轮 3 小时限制，允许推进 100 题；仍只使用该账号的实际可用配额。最新余额见上方实际读取。
- 用户明确授权了这 100 题材料上传至 Kaggle 私有数据集，并另行明确授权本分支实验代码上传、运行和更新私有 Notebook。材料上传已成功：[vfarl-l4x4-train100](https://www.kaggle.com/datasets/qingyu1232/vfarl-l4x4-train100)，约 123 MB；原始材料与代码指纹见 [材料构建/上传记录](kaggle/material-push.json)。参考修复只用于裁判和环境复现。
- 首个 [CPU 准备作业](https://www.kaggle.com/code/qingyu1232/vfarl-l4x4-cpu-stage) 已实际完成训练依赖导入和 17 个任务 profile 的安装尝试。16 个成功；profile-02 的失败来自未发布的本机开发包 `pipdeptree==0.1.dev1+...`。该包与任务运行无关，后续排除时保留修正原因。`stock` profile 也需要依据原运行环境补齐项目依赖，不能把 `task_site_packages_profile=null` 误解为只需要 pytest。
- Kaggle 能力作业实际测得：UID/EUID=0；unshare 的 mount/pid/net 均为 EPERM；Landlock 为 ENOSYS；child `chroot` 和 `no_new_privs` 成功，存在 `libseccomp.so.2`。原始记录见 [隔离能力](kaggle/capabilities-evidence/capabilities/isolation-capabilities.json)。因此 bwrap/namespace 路径不可在此环境直接使用；这一早期记录仅说明候选能力，后面的真实隔离与任务执行才提供可用证据。
- 后续 [chroot v2 实测](kaggle/jail-probe-evidence-v2/jail-capabilities.json) 已返回 `effective_mode=chroot`：真实 privilege drop、capabilities 清零、NNP/seccomp、gold 排除、只读运行时与 UID 进程清理通过。探针使用标准库 executor 适配层，956 个运行时文件，源码 SHA `cb2c8601c48ae583c33347ac3593fc6cb4ea687663893368ac5f048c03d3083a`；之后的 UV Python 路径等集成修补需由正式载体重新执行同一运行时探针，不能把旧 SHA 的通过当作当前代码已验收。
- 新私有 [linux-runtime/1](https://www.kaggle.com/code/qingyu1232/vfarl-l4x4-linux-runtime) 完成 17 个基础 profile，归档 778757268 bytes，SHA `468f2abea122dfef31ed7abdfae6d8ebe51aca2451d56e41a2362e18078e845d`。CPU [prevalidate/1](https://www.kaggle.com/code/qingyu1232/vfarl-l4x4-prevalidate) 补齐 stock 11 组后实际运行全部 100 题：36 usable、64 未通过环境验收。完整记录在 `kaggle/prevalidate-evidence-v1/vfarl-prevalidate/prevalidation.jsonl`；这不是模型成绩。
- 已从原始失败日志定位：30 题受共享库路径中的 `..` 重复打包影响；另一批因任务解释器链回落系统 Python 而缺 pytest。已修复路径去重、真实 base Python 创建 venv 和实际 `error_message` 字段读取。其余源 CLI、依赖 copy-mode 和 Moto 的 Python 3.11.8 按原环境证据恢复，不修改题目验收。
- 六道 dotenv 每题的两个 PTY 相关验收在当前 chroot 中无法运行；在首次模型采样之前，已生成 [v2 替换清单](v2/substitutions.json)：保留原 94 题，用 S600 中同课程/机制的六题补足 100。替补选择只依据环境兼容性和原训练白名单；没有模型分数参与。旧失败、旧清单均保留。
- v2 私有材料已上传，124160041 bytes；[CPU R2](https://www.kaggle.com/code/qingyu1232/vfarl-l4x4-prevalidate-r2) 已使用固定 dataset/2 与先前完成的 851821246-byte 运行时（SHA `896eb2fcceb5652492c864bde1baf1e954081e22a0b0c0cf36c8529fda1419a5`）启动。R2 实际包含 copy-mode、原源码 CLI、嵌套解释器修复及 Moto 精确 3.11.8。后续新增的 copy-mode ELF 依赖扫描不在这一已冻结载体内，按实际结果决定受影响题目的续跑。
- R2 全部 100 题实际完成：83 usable、17 环境/隔离不兼容。2026-10-26 北京时间 **03:43** 提交首个 [GPU 作业 train/1](https://www.kaggle.com/code/qingyu1232/vfarl-l4x4-train)，固定 dataset/2 和 R2/1 运行时 SHA `f32566edc1eeec8bf4101f4396d49b9a6ed2c9f03eee5d4717d8defe0936890d`。请求并实际取得 4 张 L4；原计划 4K/r32/1024 thinking、83 道已验收题、最多 5 个状态各原重放＋2 个随机分支、有异奖才更新。提交前剩余 30.00h；源码 SHA `9818d415283cce85604a671bfa071faf9ba84587410848e848f8e41c8706ca28`，jail SHA `18ca7eaddc7bae9b4792c10a91e97cf21b4abb20d17afd0772032be587804723`。[提交回执](kaggle/train-v1-push.json)与后述失败输出均保留，不覆写为修复后的结果。
- [v3 替换](v3/substitutions.json)依据 R2 的真实 socket 错误移出七道 boltons，保留其余 93 题；[七道替补 CPU 验收](kaggle/v3-seven-cpu-evidence-v1/vfarl-prevalidate/prevalidation.jsonl)全部可用。共同课 `G-optional_state_contract` 唯一槽位没有同标签替补，明确改为真实 `G-input_precondition` 并披露覆盖缺口，未伪改来源分类。67/33 和专科 S1–S5 配额保持。
- [v4 替换](v4/substitutions.json)再将一道 PTY 不兼容的 yamllint S2 换为 pyparsing S2，其余 99 题不变。所有被替换题此前都没有模型采样或模型成绩。当前 v4 为 31 仓、100 个 repair family，最长旧 thinking0 上下文 4079 tokens，其中 26 题不超过 3072。
- Python 3.11.8 的 host 解释器可用，但无 `/proc` 的 jail 中三种解释器路径均报 `undefined symbol: Py_BytesMain`。实际 `readelf` 显示其 `DT_NEEDED` 直接包含 `$ORIGIN/../lib/libpython3.11.so.1.0`；host/jail 库 SHA 相同，固定 `LD_LIBRARY_PATH` 仍不能解析这个带斜杠的依赖。[首次六臂记录](kaggle/python3118-loader-evidence-v1/python3118-loader/loader-result.json)保留。随后仅在降权 child 中设置实际 base/bin 的 `LD_ORIGIN_PATH`，base 和 task-venv 两臂均 exit 0、报告准确 3.11.8，并加载同一 canonical libpython；[Origin 实证](kaggle/python3118-origin-evidence-v1/python3118-loader/loader-result.json)。没有改回其它 Python 版本。
- [尾部十题 CPU 作业](kaggle/tail10-cpu-evidence-v1/vfarl-prevalidate/prevalidation.jsonl)实际取得 **8/10 可用**，源码中 Origin jail 为 `d1cacb126907061481d8f2275e746cd08a5fb22944eafff5c641b88dc5af072a`。Moto6755 的 3.11.8、两道 Furl 的 3.9.6 均实际得到 baseline pytest exit 1、审阅修复 exit 0，并通过完整要求，即 reward 0/1；另外五题为 Tenacity425、MonkeyType148、两道 textdistance 与新 pyparsing。未通过的 DVC 源于不需要的 rerun 插件自动开 socket，marshmallow 缺少准确的发行元数据；两者正在修环境，不修改其源任务验收。修后运行时已导出为 967830963 bytes，SHA `f5cb35ab15151e22bbd84476b1cba5539c339d248887d60ea36252b3d14f3f91`，[归档回执](kaggle/tail10-cpu-evidence-v1/vfarl-prevalidate/runtime-archive.json)。

## 首 GPU 的真实失败与修复

[实际硬件](kaggle/train-core-evidence-v1/vfarl-train/hardware.json)确认四张 NVIDIA L4。四 worker 的 `model_ready` 具有相同 policy ID `783956134dfae937a8e356034bad4267c24e016df78c188ae5d782d88b754256` 和相同 reference ID。权重进度条完成 1188/1188，用时约 **24 分 36 秒**；`model_loading` 到 `model_ready` 的日志区间约 25 分 16 秒，包含准备过程。它是一次已完成的慢加载，未发现死锁；仍不能据此宣称 31B r32 的 4K 生成或反向容量已通过。

[原始 worker 记录](kaggle/train-core-evidence-v1/vfarl-train/)共有 100 个任务条目：83 个 `A real complete model-visible token prefix is required`，17 个因先前环境未可用而没有采样。四份[更新日志](kaggle/train-core-evidence-v1/vfarl-train/update.log)均明确 `optimizer_steps=0`、没有有效状态组。平台作业结束不等于训练成功；这轮没有生成一个策略 action，不能把 83 个框架异常记成 83 个模型失败。

根因是 Transformers 5.13.1 的 `apply_chat_template` 默认给出 `BatchEncoding`，原调用没有固定返回类型；快照需要的是完整整数 token 序列。修复显式指定 `return_dict=False`，保留快照的真实前缀要求。一次性手动核对使用实际 CT v2 模板（SHA 前缀 `94899c0f`）：首 GPU 的一条真实 prompt 现在得到 1322 个整数 token，七条既有真实工具响应可继续序列化；没有生成新策略轨迹来替代失败记录。

下一载体将公共协议/前缀错误提升为立即 fatal 并打印 `collection_error`，避免同一框架错误重复走完 83 题。NF4 权重在临时盘空间足够时先按原 manifest 顺序复制一次、逐文件核对大小及 SHA，再由四 worker 读取本地副本；空间不足则明确记录并回退原挂载。加载、采样、验收、更新分别记录阶段和耗时。第二轮已经取得后述 I/O 与动作实测，仍没有学习收益结果。

上述修复已纳入 `train/2` 并于 **04:23:19** 实际提交，源码 SHA `67ce09ef846a8f50068c73b48c72d5a393bdd72373a57646307fbc0738e128b3`，固定 dataset/4、tail10/1 运行时 SHA `f5cb35ab15151e22bbd84476b1cba5539c339d248887d60ea36252b3d14f3f91`，仍为 4K/r32/1024 thinking。98 个可用身份由三份真实 CPU 报告按当前清单合并，来源哈希见[载体 manifest](kaggle/train-v2/carrier-manifest.json)。真实动作与容量结果如下。

## 第二 GPU 的真实动作与容量失败

实际 18,295,563,073-byte NF4 缓存顺序复制及所有原 producer SHA 核对共 **146.44 秒**；随后四个 worker 的 HF 权重加载进度条均约 **4 秒**，policy/reference ID 与首轮相同。这个结果支持本地副本改善本次加载，不代表生成速度或修复收益提升。[缓存记录](kaggle/train-evidence-v2/vfarl-train/local-nf4-cache.json)、[原始日志](kaggle/train-evidence-v2/vfarl-train/worker-0.log)。

共落盘 5 条完整 action：dominate 两条、flake8 fmtax 两条、flake8 w0t 一条。实际工具包括搜索文件、读取源码，以及用 run_command 创建并运行 `reproduce_issue.py`（返回真实错误）；不是人工脚本替代策略动作。已完成动作的峰值 allocated 约 21.46–21.93 GB。第二次 read_file 后，dominate 和 flake8 fmtax 的真实下一前缀分别为 **3390/3258 tokens**，4K 尚余 **706/838 tokens**，却在整段 prefill 的 q_norm/MLP 中 OOM：分别申请 212/134 MiB 失败，设备总量 22.03 GiB、剩余仅 17.12/7.12 MiB。因此这是单卡前向容量失败，不是超过 4096 或任务验收失败。

公共错误立即停止其余 worker；[清理记录](kaggle/train-evidence-v2/vfarl-train/input-cleanup.json)确认模型进程停止、无残留任务 UID，并删除本次解包输入和本地权重副本。原动作、工具响应、异常、硬件与状态记录保留于 [train-evidence-v2](kaggle/train-evidence-v2/vfarl-train/)。没有完整 episode 结果，不能据此计算成功率或奖励优势。

新策略实现（SHA `d328fc3ccccbcc24fb12ad2925f2578f24c10dbe2e3d281aa0bb1f782f08f4bf`）将完整前缀按 256-token 连续块送入同一 KV cache；实际独立参考 forward 也分块，只对所需位置每次投影 16 token。全部历史和强制控制 token 仍进入上下文，未截断、未降低 rank、未替换行为概率。一次性真实 tiny Gemma4/PEFT r32 核对覆盖滑动窗口、shared KV、跨块目标及 forced mask：FP32 对完整前向最大误差 `9.54e-7`，BF16 最大差异 `0.004294`。这仅支持计算语义，31B NF4 CUDA 显存仍待第三轮实测。

并行 [tail2 CPU](kaggle/tail2-cpu-evidence-v1/vfarl-prevalidate/prevalidation.jsonl) 中，marshmallow 用冻结源码官方 flit backend 生成准确 `3.23.1` dist-info 后得到 baseline exit 1、repaired exit 0；没有安装或覆盖目标 package 源码。DVC 移除未使用的 rerun 插件后，完整调用栈继续暴露 pygit2 导入时默认 CA 证书不可见；公共只读 CA 候选尚未完成正式验收，不算第 100 题可用。

## 第三 GPU 的分块实测与双卡调整

北京时间 **04:44:03** 已成功提交 [train-chunked/1](https://www.kaggle.com/code/qingyu1232/vfarl-l4x4-train-chunked)（本实验第 3 轮），[回执](kaggle/train-v3-push.json)。使用新 slug 保留此前作业输出，未并发启动另一份 GPU 实验。冻结源 SHA `718755367b1954f8d2db70432ec0acec4dd3cc8d03e832e5021b34515802e6d4`，v4 材料、tail2/1 运行时、99 个真实可用任务；主 jail 仍为已验证的 `d1cacb`，策略为 `d328fc`，collector 为 `d965c3`。设置 `PYTORCH_ALLOC_CONF=expandable_segments:True` 减少变化长度产生的内存碎片，同时记录实际请求长度和原始异常原因；不改变 r32、BF16、4K 或 thinking 1024。

DVC 的第一 CA 候选在提交后审阅发现目录 alias 祖先布局风险，明确不将该候选验收用于放行。正式合并前需修正候选并重新单题验收。第三 GPU 不包含这一候选，不因等待第 100 题而停顿。

第三轮真实完成 **7 条 action**，尚无完整 episode；dominate 第三次请求的 3384-token 前缀在分块 prefill 中仍 OOM，具体位于 bitsandbytes `_dequant_linear_fallback` 申请 **222 MiB** 反量化缓冲，此时设备只余 **217 MiB**。分块减少了激活需求，但没有消除基座、KV 与反量化临时内存的共同压力。因此停止继续单卡重试，改为 **每个当前策略模型跨两张 L4、同时两个 collector**；仍使用同一四卡作业、r32、BF16、完整历史，训练保持四 rank FSDP。该拓扑的实际容量尚待下一轮。

第三轮自动清理确认所有模型进程已停、无残留任务 UID，解包输入及 ephemeral sandboxes 已删除；被终止的 worker 1–3 在删除前保存了含 `.git` 的真实 workspace 压缩包及 SHA，动作和状态证据保留。[第三轮原件](kaggle/train-evidence-v3/)。累计两轮生成 **12 条 action**，不能将它们计为 12 次完整 rollout。

## 第四 GPU 双卡策略实验已启动

北京时间 **05:06:17** 已成功提交 [two-gpu-policy-training/1](https://www.kaggle.com/code/qingyu1232/vfarl-l4x4-two-gpu-policy-training)，以 Kaggle 回执中的实际 slug 为准。[提交回执](kaggle/train-v4-push.json)。载体源码 SHA `06b6a226fd2f4923a1935675294780ac8126de6db5eaf954f5afd0189720be03`，起点配额 28.48h，99 题、4K、1024 thinking、r32/BF16。两 collector 分别使用物理 L4 0–1 和 2–3，最多 3/2 个 focal state；采样结束后仍四 rank FSDP，更新后两 collector 合计重跑前四道训练题。

双卡权重仅在两个可见 CUDA 设备上均衡分配，拒绝 CPU/disk offload；每卡权重装载预算 14 GiB，实际 map 与峰值分别保存。为使采样和训练的 adapter 基础确定一致，VFARL 在基座装载后显式以配置 seed 在 CPU 创建 LoRA A/B，再移动到相应层设备；既有 SFT loader 默认行为不变。真实 tiny Gemma4 的 CPU dispatch 手动对照中，显式 seed 的单卡路径、改变前置随机消耗的重复加载和 dispatch 后创建三者 adapter SHA 一致；这一核对不代表 CUDA 双卡容量已通过。

第四轮已取得首批完整验收：dominate166 与 flake8 fmtax 的 original 均 `verified=true, reward=0`，到达 `context_budget_exhausted`；分支已开始，说明该 worker 中的 reviewed-repair 环境对照与保存状态恢复已通过。两个双卡策略实际仍为原 policy ID `783956...b754256`，在 3.3K–3.8K 前缀上继续生成，单卡峰值约 13.8 GB，越过此前单卡 OOM 点。完整 99 题与三分支结果仍在运行，不能据首批结果计算总体成功率。[实时原件](kaggle/train-evidence-v4/live.log)。日志中 `original_action_replay=true` 的首动作来自已保存轨迹，不计作一次新模型生成。

## 100 题 CPU 环境验收已完成

[DVC CA/cache 第三版原始验收](kaggle/dvc-ca-evidence-v3/vfarl-prevalidate/prevalidation.jsonl)已实际取得 `usable=true`：缺陷版本 **31 failed / 3 passed**、reward 0；参考修复 **34 passed / 0 skipped**、reward 1。官方验收调用分别 40.91/40.32 秒，pytest 部分 15.17/16.94 秒，平台长时间 RUNNING 与日志 500 不等于测试本身持续运行。提交前复读发现已完成，因此未另开恢复副本。

已核对实际载体的 jail SHA 与本地候选均为 `71336c47450c4c8e1baf9359aca3a3108641c09c1dc2e221068e9446d88d98e9`，并合入主运行时。公共 CA 保持只读，`/var/tmp/dvc` 只指向该任务快照内的私有缓存；没有放宽网络或改动目标测试。v4 **100/100 环境可用** 是跨批次实际验收合并结论。正在运行的第四 GPU 保持原冻结 99 题/d1ca 环境，最后 DVC 在后续独立补题/8K 条件中运行，不冒称已被 GPU 采样。

## 已生成动作中的概率数值差异

对第二、三轮落盘真实 action 作一次性只读计算：初始 LoRA B 为零，行为策略和独立参考在数学上对应同一基座，但自回归、整段或分块前向的实际 BF16 概率并非逐 token 相等。没有改写任何行为概率或奖励。

| 实际作业 | 完整新 action | 有效 token | 平均绝对 old/ref 差 | 最大绝对差 | 平均 sampled k3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPU/2 | 5 | 633 | 0.003149 | 0.133274 | 0.0001157 |
| GPU/3 | 7 | 814 | 0.003612 | 0.212926 | 0.0001578 |

两次的绝对差中位数约 `1.07e-6`，P95 约 `0.0141/0.0130`，P99 约 `0.0984/0.1002`；分别 9/16 个 token 超过当前训练器的 0.08 阈值。以 `exp(ref-old)` 仅作为初始 ratio 代理，两批在 `[0.8,1.2]` 外的比例均为 0，但它不是尚未实测的 FSDP current/old ratio。最大差不只发生于极低概率 token。同一真实前缀、相同 completion 的 dominate 首动作，在 GPU/2 与 /3 的执行调度之间，behavior 最大差约 0.19865、reference 最大差约 0.39697。

这些是独立于奖励的真实数值校准证据，表明仅凭权重身份相同不能假定概率逐项相等。当前不直接调高阈值、不以当前 forward 覆写 old logprob；先定位实际计算路径，再决定训练时可解释的处理。学习方向仍必须使用同执行模式的前后概率，并明确量化数值误差，不能把它冒称参数更新的收益。原件分别保留于 train-evidence-v2/v3 的 original/actions.jsonl。

## 第四轮固定五组：已观测零优势

第四轮的 3/2 状态配额已全部完成，共 **5 个真实失败 episode 的保存状态、15 条完整分支**；每组均为 `[0,0,0]`，`restoration_scope=complete_declared_scope`，没有 collection_error。原动作重放各一条不进入 on-policy loss；其余共 **10 条独立采样分支**亦均完整验收失败，因此没有 mixed-outcome group、组内优势为零。

这证明声明范围内的保存/恢复、真实分支和完整验收链路已经产生可追溯数据，但不构成有效奖励学习信号。状态配额已用完，后续 99 题原始采样不会再追加焦点组，本轮训练阶段应返回 0 optimizer step；最终仍须核对原始 focal 文件和训练返回，不能以程序执行结束冒充更新。99 题完整结果继续采集，按预登记规则进入独立 8K 条件。

## 本地必要核对

- 首版 100 题清单的唯一性、67/33 配比、train 来源、源题面哈希和重复导出一致；100 个不同 repair family、34 仓。v2/v3/v4 的替换和实际覆盖变化分别记录在各版清单，当前 v4 为 31 仓，不能沿用首版仓库数。
- 官方 SubprocessManager 的真实 task root 同路径恢复了 984 项文件/目录条目、权限、软硬链接、Git index、临时文件、venv、上下文计数及逻辑预算。改变外部依赖记录或策略身份会拒绝复用。macOS 上明确是 observable scope，不冒充 Kaggle 完整隔离。
- 新 `TaskEnvironment` 以真实 `Knio__dominate.9082227e.pr_166` 运行了官方 read_file 和两次完整验收：原始版本 reward=0、已验收修复 reward=1；两次各保存一份原始 JUnit。该命令使用临时目录，未新增测试文件；它证明实际工具与验收接线，不是模型修复结果。
- 实际 Transformers 5.13.1、PEFT 0.21.2 的 tiny Gemma4，完成缓存续算、selected logprob、eval checkpoint 梯度及 CPU 双 rank FSDP 更新/导出。分块 logprob 与完整词表误差为 0；两 rank 梯度一致，更新后正优势动作概率上升、负优势下降。该证据属于小模型数学/结构核对，未证明 31B NF4/r32 的 L4 容量。
- 新增官方 unsubmitted working-tree fallback 后，用真实任务工作区手动留下修复且不调用 submit_patch，实际捕获 510-byte 补丁，并保留 `explicit_submit=false`；避免上下文截断时把已完成的修改误当作空补丁。这里没有进行模型推理。

## 尚未取得的证据

前两次产生动作的 GPU 作业累计有 12 条真实生成 action；第四轮现已取得首批完整 rollout，但尚未得到完整同状态异奖组或 31B VFARL 更新。继续收集完整任务结果，然后报告候选状态的恢复成功率、每组两条独立新分支的同奖/异奖/unknown，以及 4K 的上下文耗尽比例。五个状态无异奖也可能只是小批奖励产率不足；只能如实记录零更新，再依实际失败决定独立 8K 条件或第二次允许采样，不能补造奖励。

首次非零更新还需保留真实旧/参考 logprob 对齐、有限梯度、参数变化、落盘 adapter 和更新后的真实 rollout。每组均报告优势加权的平均 token logprob 变化 `D_s`，用同一 current-policy no-grad 路径作 before/after，并以零更新同路径重复评分度量漂移；grad/no-grad 等跨路径差异单独列为执行偏差，不能当随机噪声门槛。还需实际 actor 路径的似然改善，才称推理策略的局部奖励学习信号。相同 completion/mask 的异奖不算动作偏好。非零 combined 梯度、参数差或 CPU 小模型概率方向不等于 31B 已学会修复；训练池内四题重跑不作泛化证据，首步 clip/KL 计算也不证明其已发挥约束作用。所有环境修复、原失败和配额消耗继续留存。
