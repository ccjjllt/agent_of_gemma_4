# 双 A800 训练与运行

研究方案、已有结果和下一步见[研究说明](../research/rl-adapter-feasibility.md)。当前使用 Unsloth BF16基座＋r128/alpha256 LoRA＋TRL GRPO；每题8条完整轨迹，同批最多更新2次。已实跑的[首题组配置](../results/grpo-20261102/004-config.toml)将每批有效题组数和批次数设为1；[默认配置](../configs/a800_2x80.toml)为每批4题、最多32批。

新实验从当前默认配置调整题组数；004配置保留作历史证据，其中旧的2 GiB缓存上限不再推荐沿用。前缀缓存每条轨迹保留一个prompt、相同prompt共享，每卡最多使用15%显存，并动态扣除活动/重组KV、复制峰值、梯度/Adam预留及2 GiB工作余量；可用`prefix_cache_bytes`再设总上限，0表示关闭。轨迹结束释放其快照，进入训练前清空生成缓存。

`sliding_cache_enabled=true`使用镜像环形KV，只双写新增token，额外镜像存储计入显存预算。成员退出时连续行与共同左padding使用view裁剪，非连续行只复制survivors；真实位置与完整上下文保持。日志中的`sliding_*`和`batch_row_*`记录分配/复制量，开启现有profiler时可单独查看`cache_membership`区域；这些计数不等于GPU耗时收益。

训练侧另采用`training_prefix_reuse="tree"`，见[精确token树](prefix_tree.py)和[训练backend](tree_backend.py)。先从真实`prompt_ids + completion_ids[:-1]`建trie，只有完全相同的token路径才合并；RoPE位置取原路径深度，full attention只见自身及祖先，sliding再限制深度差小于原窗口。每个原始completion的预测行、target、advantage和概率校正都保留，包含同一hidden预测多个分支首token的情况。组内各调用loss相加后只做一次backward，前缀梯度由所有原始损失累加；不读取生成KV，也不detach前缀。

默认最多4个调用组成一张有界共享图，packed节点最多32768；两份最终additive mask按至少float32计入每设备512 MiB预算，并要求packed dense attention面积不大于独立调用面积之和、至少省256个输入位置。超预算或共享不足时退回完整单调用，不截断上下文；单调用不构造树mask。mask预算不是模型总峰值预算，blocked attention面积及位置减少也不等于真实kernel加速。分卡mask在设备hook前按原mask/设备复用，避免每层另存一份大mask。

树模式使用安装版Transformers5.17的私有native Gemma4/SDPA路径，作用域内绑定完整forward和非reentrant层checkpoint，保留原LoRA投影模块与Accelerate设备hook；直到该组backward结束才恢复原绑定。reference和更新后探针使用相同native backend的完整单调用，仍可使用原reference数值缓存。每组先完成reference，再建立learner共享图；old概率由首次真实梯度前向冻结。组计划只存原调用索引和标量，零步恢复按同组起始RNG重放，不改变原轨迹顺序或重校准old/ref。共享图不进入checkpoint、不跨optimizer；KL早停清空本次所有已累计梯度。

`training_prefix_plan`、step指标和`training_summary.updates.training_prefix_reuse`分别记录精确共享位置、独立/实际输入位置、逻辑调用数、模型forward组数与backward组数，均不包含checkpoint重算，也不冒称GPU提速。`training_prefix_max_calls=1`保留native backend但禁用共享，用于隔离共享效果；`training_prefix_reuse="none"`保留原Unsloth独立评分路径，用于另行观察backend变化。

2026-11-03 CPU验证使用真实tiny Gemma4（混合full/sliding、滑窗跨界、非零LoRA、checkpoint开关）：合成样例33个独立输入位置合为14个树节点，全部12个loss位置保留，logp/loss一致；全参数梯度最大相对L2误差约2.45e-7，LoRA梯度最大绝对误差约6.56e-7。训练器核心的分组反向、零步恢复及KL早停已另行手动核对。上述不是31B双卡或Unsloth GPU兼容/耗时证据；本地现有日志只有长度，无法据此推断真实训练前缀重合率，需以后从服务器实际调用计划观察。[Tree Training](https://arxiv.org/abs/2511.00413v5)与[PrefixFlow](https://arxiv.org/abs/2606.01143v4)是机制参考，本实现没有照搬论文加速幅度或其零冗余分区/训练KV调度。

## 当前路径

| 内容 | 服务器路径 |
| --- | --- |
| 项目代码 | `/root/gemma4/project` |
| 学习器 Python | `/root/gemma4/envs/learner/bin/python` |
| 官方公开数据 | `/root/gemma4/official/competition` |
| QAT未量化31B模型 | `/root/autodl-fs/models/gemma4-31b-qat` |
| 裁判与工具环境 | `/opt/gemma-runtime/bin/python`、`/opt/gemma-sandbox-baseline/bin/python` |
| 运行输出 | `/root/autodl-tmp/gemma4/runs` |
| 沙箱与临时目录 | `/root/autodl-tmp/gemma4/sandboxes`、`/root/autodl-tmp/gemma4/tmp` |

训练环境、代码和官方数据位于系统盘；辅助运行库实际在数据盘，模型在共享盘。subprocess后端需要root，由不同低权限UID运行agent工具与裁判；agent接收官方题面，参考补丁和验收资产只由裁判侧读取。数据和模型手动放到服务器，训练直接读取原目录，不在本地复制训练数据，不读heldout。

## 启动与续跑

将所选配置放到服务器项目的`configs/first-group.toml`后执行；每次新运行使用新的输出目录：

```bash
cd /root/gemma4/project
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TMPDIR=/root/autodl-tmp/gemma4/tmp \
/root/gemma4/envs/learner/bin/python -m training.run \
  --config configs/first-group.toml \
  --data-root /root/gemma4/official/competition \
  --model-path /root/autodl-fs/models/gemma4-31b-qat \
  --run-dir /root/autodl-tmp/gemma4/runs/grpo-next
```

模型调用窗口32K，4个生成槽与8个CPU worker。配置中的7200秒是单次run墙钟上限，包含模型加载；不保证完成全部批次。启动记录已有来源声明与廉价文件元数据，不扫描完整权重SHA256。

仅在同一run保留完整`recovery/`状态时续跑：

```bash
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TMPDIR=/root/autodl-tmp/gemma4/tmp \
/root/gemma4/envs/learner/bin/python -m training.run \
  --resume --run-dir /root/autodl-tmp/gemma4/runs/grpo-next
```

续跑恢复轨迹、采样器、LoRA、Adam、scheduler、RNG与更新进度，不重置该run预算；追加额度用`--additional-walltime-seconds N`显式记录。只有adapter不能完整续跑，更改LoRA rank须新建run。默认训练会话时钟扣除已确认的纯GPU排队时间，并保留原墙钟；这与官方墙钟评测口径不同。

学习率采用[positive_warmup_linear_v1](lr_schedule.py)。设`k`为已完成且未skipped的optimizer调用数，`T`为固定调度总步数，`W`为固定warmup步数；下一次更新前倍率为：`k>=T`时0，`k<W`时`(k+1)/W`，其余为`(T-k)/max(1,T-W)`。warmup从正值起步，decay保持原口径；`0<W<T`时峰值连续使用两步，`W>=T`也不偷偷截短。默认`T=64,W=7`首步LR为`5e-6/7`；单步且`T=W=1`时首步仍为base LR。只支持此显式linear版本，未单独关闭某一对照臂的warmup，也不插入dummy optimizer。

每步记录实际使用的LR和scheduler前后索引，skipped不推进scheduler。恢复仍先构建scheduler、再加载Adam与scheduler状态，不能额外step或重置LR。运行状态v2将原始optimizer调用数与采样policy revision分离：只有未skipped且非零LR的调用推进后者。零LR/skipped批次不会借版本变化解除同策略筛题抑制；这些计数不证明参数改变，另保留初始adapter hash及每次导出相对初始的变化。旧调度/旧运行状态不能混入新run续跑。

### 工具状态摘要修复

2026-11-02用户决定暂不使用此修复。训练与评测继续使用`event_summarizer = "official"`，本地`tool_aware`模式和宿主补丁均不启用；代码与补丁仅保留备用，以下为接入参考。

ADK 1.36.1 默认摘要输入只取文字，遗漏结构化工具调用和返回。[ToolStateSummarizer](tool_state_summary.py)保留按事件顺序排列的文字、调用名称/ID/参数及完整返回值，包括错误和退出码；提示明确区分请求、已观察结果和未知状态。它复用ADK的事件选取、未完成调用保护及压缩事件边界，不读裁判资产，不修改原始事件，也不保证模型摘要无损。完整工具输出仍受32K上下文限制，不通过静默截断掩盖超限。

若后续决定采用修复，可在新建本地实验的配置中将`[runtime] event_summarizer`设为`"tool_aware"`；采集与BF16配对评测均传入此选择。直接运行runtime可用`--event-summarizer tool_aware`。结果的`provenance.context_management`记录模式和修复源码指纹，配对评测核对两臂一致性；不要在既有run续跑时切换模式。

默认仍为`"official"`，原因是当前比赛wheelhouse31的`adk-submission 0.2.13`只接收声明式配置，宿主未开放自定义summarizer，也没有向编译器传入callback registry；把Python文件或callback名字放入提交不能安装该修复。**本地修复模式不能称比赛侧已修复。**主办方若接入，可在创建App时设置`EventsCompactionConfig(..., summarizer=ToolStateSummarizer(llm=agent.canonical_model))`，保留原压缩参数；仅本地升级依赖不会改变官方宿主。

另提供可供宿主维护者直接应用的[ADK 1.36.1补丁](../patches/adk-1.36.1-tool-events.patch)：在对应ADK源码根目录执行`git apply /path/to/adk-1.36.1-tool-events.patch`，再由宿主维护者构建部署。补丁基于原`src/google/adk/apps/llm_event_summarizer.py`（SHA256 `51ca4af8bda8b7d2a547ca1c85008a43eb63b8213fdee68f7a250b03413cd9e4`），直接修默认类的输入格式和通用工具状态提示，无须导入本项目。自定义`prompt_template`接口保持；输入格式改为JSONL，依赖原纯文字布局的自定义提示需适配。补丁已在临时原版源码上实际应用，并手动验证默认类保留工具记录、自定义提示仍优先、formatter与本地修复一致；它不是参赛提交资产，也尚未部署到官方宿主。

比赛侧可通过声明式instruction要求模型将关键工具结果写入后续文字，或在沙箱内保存并重读工作笔记，但这只是有额外成本且依赖模型遵循的缓解，尚无真实收益证据。本次未变更训练默认行为、题面、参考补丁、验收资产或比赛提交。

2026-11-02在临时CPU环境用真实ADK1.36.1人工复现了原版遗漏，并验证修复输入保留复合part、重复ID、空返回、错误退出码和长输出；连续两次token-threshold压缩保留旧摘要种子和未返回调用边界。官方wheel原函数的私有App接入也已核对，原共享配置未改变。摘要模型使用确定性捕获替身；这些证据仅证明序列化与接入机制，不证明Gemma摘要质量、真实长任务修复率或完整服务器运行效果。未新增测试文件、未启动GPU。

## 结果与adapter

- `metrics.jsonl`记录真实采集和更新；`reward_groups.jsonl`、`reward_summary.json`记录完整候选题组、奖励与选组结果。
- `training_summary.json`从已提交checkpoint汇总有效组比例、采集/更新耗时、实际使用LR、原始/非零LR/零LR/skipped调用数、KL及adapter身份；恢复不重复计数。原始调用成本与每非零LR步成本分别计算，旧计数缺失保持未知。采样成功率明确属于更新前。
- `generation_metrics.jsonl`与调用数值摘要用于判断吞吐、排队、prefill和KV成本；不把速度或训练集成绩称为独立泛化收益。
- `recovery/`保留训练精度的完整续跑状态；`adapters/batch-*/`保存更新后的独立BF16 adapter。

提交仅取所选adapter的`adapter_config.json`与`adapter_model.safetensors`，按官方YAML引用。初始0-step adapter不是训练成果；最终必须在官方CT基座上验证挂载与修复收益。

原始调用、轨迹、补丁与JUnit只留服务器。数值日志可用[sync_experiment_logs.py](../scripts/sync_experiment_logs.py)同步；本地精简保留的性能证据和实际配置见[results/grpo-20261102](../results/grpo-20261102/)。主线优先完成完整题组和GRPO更新，吞吐重放不作为训练前置。

## 收益与单变量比较

奖励缺失处理采用`fixed_composite_intervals_v2`：所有配置分项始终保留在完整奖励区间中，回归未知不会按0处理，也不会退回共同可见分项。只使用对区间内所有补全都成立的组中心优势符号；无法确定方向的轨迹信号为0，整组无信号则不更新。共同分项信号仅用于`would_have_fallback`反事实诊断。它是保守的部分识别信号，不是缺失奖励的无偏重建，也不保证总体梯度或收益。

`scale_rewards=group`的分母为`max(std_upper_bound, advantage_std_floor)`，其中`std_upper_bound = sqrt(sum(i<j) max(|L_i-U_j|, |U_i-L_j|)^2) / n`。完整证据时等于原population std；区间扩大时分母不减、可证明信号绝对值不增，因此不会因缺失更严重反而放大优势。`none`仅省去该归一，均保留相同完整奖励、未归一学习信号、筛组与采题反馈。固定目标签名不随缺失模式变化；归一模式、floor和筛组阈值另有独立签名。

`reward_summary.json.reward_evidence`与已提交批次的`training_summary.json.rewards`按精确组件缺失状态组合、任务、目标及归一方式保留候选/选中/确认进入optimizer的记录，包括repair/regression覆盖、选题概率、归一系数和token×|advantage|暴露。缺失、不适用和禁用分开，历史未记录字段保持未知。暴露统计不含PPO比率、裁剪及参数Jacobian，不是梯度大小；optimizer调用也可能是零LR或被跳过。自适应采题与筛组后的分布仍不是均匀题目目标，不能简单乘单次`1/q`消除该偏差。

2026-11-03 CPU反例：A修复分项为0.3、回归代价未知于[-0.5,0]，B完整奖励为0；旧共同分项fallback会给A正信号，但若A实际回归为1，其完整奖励为-0.2。修复后信号为0，补全回归证据后A转为负优势。同批归一比较也已手动核对：完整奖励[0.3,0]时group优势为[1,-1]、none为[0.15,-0.15]；仅将第二条回归证据扩大为未知后，原始信号仍为[0.15,-0.15]，group优势降为[0.375,-0.375]，none不变。上述都是机制证据，不是模型收益。

概率校正由[probability_correction.py](probability_correction.py)统一实现，TRL继续管理optimizer、调度与回调，本地明确计算修正后的Dr.GRPO目标。记`q=behavior, o=old, p=current, r=reference`，`w=min(o/q, cap)`，`rho=p/o`，`K3(d)=exp(d)-1-d`：策略项为`-w*min(rho*A, clip(rho)*A)`；reference项为`w*rho*K3(log(r/p))`，其中rho必须保留梯度；复用漂移为`w*K3(log(p/o))`。全部只对生成token计损失，完整前缀照常反向，old/reference/behavior固定，不按权重和自归一。Dr.GRPO按`轨迹数×固定长度`归一，KL诊断按所测生成token数归一。

无截断且满足支持集条件时，两项分别对应已采集前缀上的`KL(p||r)`与`KL(o||p)`。实际cap=2保留截断偏差，token校正也不改变behavior前缀及终局轨迹分布，所以这些值不能称完整当前策略/全轨迹KL；`target_kl`是有偏诊断的复用阈值，不是严格信赖域。`behavior_correction=false`同时关闭三处old/behavior权重并如实记录；不再支持单独关闭reference的current/old校正。

复用时利用原梯度前向累计非负加权K3；若`已累计总和/整批生成token数`已超阈值，立即清空该次全部梯度并停止剩余调用，不再做该调用反向。`last_reuse_attempt`记录部分覆盖及`old_policy_kl_lower_bound`，不冒充整批均值或新增optimizer步。记录同时保留权重均值、截断比例和ESS；checkpoint携带校正语义版本，`training_summary.kl.by_correction_version`将旧未标记记录与新值隔离。

更新结束用`post_update_kl_max_calls=4`在均匀选取的固定完整调用上补测同一公式的before/after/delta；不截断上下文、不重新生成。它只覆盖所选调用，仍带截断及固定behavior前缀限制。设为0可关闭；诊断失败保留未知，不阻止已完成更新的adapter导出。

2026-11-03的一次性CPU枚举中，三动作分布的精确条件reference KL为0.235613188862，修正值及logit梯度与精确式一致；旧缺失old/behavior项为0.271715699258。cap=2时诊断为0.196699237654、权重质量为0.9，直接显示截断偏差。此为估计器机制证据，不是GPU或任务修复收益。仅生成token向量使用float64，指数乘积在log域组合，避免中间溢出；不改变模型BF16计算或增加训练前向。

本地只准备一个对照配置，不启动GPU：

```bash
python3.11 -m training.ablation prepare --baseline configs/a800_2x80.toml \
  --output-dir tmp/reuse-one --factor reuse --value 1
```

同一目录保存`parent.toml`、`candidate.toml`与语义差异。支持`rank`、`reward.<单项weight>`、`sampling=uniform`、`reuse`、`scale_rewards=group/none`。rank保持alpha/r；对照显式固定等价的`grpo.scheduler_total_steps`，避免同时改变warmup及LR曲线。分别从同一初始基座新建run；不能在原run续跑时修改因素。

短程复用已准备[两次基线](../configs/experiments/reuse-one-batch/parent.toml)与[一次候选](../configs/experiments/reuse-one-batch/candidate.toml)：共同为单批单题组、`T=2,W=1,base_lr=5e-6`，唯一差异是`num_iterations=2/1`。计划LR分别为`[5e-6,5e-6]`与`[5e-6]`，第一步完全一致；两臂不是等更新次数或等算力实验。2026-11-03真实CPU合成LoRA演练中，旧调度reuse=1权重不变、reuse=2第二步才变；新调度两臂首步都改变权重且首步结果相同。首步前后恢复、skipped和显式零LR计数亦已手动核对。这些是调度机制证据，非Gemma任务收益。

`ablation compare`对复用收益另核对被评测adapter的来源：两臂实际LR合同一致、从初始到该导出批次的有效计数完整、非零LR步数为正、导出权重确实不同于初始。后续批次的更新不能替早期零更新adapter补资格；最后一批没变也不抹掉此前已完成训练。无有效更新时标为无效，缺证据时保留未知，原始评测数值和成本仍保留；不因零LR对照产生表面差异便宣称复用收益。尚未运行该GPU对照。

已准备[group基线](../configs/experiments/reward-scale-none/parent.toml)与[none候选](../configs/experiments/reward-scale-none/candidate.toml)，两者仅`grpo.scale_rewards`不同，动态采题及筛组阈值保持不变。重新准备的命令为`python -m training.ablation prepare --baseline configs/a800_2x80.toml --output-dir <新目录> --factor scale_rewards --value none`。后续比较需同时观察缺失模式、每题暴露、有效组/小时、非零LR更新及相同任务/seed的修复与回归；模型更新后轨迹和自适应采题可以间接变化，不能把两次完整训练称固定同批对照。当前没有该对照的训练后adapter或配对验收结果，本次未启动GPU；默认保留group，不据CPU信号大小选胜者。[Dr.GRPO对组内标准差隐含题目加权的分析](https://arxiv.org/html/2503.20783v2)不替代本项目的收益证据。

拿到adapter后，在服务器显式执行公开任务配对评测（`TASK_ID`替换为选定的公开题ID，可重复`--task-id`）：

```bash
python -m training.evaluation --config configs/first-group.toml \
  --data-root /root/gemma4/official/competition \
  --model-path /root/autodl-fs/models/gemma4-31b-qat \
  --adapter /root/autodl-tmp/gemma4/runs/grpo-next/adapters/batch-0001 \
  --training-run-config /root/autodl-tmp/gemma4/runs/grpo-next/run_config.json \
  --task-id TASK_ID --seed 3407 --seed 3408 --walltime-seconds 7200 \
  --output-dir /root/autodl-tmp/gemma4/runs/eval-next \
  --scratch-dir /root/autodl-tmp/gemma4/tmp/eval-next
```

此入口顺序比较禁用adapter的基座与所选adapter，每臂重置随机状态与缓存，使用相同墙钟预算、官方题面和完整验收。`paired-summary.json`给出配对修复/退化、repair/regression、证据覆盖率与尝试成本；它属于BF16后端，不能代替官方CT结果。默认按训练题成绩解释；不读取隐藏评测数据。仅有训练摘要时收益保持未知。

`python3.11 -m training.ablation compare --help`列出摘要与配对评测输入；比较时同时读取实际run配置与adapter身份，只在同一因素、同一评测题目/seed/流程下计算差异。已有数值记录可随上述同步脚本取回本地。
