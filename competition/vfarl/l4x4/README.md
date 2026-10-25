# VFARL：Kaggle L4×4 训练验证

2026-10-26。分支 `codex/vfarl-l4x4-training-validation`，从 `ea9fa421` 创建。目标是证明真实 Agent 采样、状态恢复、完整验收、分叉优势与 LoRA 更新能闭环运行，并观察有效奖励学习信号；本轮不以高分专家或比赛增益为完成标准。

用户授权使用 `qingyu1232` 的 GPU 配额推进 100 题，随后明确取消先前的 3 小时限制，并明确授权将约 123 MB 的这批训练材料上传至该账号的 Kaggle 私有数据集。仍受平台实际可用配额与单次作业上限约束，不切换账号或启动付费机器。启动前只读配额为 30.00h；实际扣减以之后平台读数为准。

## 当前研究事实

- S600 已完成 600 题（400 共同、200 专科）的真实六工具冷启动示范，4892 条动作、621344 个助手监督 token。它们是审阅后执行的 off-policy 示范，导出条件为 `thinking=0`，不是学生当前策略样本。
- VFARL/S600 在本分支开始时的模型 rollout、同状态验证动作对和 RL 更新均为 0。这个统计不代表旧研究从未做过 rollout 或 LoRA 更新：P015 有真实采样与 action RFT，P017 有 CT 权重对照，不能把这些不同实验改名为 VFARL 数据。
- 现有 QLoRA 使用官方 31B QAT-unquantized v2 同源派生的 NF4 缓存；CT/W4A16 是部署格式。本轮采样和训练统一使用同一 NF4＋LoRA 策略、官方 CT v2 tokenizer/template。不能把 CT 采样概率与 NF4 训练概率视为天然相同。
- 旧 L4×4 FSDP 证据达到 6846 tokens，但属于 r8；本轮 r32 七投影的 4K/8K 容量仍需实际测量。

## 固定输入与运行条件

[v2/selection.json](v2/selection.json) 是当前 100 个 S600 train 任务：67 共同、33 专科，33 仓、100 个 repair family；专科 S1–S5 分别 7/7/7/6/6。首版 [selection.json](selection.json) 保留原 34 仓的选择及真实失败证据。首次 Linux 验收证实六道 dotenv 题需要当前 chroot 不具备的 PTY，因此在任何模型采样前，以同课程/同机制的两道 PyPika、四道 funcy 替换，其余 94 题不变；新题仍须单独通过真实 Linux 验收。[替换来源与原因](v2/substitutions.json)。只读 train 白名单，没有访问 heldout 内容或将预留评测题用于训练。

[v2/questions.jsonl](v2/questions.jsonl) 仅保存题目身份、原官方初始对白与哈希。源码来源、参考修复与验收要求属于裁判资产，不传给策略。隔离 shell 隐藏裁判材料和其他运行输出；当前任务源码及真实工具响应才是策略观察。

全部选中题的旧示范最长不超过 4096 tokens（最大 4079）；只有 37 题不超过 3072。这个筛选不能保证启用 1024 thinking 的新轨迹仍能装入 4K。采样按实际完整 token 前缀计算可用空间，不截历史后沿用原 logprob。4K 是联合上下文上限，thinking 计入输出；8K 应使用独立配置和结果。

配置在 [training/configs/vfarl_l4x4.yaml](../../../training/configs/vfarl_l4x4.yaml)：4-bit NF4、BF16 计算、LoRA r32/alpha64、文本七投影、dropout0，初始 lr=5e-6。采样 T=1、top_p=1、top_k=0，使行为与训练分布定义一致。实际生成的思考保留；达到 1024 thinking 后强制关闭 channel 的控制 token 不参与策略 loss。旧 thinking0 示范保持冻结，不伪造其思考。

## 闭环与证据

1. CPU 准备各题 Linux 依赖，使用原官方 `verify_task` 对原始版本与已验收修复执行完整验收；环境失败记 unknown，修环境后再采样。
2. 4 个单卡 NF4 worker 对 100 题各采样 1 次，可再安排第 2 次。每次保存 exact prompt/completion IDs、采样分布、行为/参考逐 token logprob、真实六工具响应、预算及完整验收结果。
3. 总共最多选择 5 个真实失败 episode 的 pre-action 状态，每状态 3 条分支：原动作重放＋2 条独立当前策略动作。之后使用相同 continuation 策略和匹配的 seed、剩余逻辑预算继续到完整验收。不同分支不是不同独立题。
4. [state.py](state.py) 同路径恢复完整任务 root、git index、ignored/untracked 文件、临时目录、venv、可见 token 前缀与预算。Kaggle namespace 不可用，已增加 [chroot 后端](jail.py)：只读运行时单独绑定内容指纹，独立低权限 UID、capabilities 清零、seccomp 与命令后的 UID 进程清理；实际探针与任务调用均成立的声明范围才标 `complete_declared_scope`。隔离不可用时不加载模型或生成假同状态数据。宿主时钟、熵和 inode 等不可恢复部分保留说明。
5. [training/train_vfarl.py](../../../training/train_vfarl.py) 只对完整验证、同状态的独立随机分支计算 `A=R−mean(R)`。原动作强制重放和 continuation 不进入 focal loss；首版只用状态组终局优势，避免把同一终局标签在 task 与 branch 分量中重复计权。
6. 对有异奖的状态执行一次真实 FSDP PPO clipped objective＋reference sampled k3 KL 更新，保存梯度、实际 LoRA 参数变化、前后动作 logprob 和独立 adapter。若没有异奖状态，则优势为零，记录没有奖励更新，不用 KL 或合成标签冒充有效学习。

逐 token 行为概率来自实际生成分布；reference 另做真实前向。首轮 reference 为冻结 NF4 基座，与零初始化 LoRA 的初始策略等价。后续多轮迭代需要明确加载新的行为策略及固定 reference，不能把首轮专用训练入口直接重复当成任意 continuation trainer。

## 手动验证与执行状态

已完成 100 题来源/唯一性/可重复导出核对、原生工具参数解析、真实官方任务 root 的快照恢复，以及一次性 CPU 小模型的 logprob/梯度核对。小模型分块 logprob 对完整词表误差为 0；正优势动作概率上升、负优势动作概率下降。这只支持数学与反向实现，不代表 31B NF4/FSDP 已运行。

纯 CPU 准备作业及私有资产上传已开始；实际 CPU/GPU 结果将在本目录的执行说明中更新。没有新增专用测试文件，也不以检查数量衡量研究进展。当前尚未声称 100 题模型采样完成、同状态有效奖励产生或 GPU 参数更新完成。

代码入口：

```bash
python3 competition/vfarl/l4x4/prepare.py --seed 20261026
python3 competition/vfarl/l4x4/replace_unavailable.py
python -m competition.vfarl.l4x4.run --prevalidate --selection SELECTION --questions QUESTIONS --assets ASSETS --out CPU_RESULTS
python -m competition.vfarl.l4x4.run --selection SELECTION --questions QUESTIONS --assets ASSETS --config CONFIG --prevalidation CPU_RESULTS/prevalidation.jsonl --out ROLLOUTS
torchrun --standalone --nproc-per-node=4 -m training.train_vfarl --config CONFIG --data ROLLOUTS/focal.jsonl --out TRAINING_RUN
```

实际 Kaggle 载体负责固定模型/cache/依赖、四 worker 分片与总状态配额，再释放采样进程并启动四卡更新，详见 [kaggle](kaggle/)。命令里的路径占位符必须替换为本次已核对的资产，不是可直接执行的默认路径。
