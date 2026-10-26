# VFARL：Kaggle L4×4 训练验证

2026-10-26。分支 `codex/vfarl-l4x4-training-validation`，从 `ea9fa421` 创建。目标是证明真实 Agent 采样、状态恢复、完整验收、分叉优势与 LoRA 更新能闭环运行，并观察有效奖励学习信号；本轮不以高分专家或比赛增益为完成标准。

当前：第四轮已完成 **99 道真实任务的完整验收：6 成功、93 失败、0 unknown**。两个采样器各跨两张 L4，已完成本轮 4K/r32/BF16 真实采样，越过此前单卡 OOM；这尚未验证四卡反向或 8K 容量。预定 5 个同状态分叉组均为 `[0,0,0]`，没有异奖优势，实际 **optimizer step 为 0**，VFARL 奖励学习闭环尚未完成。CPU 环境累计 **100/100 可用**，其中 DVC 尚未进入 GPU 采样。第四轮墙钟 **16669.047 秒**，平台配额本轮消耗 **9.26h**、累计已用 **10.78h**、剩余 **19.22h**。完整结论与原始证据见 [第四轮结果](results-v4.md)；历史过程见 [执行记录](execution.md)。

8K 对照、DVC 补题和无 optimizer 的概率校准均尚未运行。下文与执行记录保留的“正在运行／待结果”等表述是各阶段当时的事实，最终状态以上述结果为准。

用户授权使用 `qingyu1232` 的 GPU 配额推进 100 题，随后明确取消先前的 3 小时限制，并明确授权将约 123 MB 的这批训练材料上传至该账号的 Kaggle 私有数据集。仍受平台实际可用配额与单次作业上限约束，不切换账号或启动付费机器。启动前只读配额为 30.00h；实际扣减以之后平台读数为准。

## 当前研究事实

- S600 已完成 600 题（400 共同、200 专科）的真实六工具冷启动示范，4892 条动作、621344 个助手监督 token。它们是审阅后执行的 off-policy 示范，导出条件为 `thinking=0`，不是学生当前策略样本。
- VFARL/S600 在本分支开始时的模型 rollout、同状态验证动作对和 RL 更新均为 0。这个统计不代表旧研究从未做过 rollout 或 LoRA 更新：P015 有真实采样与 action RFT，P017 有 CT 权重对照，不能把这些不同实验改名为 VFARL 数据。
- 现有 QLoRA 使用官方 31B QAT-unquantized v2 同源派生的 NF4 缓存；CT/W4A16 是部署格式。本轮采样和训练统一使用同一 NF4＋LoRA 策略、官方 CT v2 tokenizer/template。不能把 CT 采样概率与 NF4 训练概率视为天然相同。
- 旧 L4×4 FSDP 证据达到 6846 tokens，但属于 r8；本轮 r32 七投影已有双卡 4K 真实采样证据，四卡反向与 8K 容量仍待实测。

## 固定输入与运行条件

[v4/selection.json](v4/selection.json) 是当前 100 个 S600 train 任务：67 共同、33 专科，31 仓、100 个 repair family；专科 S1–S5 始终分别为 7/7/7/6/6。首版 [selection.json](selection.json) 与各版原始 CPU 失败均保留。替换只依据实际环境能力，发生在被替换题的任何模型采样之前，没有使用模型成绩：

- [v2](v2/substitutions.json) 替换六道缺少 PTY 的 dotenv，保留其余 94 题及各机制槽位。
- [v3](v3/substitutions.json) 替换七道依赖网络 socket 的 boltons，保留其余 93 题。共同课唯一的 `G-optional_state_contract` 在 S600 内没有合格同标签替补，明确换成真实 `G-input_precondition`，没有伪改标签；该共同机制覆盖缺口保留披露。
- [v4](v4/substitutions.json) 将一道 PTY 不兼容的 yamllint S2 题替换为真实 pyparsing S2，保留其余 99 题。

只使用 S600 原 train 白名单，没有访问 heldout 内容或将预留评测题用于训练。按 v4 身份合并 R2、七题替补和尾部十题的原始 CPU 验收，加上 tail2 中 marshmallow 的真实通过记录，再合并 DVC CA/cache 第三版的真实完整对照后，当前为 100/100 可用；这是跨批次环境证据，不是单次模型成绩。原验收未放宽。

[v4/questions.jsonl](v4/questions.jsonl) 仅保存题目身份、原官方初始对白与哈希。源码来源、参考修复与验收要求属于裁判资产，不传给策略。实际采样用官方 `build_agent_prompt` 重建本次真实预算并记录新旧 prompt hash，不能沿用历史 60 分钟/100 工具调用而暗中执行另一套预算。隔离 shell 隐藏裁判材料和其他运行输出；当前任务源码及真实工具响应才是策略观察。

全部 v4 选中题的旧示范最长不超过 4096 tokens（最大 4079）；26 题不超过 3072，首版的相应数量是 37。这个筛选不能保证启用 1024 thinking 的新轨迹仍能装入 4K。采样按实际完整 token 前缀计算可用空间，不截历史后沿用原 logprob。4K 是联合上下文上限，thinking 计入输出；8K 应使用独立配置和结果。

配置在 [training/configs/vfarl_l4x4.yaml](../../../training/configs/vfarl_l4x4.yaml)：4-bit NF4、BF16 计算、LoRA r32/alpha64、文本七投影、dropout0，初始 lr=5e-6。采样 T=1、top_p=1、top_k=0，使行为与训练分布定义一致。实际生成的思考保留；达到 1024 thinking 后强制关闭 channel 的控制 token 不参与策略 loss。旧 thinking0 示范保持冻结，不伪造其思考。

## 闭环与证据

1. CPU 准备各题 Linux 依赖，使用原官方 `verify_task` 对原始版本与已验收修复执行完整验收；环境失败记 unknown，修环境后再采样。
2. 两个 NF4 collector 各使用两张 L4，对 100 题各采样 1 次，可再安排第 2 次。每次保存 exact prompt/completion IDs、采样分布、行为/参考逐 token logprob、真实六工具响应、预算及完整验收结果。
3. 两个 collector 按 3/2 分配，总共最多选择 5 个真实失败 episode 的 pre-action 状态，每状态 3 条分支：原动作重放＋2 条独立当前策略动作。之后使用相同 continuation 策略和匹配的 seed、剩余逻辑预算继续到完整验收。不同分支不是不同独立题。
4. [state.py](state.py) 同路径恢复完整任务 root、git index、ignored/untracked 文件、临时目录、venv、可见 token 前缀与预算。Kaggle namespace 不可用，已增加 [chroot 后端](jail.py)：只读运行时单独绑定内容指纹，独立低权限 UID、capabilities 清零、seccomp 与命令后的 UID 进程清理；实际探针与任务调用均成立的声明范围才标 `complete_declared_scope`。隔离不可用时不加载模型或生成假同状态数据。宿主时钟、熵和 inode 等不可恢复部分保留说明。
5. [training/train_vfarl.py](../../../training/train_vfarl.py) 只对完整验证、同状态的独立随机分支计算 `A=R−mean(R)`。原动作强制重放和 continuation 不进入 focal loss；首版只用状态组终局优势，避免把同一终局标签在 task 与 branch 分量中重复计权。
6. 对有异奖的状态执行一次真实 FSDP PPO clipped objective＋reference sampled k3 KL 更新，保存梯度、实际 LoRA 参数变化、前后动作 logprob 和独立 adapter。若没有异奖状态，则优势为零，记录没有奖励更新，不用 KL 或合成标签冒充有效学习。
7. 实际更新和 adapter 已落盘后，再加载该 adapter 做独立的真实 Agent rollout，绑定新的策略身份。记录未变好、回退或上下文耗尽；少量同训练题上的变化只用于验证闭环，不作泛化或比赛提分声明。

逐 token 行为概率来自实际生成分布；reference 另做真实前向。首轮 reference 为冻结 NF4 基座，与零初始化 LoRA 的初始策略等价。后续多轮迭代需要明确加载新的行为策略及固定 reference，不能把首轮专用训练入口直接重复当成任意 continuation trainer。

## 学习信号的解释边界

每状态的三条分支中，原动作强制重放只提供对照，进入 focal loss 的是两条独立新采样。最小奖励组要求同一恢复状态、完整前缀和策略下，两条 sampled 分支真实得到 `[0,1]`；原重放不能补足。若两条 completion tokens 和 mask 完全相同却异奖，只说明后续过程或环境的随机性，不构成动作偏好。最多五个状态是小批产率实验，不能保证产生异奖；应分别报告同奖、异奖和 unknown，而不是把零更新自动解释为算法无效。

4K 下要实际记录完整前缀、thinking、动作和工具观察占用，是否到达生产编辑，以及终止是否由上下文耗尽造成。若多数真实失败只是没有剩余恢复空间，应在同一允许训练池按独立 8K 配置诊断，保留原 4K 结果；不能删历史、压短真实思考或补造奖励来产生更新。

有非零 combined PPO＋KL 梯度和 LoRA 参数差只证明更新发生。每个参与组均报告 `D_s = mean_i[A_i × mean_token(logp_after − logp_before)]`；before/after 必须使用同一 current-policy no-grad 执行路径、形状与配置，reference 不能仅凭零 B 的数学等价自动替代 before。零参数更新的同路径重复评分给出经验数值漂移；grad/no-grad、actor/FSDP、whole/chunk 差异另列为执行偏差，不混作随机噪声。分别报告 learner 路径与实际 actor 缓存路径的 D_s；只有 actor 路径改善并明显超出其同路径漂移，才称推理策略有局部奖励一致似然信号。所有组均保留，不能只挑改善组。更新后训练池内四题重跑用于验证 adapter 加载和行为，不能当作泛化收益。

当前目标是动作内 token 均值的 PPO surrogate，加上只包含异奖状态组的采样目标，并非 100 题总体的无偏策略梯度；两条样本组均值对应的优势 ±0.5 是明确的固定缩放。第一步前 `policy≈old≈reference`，clip 通常不激活，KL 的理论梯度为零。因此计算链完成不等于已证明裁剪或 KL 抑制漂移的效果；sampled k3 也不是全状态/全词表的精确 KL。当前尚未取得这些 31B 实验结果。

## 后续补题与 8K 对照规则

第四轮完成后，DVC 使用已经完整验收的 CA/cache 运行时单独补跑 4K，计作第 100 道真实策略任务；这一臂只补覆盖，不依据奖励选择是否更新。8K 使用第四轮相同的 d1ca launcher 与 tail2/1 运行时，避免公共 CA/cache 修改混入七题配对。初始 zero-B 策略、全局任务 seed、每响应 1536、thinking 1024 和工具预算保持一致；只提高上下文并在 prompt 中如实改预算。8K 是新的 episode 和状态，不能修改旧快照预算冒称同状态续跑。

七题选择只读取第四轮初始 worker 的 original/draw-0 结果：完整 verified、reward 0，且终止确为上下文耗尽，按 v4 冻结全局序号取前七；不看原动作重放、随机分支或更新后成绩。`context_or_output_budget_exhausted` 还须核对末尾历史达到窗口阈值、episode 输出预算仍有余额；1536 单响应截断与环境错误不能补入。若合格不足七，就报告实际数量。

逐题比较原始奖励、验收有效性、终止原因、动作数、完整历史长度、输出/思考 token、时间和各卡显存。超过 4096 的真实模型调用证明容量，完整 verified 的 0→1 才支持这些配对题的预算改善；不声称总体或 heldout 增益。若 4K 已完成合法更新，8K 只做采样/完整验收；若 4K 没有异奖组，8K 固定最多五个新失败状态及每状态原重放＋两随机分支，作为独立求取学习信号的一次更新实验，完整保留同奖组，不与 4K 数据混合训练。

## 手动验证与执行状态

已完成 100 题来源/唯一性/可重复导出核对、原生工具参数解析、真实官方任务 root 的快照恢复，以及一次性 CPU 小模型的 logprob/梯度核对。小模型分块 logprob 对完整词表误差为 0；正优势动作概率上升、负优势动作概率下降。这只支持数学与反向实现，不代表 31B NF4/FSDP 已运行。

首 GPU 的 83 道已验收候选均在采样前失败：Transformers 5.13.1 的 `apply_chat_template` 默认返回 `BatchEncoding`，原代码未显式要求 token list，快照因而拒绝字符串键构成的前缀。已显式使用 `return_dict=False`，并用真实 CT v2 tokenizer/template 手动核对一条首轮真实 prompt 的 1322 个整数 token 和七条既有真实工具响应的序列化。这些核对没有生成新策略 action。

首轮直接读取挂载权重耗时约 24 分 36 秒；之后先顺序复制并核对约 18.3 GB 缓存，实际复制约 127–146 秒，本地权重加载约 4 秒。单卡在长前缀的激活与反量化临时内存处分配失败，因此当前按 256-token 分块预填充，并独立分块计算参考概率，两个模型各跨两卡采样；四卡 FSDP 训练布局保持独立。公共框架或容量错误立即终止并保留真实证据。实际运行与账单见 [execution.md](execution.md)。没有新增专用测试文件，也不以检查数量衡量研究进展。

代码入口：

```bash
python3 competition/vfarl/l4x4/prepare.py --seed 20261026
python3 competition/vfarl/l4x4/replace_unavailable.py
python -m competition.vfarl.l4x4.run --prevalidate --selection SELECTION --questions QUESTIONS --assets ASSETS --out CPU_RESULTS
CUDA_VISIBLE_DEVICES=0,1 python -m competition.vfarl.l4x4.run --policy-gpus 2 --selection SELECTION --questions QUESTIONS --assets ASSETS --config CONFIG --prevalidation CPU_RESULTS/prevalidation.jsonl --out ROLLOUTS
torchrun --standalone --nproc-per-node=4 -m training.train_vfarl --config CONFIG --data ROLLOUTS/focal.jsonl --out TRAINING_RUN
```

实际 Kaggle 载体负责固定模型/cache/依赖、两个双卡 collector 的任务分片与总状态配额，再释放采样进程并启动四卡更新，详见 [kaggle](kaggle/)。命令里的路径占位符必须替换为本次已核对的资产，不是可直接执行的默认路径。
