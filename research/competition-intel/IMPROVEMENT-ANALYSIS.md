# Gemma 4 Developer Agent：公开证据盘点与改进路线判定

状态：只读调研，2026-11-15。数据来源为 Kaggle 竞赛 `gemma-4-developer-agent`（id 149921，截止 2026-12-02）的公开讨论、公开 notebook 及其 kernel 元数据、公开 LB 读取。
本文不改动任何提交、不启动 GPU、不创建 notebook；所有数字均来自下列来源，推断处明确标注「推测」。

---

## 0. 结论摘要

1. **公开 LB 现状（本次实测读取）**：0.29 仅 1 队（Yurnero，2026-10-29 提交）；0.24 有 13 队；0.22 十余队。前 30 名里 0.24 是最密集的台阶，0.29 是孤立高点。Top-1 与密集台阶相差 3 题（58 题制，1 题 = 0.017）[LB-READ][743506]。
2. **公开证据中没有任何一次「LoRA 带来可确认 LB 增益」的记录**。11 组有据可查的 31B/12B/E4B adapter 实验里：0 次正向确认，最好一次是 DPO 的 9 题 vs 8 题（CI −2.7~+4.7，等于噪声）[MOBASSIR]；3 次 LB 0.00（其中一次是 alpha/r=2.0 导致复读死循环）[750713][755677]。公开 notebook 的最高分 0.22 是无 adapter 的 31B 单 agent bundle（flexonafft：28 calls / 8 min）[KERNEL-META]。
3. **框架侧已测到的最大杠杆（思考开关、thought summaries、工具预算）基本都在我们现有 bundle 里了**。剩下的可动项是：12 小时总时限安全、上下文压缩（compaction）、提交卫生（不得改测试文件）、早编辑硬约束、工具面裁剪。这几项单项 1~3 题、合起来预期 +1~3 题/58（推测），但可本地配对验证、成本低。
4. **判定**：
   - **框架改进 = 主线**。它是唯一能在剩余 17 天里用本地配对实验闭环验证、且下行风险可控的杠杆。
   - **自训 LoRA = 条件性旁路，不做主线**。只有在「严格门控 + 低成本 + 目标明确到行为层」时才做；31B 主线冲榜的直接收益期望≈0（推测，区间 −4~+2 题）。若要做，**优先 E4B/E2B 小模型奖**（$10k，当前公开最好仅 0.06，赔率远高于主线）[751104][750974]。
5. **对我们最具体的一条警告**：我们 `submission-candidate-0.24` 用的 honghanhhh bundle，其 Kaggle kernel 元数据 `best_public_score = 0.18`，且第三方独立复跑得到 0.17 [KERNEL-META][757588]。**不要按 0.24 预估我们的基线**，应按 0.17~0.18 规划。
6. **对「(a) 还是 (b)」的直接回答：两者都做，但必须严格串行。** 先把框架锁死在可复现的本地基线上（P0~P5），再用剩余预算做**一次**门控 LoRA（P7）；若只剩一次提交机会，选框架。LoRA 不是"另一条更快到达 0.24 的路"，而是"在框架已经封顶之后，用 25~40 GPU-h 买一张 +1~2 题、方差极大的彩票"。

---

## 1. 计量口径（先统一，否则所有结论都会错）

| 事实 | 数值 | 来源 |
|---|---|---|
| 公开 LB 题数 | **58 题**（唯一能同时重现历史显示分数的题数） | [743506] |
| 1 题价值 | **0.0172**（0.29 = 17/58） | [743506][HITARTH] |
| 私有集 | 约 60 题，与公开集在同一 12 小时窗口内**顺序**评测 | [743063] |
| 单次提交的 run-to-run σ | **1.5~3 题**（≈0.026~0.052） | [MOBASSIR][758954][758969] |
| 区分「差 1 题」所需重复次数（80% power） | **每臂 18~37 次** | [MOBASSIR] |
| 同一份文件的实测极差 | 4/6/7 题；7/7/9/11/13 题；8~11 题（10 次均值 10.0/58） | [GLUZDOV] |
| 单次 LB 分数的 95% 区间 | 约 ±0.08~0.11 | [HITARTH] |
| 12 小时上限行为 | 超时**整份提交报错**（官方曾计划改为未完成计 0，10-07 时仍在重新考虑）；只读 `timeout_seconds`/`max_tool_calls`/`max_time_minutes`/`max_turns` | [743063] |
| 公开集内部不可解题 | 6/129 题要求 prompt 与仓库中都不存在的名字；放宽口径 16/129 | [745855] |
| 本地金标可过率（普通 notebook） | 52/129（fastapi 仅 10/67） | [751487] |
| 本地金标可过率（scorer-like 沙箱） | **127/129**（fastapi 66/67） | [751487] |

**推论（重要）**：任何 LB 上小于 ~0.05 的差异，单次提交都不可判定；决赛只看一次私有集抽样，因此**只有本地配对增益 ≥4 题/90 题（≈+0.07）的改动才值得占用最终名额**（推测，基于 σ=1.5~3 题与 80% power 表）。

---

## 2. 证据表

可信度定义：**A** 官方声明或多方独立复现（≥100 题）；**B** 单一作者但配对本地评测、样本 30~130 题、方向与机制一致；**C** 单次运行/样本<30/仅单次 LB 抽样/自报无原始数据；**D** 推测。

### 2.1 采样与思考（对分数影响最大的一类）

| 技术点 | 来源 | 测量效果 | 可信度 |
|---|---|---|---|
| **关闭 thinking** | [746250] E17 | 23/129（17.8%）vs 基准 48/129 → **−25 题/129 ≈ −11 题/58** | A |
| **关闭 thought summaries** | [746250] E18/E19 | 35/129、34/129 vs 51/129 → **−16~−17 题/129 ≈ −7 题/58** | A |
| temperature 0.2→0.4/1.0 | [746250] E11/E12 | 51/129 vs 48 → +3（噪声内） | B |
| top-p 0.8/1.0 | [746250] E13/E14 | 48/129，无变化 | B |
| thinking budget 2048 / 6144 | [746250] E15/E16 | 50 / 48 vs 48 → 0~+2（噪声内） | B |
| **thinking 4096→2048（配合压缩控制）** | [MOBASSIR] §5.1 | 解决数不变、**超时 10→2 次/run、快 8%、空补丁大减** | B |
| **丢弃历史 reasoning（`include_thoughts:false`）** | [MOBASSIR] §6.2 | **超时 3→15、慢 19%**，模型每轮重新推导计划 → 反向 | B |
| temperature 0.2 vs 1.0（另一套 bundle） | [747696] v13 | 17/31 vs 17/31，本地无差 | C |
| temperature 0.6（E4B） | [758969] | 0.03，明显差于 0.05 基线 | C |

### 2.2 时间预算与 12 小时总时限

| 技术点 | 来源 | 测量效果 | 可信度 |
|---|---|---|---|
| 每任务时长扫描（快卡，129 题） | [746250] E20~E23 | 6min 37、10min 46、15min **54**、20min 49、60min 51；6min 有 61 次会话超时 | B |
| Kaggle 4×L4 速度标定 | [746250] | 900.8 s/百万 token、1110 tok/s、8.68 s/次调用；**120 题 × 5.5 min ≈ 11 h** | A（官方环境实测） |
| 4.5 vs 4 min（配对，128 题） | [747857] | 46/128 vs 43/128，McNemar **p=0.58**；短跑省 8.5% token | B |
| 7 / 4.5 / 4 min 全量 | [747857] | 43 / **47** / 43（129 题），超时 21/53/66 | B |
| 命令超时 300→100/150/250 s | [747857] | 47/43/44/40；只有 0.06~0.47% 命令真的超时 → 不是主要解释变量 | B |
| 5.5 vs 4 min（另一独立复现） | [759154] | +1.6 pt（95% CI −1.6~+4.7），无定论 | B |
| 8 min cap 的整份提交时长投影 | [SCOTT] | 120 题顺序 ≈ **12.4~14.2 h** → 会超 12 h；建议 6 min | B |
| 并发 8 vs 2（本地评测伪影） | [SCOTT] | 48 题 rich fold：33 vs 19 次超时；4×L4 的 KV cache 仅容 ~1.4 条满长序列 | B |
| 评分器并发 | [743063] | **顺序执行**，无并发 | A |

### 2.3 工具面、检索与定位

| 技术点 | 来源 | 测量效果 | 可信度 |
|---|---|---|---|
| **工具预算 50→25** | [746250] E35 vs E37 | 50/129 → 36/129，**−14 题**；运行时间 −29% | A |
| 工具预算 28→40（另一 bundle） | [758969] | 0.15 vs 0.17，无增益 | C |
| 增加 analyzer 子 agent | [746250] E36 | 50→51/129（+1，噪声） | A |
| **代码图检索工具** | [746250] E32 vs E30 | 50/129 vs 55/129（历史对照）；21/129 题用到；**21 次语义检索全部为空**；子图从未被调用 | B |
| 移除图工具（独立复现） | [747696] v3 | 13/31、11/31，LB 0.08（父版本 0.13）→ 负向 | C |
| graph-first 提示词 | [SCOTT] | 13/48 vs 14/48；调用 21 vs 26、超时 14 vs 19，但**测试失败多 4 次** | B |
| 异步函数不在代码图里 | [758854] | fastapi 快照 914 个 async 函数，图中 0 个；rich/httpx 同样为 0 | A（可从数据复算） |
| 「先列候选再读」的检索证据（无模型） | [746250] 评论 3536553 | 首个 identifier-grep 命中率 30%；grep+BM25 top-5 61%、top-10 74%；35/128 题面**不含任何代码标识符** | B |
| 符号名+邻居扩展（无模型） | [746250] 评论 3543807 | 2 跳内命中编辑点 47.9% vs TF-IDF Hit@5 14.5% | C（单作者） |
| 金标文件被打开的位置 | [752848] 评论 3547651 | 163 次运行中 **84% 在第 2~3 次调用就打开正确文件**；首个编辑耗时：解决 106 s vs 失败 183 s；编辑后耗时两者都 ~80 s | B |
| 「早编辑」提示词的实际执行率 | [752848] 同上 | 提示词要求早编辑**大多被忽略** | B |

### 2.4 提示词、流程与提交卫生

| 技术点 | 来源 | 测量效果 | 可信度 |
|---|---|---|---|
| **空 system prompt** | [747857] F150 vs N150 | token −15.6%；解决 44 vs 43（无差）；但**测试冲突 2 vs 8**；三组配对 2~4 vs 8~11 | A |
| 失败驱动 prompt 规则 v2 | [MOBASSIR] §5 | 本地 +1~+4 pt；**空补丁 16→6 次/run**；LB 10 题 = 基线 11 次抽样均值 → 不可见 | B |
| patch hygiene / finalize 步骤 | [747857] | 有 finalize 的 prompt 测试冲突 2~4，空 prompt 8~11；任务调用 patch hygiene 54 次 vs 0 | A |
| 提交前 `git status` 未跟踪文件检查、/tmp 放临时脚本 | [HITARTH] §6；[CAND-024] | 被列为高分组共同点（0.18~0.24 的 bundle） | C（关联非因果） |
| 单 agent vs analyzer+coder | [HITARTH] §6 | 高分组全部为单 agent；analyzer 仅 +1 题 | B |
| E4B「跑相关测试-读错误-再修」循环 | [758969] | 0.05 → 0.08 / 0.05（均值 ~0.065，约 +1 题） | C |
| E4B「前 3 分钟不许提交」 | [758969] | 0.01（显著变差） | C |
| E4B 长 prompt / 去图工具 | [758969] | 均 0.03 | C |
| skill 名必须 lowercase-kebab-case | [758854] | 不合规 → agent 不启动，LB 静默 0.00（不是报错） | A（作者本地复现+官方行为一致） |
| 一个 zip 里混入第二个 agent.yaml（不同模型） | [758969] | 两次 "Notebook Threw Exception"；删除后同一配置 0.17 | C |

### 2.5 验证环与失败模式

| 技术点 | 来源 | 测量效果 | 可信度 |
|---|---|---|---|
| **工具结果双重 JSON 转义** | [744272] | 62/299 次 `edit_file` 失败，其中 39 次可证为转义；30 个固定上下文回放：双重 JSON 失败 22%、单层 62%、**原文 0%** | A（已由官方 09-30 wheelhouse 修复） |
| `edit_file` 的 `new_string` 仍会被写入转义 | [744272] 评论 3551888 | 构造用例可复现；真实 trace（rich_3518, 12B）出现，靠语法检查兜住 | B |
| **gemma4 tool parser 数组参数死循环** | [753973] | 一次畸形调用冻结推理服务器 → 后续 3 题各 480 s、0 次 LLM 调用、全部 0 分；复现：`args:["Fix list[str] handling"]` 永不返回 | A（有最小复现） |
| 上下文压缩（14,336 token 阈值） | [MOBASSIR] §6 | ~80% run 触发压缩；解决率 80%（无压缩）→ 30%（≥2 次）；控制难度后 OR 0.50（0.35~0.64）/0.39（0.29~0.50） | B |
| 失败分类 | [MOBASSIR] §5.2 | 规则与思考预算削减掉的主要是「空补丁/超时」，**剩余最大失败类仍是「错补丁」** | B |
| 本地金标不可过 → 本地分数系统性偏低 | [751487] | fastapi 10/67 → 66/67；总 52/129 → 127/129 | A |

### 2.6 LoRA / 后训练（全部公开结果）

| 实验 | 模型/配置 | 数据 | 成本 | 结果 | 是否 LB |
|---|---|---|---|---|---|
| alpha 扫参事故 [750713] | 31B QLoRA，r16 α32（scale 2.0），7 投影×60 层，122.4M 可训 | 1,500 条轨迹 → 过滤后 **697 条/99 题**，另 60 条/11 题 holdout | 2×GX10（各 1 卡），220 optimizer steps，3 epoch 计划 | **LB 0.00**：4 分钟不出工具调用、复读 `git log` 直到预算耗尽 | **LB** |
| 同上，α16（scale 1.0） | 同上 | 同上 | 同上 | 5/8、7/11（训练集内）；思考块涨到 2.5k~14.7k 字符；中位 290 s/题 | 本地 |
| 同上，α8（scale 0.5） | 同上 | 同上 | 同上 | 6/11 vs 基线 **8/11**；中位 163 s；每题都出补丁 | 本地 |
| 全量 SFT r16 α32 [755677] | 31B，7 投影 | 2,377 条 gold-verified SWE-bench 轨迹，509 MB | 未报 | 本地 17/129（13.2%）→ **LB 0.00** | **LB** |
| 同上压缩到 r4 α8 | 同上 | SVD 压缩到 127 MB | 未报 | 本地 14/129（10.9%）→ **LB 0.00**；同 bundle 无 adapter LB **0.18** | **LB** |
| 同形状 adapter，改预算 [755677] 评论 | r16 α32 | 同上 | 未报 | 8 min/4096：complete 无分；**6 min/2048：0.17（= 无 adapter 基线）**；每任务 +29% 时长、每步 +45% token | **LB** |
| rank32 adapter [755677] 评论 | r32 | 未报 | 未报 | **LB 0.06**（4 min cap） | **LB** |
| **轨迹 DPO** [MOBASSIR] §4 | 31B int4，r16 全 attn+MLP，β=0.1，IRPO 式 + NLL 锚，thought 权重 0.2 | 4,207 条自建 mined 任务，on-policy 配对（同题 solved vs unsolved） | 未报 | 本地 **+2~3.5 pt**（sign test p≈0.008，需 2 holdout×2 seed 才显著）；**LB 9 题 vs 8 题**，CI −2.7~+4.7 | **LB（1 题）** |
| 「edit now」习惯 adapter [758968] | r8，442 个 edit-ready 决策点 | 自己 solved run | 未报 | 离线 edit 率 0~2% → **67%**（93% 编辑有效）；**LB 0.12 vs 基线 0.15/0.13**；**零权重 adapter 也是 0.12** | **LB（负）** |
| 同上「post-edit」adapter [758968] | 同上 | 编辑后片段 | 未报 | 学会反复跑同一条测试 **144 次**，9 题解 1 题 | 本地（负） |
| E4B 轨迹 SFT [750961] | E4B，r16 α32，4 epoch，全程监督、mask 工具返回 | 170 条教师轨迹/62 题 | 未报 | 基线 2/4、1/8；最好 SFT 1/4、1/8（**不如基线**） | 本地（负） |
| E4B 三种路线 [753793] | E4B | ①公开 SWE-Zero 过滤后 ~900 条（645 issue/291 repo）②教师轨迹 13 条×4 epoch ③自采样 40→200→470 | 未报 | **没有任何 adapter 超过基线**；学到的是「风格」（更短、更少调用），解码更慢 | 本地（负） |
| 31B tool-use LoRA [SCOTT] | r16，410 投影，16k 上下文 | 模型自己 solved 的 **83 个 turn / 14 题** | 1 个 4×L4 会话采样 + 1 个训练 | loss 0.17→0.11，held-out 工具一致率 0.990→0.985；**resolve 评测未跑** | 未测 |
| edit-only LoRA [AYUSH-LORA] | 31B，r16，层 30~59，q/o/down，只学 `edit_file`（≤1536 token），ctx 12,288 | 官方 kit 的 resolved + external 轨迹 | **4 h × 4×L4 = 16 L4-h** | 不宣称分数；只做 adapter 合法性检查后打包 | 未测 |
| surgical top-12 LoRA [BORISTOWN] | 31B，r8 α2（scale 0.25），仅层 48~59 的 q/o（24 模块） | 自己 18 条 live 通过轨迹，MAX 8 samples | 1 个 4×L4 会话 | 31B 上 **`ATTACH_SURGICAL_LORA_IN_SUBMISSION=False`**（为了不损失 W4A16 速度），仅在 E4B 轨道启用；无 LoRA 增益声明 | 未测 |

### 2.7 模型规模与教师

| 技术点 | 来源 | 测量效果 | 可信度 |
|---|---|---|---|
| 31B vs 12B vs E4B（同 agent 文件，31 题） | [750974] | 31B **9、11**/31；12B 4/31（5 min：4、4）；12B+12B reviewer 3/31；**E4B 0/31** | B |
| 同上独立复现（24 题×多次） | [750974] 评论 3542422 | 31B 15~21/48（31~44%），至少解出 9~13 题；12B 8/48（17%）、6 题；**12B 从未解出 31B 没解出的题** | B |
| 12B 的速度真相 | 同上 | decode 快 1.9×，但每次调用多写 1.8× token → 每次调用**更慢**（10.3 s vs 11.5 s）；非空补丁率 78% vs 46% | B |
| **强教师在同 harness 下的上限** | [747168] | Gemma 54/124、53/123；DeepSeek V4 Pro 58/124；Qwen3.6-27B 59/123；**仅 +4~5 题**；两次相同 Gemma 有 19/123（15%）翻转；两人都失败的 60 题里教师只能多解 10 题，**48 题无人可解** | A |
| 教师轨迹（DeepSeek）在同 harness | [747696] | 15 min 预算下 66/129（51.2%）vs Gemma 44~49/129 | B |
| 小模型公开最好分（2026-10-20） | [750974] 评论 / [751104] | **0.06** | B（时点数据） |

### 2.8 LB 噪声与"复现落差"

| 事实 | 来源 | 数值 | 可信度 |
|---|---|---|---|
| 公开 0.24 bundle 原样复跑 | [757588] | honghanhhh → **0.17**；emmanuel → 0.15 | B |
| honghanhhh notebook 自身挂接提交的最好分 | [KERNEL-META] | **0.18**（标题写 0.24） | A（Kaggle 元数据） |
| Budget-Fit 原样重复 | [758969] | 0.17 → **0.12**（只改了一行注释） | C |
| Budget-Aware v2 | [758969] | 0.17、0.17（公开 notebook 标 0.22） | C |
| 同一 zip 的分数区间 | [759154] | 0.12~0.17 | C |
| LB 分布（2026-11-10，5,250 队） | [757588] | 0.29×1、0.24×11、0.22×74、0.20×416、0.17×964、0.15×764，中位 0.15 | B |
| 「整条 LB 只是二项噪声」 | [748051] 评论 3538259 | 认为分布可由两个二项分布解释，无人真正突破 | C（与 0.24 密集台阶相冲突，仅记录） |
| 「加 LoRA 才能破 0.18 墙」 | [748051] 首帖 | 作者主观判断；同帖多人反驳（"top 没人靠微调"） | C（争议） |
| 「0.31 相当可达」 | [752848] 评论 3545049 | Top-1（Yurnero）主观判断，无方法披露 | D（推测） |

---

## 3. 自己训练 LoRA 是否值得

### 3.1 公开证据的总账

把 2.6 的表压缩成一句话：**11 组有分数的 adapter 实验，0 组给出可确认的正向 LB 增益；1 组给出 +1 题（CI 跨 0）；3 组给出 LB 0.00；1 组 0.17（= 同 bundle 无 adapter 基线）；1 组 0.12（低于基线 0.15/0.13）；1 组 0.06。**（"11 组"含只跑本地评测的 E4B 实验；有 LB 数字的为 7 组：0.00×3、0.17、0.06、0.12，以及 DPO 的 9 题 vs 8 题。）

其中三组失败有明确的、可复现的机制，而不是"运气不好"：

1. **alpha/r 比例把模型推出分布**：α32/r16 = scale 2.0 → 复读死循环、4 分钟不出第一次工具调用、LB 0.00；同一 adapter 改成 α16 立刻恢复 [750713]。而 α8（scale 0.5）只是"基线与微调之间的插值"，6/11 vs 8/11，仍不如基线 [750713]。
2. **adapter 静默掐掉 thinking 通道**：训练目标里没有 thought token（0/993 行），加载后 **0/109 次调用带 reasoning**，工具调用以纯文本泄漏（5/9 run）[758968]。反例：另一队的 r8 adapter 保留了 thinking，就能稳定出 0.17 [758968] 评论 3554629。
3. **adapter 的运行时税直接吃掉 12 小时预算**：每任务 +29% 时长、每步 +45% token；8 min cap 下整份提交超时"complete 无分"，必须降到 6 min/2048 才拿到 0.17（= 基线）[755677] 评论 3550223。LoRA 不是"免费的能力"，它从时间预算里扣钱。

再加上两条环境事实：教师模型在同 harness 下只比 Gemma 多 4~5 题，且 48/123 题谁也解不出 [747168]；31B 在本 harness 的真实瓶颈是 **14,336 token 压缩阈值 + 墙钟**，不是策略质量 [MOBASSIR]。这解释了为什么"更强的策略"很难变现。

### 3.2 预期收益（含推算，标注推测）

| 场景 | 预期 LB 变化（58 题） | 依据 |
|---|---|---|
| 31B 轨迹 SFT（r8~16，常规配方） | **0 题**，区间 −4~+1 | 11 组公开实验的点估计；DPO 的 +1 题 CI 跨 0 [MOBASSIR]；SFT 组全部持平或负 [750713][755677][753793] |
| 31B + 严格门控（thought 通道对齐、α=r、行为目标窄、6 min 预算内不减速） | **+1~+2 题**（≈+0.02~0.03），区间 −2~+4 | 机制上消除了三个已知失败源；但 harness 上限仍在 [747168]（推测） |
| 31B + RL（DPO/SPO，rank 128） | **未知，点估计 0** | 公开只有 DPO r16 的 +1 题；rank 128 无任何公开结果；RL 的收益需 ≥2 holdout×2 seed 才可见 [MOBASSIR]（推测） |
| E4B/E2B（小模型奖） | 相对值可能 +2~5 题，绝对值仍 <0.15 | 公开最好 0.06；E4B 主要失败是"改错文件"（13/31）而非"不会改" [750974]；奖励只看小模型内部排名 [751104]（推测） |

**关键判断**：主线要赢，需要跨过 0.24 密集台阶到 0.29+，即 **至少 +3 题**。LoRA 的公开点估计是 0 题、上限约 +2 题，**即使成功也大概率只是把 0.17 变成 0.19~0.20**，落在 416 队的 0.20 台阶里，不改变名次结构。

### 3.3 需要的数据（不是瓶颈）

- 官方公开集 129 题；按公开 31B 在本任务约 36~40% 的本地通过率估算 [747857]，可得 40~50 个已解题，每题 3~5 次采样 → **150~250 条可用轨迹**（自采）。
- 公开可用的现成轨迹（均已确认可用于 LoRA）：`dmitriigluzdov/gemma-4-serving-and-lora-verification-kit`（103 条 31B resolved + ~1,600 条外部）[750974]、scottweeden 的 tool-use kit [SCOTT]、SWE-smith / SWE-Gym / SWE-Next 等 HF 轨迹（官方明确允许，见 [746046] 与 [759238]）。
- 公开实验已证明 **数据量不是瓶颈**：442 个决策点就能显著改变行为（edit 率 0~2%→67%）[758968]；697 条/99 题也足够把 loss 从 0.61 压到 0.51 [750713]。**瓶颈是"学哪段 token"和"是否包含服务端的 thought 原文"。**

**数据配方（若要做）**：
1. 目标 token 必须是 **thought 通道 + 可见文本 + 工具调用** 的完整 assistant turn，与服务端实际发出的完全一致（用日志代理抓 request/response，并按 chat template 复渲染校验长度，误差 ≤1 token）[SCOTT]。
2. 过滤：只保留 resolved；每题只留**工具调用数最短**的一条（避免教"闲逛"）[750713]。
3. 预算形状必须与推理一致：把 `max_tool_calls` / `max_time_minutes` 的真实数字写进系统提示并作为训练分布的一部分；"预算是提示词的一部分"，改 cap 而不改数据等于让模型按比例多探索 [750713]。
4. 目标窄化优先：只学 `edit_file`（≤1,536 token，ctx 12k）比学整条轨迹便宜 10 倍且不易学到闲逛 [AYUSH-LORA]。

### 3.4 GPU 成本

| 方案 | 成本 | 来源 |
|---|---|---|
| 31B QLoRA r16，~200 轨迹/epoch | **~9 A100-h/epoch**（2×A100） | [750974] 评论 3542686（作者估算） |
| 12B 同类 | 约 2.5× 便宜，单 A100 可跑；"2 epoch×200 轨迹 + 2 次×200 题评测" ≈ **12 A100-h** | 同上 |
| 31B 完整实验（2 epoch + 2 次 200 题评测） | 约 30~50 A100-h（训练 ~18 h + 两次评测；推算，评测按 ~11 h/200 题） | 推算 |
| 31B edit-only r16（层 30~59，3 投影） | **16 L4-h**（4 h × 4×L4） | [AYUSH-LORA] |
| 31B 轨迹 SFT r16（697 条） | 2×单卡工作站，220 step，未报小时数 | [750713] |
| E4B | 单张 24 GB L4 可训 | [750974] 评论 |
| **一次 129 题全量本地评测** | ~11 GPU-h（并发 1）；并发 2 约 5.5~6 h | [746250] 标定推算 |

**真正的成本在评测，不在训练**：一次"训练 + 配对两臂全量评测"约 30~40 GPU-h；用 60 题分层面板可以把单轮压到 6~8 GPU-h。剩余 17 天、每天 1 次提交的节奏下，**能做的严格配对实验不超过 10~15 轮**。

### 3.5 风险清单

| 风险 | 证据 | 影响 |
|---|---|---|
| **rank/规模限制** | 官方对 rank 与包大小有上限（`HARNESS_README`/Getting Started；本任务给出的口径为 rank ≤128），提交包 ≤3 GB，adapter 必须落在 `adapters/<name>/` 并由 vLLM 以 `--lora-modules` 注册 [743964][755677] | rank 128 + 全 7 投影 ≈ 数 GB，接近 3 GB 上限；且公开实验里 rank 只有 4/8/16/32，**rank 128 无任何公开结果** |
| **alpha 未扫** | α32/r16 → LB 0.00；α16 恢复 [750713] | 一次错误 alpha 就能烧掉一周 |
| **thinking 通道被学没** | 0/109 次调用带 reasoning；工具调用文本泄漏 [758968] | 灾难性、且离线指标看不出来 |
| **解码变慢** | +29% 时长、+45% token/step [755677]；300 模块 LoRA 曾致 35% Punika 开销 [BORISTOWN] | 8 min cap 直接超 12 h；必须同时降预算，等于把收益还回去 |
| **对公开题过拟合 / 评测污染** | 公开集只有 129 题且被反复用于选型（"about 18 looks"）[MOBASSIR]；DPO 增益集中在"复述修复"的 PR body 题面（+5.8 vs +1.8）[MOBASSIR] | 本地涨、私有集不涨；这正是 DPO 组 LB 只有 +1 题的机制解释 |
| **单次私有抽样** | 决赛只跑一次 ~60 题 [754295][743063] | +1~2 题的增益在决赛里 50% 概率变负 |
| **许可/合规** | 公开数据可自由使用 [742807 评论]；其他开源模型轨迹、HF 公开 SWE 轨迹均被明确允许 [746046]；但"用第三方已发布的闭源模型轨迹"截至 11-14 仍只有同侪回答、无官方定论 [759238]；获奖 adapter 的释放义务（2.5/2.8 条）官方未答 [742807] | 若冲奖，需预留合规解释成本 |
| **静默 0.00** | skill 名不合规、zip 混入第二个 agent.yaml、adapter 路径错都会 0.00 而非报错 [758854][758969] | 一次 0.00 浪费一天且无法区分"adapter 差"与"打包错" |

### 3.6 结论

**不建议把 31B LoRA/RL 作为主线冲榜手段；建议作为「一次、严格门控、可随时放弃」的旁路实验，并把小模型奖作为首选 LoRA 落点。**

判定依据（按权重）：
1. 公开 11 组实验 0 组正向确认，最好的 +1 题在噪声内（§2.6）。
2. 机制上，harness 上限（教师只多 4~5 题；48/123 无人可解）压住了策略增益 [747168]，而 adapter 还要再交 29% 时长税 [755677]。
3. 计量上，σ=1.5~3 题、决赛单次抽样，+2 题的改动无法被验证，也无法被兑现 [MOBASSIR][754295]。
4. 成本上，一次可信的 31B 实验（训练+两臂全量评测）≈25~40 GPU-h，机会成本等于 3~4 轮框架 A/B。

**只有在下列条件全部满足时才启动 31B LoRA**：
- 已完成 §4 的框架收敛，基线稳定在本地 129 题的一个已知均值；
- 有 ≥90 题的配对面板和 ≥2 个 seed 的评测预算；
- 训练目标包含 thought 通道，α=r，rank ≤32，且加载后实测解码减速 ≤10%；
- 预注册成功门槛：**本地配对 ≥+4 题/90 题**（≈+0.07），否则不上最终名额。

---

## 4. 修改 agent 框架是否值得

### 4.1 已经"用掉"的杠杆（不要再花时间）

| 杠杆 | 现状 | 证据 |
|---|---|---|
| thinking 开 + thought summaries 开 | 我们的 bundle 已是 `thinking_budget 4096 / include_thoughts true` [CAND-024] | 关掉任一项 −7~−11 题/58 [746250] |
| 单 agent、无 analyzer | 已是单 agent | analyzer 仅 +1/129 [746250] |
| 图工具 | 已注册但基本不被调用；公开高分组都不用 | 21 次语义检索全空 [746250]；graph-first 反而多 4 次测试失败 [SCOTT] |
| 温度/top-p | 0.2/0.95 | 0.4 与 1.0 并列，差异在噪声内 [746250] |

**结论：这几个方向继续投入的边际收益 ≈0。从公开对比看，0.17→0.22 的差距更可能来自流程/预算/抽样，而不是这些开关（推测）。**

### 4.2 仍然值得做的六项（按证据强度排序）

**(1) 12 小时安全与预算节奏（最高优先，风险项而非增益项）**
- 8 min cap × 120 题顺序执行 = 12.4~14.2 h，**会整份报错** [SCOTT]；官方确认超时当前是 error [743063]。
- 4.5 vs 4 min 的配对差 p=0.58，说明 4~6 min 区间内"多给时间"并没有可测收益 [747857]；而 6 min 能把 120 题压到 ~11 h。
- **动作**：把 `max_time_minutes` 从 8 降到 5.5~6，同步把 `max_tool_calls` 从 28 提到 32~34（多数任务在 26 次调用内完成 [SCOTT]；28 次上限有 35% 的 run 用满 [750974] 评论）。预期：解决数持平，超时与整份失败风险大幅下降。
- 可信度 B；预期 LB 0~+1 题，但**消除的是 −0.17 级别的尾部风险**。

**(2) 压缩与上下文增长控制（唯一还有"机制性增益"空间的项）**
- 14,336 token 触发压缩；~80% run 会压缩；控制难度后压缩仍与失败强相关（OR 0.50/0.39）[MOBASSIR]。
- 有效手段：**降低单轮思考预算到 2048**（解决数不变、超时 10→2、快 8%）[MOBASSIR]；**不要**关闭历史 reasoning（超时 3→15）[MOBASSIR]。
- 配套：限制 `read_file` 单次行数、禁止整文件 dump、工具输出保持简短（我们自己 bundle 的 prompt 已要求 bounded read，需核对是否真的执行）。
- 可信度 B；预期 +1~2 题/58（推测，主要来自"少超时 = 多跑完几题"）。

**(3) 提交卫生（最便宜、最确定）**
- 空 prompt 的测试冲突 8~11 次 vs 有 finalize 的 2~4 次 [747857]；改测试文件 = 直接 0 分（规则）。
- **动作**：核对 prompt 是否明确包含：不得修改/新建/删除 `tests/`、`*_test.py`、`test_*.py`；临时脚本走 `/tmp` 且用 `run_command` 写（`write_file` 只能写 workspace 内相对路径 [BORISTOWN]）；提交前 `git status` 检查未跟踪文件、`git diff --stat` 检查 patch 非空；`submit_patch` 后确认 `patch_size>0 && files_changed>0`。
- 可信度 A（机制确定）；预期 +0~1 题，但同样消除 −0.17 级尾部风险。

**(4) 早编辑硬约束（有效但被高估）**
- 84% 的 run 在第 2~3 次调用就打开正确文件；**解决与失败的差别在"首次编辑时间"（106 s vs 183 s）**，编辑后耗时相同 [752848]。
- 但"提示词要求早编辑"大多被忽略 [752848]；而预算按比例分配时，提高 cap 只会让模型按比例多探索 [750713]。
- **动作**：不要写"尽早编辑"这种软话，写**固定数字**："第 8 次调用前必须有一次源码编辑；第 20 次调用后不得再开始新的探索"；并在 system prompt 的预算块里同时给出 calls 与 minutes 的真实数字。若本地仍不执行，改用 **skill/回调**（注意回调注册表在 0.2.11 为空，需自行确认可用性 [743964] 评论 3529772）。
- 可信度 B；预期 0~+2 题。

**(5) 工具面裁剪（低风险、省时间）**
- 28 次调用里图工具几乎不被使用；保留它们只增加 schema token 与误用风险。E4B 上"去图工具"单独看是 0.03（但那是 E4B 且同时改了别的东西）[758969]。
- **动作**：A/B「6 工具（`run_command/read_file/edit_file/write_file/get_status/submit_patch`）」vs「9 工具」。预期解决数持平、中位调用数 −10%、少一次"空检索"。
- 可信度 B；预期 0~+1 题。

**(6) 验证环（对 31B 证据弱，不要期待）**
- E4B 上"跑相关测试-读错误-再修"把 0.05 抬到 0.065 均值 [758969]（+1 题，单次）。
- 31B 上"post-edit 训练"反而学会重复同一条测试 144 次 [758968]。
- 且 12 小时预算下，多跑一次全量 pytest 往往直接超时（命令超时 300 s 会与评分器 pytest 共用 [BORISTOWN]）。
- **动作**：只保留"一次 compile check 或一次定向测试"，禁止 full-repo pytest。
- 可信度 B；预期 0~+1 题。

### 4.3 结论

**值得，且是主线。** 理由不是"框架能涨很多"，而是三点：
1. 已测到的框架大杠杆（thinking / summaries / 单 agent / 去图工具）我们已经在用，**剩下的是尾部风险与压缩成本**；这些恰恰是"不涨分但会掉分"的项，必须先锁死。
2. 框架改动能用**本地配对 + 129 题 scorer-like 沙箱**验证，成本 6~12 GPU-h/轮，且结果可复现（金标 127/129 [751487]）。
3. 相比之下，LoRA 的验证成本高 3~5 倍、公开成功率 0/11、且要额外承担 29% 时长税。

**预期**：框架收敛后本地 129 题 +3~6 题（≈+1~3 题/58），LB 从 ~0.17 抬到 0.20~0.22 区间，**进入 0.22 的 74 队台阶是现实目标；0.24+ 需要运气或一次成功的窄域 adapter**（推测）。

---

## 5. 排序实验计划（最便宜的先做）

> 统一约束：本地评测必须用 scorer-like 沙箱（金标 ≥125/129、空补丁 ≤2/129）[751487]；并发 1~2（与评分器一致）[SCOTT]；所有比较为**同题配对**，报告逐题翻转表 + McNemar/配对 bootstrap；每个实验预注册成功/放弃门槛。

| # | 实验 | 成本 | 主指标与成功门槛 | 放弃/回滚条件 |
|---|---|---|---|---|
| **P0** | **本地评测口径校准**：搭 scorer-like 沙箱，跑 129 题金标 + 空补丁；再用同一 harness 复跑我们候选 bundle 与 Budget-Fit 各 1 次 | 2~3 CPU-h + 5~6 GPU-h（60 题面板 ×2 bundle，并发 2） | 金标 ≥125/129；空补丁 ≤2/129；我们候选本地解决率与公开 0.17~0.18 同量级（本地通常高 8~12 pp） | 金标 <120 → 先修环境，停止一切比较 |
| **P1** | **打包/合规静态审计**（0 GPU）：4 个 eval 字段；skill 名 kebab-case；全 zip 只有一个 `model:`；adapter 路径与 base_model 名；3 GB 限制；prompt 含"禁改测试/临时文件走 /tmp/提交前 git status"；12 h 预算表（最坏 cap×120） | 1 h | 全部通过；最坏情形投影 ≤11 h | 任一项失败 → 先修，不提交 |
| **P2** | **自身噪声标定**：候选 bundle 在 60 题分层面板上跑 3 次（同一 zip，不改任何字段） | ~8 GPU-h | 得到 σ 估计与"flaky 题清单"；≥90% 题三次结果一致 | 若 σ>3 题 → 后续一切 <4 题的结论都不采信 |
| **P3** | **预算节奏 A/B**：A=8 min/28c/thinking4096（现状） vs B=6 min/34c/thinking2048，129 题配对 | ~12 GPU-h | B ≥ A−1 题，且超时 −50%，压缩率 −20%，12 h 投影 ≤10.5 h | B < A−3 题 → 回滚 |
| **P4** | **卫生+早编辑 A/B**：在 P3 胜者上加"finalize 卫生块 + 第 8 次调用硬编辑 + 第 20 次停止探索"（固定数字） | ~12 GPU-h | 测试冲突 ≤1；空补丁 −50%；解决 +2 题 | 解决 −2 题 → 回滚 |
| **P5** | **工具面 A/B**：9 工具 vs 6 工具（去图工具） | 8~11 GPU-h | 解决数 ≥ 持平；中位调用 −10%；无空检索 | 解决 −2 题 → 保留图工具但 prompt 禁用 |
| **P6** | **（并行、低成本彩票）E4B 小模型奖**：E4B + 单卡 24 GB LoRA（r16 α16，含 thought 通道，目标=edit 决策点或完整 turn），4×L4 一个会话 | 16 L4-h + 1 次小模型提交 | 31 题面板 ≥ 基线 +2 题；补丁率 ≥95%；LB >0.06 | 本地 <基线 → 放弃该奖 |
| **P7** | **（门控）31B LoRA 试点**：仅在 P3~P5 收敛后；r8~16、α=r、层 48~59 的 q/o 或全 7 投影、数据=自采 150~250 条 resolved 轨迹（含 served thought 原文）、≤32k、2 epoch | 25~35 GPU-h（训练 + 两臂 90 题×2 seed 评测） | 本地配对 **≥+4 题/90 题**；解码减速 ≤10%；thought 通道非空率 100% | 任一不满足 → 不占最终名额 |
| **P8** | **最终名额配置**：1 个"稳定 0.22 级"保守 bundle + 1 个"实验"bundle；各自选多次抽样均值而非最好一次 | 2 次提交 | 选均值最高者；保留一次同 zip 复跑用于自测 σ | 禁止用两次提交赌同一假设 |

**时间预算（17 天）**：P0~P2 约 3 天；P3~P5 约 6 天（可串行）；P6 并行 1~2 天；P7 只有在前序全部通过且剩余 ≥6 天时才启动。

**提交纪律**：每天 1 次、同时只能 1 个 pending [747519]；任何"小于 4 题"的改动不要用提交去验证，用本地配对；同一 zip 至少复跑 1 次以估计自身 σ。

---

## 6. 不要做清单（有明确反证）

1. **不要**关 thinking 或关 thought summaries（−7~−11 题/58）[746250]。
2. **不要**设 `include_thoughts:false` 来省 context（超时 3→15、慢 19%）[MOBASSIR]。
3. **不要**加 analyzer / 多 agent 编排（+1/129，且在噪声内）[746250][HITARTH]。
4. **不要**指望代码图工具（语义检索 21/21 为空；async 函数根本不在图里）[746250][758854]。
5. **不要**在没降时间 cap 的前提下提工具 cap（8 min × 120 题 = 12.4~14.2 h，整份报错）[SCOTT][743063]。
6. **不要**用单次 LB 判断任何 <0.05 的变化（σ=1.5~3 题）[MOBASSIR][758954]。
7. **不要**在训练目标里丢掉 thought token（0/109 次推理 → 工具调用文本泄漏）[758968]。
8. **不要**用 α/r>1 而不扫 α（α32/r16 → LB 0.00）[750713]。
9. **不要**相信 eval loss（0.5088 的 loss 产出了复读模型）[750713]。
10. **不要**在本地用并发 8 评测（4×L4 KV cache 仅 ~1.4 条序列 → 超时爆炸）[SCOTT]。
11. **不要**改测试文件或让临时文件污染 workspace（直接 0 分；空 prompt 冲突 8~11 次）[747857]。
12. **不要**把 honghanhhh 的 "0.24" 当作我们的基线（其 notebook 元数据 0.18，第三方复跑 0.17）[KERNEL-META][757588]。

---

## 7. 证据缺口（决定我们要自己测什么）

| 缺口 | 为什么重要 | 我们该怎么补 |
|---|---|---|
| 没有任何"top-1 0.29 用了什么"的公开说明 | 0.29 与 0.24 台阶差 3 题，可能来自方法也可能来自抽样 | 只能靠自己的配对实验；不要假设对方用了 LoRA |
| 没有 rank ≥64 的公开结果 | 我们计划 rank 128，风险未知 | P7 之前先做 rank 8/16 的行为验证，再决定是否上 128 |
| 没有"压缩率与解决率"的因果实验（只有相关性） | 这是我们认为最有机制空间的项 | P3 直接以"压缩次数/run"为次要指标做 A/B |
| 没有 12 h 超时后的官方最终行为定论 | 影响是否敢用 8 min cap | 按"会 error"规划（保守）[743063] |
| 小模型奖当前最好分未知（0.06 为 10-20 时点） | 决定 E4B 彩票是否还值得 | 提交前读一次小模型 LB notebook [751104] |
| 私有集与公开集的难度差未知 | 本地 129 题 → LB 的映射不稳定 | P0/P2 用两个已知 bundle 校准映射 |

**本次调研的局限**：小模型奖 LB notebook 的 `leaderboard.csv` 下载返回 403，无法刷新 0.06 之后的排名，故该数字仅代表 2026-10-20 时点；`best_public_score` 来自 Kaggle `search_content` 的 kernel 元数据，是「该 notebook 关联提交的最好公开分」，不等于作者在标题里声称的分数（honghanhhh 就是反例）。

---

## 8. 来源索引

**讨论帖（Kaggle forum topic）**

- [742807] Clarification on distillation from external LLMs — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/742807
- [743063] Scoring-run questions: task concurrency, eval_config keys, and the 12 h limit — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743063
- [743140] What Results Did You Get with Representative Submissions? — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743140
- [743506] The public leaderboard has exactly 58 tasks — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743506
- [743964] Clarifications on 12h overrun, ParallelAgent concurrency, LoRA loading, and self-generated training data — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743964
- [744272] Tool results reach the model double-JSON-encoded — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744272
- [744354] Thinking mode drops the model's thoughts between tool calls — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744354
- [745855] The first task (fastapi_15661) seems unsolvable from the prompt alone — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/745855
- [746046] Allowed training data for LoRA adapters: trajectories from other open models? — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/746046
- [746250] Gemma 4 agent experiments: sampling, thinking, time limits, and tool ablations — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/746250
- [747168] Stronger models barely beat Gemma 4 31B in this harness — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747168
- [747696] Time, Budget, and Thinking: Gemma 4 Coding-Agent Experiments — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747696
- [747857] Gemma 4 follow-up: Kaggle runtime calibration, command timeouts, and empty-prompt ablations — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747857
- [748051] The top of the leaderboard is dominated by those who trained LoRA's — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748051
- [748287] New Gemma 4 Models Available: 12B, E4B, and E2B — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748287
- [750713] Fine-tuning Gemma 4 31B on 1,500 agent trajectories — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/750713
- [750961] Has anyone achieved reliable gains from LoRA trajectory SFT on Gemma 4 coding agents? — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/750961
- [750974] 31B vs 12B vs E4B with the same agent files: a local comparison — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/750974
- [751104] Bonus Prize: Small Model Development ($10,000) — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/751104
- [751487] Local eval in a scorer-like sandbox: gold patches pass on 127/129 public tasks — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/751487
- [752848] Why does nobody have a score of over 0.24? — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/752848
- [753793] Is LoRA really helpful? — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/753793
- [753973] [BUG REPORT] tool parser can loop forever on an array tool argument — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/753973
- [754295] Will private leaderboard evaluation use multiple runs to reduce variance? — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/754295
- [755677] Scoring zero when using LoRA adapters — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/755677
- [757588] I reran a public 0.24 bundle unchanged and got 0.17 — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/757588
- [758817] Open toolkit for measuring noise in your agent runs — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/758817
- [758821] Open code and data: agent configs, weight-editing library and held-out task sets (E4B) — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/758821
- [758854] Harness pitfalls: no async functions in the code graph; bad skill names score 0.00 — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/758854
- [758954] Scorer minutes: a toolkit to convert time caps between GPUs and measure run to run noise — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/758954
- [758968] LoRA habit adapters changed the agent's behaviour offline but not the live score — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/758968
- [758969] Three lessons from 22 submissions: a zip bug, LB noise, and E4B quitting early — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/758969
- [759154] Testing configs offline on the 129 training tasks: my harness and results — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/759154
- [759238] Public trajectory datasets generated by proprietary models: allowed for LoRA training? — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/759238
- [747519] Update on "Kaggle Error" Submissions & Queue Delays — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747519

**Notebook / 数据集**

- [MOBASSIR] `mobassir/why-your-leaderboard-can-t-see-a-3-point-gain` — https://www.kaggle.com/code/mobassir/why-your-leaderboard-can-t-see-a-3-point-gain （论文 "Too Small to See"，含 DPO 实验、干预台账、压缩分析）
- [HITARTH] `hitarthjain0/gemma-4-agent-what-took-lb-0-12-0-24` — https://www.kaggle.com/code/hitarthjain0/gemma-4-agent-what-took-lb-0-12-0-24
- [GLUZDOV] `dmitriigluzdov/gemma-4-public-agent-ideas-and-evidence` — https://www.kaggle.com/code/dmitriigluzdov/gemma-4-public-agent-ideas-and-evidence
- [SCOTT] `scottweeden/gemma-4-developer-agent-tool-lora-writeup` — https://www.kaggle.com/code/scottweeden/gemma-4-developer-agent-tool-lora-writeup
- [BORISTOWN] `boristown/gemma-4-e2b-lora-fine-tuning-swe-agent-baseline` — https://www.kaggle.com/code/boristown/gemma-4-e2b-lora-fine-tuning-swe-agent-baseline （标题写 E2B，代码实为 31B flagship V20；31B 上不挂 adapter）
- [AYUSH-LORA] `ayush04barai/lora-step-2` 与 `ayush04barai/version-1-added-adaptor` — https://www.kaggle.com/code/ayush04barai/lora-step-2 ，https://www.kaggle.com/code/ayush04barai/version-1-added-adaptor
- [KERNEL-META] Kaggle `search_content` 返回的 kernel 元数据 `best_public_score`（2026-11-15 读取）：flexonafft 0.22、denpugovkin 0.22、honghanhhh "LB 0.24" 0.18、budget-fit 0.18、boristown E2B LoRA 0.15、romanrozen 0.12
- [LB-READ] `get_competition_leaderboard`（2026-11-15 读取，public）：Yurnero 0.29；0.24 共 13 队；0.22 多队
- [CAND-024] 本仓库 `submission-candidate-0.24/provenance.json`（honghanhhh v8，8 min / 28 calls / 240 s / 80 turns）
- [750974 数据集] `dmitriigluzdov/gemma-4-serving-and-lora-verification-kit`（103 条 31B resolved 轨迹 + ~1,600 条外部 + `lora_index.csv`）

**规则/官方**

- [HARNESS_README] 竞赛数据内 `HARNESS_README.md`（模型别名、eval_config 四字段、adapter 注册方式）
- [Evaluation] https://www.kaggle.com/competitions/gemma-4-developer-agent/overview/evaluation
- [Paper Track] `too-small-to-see-resolution-limits-leaky-stateme` — https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/writeups/too-small-to-see-resolution-limits-leaky-stateme
