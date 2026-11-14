# Gemma 4 Developer Agent（Kaggle 149921）公开方案情报

采集时间：2026-11-15（本地时间）。只读 Kaggle，未提交、未投票、未创建 notebook。

---

## 1. 结论摘要

1. **榜首 0.29（team "Yurnero"）没有任何公开产物**：本人即 Kaggle 用户 `samson8`（Grandmaster，competition ranking 33）。其名下只有 1 个 2023 年 LLM Science Exam 的 notebook（`samson8/how-to-create-wikipedia-embeddings`）和 2 个 2023/2025 年无关 dataset；本赛题下 0 个 notebook、0 个 dataset、0 篇 writeup。全站 344 个赛题 topic 标题已全量扫描，没有 solution/winner/0.29 类文章。
2. **公开可复现的最高分产物是 `flexonafft/gemma-4-budget-aware-coding-agent-v2` 的 v10**：Kaggle 记录的 `best_public_score = 0.22`，且独立第三方普查（145 个"有分数的公开 notebook 版本"）里**唯一**一个 0.22。已下载其 `submission.zip` 并解包验证为完整 agent bundle。
3. **但 0.22 是"多次抽样的最大值"，不是稳定水平**：同一份文件（`files_sha256 = 56010e9e…`）在普查中共 6 次抽样，结果为 0.22 / 0.17×5，均值 0.178、中位数 0.17。复现时应按 0.17–0.18 设预期。
4. **所有"LB 0.24"的公开 notebook 都是标题宣称，Kaggle 记录的真实 best 只有 0.17–0.18**：`honghanhhh` 记录 0.18、`kawh1111` 0.18、`ozertuu` 0.17；其中 `honghanhhh` 甚至被他人原样重跑得到 0.17（topic 757588）。
5. 评分刻度（社区反推，非官方声明）：58 道公开题，`score = floor(100 × 解出题数 / 58) / 100`。→ 0.22 = 13 题、0.24 = 14 题、0.29 = 17 题、0.31 = 18 题。榜首比最好的公开 bundle **多约 4 道题**，这 4 道不是公开信息能补上的。

---

## 2. 实时榜单前 5（本次重新读取，2026-11-15）

| rank | team | public score | 提交时间 (UTC) | 公开产物 |
|---|---|---|---|---|
| 1 | Yurnero | **0.29** | 2026-10-29 08:07:28 | 无（见 §1） |
| 2 | Yike Xu | 0.24 | 2026-11-12 03:45:13 | 未发现 |
| 3 | MUHAMMED IDRIS BIN YACOOB | 0.24 | 2026-11-07 23:49:25 | 未发现 |
| 4 | SAVE | 0.24 | 2026-11-13 20:51:18 | 未发现 |
| 5 | chizhu | 0.24 | 2026-11-06 07:28:24 | 未发现 |

赛题元信息：5716 队、截止 2026-12-02、每日 1 次提交、团队上限 5 人、奖池 75,000 USD。
对照：2026-11-10 的榜单分布（topic 757588 记录）为 0.29×1、0.24×11、0.22×74、0.20×416、0.17×964、0.15×764，中位数 0.15——**0.29 是孤例，0.24 只有 11 队**。

---

## 3. 候选排序表

"分数证据"一栏严格区分三类：**Kaggle 记录值**（notebook metadata 的 `best_public_score`，客观）、**标题宣称**（作者自己写的）、**第三方重跑**。

| # | 名称 | 分数证据 | URL | 是什么 | 本地路径 |
|---|---|---|---|---|---|
| 0 | Yurnero（samson8） | LB **0.29**（榜单，非产物） | https://www.kaggle.com/samson8 | 榜首；**无任何本赛题产物** | — |
| 1 | flexonafft / Budget-Aware Coding Agent v2 (v10) | **Kaggle 记录 0.22**（13/58）；普查中唯一 0.22；同文件另有 5 次 0.17 | https://www.kaggle.com/code/flexonafft/gemma-4-budget-aware-coding-agent-v2?scriptVersionId=357955943 | 单 agent，9 工具（含 3 个代码图工具但提示词禁用），8 min / 28 calls / 240 s / 80 turns，temp 0.2 | `flexonafft__gemma-4-budget-aware-coding-agent-v2/` |
| 2 | flexonafft / Lean Conte (v25) | 无记录分（未评分新候选） | https://www.kaggle.com/code/flexonafft/gemma-4-budget-aware-coding-agent-lean-conte | 同作者最新分支：去掉 3 个图工具（6 工具）、时限 8→10 min、新增 14k token 压缩预算规则 | `flexonafft__gemma-4-budget-aware-coding-agent-lean-conte/` |
| 3 | honghanhhh / "Gemma 4 Baseline \| LB 0.24" (v8) | 标题宣称 0.24；**Kaggle 记录仅 0.18**；第三方重跑 **0.17** | https://www.kaggle.com/code/honghanhhh/gemma-4-baseline-lb-0-24 | 候选 1 的字节级副本 + 3 处提示词修改（保护 pytest.ini/conftest.py、/tmp 临时文件、sync/async 全覆盖、`git status --short && git diff`） | `honghanhhh__gemma-4-baseline-lb-0-24/` |
| 4 | denpugovkin / Repair Lab (v8) | **Kaggle 记录 0.22**（其 exp021 版本） | https://www.kaggle.com/code/denpugovkin/gemma-4-repair-lab-compare-before-you-submit | 实验台；当前 `submission.zip` 是**未评分的 exp024**，0.22 的 exp021 以 `score_022_submission.zip` 另存（该文件经 API 下载返回 404，未取得） | `denpugovkin__gemma-4-repair-lab-compare-before-you-submit/` |
| 5 | ozertuu / gemma-4-gold-submit (v4) | 标题宣称 "0.24+"；**Kaggle 记录 0.17** | https://www.kaggle.com/code/ozertuu/gemma-4-gold-submit | 6 工具（删掉 3 个图工具）、temp 1.0 / top_k 64 / 16384 输出；解释了 58 题刻度 | `ozertuu__gemma-4-gold-submit/` |
| 6 | kawh1111 / G4 baseline 024 | 标题 0.24；记录 0.18 | https://www.kaggle.com/code/kawh1111/g4-baseline-024 | 候选 3 的 fork，仅改署名行 | 未下载（与候选 3 等价） |
| 7 | verracodeguacas / Budget-Fit Single Agent | 记录 0.18 | https://www.kaggle.com/code/verracodeguacas/gemma-4-agent-budget-fit-single-agent | 另一条主线（matterhorn3838 SuperAgent 系）的起点 | 下载返回 403，未取得 |
| 8 | jvnhonglei / SuperAgent Reproduction | 记录 0.20 | https://www.kaggle.com/code/jvnhonglei/gemma-4-superagent-reproduction | 0.20 档，介于 0.18 与 0.22 之间 | 未下载 |

---

## 4. 唯一推荐复现的候选

**`flexonafft/gemma-4-budget-aware-coding-agent-v2` v10（scriptVersionId 357955943）**

理由，按证据强度排序：

1. Kaggle 自己的 notebook 元数据记录 `best_public_score = 0.22`——这是平台上**客观记录**、且是全赛题公开 notebook 的最高值（我用 5 组不同关键词检索了 68 个 kernel 文档，最高只有 0.22；按 `ScoreDescending` 排竞赛 notebook，第一名也是它）。
2. 独立第三方普查（数据集 `dariushafshar/gemma-4-agent-public-config-draws`，2026-10-22 抓取，覆盖 274 个公开 notebook、145 个有分数的版本）里，**0.22 只出现 1 次**，就是它的 v10；`files_sha256 = 56010e9e9d4c7f33856ce49d12ec3edae5c912261986e999a4aa989b3cd463cb`。
3. 它是当前公开生态的**事实祖先**：`honghanhhh`（"LB 0.24"）、`kawh1111`（"024"）都声明 `agent.yaml` / `sampling.yaml` / `eval_config.yaml` 从它字节级复制；`ozertuu` 与 `lean-conte` 是同一配置去掉图工具。
4. 我下载的 zip（`kf/357955943/...`）与该版本号完全对应，解包后是结构完整的合法提交：`agent.yaml` + `prompts/system.md` + `configs/sampling.yaml` + `eval_config.yaml`。

**预期修正**：同一 `files_sha256` 的 6 次实际抽样为 0.22、0.17、0.17、0.17、0.17、0.17（出处见 §6 普查行）。所以正确预期是**中位数 0.17、均值 0.178**，0.22 是运气上沿。要跨过 0.24（14 题）需要在这份 bundle 上稳定多解 1–2 题，要碰 0.29（17 题）需要多解 4 题。

**这份 bundle 的具体配置**（完整文件已落盘）：

```yaml
# agent.yaml
name: swe_agent
model: gemma-4-31b-it-qat-w4a16-ct
tools: [run_command, read_file, edit_file, write_file, get_status, submit_patch,
        get_code_neighbors, search_similar_code, get_code_subgraph]
# eval_config.yaml
evaluation: {timeout_seconds: 240, max_tool_calls: 28, max_time_minutes: 8, max_turns: 80}
# configs/sampling.yaml
temperature: 0.2, top_p: 0.95, max_output_tokens: 8192, thinking_budget: 4096, include_thoughts: true
```

提示词要点（54 行）：硬性预算（28 calls / 8 min，"未编辑的树永远是 0 分"）；禁改测试文件；禁跑全量 pytest；**明令禁用 3 个代码图工具**（"它们返回不了有用的东西"）；call 1–8 定位、9–12 必须动手、12 之前必须已编辑、20 之后不再探索；`edit_file` 匹配失败两次后改用 heredoc Python 精确替换；结尾 `git diff` + `submit_patch`。

---

## 5. 全部已下载文件（精确路径）

根目录：`/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/.worktrees/training-plan-refactor/research/competition-intel/`

```
flexonafft__gemma-4-budget-aware-coding-agent-v2/
  submission.zip                                   3,713 B  (zip/deflate, kf/357955943)
  extracted/agent.yaml
  extracted/configs/sampling.yaml
  extracted/eval_config.yaml
  extracted/prompts/system.md                      54 行

honghanhhh__gemma-4-baseline-lb-0-24/
  submission.zip                                   3,990 B
  extracted/agent.yaml                             (与候选 1 字节相同)
  extracted/configs/sampling.yaml                  (字节相同)
  extracted/eval_config.yaml                       (字节相同)
  extracted/prompts/system.md                      56 行（3 处差异，见 §3）

flexonafft__gemma-4-budget-aware-coding-agent-lean-conte/
  submission.zip                                   3,873 B
  extracted/{agent.yaml, configs/sampling.yaml, eval_config.yaml, prompts/system.md}   59 行

ozertuu__gemma-4-gold-submit/
  submission.zip                                   2,755 B
  extracted/{agent.yaml, configs/sampling.yaml, eval_config.yaml, prompts/system.md}   45 行

denpugovkin__gemma-4-repair-lab-compare-before-you-submit/
  submission.zip                                  20,902 B  (含 LICENSE.txt/NOTICE.txt；未评分 exp024)
  extracted/{agent.yaml, configs/sampling.yaml, eval_config.yaml, prompts/system.md, LICENSE.txt, NOTICE.txt}

evidence__public-config-draws/
  census_excerpt_ge_0.17.csv                       33 行摘录（0.17 及以上全部行，含 files_sha256）
```

同目录还留有上一轮已抓取、本次校验为**完全相同**的副本（未改动）：
`flexonafft-v10/unpacked/**`（= 候选 1 的 extracted）、`honghanhhh-0-24/unpacked/**`（= 候选 3 的 extracted）、`baseline-58181046/**`、`official-starter/**`（官方 sample agent，12B 模型 + `sub_agents/code_analyzer.yaml`）。

---

## 6. 讨论区证据（topic id + URL + 具体主张）

- **752848** "Why does nobody have a score of over 0.24?" — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/752848
  榜首 Yurnero（`samson8`，显示 competition_ranking=1）留言："这是难任务……**我认为 0.31 相当可达**"；并称"公开任务的 gold patch 并非全部可用，但私有任务的 gold patch 全部正确，因为主办方验证过"。另有用户问 "The guy above got 0.29, what method do you think he used?"，**无人回答**——即 0.29 的做法从未公开。
- **757588** "I reran a public 0.24 bundle unchanged and got 0.17" — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/757588
  原样重跑 `honghanhhh` 的 "public claim 0.24" 得到 **0.17**；`emmanuelmirekumensah` 的 budget-v2 得 0.15；给出 2026-11-10 榜单分布与"1 题 = 0.017、run-to-run sd 约 1.5–3 题"。
- **758969** "Three lessons from 22 submissions" — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/758969
  同一份 Budget-Fit 提交 0.17 → 0.12（仅改注释）；Budget-Aware v2 自测得 0.17/0.17（"public notebook shows 0.22"）；40 calls / 5.5 min 版只得 0.15。**结论：<0.05 的差异视为噪声。**
- **759154** "Testing configs offline on the 129 training tasks" — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/759154
  同一 zip 历史得分 0.12–0.17；5.5 min 比 4 min 只 +1.6 分（95% CI −1.6~+4.7），代码上下文 skill 与自写 lessons 均无收益。
- **746250** "sampling, thinking, time limits, and tool ablations"（49 票）— https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/746250
  129 题本地集：**保留 thinking 与 thought summaries**；temp 0.4 与 1.0 打平（作者保留 1.0 / top_p 0.95 / thinking 4096）；**graph retrieval 无可观测收益**；analyzer 子 agent 无明显增益；工具预算 50→25 少解 14 题。
- **747168** "Stronger models barely beat Gemma 4 31B in this harness"（39 票）— https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747168
  同一 harness 下 Gemma 31B 54/124，DeepSeek V4 Pro 58/124，Qwen3.6-27B 59/123；Gemma 两次相同 pass 有 19/123 题翻转。**说明瓶颈在 harness/题目而非模型选择。**
- **751487** "gold patches pass on 127/129 public tasks (was 52)" — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/751487
  普通 notebook 里 gold patch 只有 52/129 通过（fastapi 10/67）；搭 scorer-like 沙箱后 127/129。**本地评测必须沙箱对齐，否则低分是环境造成的。**
- **745998** 主办方 "Note on Code Graphs for Async Functions" — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/745998
  `async def` 不进代码图；但**私有题库没有任何 async def**（至多 1.8% 函数），故不修——这条与"公开 0.17、私有 0.24"的落差方向一致，值得注意。
- **748051** "The top of the leaderboard is dominated by those who trained LoRA's" — https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748051
  作者称 0.18 墙只能靠 LoRA 突破；但同帖最高票回复反驳："我认为榜首没有一个靠微调拿到提升"。**LoRA 有效性在公开信息中无一致证据**（758968 的实测：LoRA 让离线行为大变，线上仍是 0.12 < 基线 0.15）。

---

## 7. 明确"未找到 / 未验证"清单

- **没有找到任何 0.29 的公开产物**：无 notebook、无 `submission.zip`、无 dataset、无 writeup。榜首 `samson8` 在本赛题下零公开内容（已核对其全部 kernel 与 dataset 列表）。
- **没有找到 0.24 的公开产物**：所有 "LB 0.24 / 024 / 0.24+" 标题的 notebook，Kaggle 记录值均为 0.17–0.18，且无一有第三方复现到 0.24 的记录。
- **没有找到官方"solution/winner/what worked"类帖子**：已遍历本赛题全部 344 个 topic 标题（`list_competition_topics` 分页 18 页，Recent/Top 两种排序）。
- **未取得**：`verracodeguacas/gemma-4-agent-budget-fit-single-agent` 的 zip（签名 URL 返回 403）；`denpugovkin` 的 `score_022_submission.zip` 与 `exp021_runtime_control.zip`（签名 URL 返回 404）。二者分数来自 Kaggle 记录值 0.18 / 0.22，**不是**我本地验证的结果。
- **未验证**：`flexonafft/lean-conte`、`ozertuu`、`honghanhhh` 的最新版本从未在 Kaggle 上评分，任何分数都只是标题宣称。
- **"58 题 / floor(100×k/58)" 刻度**来自参赛者 notebook（`ozertuu`）与多篇讨论的交叉一致陈述，**不是主办方公告**；0.29 = 17 题是我据此推导，未获官方确认。
