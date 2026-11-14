# 公开方案复现、提交与改进判定

日期：2026-11-15。范围：Kaggle `gemma-4-developer-agent`（id 149921）。本文件回答三件事：用了哪个公开方案提交、这次提交的真实预期、以及“自训 LoRA / 改 agent 框架”哪条能提升分数。

---

## 1. 结论

1. **榜首 0.29 不可复现**：team `Yurnero` 即用户 `samson8`，在本赛题 0 个 notebook、0 个 dataset、0 篇 writeup；344 个赛题 topic 无 solution/winner 类文章。0.29 没有任何公开产物。
2. **公开可下载的最高记录是 0.22，不是 0.24**：`flexonafft/gemma-4-budget-aware-coding-agent-v2` v10 的 Kaggle notebook 元数据 `best_public_score = 0.22`，是全部公开 notebook 中唯一一个 0.22。所有标题写 “LB 0.24” 的 notebook（`honghanhhh`、`kawh1111`、`ozertuu`）真实记录只有 0.17–0.18。
3. **0.22 本身是抽样上沿**：同一 `files_sha256 = 56010e9e…` 在第三方普查（274 个公开 notebook / 145 个有分版本）中 6 次抽样为 **0.22、0.17×5**，均值 0.178、中位数 0.17。正确预期是 **0.17–0.18**。
4. **本次提交**：使用 `honghanhhh/gemma-4-baseline-lb-0-24` v8 —— 它是 flexonafft v10 的**字节级配置副本 + 3 处提示词修改**。提交 ref **58350128**，状态 PENDING。预期 0.17–0.18；它的标题 0.24 不成立（记录 0.18，第三方复跑 0.17）。
5. **自训 LoRA：不作为主线**。公开 11 组 adapter 实验 **0 组**给出可确认的 LB 正向；最好一次在噪声内；3 次 LB 0.00。只保留“一次门控旁路”，且优先 E4B 小模型奖。
6. **改 agent 框架：是主线，但只剩尾部项**。已用掉的大杠杆（thinking、thought summaries、单 agent、禁图工具）不要再动；剩下的是 12 小时预算安全、上下文压缩、提交卫生、早编辑硬约束、工具面裁剪，合计预期 **+1~3 题/58**，目标把稳定水平从 0.17 抬到 0.20–0.22。

---

## 2. 本次提交的精确内容

| 项 | 值 |
| --- | --- |
| 来源 | `honghanhhh/gemma-4-baseline-lb-0-24` v8（kf/361393189） |
| 本地文件 | `submission-candidate-0.24/submission.zip` |
| SHA256 | `205a677fdcbd122f50ac3138b1cda0403aeb625453c93edf4c53b7b691926186` |
| 大小 | 3,990 B（解包 7,512 B，4 个文件） |
| 提交 ref | **58350128**（2026-11-15，PENDING） |
| 模型 | `gemma-4-31b-it-qat-w4a16-ct` |
| 预算 | 240 s 命令 / 28 calls / 8 min / 80 turns |
| 采样 | temp 0.2 / top_p 0.95 / 8192 输出 / thinking 4096 |
| 与 0.17 基线差异 | 仅 `prompts/system.md`：保护 `pytest.ini`/`conftest.py`、临时文件只放 `/tmp`、结尾 `git status --short && git diff`、sync/async 等并行实现全覆盖 |
| 分数证据 | 标题宣称 0.24；Kaggle 记录 0.18；第三方原样复跑 0.17（topic 757588） |

校准尺：公开 LB 58 题，`score = floor(100×k/58)/100`，1 题 = 0.0172。0.22 = 13 题，0.24 = 14 题，0.29 = 17 题。

---

## 3. 自训 LoRA 是否值得

**判定：不值得作为冲榜主线；最多做一次严格门控的旁路，且优先小模型奖。**

证据：

- 公开 11 组 31B/12B/E4B adapter 实验，**0 组**可确认正向；最好的是 DPO 本地 +2~3.5pt，LB 9 题 vs 8 题（CI −2.7~+4.7，等于噪声）。
- 3 次 LB 0.00：其中一次 `alpha/r = 2.0` 导致复读死循环；rank32 一次只有 0.06；“习惯 adapter” 0.12 低于无 adapter 基线 0.15/0.13。
- 机制上限：同 harness 下 DeepSeek V4 Pro / Qwen3.6-27B 只比 Gemma 多 4~5 题，48/123 题任何模型都解不出；adapter 还要交约 29% 时长税、45% token/step 税，8 min cap 下更容易撞 12 小时总时限。
- 计量现实：单次 LB 的 run-to-run σ = 1.5~3 题；决赛只跑一次。小于约 4 题（≈0.07）的改动既无法验证也无法兑现。
- 成本：一次可信的 31B 实验（训练 + 两臂全量评测）约 20~40 GPU-h，等于 3~4 轮框架 A/B。
- 本仓库现状：只有 rank 128 / alpha 256 的 `adapter_config.json` 空壳，**没有任何已训练权重**；走 31B LoRA 必须从零开 GPU。

若仍要做，硬门槛：训练目标必须含 served thought 原文；`alpha = r`，rank ≤ 32；加载后解码减速 ≤ 10%；本地 90 题配对 **≥ +4 题**；否则不占最终名额。更划算的落点是 **E4B/E2B 小模型奖（$10k，公开最好仅 0.06）**，约 16 L4-h 可试。

---

## 4. 修改 agent 框架是否值得

**判定：值得，且是唯一主线；但不要再改已被证伪的开关。**

已经用掉、不要回退的杠杆：

- thinking 关闭或 `include_thoughts:false`：−7~−11 题/58，超时 3→15。
- 多 agent / analyzer 编排：+1/129，在噪声内。
- 代码图工具：语义检索 21/21 为空，async 函数不在图里；当前 prompt 禁用它们是正确做法。
- 提高工具 cap 而不降时间 cap：8 min × 120 题 = 12.4~14.2 h，**整份提交会因 12 小时总时限报错**；历史 submission 57438326 就是 `COMPLETE` 但 `public_score` 为空、`error_description` 为 runtime exceeded。

仍然值得做的（按证据强度）：

| 优先级 | 改动 | 预期 | 验证方式 |
| --- | --- | --- | --- |
| P1 | 预算重排：8 min/28c → **6 min/34c/thinking 2048**，最坏投影 ≤ 11 h | 超时 −50%，解决数 ≥ 现状 −1 题 | 129 题本地配对 |
| P1 | 提交卫生：禁改测试文件、临时文件只放 `/tmp`、结尾 `git status --short && git diff` | 测试冲突 ≤ 1，空补丁 −50% | 本地金标/空补丁 |
| P2 | 早编辑硬约束（第 8 次调用必须动手、第 20 次停止探索）+ 14k token 上下文压缩 | +1~2 题 | 129 题配对 |
| P3 | 工具面 9 → 6（删 3 个图工具） | 持平或略升，调用数下降 | 8 GPU-h A/B |

合计现实目标：本地 129 题 +3~6 题（≈ +1~3 题/58），LB 稳定到 **0.20–0.22**；0.24+ 需要运气或一次成功的窄域 adapter。

---

## 5. 下一步（按成本排序）

1. **P0 本地口径校准**（2~3 CPU-h + 6 GPU-h）：搭 scorer-like 沙箱，金标应 ≥125/129、空补丁 ≤2/129；用 58350128 的 zip 与 Budget-Fit 各复跑一次，建立“本地→LB”映射。
2. **P1 打包审计（0 GPU）**：单一 `model:`、`agent.yaml` 根名、扩展名白名单、3 GiB、adapter 命名；按 6 min×120 题重算 12 h 投影。
3. **P2 自身噪声标定**：同一 zip 跑 3 次 60 题分层面板，得到 σ；σ>3 题则后续 <4 题的结论全部不采信。
4. **P3 预算/卫生 A/B**：见上表；只有本地配对 ≥ +2 题才进入下一次提交。
5. **P6 E4B 小模型奖**（并行彩票）：E4B + r16 LoRA，16 L4-h；本地 31 题 ≥ 基线 +2 且补丁率 ≥95% 才提交。

---

## 6. 证据与文件

- `FINDINGS.md`：榜单、候选排序、0.29 无产物的核查、普查哈希。
- `SUBMISSION-CONTRACT.md`：官方 schema/限制/打包审计命令/异常坑。
- `IMPROVEMENT-ANALYSIS.md`：11 组 adapter 实验、框架证据表、P0~P8 完整计划、12 条“不要做”。
- `evidence__public-config-draws/census_excerpt_ge_0.17.csv`：0.17 以上全部抽样行（含文件哈希）。
- 原始包：`flexonafft__gemma-4-budget-aware-coding-agent-v2/`、`honghanhhh__gemma-4-baseline-lb-0-24/`、`ozertuu__gemma-4-gold-submit/`、`denpugovkin__gemma-4-repair-lab-compare-before-you-submit/`。

关键外部来源：LB 读数（2026-11-15）；topic 743506（58 题）；topic 757588（0.24 复跑 0.17）；topic 743063（12 h 与私有集）；topic 746250（采样/思考/工具消融）；topic 747168（教师对比）；topic 751104（小模型奖）；`dariushafshar/gemma-4-agent-public-config-draws`（普查）。
