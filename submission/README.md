# Submission

当前 `submission/` 是 2026-11-15 实际提交到 Kaggle 的 agent bundle（解包形态）。打包成 zip 后根目录必须直接包含 `agent.yaml`。

## 当前提交

| 项 | 值 |
| --- | --- |
| Kaggle ref | `58350128`（`PENDING`） |
| zip SHA256 | `205a677fdcbd122f50ac3138b1cda0403aeb625453c93edf4c53b7b691926186` |
| 来源 | `honghanhhh/gemma-4-baseline-lb-0-24` v8 |
| 模型 | `gemma-4-31b-it-qat-w4a16-ct` |
| 预算 | 240 s / 28 calls / 8 min / 80 turns |
| 采样 | temp 0.2 / top_p 0.95 / 8192 输出 / thinking 4096 |

该 bundle 的 `agent.yaml`、`configs/sampling.yaml`、`eval_config.yaml` 与 `flexonafft/gemma-4-budget-aware-coding-agent-v2` v10 字节级一致，仅 `prompts/system.md` 不同（测试文件保护、`/tmp` 临时文件、`git status --short && git diff`、并行实现全覆盖）。

## 真实预期

来源标题写 “LB 0.24”，但 Kaggle 记录只有 0.18，第三方原样复跑为 0.17；同一 flexonafft 哈希 6 次抽样为 0.22 + 0.17×5。规划基线按 **0.17–0.18**，不要按 0.24。

已知风险：8 min × 约 120 道隐藏题 ≈ 12.4–14.2 h，超过官方 12 h 总时限；历史 submission `57438326` 曾因此出现 `public_score` 为空。下一版应改为 6 min/task。

## 复现打包

```bash
cd submission && zip -r ../submission.zip agent.yaml configs eval_config.yaml prompts
sha256sum ../submission.zip   # 应匹配 provenance.json
```

不要把 `submission.zip` 自身、`.DS_Store`、`__MACOSX` 或任何测试文件放进包内；根配置只保留 `agent.yaml`。

完整调研与改进判定见本分支 `research/competition-intel/`。
