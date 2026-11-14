# Gemma 4 Developer Agent — Submission Branch

`main` 只用于构建和提交 Kaggle submission，不保存训练、研究或实验记录。训练与研究代码保留在 `codex/training-plan-refactor` 分支。

## 目录

- [submission/](submission/README.md)：当前 Kaggle 提交（ref `58350128`）的 agent bundle 与来源指纹。
- [research/competition-intel/](research/competition-intel/SOLUTION-ANALYSIS.md)：公开方案调研、提交契约、LoRA 与 agent 框架改进判定。
- [official/](official/README.md)：官方接口说明、模型配置与依赖缓存，附来源和文件指纹。
- [patches/](patches/adk-1.36.1-tool-events.patch)：host runtime 的 ADK 事件摘要修复。
- [AGENTS.md](AGENTS.md)：项目工作约定；研究约定仅适用于 `codex/training-plan-refactor` 分支。

## Submission 约束

以 `official/HARNESS_README.md` 为准，关键边界包括：

- 提交是声明式 YAML 目录，不提交 `agent.py` / Python 入口；根配置只使用 `agent.yaml`（官方代码实际只识别 `agent.yaml` / `root_agent.yaml`，`.yml` 不被识别）。
- 全部 agent 只能声明一个允许的 base model，并从官方四个量化模型中选择。
- 解压后总大小必须小于 3 GiB（含 `adapters/`）；LoRA rank 上限 128。
- 路径不得包含绝对路径、`..`、空字节或指向根目录外的 symlink。
- 评测题、隐藏任务和官方私有资产不得写入 submission。

## 当前状态

`main` 不包含训练代码、训练数据、实验日志或 adapter 权重；这些资产不随本分支提交。GPU 训练与模型研究不在本分支进行。

当前 submission 为 `honghanhhh/gemma-4-baseline-lb-0-24` v8（flexonafft v10 的提示词分支），Kaggle ref `58350128`，状态以 Kaggle 为准；按公开记录其真实水平为 **0.17–0.18**，不是标题宣称的 0.24。详细证据与改进判定见本分支 `research/competition-intel/`。

比赛入口：[Gemma 4 Developer Agent Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent)。
