# Gemma 4 Developer Agent — Submission Branch

`main` 只用于构建和提交 Kaggle submission，不保存训练、研究或实验记录。训练与研究代码保留在 `codex/training-plan-refactor` 分支。

## 目录

- [submission/](submission/README.md)：submission 根目录占位；实际提交内容按官方 `adk-submission` 契约放入。
- [official/](official/README.md)：官方接口说明、模型配置与依赖缓存，附来源和文件指纹。
- [patches/](patches/adk-1.36.1-tool-events.patch)：host runtime 的 ADK 事件摘要修复。
- [AGENTS.md](AGENTS.md)：项目工作约定；研究约定仅适用于 `codex/training-plan-refactor` 分支。

## Submission 约束

以 `official/HARNESS_README.md` 为准，关键边界包括：

- 提交是声明式 YAML 目录，不提交 `agent.py` / Python 入口；根配置为 `agent.yaml`、`agent.yml`、`root_agent.yaml` 或 `root_agent.yml` 之一。
- 全部 agent 只能声明一个允许的 base model，并从官方四个量化模型中选择。
- 解压后总大小必须小于 3 GiB（含 `adapters/`）；LoRA rank 上限 128。
- 路径不得包含绝对路径、`..`、空字节或指向根目录外的 symlink。
- 评测题、隐藏任务和官方私有资产不得写入 submission。

## 当前状态

`main` 不包含训练代码、训练数据、实验日志或 adapter 权重；这些资产不随本分支提交。GPU 训练与模型研究不在本分支进行。

比赛入口：[Gemma 4 Developer Agent Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent)。
