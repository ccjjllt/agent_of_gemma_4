# Gemma 4 Developer Agent

单一路线：Gemma 4 31B BF16 + LoRA Rank 128 / Alpha 256 + Single-stream SPO + 异步 Agent RL。推理走 SGLang，同机双 A800 模型并行。不做算法对比，不降 Rank。

实施入口：[双 A800 训练与运行说明](training/README.md)。首轮只跑在线 SPO；SFT 冷启动、教师蒸馏、OPD 已实现（`training/data/`）但明确后置，待首次真实 SPO 更新证明循环成立后再接入。数据与模型在服务器手动获取。

方案细节：[SPO 异步训练设计](research/spo-async-training-design.md)。当前配置：[configs/spo_a800_2x80.toml](configs/spo_a800_2x80.toml)。

当前目录：

- [AGENTS.md](AGENTS.md)：项目工作约定。
- [official/](official/README.md)：少量官方原始说明、模型配置与依赖缓存，附来源和文件指纹。
- [scripts/](scripts/)：服务器环境安装；数据与模型手动获取。
- [training/](training/)：SPO 训练链（调度、执行、奖励、更新、adapter 管理）；旧 GRPO 文件已从本分支删除，历史可查 Git 记录。
- [docs/plan/007-training-throughput-optimization.md](docs/plan/007-training-throughput-optimization.md)：Checkpoint、KL 前向复用与 Prefix Tree 吞吐优化记录。
- [requirements/](requirements/)：学习器候选依赖及已解析的版本清单。
- `.venv/`：保留的本地 Python 环境。
- `.git/`：保留现有 Git 历史；已提交的旧内容仍可从历史中查阅。

训练数据在运行时直接挂载比赛官方公开集，不在本地下载、复制或另建训练数据包。GPU 实验按新的具体授权与预算执行；没有启动训练或评测，也没有操作远程资源。

比赛入口：[Gemma 4 Developer Agent Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent)。
