# Gemma 4 Developer Agent

2026-10-28，按用户要求清空此前研究，重新开始。旧研究文档、训练与采样实现、自建数据、实验结果和提交载体已从工作目录删除。

实施入口：[AutoDL 双 A800 的训练准备与运行说明](training/README.md)。当前采用 Unsloth BF16 LoRA＋TRL GRPO，包含官方任务验收、复合奖励、动态采题、同批轨迹复用和 adapter 导出。数据与模型在服务器手动获取；GPU兼容性、32K容量和实际收益尚待接入机器后验证。

方案概览：[当前 GRPO 方案](research/rl-adapter-feasibility.md)。集中记录当前算法、奖励、采题策略和运行配置。

当前目录：

- [AGENTS.md](AGENTS.md)：项目工作约定。
- [official/](official/README.md)：少量官方原始说明、模型配置与依赖缓存，附来源和文件指纹。
- [scripts/](scripts/)：服务器环境安装；数据与模型手动获取。
- [training/](training/)：模型服务、官方任务进程与 LoRA 更新程序。
- [configs/a800_2x80.toml](configs/a800_2x80.toml)：每题8条、目标每批4题、同批最多2次更新。
- [requirements/](requirements/)：学习器候选依赖及已解析的版本清单。
- `.venv/`：保留的本地 Python 环境。
- `.git/`：保留现有 Git 历史；已提交的旧内容仍可从历史中查阅。

训练数据在运行时直接挂载比赛官方公开集，不在本地下载、复制或另建训练数据包。GPU 实验按新的具体授权与预算执行；本次清理没有启动训练或评测，也没有操作远程资源。

比赛入口：[Gemma 4 Developer Agent Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent)。
