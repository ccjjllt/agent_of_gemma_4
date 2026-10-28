# Gemma 4 Developer Agent

2026-10-28，按用户要求清空此前研究，重新开始。旧研究文档、训练与采样实现、自建数据、实验结果和提交载体已从工作目录删除。

新的研究入口：[官方31B CT基座的RL adapter路线研究](research/rl-adapter-feasibility.md)。包含基座对齐、开源实现审计、训练目标、租卡预算与待实测边界；尚未租卡或实现新的训练流水线。

当前目录：

- [AGENTS.md](AGENTS.md)：项目工作约定。
- [official/](official/README.md)：少量官方原始说明、模型配置与依赖缓存，附来源和文件指纹。
- `.venv/`：保留的本地 Python 环境。
- `.git/`：保留现有 Git 历史；已提交的旧内容仍可从历史中查阅。

训练数据在运行时直接挂载比赛官方公开集，不在本地下载、复制或另建训练数据包。GPU 实验按新的具体授权与预算执行；本次清理没有启动训练或评测，也没有操作远程资源。

比赛入口：[Gemma 4 Developer Agent Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent)。
