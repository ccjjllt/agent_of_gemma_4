# Gemma 4 软件修复 Agent 研究

围绕比赛固定的 Gemma 4 31B CT 模型，研究仓库级软件修复、Agent 行为和同协议后训练。先从本页进入，再按研究编号查计划、代码和实际证据。

## 当前进展

截至仓库最新记录（2026-10-22）：

- **正式最好成绩为 0.15**。最新 runtime-v2 提交 `57471497` 已 COMPLETE，与旧最好持平；[官方回执](research/014/submission/runtime-v2/README.md)。
- **当前主线为 P015**：31B 后训练、同一 Agent 的权重 A/B，以及失败反馈纠正；[执行记录](research/015/README.md)、[研究计划](docs/plan/015-31b-posttraining-and-score-handoff.md)。
- 已完成 31B FSDP 三次真实更新、独立 adapter 导出和 CT 请求路由确认；新开发面板 base 为 **5/12**。首轮 action RFT 仅含 **3 个独立问题、14 条动作、1,014 个监督 token**，已推送任务；最新记录尚待训练回执，A/B 尚未启动，不能据此宣称修复收益。

这里汇总的是仓库已保存的事实；作业与配额的最新状态以对应轮次原始记录为准。

## 目录入口

| 目录 | 用途与入口 |
| --- | --- |
| [research/](research/README.md) | 按轮次保存研究实现、来源、冻结清单、实验结果与提交回执；当前看 `015/` |
| [training/](training/README.md) | SFT、DDP/FSDP、编码与导出代码；`configs/` 为配方，`artifacts/` 为本地数据产物 |
| [notebooks/](notebooks/README.md) | Kaggle 历史载体按轮次归类，`builders/` 集中存放生成器 |
| [docs/](docs/README.md) | 研究计划、进展记录及本次目录整理说明 |
| [agent/](agent/agent.yaml) | 早期单 Agent 基线配置；最新正式冻结方案见 P014，P015 协议见 [protocol.json](research/015/protocol.json) |
| `results/` | 本地保存的早期真实运行结果、轨迹和补丁，虽然被 Git 忽略仍需保留 |
| `.venv/` | 项目本地 Python 环境 |

## 工作约定

遵守 [AGENTS.md](AGENTS.md)：专注机制证据、有效数据和真实修复收益；不新增测试文件或套件，只做与改动相关的一次性手动核对。不读取 heldout 内容，不将 select、dev 或预留题转作训练；GPU 按会话授权和预算运行。

研究目录保留原编号和证据路径。大文件、依赖缓存与数据常被 Git 忽略，**是否被忽略不代表是否可删除**。本次清理范围、恢复来源和 Notebook 旧路径映射见 [目录整理记录](docs/project-layout.md)。
