# Gemma 4 软件修复 Agent 研究

围绕比赛固定的 Gemma 4 31B CT 模型，研究仓库级软件修复、Agent 行为和同协议后训练。先从本页进入，再按研究编号查计划、代码和实际证据。

## 当前进展

截至仓库最新记录（2026-10-24）：

- **正式最好成绩为 0.15**。最新 runtime-v2 提交 `57471497` 已 COMPLETE，与旧最好持平；[官方回执](research/014/submission/runtime-v2/README.md)。
- **P016首轮主训练已完成**：4×A40上完成 **537个完整问题、68仓、68/68次更新、1 epoch**，共7,119,739输入/1,291,819监督tokens，全部≤16K；2026-10-23 17:40正常退出，最终adapter和七个完整checkpoint已备份至Drive。真实断点恢复已在A40验证；[执行记录](research/016/README.md)、[最终摘要](research/016/training/main-short-training/final-result-summary.json)。原677题版本是另一数据版本，不能当成本次实际训练量。
- **下一阶段为P017，方案已形成、尚未执行**：[Kaggle权重迁移与提分交接](docs/plan/017-kaggle-weight-transfer-and-score-handoff.md)。固定runtime-v2做同期Base／Step-68修复比较，再用训练外新面板确认，按工具协议、NF4→CT、验证恢复及耗时证据决定后续改进。最终adapter尚未在官方运行时加载，无新增修复或正式提分证据。
- P015 已完成 31B FSDP 三次真实容量更新、独立 adapter 导出和 CT 请求路由；开发面板 base 为 **5/12**。后续三题 action RFT 作业现已回收，状态为取消，只确认第一步更新，未见最终导出；[真实回收记录](research/016/p015-recovery/actual-result.json)。权重 A/B 尚无结果，不能据此宣称修复收益。

这里汇总的是仓库已保存的事实；作业与配额的最新状态以对应轮次原始记录为准。

## 目录入口

| 目录 | 用途与入口 |
| --- | --- |
| [research/](research/README.md) | 按轮次保存研究实现、来源、冻结清单、实验结果与提交回执；当前看 `016/` |
| [training/](training/README.md) | SFT、DDP/FSDP、编码与导出代码；`configs/` 为配方，`artifacts/` 为本地数据产物 |
| [notebooks/](notebooks/README.md) | Kaggle 历史载体按轮次归类，`builders/` 集中存放生成器 |
| [docs/](docs/README.md) | 研究计划、进展记录及本次目录整理说明 |
| [agent/](agent/agent.yaml) | 早期单 Agent 基线配置；最新正式冻结方案见 P014，P015 协议见 [protocol.json](research/015/protocol.json) |
| `results/` | 本地保存的早期真实运行结果、轨迹和补丁，虽然被 Git 忽略仍需保留 |
| `.venv/` | 项目本地 Python 环境 |

## 工作约定

遵守 [AGENTS.md](AGENTS.md)：专注机制证据、有效数据和真实修复收益；不新增测试文件或套件，只做与改动相关的一次性手动核对。不读取 heldout 内容，不将 select、dev 或预留题转作训练；GPU 按会话授权和预算运行。

研究目录保留原编号和证据路径。大文件、依赖缓存与数据常被 Git 忽略，**是否被忽略不代表是否可删除**。本次清理范围、恢复来源和 Notebook 旧路径映射见 [目录整理记录](docs/project-layout.md)。
