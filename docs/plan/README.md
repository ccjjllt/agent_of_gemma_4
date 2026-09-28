# 首批研究计划

资料核对日期：2026-09-28。

本项目研究指定 Gemma 4 模型在仓库级软件修复任务中的行为、代码定位效率和时间预算利用。当前已完成官方资料读取与公开任务清单分析，尚未实现研究代码，也没有模型运行或微调结果。

## 研究顺序

| 轮次 | 计划 | 核心产出 |
| --- | --- | --- |
| 001 | [无微调基线与失败分析](001-baseline.md) | 可复现的单 Agent 配置、固定任务划分、完整轨迹、环境与模型失败分类 |
| 002 | [代码定位与图检索](002-code-localization.md) | 文本定位、符号相似检索、图邻域扩展的同条件对照 |
| 003 | [上下文与时间预算](003-budget-and-context.md) | 读取策略与单题预算对解决率、耗时的影响 |
| 004 | [针对失败行为的 LoRA](004-targeted-lora.md) | 与比赛基座兼容的 adapter，以及针对单一行为缺陷的训练前后对照 |

先实现 001。002、003 复用其任务划分、环境和日志；004 的训练目标由实际失败决定。实现与运行结果分别回填到 `docs/progress/` 中同编号、同主题文件，届时再创建对应结果文档。

## 已读取的资料

- Kaggle 官方 MCP：比赛元数据、全部内容页、数据文件清单第一页、公开讨论、官方 Getting Started Notebook v2 源码、当前账号加速器配额。
- 官方 `HARNESS_README.md`：49,356 字节，SHA-256 `3d6e57a13234cb4e783ba24eaab486459af76e0923ea4c6607cd41cc8961bbbb`。
- 官方 `tasks.jsonl`：1,984,455 字节，SHA-256 `e4b3fd60f69dbc2b9213e54eeb9636db78aefe92c1d06269d73d9f5f8f3c8ad6`。
- 实际任务记录：129 条，127 个唯一 `(repo, base_commit)`；FastAPI 67、Rich 48、Requests 13、HTTPX 1。

文件本次仅下载到临时研究目录，未将任务答案、训练数据或官方 Notebook 复制进项目。实现时从官方资源重新取得，并比较上述版本记录；若资料更新，记录变动后采用新版本。

## 会影响实验的当前信息

- 比赛只支持 `gemma-4-31b-it-qat-w4a16-ct`。官方 Notebook v2 使用模型资源 `google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2`。
- 官方说明中的评分环境是 4 × NVIDIA L4、32,768 token 上下文。这不表示个人 Notebook 一定能分配相同资源，也不是训练显存承诺。
- 主办方 9 月 26 日表示空 embedding 文件已修复，缺失的 `async` 函数尚未补齐；重新下载后仍需检查实际版本。[图数据讨论](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/742911)
- 主办方 9 月 27 日表示公开 Notebook 的 subprocess 沙箱可能有问题，将改进可靠性；它与正式 Docker 评分环境不能直接视为等价。[运行环境讨论](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/742882)
- 闭源教师模型蒸馏的专项讨论中，当前读取到的主办方答复仍是等待进一步说明；一般外部数据条款不能代替该具体问题的结论。[蒸馏讨论](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/742807)

## 官方入口

- [比赛概览](https://www.kaggle.com/competitions/gemma-4-developer-agent)
- [比赛数据与 HARNESS_README.md](https://www.kaggle.com/competitions/gemma-4-developer-agent/data)
- [官方 Getting Started](https://www.kaggle.com/code/ryanholbrook/getting-started-gemma-4-developer-agent)
- [官方评估依赖包](https://www.kaggle.com/datasets/metric/gemma-4-developer-agent-wheelhouse)
- [比赛规则](https://www.kaggle.com/competitions/gemma-4-developer-agent/rules)
