# P014 首个冻结候选：p014_single_v1

日期：2026-10-20。构建源 HEAD `a2b1298a894f2f7ac8625b8bd0975c55e1b8f830`。本候选在看到任何 P014 新题模型结果前确定并冻结，尚无提分证据。

## 直接使用

- Agent 根目录：`research/014/candidate/agent/`；根文件 `agent.yaml`。
- 提交结构候选包：`research/014/candidate/candidate.zip`，3089 bytes。
- ZIP SHA256：`17435502e614518e29bd74077e1f1a05515c86708a269375b20b6d9f8c7ab3c9`。
- 配置与逐文件指纹：[manifest.json](manifest.json)。ZIP 只含 `agent.yaml`、`eval_config.yaml`、`configs/sampling.yaml`、`prompts/system.md`，内容与该目录一一对应。
- 固定 31B CT v2、无 adapter、8 分钟 / 28 calls / 80 turns、temperature 0.2、top_p 0.95、8192 输出 / 4096 thinking。配置预算与原样 public022_v10 完全一致。

## 为何选这一版

P013 两轮旧四题没有新增修复；部分题的 repair 吞掉整题时间，review 只剩收尾机会。SequentialAgent 没有硬性阶段时限，继续依赖提示词交接不能保证有效审查。因此首个 B 采用单 Agent，让一个上下文完成定位、生产修改与验收；不在新题上再扫描单/双阶段矩阵。

流程继承 public022_v10 的需求提取、定位契约、精确编辑及 diff 收尾，集中保留 P013 的已知失分修正：

1. 预编辑复现只用于区分具体原因，inline 命令 10 秒超时，相关已有测试 60–120 秒；超时不直接触发撤销补丁。
2. 每次可信 Python 行为检查指向工作区，并在同命令核对模块来源；区分外部依赖缺失与题目要求的缺符号。
3. 每处关联修改前观察源码，集中完成相邻分支、导出/注册或参数透传；避免只改 import 或一直写复现。
4. 只清理由本次运行创建且确认属于临时产物的文件；检查完整 diff，零补丁不能直接连交。

使用六个普通工具；移除 public022 提示中已经禁止调用的三种图检索工具。没有增加角色、skills、预算或新工具。提示词为 4863 bytes，没有写入新题题目、判据或参考补丁。

另已在 `research/013/agent/prompts/review.md` 插入原题变量 `{problem_description}`，并用官方 schema 读取确认。该修正只改变 P013 源码；旧诊断嵌入包仍保留为历史证据。首个 B 本身为单 Agent，不依赖 reviewer。

## 本次手动核对

一次性命令使用官方 `adk_submission.yaml_loader.load_yaml` 与 `SandboxedLlmAgentConfig`，核对 include 在包根内、根配置字段与工具格式；比较采样/预算与 public022 相等，逐 ZIP 成员复算 SHA256 与源文件一致。以上通过。

完整 `compile_submission` 依赖正式 ADK/模型/工具注册表，本机未执行；评测载体必须在官方运行环境中执行。没有新增测试文件，没有模型尝试、GPU 作业或正式提交。修复率、公榜收益及单 Agent 对交接损耗的改善均待真实对照。

指令内容由 public022_v10 与 P013 改写，保留上游 Apache-2.0 许可为 [LICENSE-public022-Apache-2.0.txt](LICENSE-public022-Apache-2.0.txt)。
