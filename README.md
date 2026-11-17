# Gemma 4 Developer Agent

单一路线：Gemma 4 31B BF16 + LoRA Rank 128 / Alpha 256 + Single-stream SPO + 异步 Agent RL。推理走 vLLM，同机双 A800 模型并行；vLLM 使用独立 Python 环境，不并入 Unsloth learner。不做算法对比，不降 Rank。

实施入口：[双 A800 训练与运行说明](training/README.md)。首轮只跑在线 SPO；SFT 冷启动、教师蒸馏、OPD 已实现（`training/data/`）但明确后置，仍未接入正式训练入口。数据与模型在服务器手动获取。

方案细节：[SPO 异步训练设计](research/spo-async-training-design.md)、[运行时与 checkpoint 契约](docs/plan/009-cpu-acceptance-reward-feedback.md)。当前配置：[configs/spo_a800_2x80.toml](configs/spo_a800_2x80.toml)；推理环境：[requirements/inference-vllm.lock.txt](requirements/inference-vllm.lock.txt)。

当前目录：

- [AGENTS.md](AGENTS.md)：项目工作约定。
- [official/](official/README.md)：少量官方原始说明、模型配置与依赖缓存，附来源和文件指纹。
- [scripts/](scripts/)：服务器环境安装；数据与模型手动获取。
- [training/](training/)：SPO 训练链（调度、执行、奖励、更新、adapter 管理）；旧 GRPO 文件已从本分支删除，历史可查 Git 记录。
- [docs/plan/009-cpu-acceptance-reward-feedback.md](docs/plan/009-cpu-acceptance-reward-feedback.md)：CPU 环境/验收反馈、五步 run 结果、运行时与 checkpoint 契约（已合并旧 007/008）。
- [docs/plan/011-adapter-evaluation.md](docs/plan/011-adapter-evaluation.md)：已发布 adapter 的独立 rollout+裁判评估入口（base/任意版本）。
- [docs/plan/013-long-run-plan.md](docs/plan/013-long-run-plan.md)：in-run retention、更广题池长跑计划、从已有 adapter 继续训练。
- [requirements/](requirements/)：学习器与独立 vLLM 推理环境的依赖清单。
- `.venv/`：保留的本地 Python 环境。
- `.git/`：保留现有 Git 历史；已提交的旧内容仍可从历史中查阅。

训练数据在运行时直接挂载比赛官方公开集，不在本地下载、复制或另建训练数据包。2026-11-15已按授权完成首次真实SPO更新、v1发布及生成核对；实验进程已停止。尚无修复收益证据，非零adapter的logprob重复性有小幅波动，详见[五步 run 与实验记录](docs/plan/009-cpu-acceptance-reward-feedback.md)。后续GPU工作仍按具体授权与预算执行。

2026-11-17 完成 rank-128 五步 SPO run（5/5 更新、最终 v5），但同题单样本对照未显示可测改善；2026-11-18 补齐任意已发布版本的独立评估入口、运行中 adapter 保留（含硬链接发布）和从 v5 继续训练的入口，文档合并为 009/011/013。真实 GPU 54 条评估、长训与 v5 continuation 尚未启动。

同日 CPU 学习信号研究已完成 129 道公开题的空补丁/参考补丁对照，以及 10 份历史模型补丁重验。健康空基线仅 63 题；P 有粒度遗漏，当前 G 不可测。真实候选未提供可用的正修复信号，部分验收破坏也未被标量奖励区分；首次更新的同奖励优势差异来自价值历史。下一步先完善 CPU 环境与失败反馈，再进行短 GPU 采样，暂不直接扩大训练。关键结论和条件耗时估计见[学习信号研究](research/spo-async-training-design.md#17-cpu-学习信号研究2026-11-15)。

2026-11-15 续：CPU 验收环境已改为按仓库版本约束选择离线 wheel、执行 `-r/-c` include、按 `Requires-Python` 过滤并补充可离线满足的 extras/测试依赖；候选导致的测试收集/导入失败现在 `validity=0`（0.15→0.10，仍 eligible），奖励版本升为 `official_differential_spo_v2`；exact 节点 `shadow_regression`/`shadow_repair` 只进裁判侧 audit（正式训练写入 ledger audit event），不替换官方 `G`。目标集为 51 道旧不健康题 + 7 道旧健康对照；58/58 完成但跨 3 个环境身份，最终身份直接验证 46 道（27 健康、26 恢复、19 不健康），全集健康估计 63→95 是跨阶段估计。3 份真实候选已在同一最终环境下重验；新增 `require_positive_signal`，无 P>0 且无 shadow repair 时不执行优化步。详见[验收环境与奖励反馈修正](docs/plan/009-cpu-acceptance-reward-feedback.md)。仍未取得模型修复收益证据。

比赛入口：[Gemma 4 Developer Agent Competition](https://www.kaggle.com/competitions/gemma-4-developer-agent)。
