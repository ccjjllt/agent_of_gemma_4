# Gemma 4 Developer Agent：从官方评分重新开始

2026-10-24 按用户要求重启研究。当前只研究两件事：官方到底如何评分，以及 129 个公共开发任务需要哪些可迁移的修复能力。

当前入口：[官方评分与任务机制分析](competition/README.md)。逐题依据与分类见 [129 题目录](competition/task-catalog.md)；原始来源及文件指纹见 [来源清单](competition/sources/provenance.json)。

2026-10-28正在进行：[VFARL算法与固定100题深审](competition/vfarl/l4x4/round100/optimization-overnight/README.md)。真实采样与两次单步更新证据已经保留；当前先修复算法/数据缺陷，正式GPU训练关闭，独立修复收益仍待验证。

旧研究结论不再作为当前决策前提。已删除 `docs/plan/`、`docs/progress/` 和早期 `agent/` 中 41 个文件；这些文件在 Git 提交 `984f2b3` 中可恢复。真实实验结果、失败记录、提交回执、训练产物和依赖缓存仍保存在历史目录，防止丢失事实或重复花费下载/GPU 配额。

遵守 [AGENTS.md](AGENTS.md)：不新增测试文件，只做必要的一次性手动核对；不读取官方隐藏评测内容；不自动启动 GPU。
