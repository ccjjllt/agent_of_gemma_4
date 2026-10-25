# F600 题目与冷启动数据

2026-10-26。已完成 **600 / 600 题：400 共同＋200 F 专科**，对应 **5,083 条真实助手决策、664,354 个监督 tokens**。题面、六工具重放、完整官方修后验收和 Gemma CT v2 对话/工具调用导出均已完成；未启动模型采样或 GPU 训练。

| F 专科机制 | 已准备题数 |
| --- | ---: |
| F1 参数与上下文传播 | 83 |
| F2 身份、引用与所有权 | 21 |
| F3 缓存、失效与重用 | 24 |
| F4 资源与退出 | 26 |
| F5 事件、协议与恢复时序 | 46 |

每类 40 是搜索参考，不是准入配额。按真实补丁改变的能力归类；当前 F2/F3 少于参考值、F1/F5 更多，未用关键字或缓存类名补足数字，也未设训练混合权重。

共同课逐字节复用已验收的 S_common 400，保留真实 G/S/F/R 标签，不按主责 F 再次分配。F 专科与共同课、S 的 200 专科均无题号重合。S600 与 F600 合计 800 个不同训练任务，不把共享 400 重复算成新增。

## 可审阅与使用的文件

- [课程清单、分课数量与哈希](/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/competition/vfarl/f600/course-manifest.json)
- [共同 400 题题面](/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/competition/vfarl/replay/artifacts/f600/F_common_questions.jsonl)、[F 专科 200 题题面](/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/competition/vfarl/replay/artifacts/f600/F_specialist_questions.jsonl)
- [共同课动作](/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/competition/vfarl/replay/artifacts/f600/F_common.jsonl)、[专科动作](/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/competition/vfarl/replay/artifacts/f600/F_specialist.jsonl)，格式为 `gemma-human-action-sft-v2`，含原 token 与助手损失掩码。
- [逐题官方工具及完整验收证据](/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/competition/vfarl/f600/canonical-results-manifest.json)、[Gemma 导出清单](/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/competition/vfarl/replay/gemma-f-manifest.json)
- [最终 200 题选择理由](/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/competition/vfarl/f600/specialist-selection.json)、[205 个已完成 F 训练候选](/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/competition/vfarl/f600/reviewed-specialist-pool-v2.json)

每条题卡绑定真实官方初始提示、任务来源、审阅哈希和实际运行记录。策略输入只使用 `official_initial_user_prompt`；外层机制标签、参考来源及事后结果不是策略提示。另有 5 道完整通过的 F1 训练候选未放入首轮 200 专科，保留供后续扩展，属于公开 train，不是预留评测题。

## 本轮修正

旧 `preview-120.json` 和各批次旧的“40 题完成”说明属于历史预览，不能代表最终已验收数量。复核剔除了机制不符、同一缺陷重复计数、仅有准备目录、原官方验证失败等记录。旧来源、旧审阅和真实失败记录均保留。

修复了 F 导出器的主责标签回退分配：F 的共同课现在只取冻结的共同 400 名单，专科必须显式入选。新增单独的 F 导出流程，仅打开所选 train 报告，避免重写 S600。规范化导出的机制以最新哈希绑定的来源审阅为准，旧报告机制另行保留。

错误函数名和 family 已按实际补丁纠正。比如 Arrow 修复的是 `format`，不是 `__format__`；部分 DVC 的缓存方法标签也已纠正。采集器同时修复了 EOF 换行序列化、纯空白 hunk 的作用域识别、根目录原生验证资产恢复及显式题面版本支持。已有 EOF 不一致的三个来源另做真实修正重放，没有只改成功标记。

归类以因果修复为准：TTL/scoped/capacity 仅在调用边被改写的任务归 F1；缓存队列/淘汰/失效协调才归 F3。仅更改映射槽位、容器形状、Python 上下文返回协议或额外 NUL payload 的 5 个成功来源退出 F；成功本身不等于适合 F 专科。

全部候选经真实源码快照与提交补丁重建，包含嵌套函数；同作用域相同 before→after AST 的重复修复未入选。9 组函数重合具有不同已审核心契约或共享恢复组成项，保留 snapshot 与显式关联审阅，不能声称补丁完全独立，未创建随机评测划分。详见 [全池作用域复核](/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/competition/vfarl/acquisition/f600-review-120/final-pool-semantic-dedup-v1.json)、[机制边界独立复核](/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/competition/vfarl/acquisition/f600-review-120/independent-f3-resource-review-v3.json)。

## 官方对齐与实际边界

600 题来自 70 个仓库，原始物理 split 均为 train：SWE-smith-py 565、SWE-rebench-V2 21、SWE-Gym 14。四个比赛仓库未入池，未读 heldout/129 官方任务正文。

600 题均保留原验证节点身份并通过原官方完整文件范围判定，未新增匹配器例外；共同课继承的 2 个已审原 F2P 标签噪声 profile 继续单独披露，其余 598 题原 F2P 全部实际失败过。564 题保留发布者原题面，36 题使用明确版本化、实际按该版本执行的行为澄清题面。

全部为 Codex 编排并审阅的实际 off-policy 冷启动示范，当前模型 rollout、同状态已验证关键动作对均为 0。历史导出保持 `thinking_budget=0` 和真实空 thought 前缀，未编造推理。用户选定的 thinking 1024–4096 策略用于后续真实模型采样，不能靠修改标签把本批变成 thinking-on 轨迹。

最长完整动作上下文 **16,971 tokens**，最长生成动作 **2,653 tokens**，均满足 32K 上下文与单次 8192 输出。F 的训练窗口需覆盖实际完整上下文；原 S 专用的 12288 候选窗口不能直接用于裁剪 F 数据。

本批使用官方 `SubprocessManager` 的本地 macOS CPU 后端，尚未证明比赛 Docker/Linux 等价。真实六工具、原验收和模板对齐已完成；GPU 训练、训练加载器接入及模型策略失败/恢复分支采集属于后续工作。
