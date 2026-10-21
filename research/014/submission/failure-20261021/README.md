# 57438326 整体超时：定位与最小修复

2026-10-21查询完整官方API：`status=COMPLETE`，public/private score均为空，`error_description` 为整体运行超时。CLI默认提交列表没有输出这个字段，旧“COMPLETE即成功”的查询说明存在缺陷，已经修正。旧最好0.15仍有效。本次没有读取heldout内容。

当前官方Evaluation页经API核对：所有任务提交补丁合计上限12小时，包含sandbox setup，排除patch validation。原件见[页面快照](pages.json)。未取得失败会话的逐题日志，因此不能断言是哪一题、哪一次命令导致最终超时，也不能把开发八题4并发耗时直接外推成正式串行总时长。

同一公开页面的data-description说明评测约120题（仅公开规模说明，没有读取任何hidden题内容）。若每题都跑满8分钟，仅Agent循环最坏就达16小时，已超过12小时；5分钟对应约10小时，为setup/加载等留下约2小时。此为预算情景计算，既不是实际失败会话耗时分解，也不是整场完成保证。

## 已证实的运行风险

- `max_time_minutes=8`只包Agent循环。官方 `agent_runner.py`先复制wheels、解压、安装、准备工作树，再进入 `asyncio.timeout`；因此它不是单题全流程上界。源码见 `research/006/official-data-20261013/wheels-source/swegemma/harness/agent_runner.py`。
- `run_command`允许单次240秒，调用同步 `docker.exec`；长命令可能占用大量单题预算。官方effective timeout会取剩余时间，但仍不包含外部setup。
- ADK在执行工具前拒绝的缺参数请求不消耗28次工具预算。原31B候选已真实出现12次缺old_string的拒绝循环；80 turns留下较大无效生成空间。
- 原八题两臂的最大LLM calls为33，候选最大31；40轮没有低于这些已观察计数，但不保证未来任务不受影响。

## 冻结修复 `p014_single_v2_runtime`

| 配置 | 失败包 | 修复包 |
|---|---:|---:|
| Agent循环分钟 | 8 | 5 |
| 单命令秒 | 240 | 60 |
| 模型轮次 | 80 | 40 |
| 实际工具调用额度 | 28 | 28 |

模型、采样、输出上限均保留。prompt同步5分钟，要求第10次调用前完成主要编辑；明确edit_file的三个必填参数；缺参数纠正一次仍失败就切换等效run_command；非空提交后停止，不重复验证或预算耗尽后继续调用。修复包位于 `research/014/candidate/runtime-v2/`，SHA256 `68dfd7240c5f3bbd765274db817d06879c072f737191adc9249fa71ceb3d88db`。

Agent最坏循环时长减少37.5%，单命令上限减少75%。这是针对实际整体超时与已观察错误循环的运行修复；未知正式任务数量、外部setup时长和服务波动仍使本地无法保证整场12小时。需要修复版真实运行与新提交确认。

官方公开GettingStarted入口（本地 `research/003/runtime-review-20260930/official-starter-source.txt` 第225–270行）读取ZIP根目录eval_config.yaml的evaluation块，将timeout_seconds、max_time_minutes、max_turns传入EvalConfig；主办方在已取回topic-748428答复也确认eval_config在正式Agent运行时应用。当前私有评分入口源码未取得，不能声称逐字审查其实现，但没有忽略这些字段的现有证据。

只做一次原冻结八题的修复版单臂运行，复用CPU材料和原载体，单模型四卡4并发；不重跑基线、不换题、不展开prompt矩阵，不新增测试文件。该运行用于发现缩短预算后的明显修复损失与官方编译问题，不能称作新泛化/净提分证据。

GPU冷启动比上一轮慢，预算配置和官方读取路径手动核实后，已并行用ccjjllt既有参赛身份提交修复包：**57471497，10月21日15:24:23.717北京时间，PENDING、错误描述为空**。不是已确认全场消除超时。单臂八题随后全部完成并 **8/8通过**，fresh judge源码来源均有效；一题到达5分钟停止后补丁仍通过，表明时间限制实际生效。本批没有观察到修复损失，不能外推成独立提分证据。见[修复版真实回执](../runtime-v2/README.md)及[实际结果](../../eval-31b/runtime-v2-result.md)。用户表示分数返回后会通知，当前不创建自动轮询。
