# P016 大规模后训练准备与实际研究记录

更新至2026-10-23，北京时间。Runpod 4×A40 的**首轮正式主训练已于17:40:24正常完成，68/68次更新、537个完整train问题、68仓、1 epoch全部跑完**。11:23:32启动至进程正常退出共6小时16分52.59秒，低于8小时上限；四rank最终进度一致，最终adapter与第68步完整checkpoint已导出并核对哈希。原基座初始化全新LoRA，没有使用校准adapter续训。尚未在Kaggle官方运行时加载或执行比赛Agent对照，不能据此宣称提分，正式最好仍为 **0.15**。

用户后续要求在本轮训练退出后通过服务器一次性程序直接备份到Google Drive；备份完成后由用户自行关闭服务器。训练完成与云端备份完成是独立事实，备份应以云端大小/MD5核对及最终回执为准，本次训练摘要不代替备份成功证明。没有Codex定时任务，也未自动停止Pod。实际命令、授权边界与完成证据见 [主训练执行记录](training/main-short-run-plan.md)、[最终训练摘要](training/main-short-training/final-result-summary.json)，资源/SSH/环境见 [Runpod交接记录](training/runpod-handoff.md)。

本轮执行改变了数据优先级：**先做好一套 SWE-smith 原生三工具轨迹 SFT，再用真实比赛六工具数据做必要适配；评估 Agent 保持原样。** 这允许复用来源真实执行语义，避免全量六工具重放成为新的瓶颈。它是根据数据产率和协议差异形成的研究选择，不是已经证明迁移有效。

最终产物已直接上传并完成文件校验：[Google Drive完整备份](https://drive.google.com/drive/folders/1JvQjxGl3vTE5Ss1pGQfvybEmQ4X5I61X)，包含全部七个checkpoint和最终adapter，约4.84GB。目录ID读取异常经Drive界面确认补齐，[原始结果与补充核对](training/final-backup-result.json)均留存；临时授权已清理，用户可停止Pod。

## 已取得的事实

| 工作 | 实际结果 | 证据边界 |
|---|---|---|
| 回收旧 P015 作业 | `p015-31b-task-rft/1` 当前 `CANCEL_ACKNOWLEDGED`；仅确认第一步完成，242 监督 token、132.39 秒；第二步日志到 8,165 长度的 backward | 未见第二步更新结束、最终导出或修复收益；取消者和原因未知，不自动重跑 |
| 三源首批 | 160 个去重问题、53 仓，约 28.25MB 原件 | 来源成功标签不等于本地重放；诊断长度不等于实际训练长度 |
| 缓存扩量 | 新增 SWE-smith 1,000 个问题、70 仓，111,784,674 bytes；约 3.08 秒本地抽取 | 与首批身份零重叠；未新增网络下载/GPU |
| 最终原生SFT v2 | **677个独立问题、69仓、约1,332万输入/238万监督token**；224题≤16K、453题16K–32K，未截断 | 已修XML提示与标签冲突，排除具体参考实现污染；复用来源成功证据，尚未实测比赛迁移 |
| 六工具适配种子 | 复用六个成功train题：70个动作、9段、6,230监督token，最长15,273；同目标输入从644,775降至110,996 | 16次生产编辑、5次提交、9个恢复动作；0上游验证/真实final，单题占监督68.84%，尚不足以证明适配完成 |
| FSDP 续训 | 四张A40完成16K校准4次更新，真实保存/退出/重载/继续并导出adapter；已修复非0 rank加载adapter错误 | 仅证明该运行链路；尚无正式比赛修复收益 |
| 首轮主训练 | 537个完整短轨迹问题、68仓，7,119,739输入/1,291,819监督token，全部≤16K；68次更新完成、正常退出，最终adapter 212,261,592 bytes | 159 PR、375命名变异、3旧pilot数字身份；最终batch loss 0.18059不是评测分数，尚未验证官方运行时兼容性或比赛修复收益 |
| 编码 | 补齐因果前缀核对、原始生成流入口、明确终局无回执的离线工具调用边界 | 不把 parsed reasoning/content 拼成虚构原始流；P015 非工具 final 的原始边界仍缺证据 |
| Open-SWE 限量候选 | 只读 5,000 条元数据，冻结 44 个 train 身份 | 官方筛选接口超时、行号接口 HTTP500，0 条正文；本轮暂缓 |

真实结果：[旧作业回收](p015-recovery/actual-result.json)、[首批数据](data-preparation/README.md)、[新增千题](data-preparation/smith-next-1000-report.md)、[最终语料v2](encoding/native-sft-v2.md)、[六工具适配种子](target-protocol/README.md)、[续训 CPU 实跑](training/resume-notes.md)、[编码说明](encoding/notes.md)、[Open-SWE 限量调查](open-swe/README.md)。原生数据以v2实际产物为准，旧v1仅保留过程证据。六工具合并只减少82.79%的重复前缀输入；不是已测GPU加速或新增解题，独立CPU重编码与实际九段逐字段一致。

## 为什么调整数据主次

首批原表示诊断中，SWE-Lego Real 的 48 条均超过32K，Synthetic只有7/48不超过32K；SWE-smith有46/64。Lego六例角色分解显示，长主要来自真实观察与动作，额外挂六工具schema只增加332 tokens；因此不能指望简单去掉工具说明解决长度。

更重要的是协议语义：源 view 可以读到248行、关键编辑在224行，而目标 read_file 默认只返回150行；原 submit 有时先要求复查后再次提交；Lego可以先 cd 后执行相对命令，依赖持久终端。直接改名会制造目标 Agent 不可能看到的状态。具体案例与最小适配证据见 [协议迁移判断](data-preparation/protocol-transfer-decision.md)。

第一阶段只使用清楚的 Smith 三工具协议，不为了大混合改造任意多schema框架。数据采用固定 CT tokenizer/template 序列化来源专家示范，保留当时观察、完整工具参数及源码状态；不是 Gemma 自己采样的 wire，也不是六工具已经执行。后续同一比赛 Agent 的权重 A/B 才回答迁移收益。

## 工程边界与下一步

复用已成功的 r8/alpha16、2e-5 起点。FSDP 的真 resume 绑定同一数据/配置/协议与拓扑；已实现`--init-run`从已完成adapter开始新阶段，重置优化器与进度，允许明确切换数据/工具协议，双rank CPU核对通过。现有 P015 两个 builder 已补入新功能模块 `fsdp_checkpoint.py`，未重建或改写历史冻结的 kernel 包。

GPU校准实际执行各段累计约46.81分钟，未超先前2小时校准预算。16K完成4步并实测真resume和导出；关闭NCCL P2P解决该机器首次通信停滞，将adapter先载入CPU解决多rank不一致。25K批次加入expandable_segments后完成更新，但32,382长度仍反向OOM，因此首轮主训练使用完整≤16K轨迹，暂停长上下文容量调试。

首轮主训练使用冻结的 `smith-pilot-calibration-v5` 代码和独立 `main-sft-short-v1` 语料，537题来自原224个短题与新准入313题。第10、20、30、40、50、60、68步均保存完整checkpoint；最终四rank结果和68条metrics的token总数一致。实际训练计时22,583.32秒，输入吞吐约315.27 tokens/s；完整启动至退出6小时16分52.59秒符合先前6–7小时估算。实际代码/数据哈希、命令和8小时预算见 [主训练执行记录](training/main-short-run-plan.md)，GPU失败与成功原始证据见 [校准记录](training/runpod-calibration/README.md)。下一步是官方运行时加载确认及固定Agent的权重对照，不自动启动下一轮训练。

18.30GB NF4缓存已在服务器完整下载并核对所有payload SHA，来源为此前私有Kaggle `ccjjllt/p005-cpu-nf4-cache` 的官方QAT派生缓存。训练使用固定CT tokenizer/template；没有重新转换基座或更换数据协议。

用户已有 heldout/select/dev/预留题用途保持不变。只使用预先冻结的 train 身份，原件与 SHA 留存；缺完整 Git SHA 的离线示范如实记录固定 dataset revision、instance/traj和原件哈希，不假填 SHA，也不把环境重放要求变成所有离线数据的新增门槛。没有新增测试文件或套件，必要核对均为一次性手动命令。
