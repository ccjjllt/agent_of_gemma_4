# P016 大规模后训练准备与实际研究记录

更新至2026-10-23，北京时间。用户已授权开始推进 [P016 方案](../../docs/plan/016-large-scale-posttraining-handoff.md)，现已确认提供 **Runpod 4×A40（每卡48GB）** 并授权通过Runpod官方Agent入口接入。下述产物截至CPU冻结时尚未启动GPU、教师调用或新提交；连接、模型传输与GPU结果需另据实际执行记录更新，正式最好仍为 **0.15**。

Runpod接入已完成本地16个技能安装、官方MCP直接注册和OAuth登录；CLI确认enabled、`auth_status=o_auth`。本会话尚未加载Runpod工具，需完全退出并重开Codex后读取现有Pod；尚未查看或改动远端资源。恢复入口与准确包路径见 [Runpod交接记录](training/runpod-handoff.md)。

本轮执行改变了数据优先级：**先做好一套 SWE-smith 原生三工具轨迹 SFT，再用真实比赛六工具数据做必要适配；评估 Agent 保持原样。** 这允许复用来源真实执行语义，避免全量六工具重放成为新的瓶颈。它是根据数据产率和协议差异形成的研究选择，不是已经证明迁移有效。

## 已取得的事实

| 工作 | 实际结果 | 证据边界 |
|---|---|---|
| 回收旧 P015 作业 | `p015-31b-task-rft/1` 当前 `CANCEL_ACKNOWLEDGED`；仅确认第一步完成，242 监督 token、132.39 秒；第二步日志到 8,165 长度的 backward | 未见第二步更新结束、最终导出或修复收益；取消者和原因未知，不自动重跑 |
| 三源首批 | 160 个去重问题、53 仓，约 28.25MB 原件 | 来源成功标签不等于本地重放；诊断长度不等于实际训练长度 |
| 缓存扩量 | 新增 SWE-smith 1,000 个问题、70 仓，111,784,674 bytes；约 3.08 秒本地抽取 | 与首批身份零重叠；未新增网络下载/GPU |
| 最终原生SFT v2 | **677个独立问题、69仓、约1,332万输入/238万监督token**；224题≤16K、453题16K–32K，未截断 | 已修XML提示与标签冲突，排除具体参考实现污染；复用来源成功证据，尚未实测比赛迁移 |
| 六工具适配种子 | 复用六个成功train题：70个动作、9段、6,230监督token，最长15,273；同目标输入从644,775降至110,996 | 16次生产编辑、5次提交、9个恢复动作；0上游验证/真实final，单题占监督68.84%，尚不足以证明适配完成 |
| FSDP 续训 | 实现 step 边界完整恢复；两 rank CPU 真退出后重启，参数/优化器/RNG/进度与连续路径一致 | 尚未实测 GPU、NF4、NCCL 与目标机器的续训 |
| 编码 | 补齐因果前缀核对、原始生成流入口、明确终局无回执的离线工具调用边界 | 不把 parsed reasoning/content 拼成虚构原始流；P015 非工具 final 的原始边界仍缺证据 |
| Open-SWE 限量候选 | 只读 5,000 条元数据，冻结 44 个 train 身份 | 官方筛选接口超时、行号接口 HTTP500，0 条正文；本轮暂缓 |

真实结果：[旧作业回收](p015-recovery/actual-result.json)、[首批数据](data-preparation/README.md)、[新增千题](data-preparation/smith-next-1000-report.md)、[最终语料v2](encoding/native-sft-v2.md)、[六工具适配种子](target-protocol/README.md)、[续训 CPU 实跑](training/resume-notes.md)、[编码说明](encoding/notes.md)、[Open-SWE 限量调查](open-swe/README.md)。原生数据以v2实际产物为准，旧v1仅保留过程证据。六工具合并只减少82.79%的重复前缀输入；不是已测GPU加速或新增解题，独立CPU重编码与实际九段逐字段一致。

## 为什么调整数据主次

首批原表示诊断中，SWE-Lego Real 的 48 条均超过32K，Synthetic只有7/48不超过32K；SWE-smith有46/64。Lego六例角色分解显示，长主要来自真实观察与动作，额外挂六工具schema只增加332 tokens；因此不能指望简单去掉工具说明解决长度。

更重要的是协议语义：源 view 可以读到248行、关键编辑在224行，而目标 read_file 默认只返回150行；原 submit 有时先要求复查后再次提交；Lego可以先 cd 后执行相对命令，依赖持久终端。直接改名会制造目标 Agent 不可能看到的状态。具体案例与最小适配证据见 [协议迁移判断](data-preparation/protocol-transfer-decision.md)。

第一阶段只使用清楚的 Smith 三工具协议，不为了大混合改造任意多schema框架。数据采用固定 CT tokenizer/template 序列化来源专家示范，保留当时观察、完整工具参数及源码状态；不是 Gemma 自己采样的 wire，也不是六工具已经执行。后续同一比赛 Agent 的权重 A/B 才回答迁移收益。

## 工程边界与下一步

复用已成功的 r8/alpha16、2e-5 起点。FSDP 的真 resume 绑定同一数据/配置/协议与拓扑；已实现`--init-run`从已完成adapter开始新阶段，重置优化器与进度，允许明确切换数据/工具协议，双rank CPU核对通过。现有 P015 两个 builder 已补入新功能模块 `fsdp_checkpoint.py`，未重建或改写历史冻结的 kernel 包。

千题批次实际编码与筛选、主语料合并及16K/32K校准输入已完成；用户已确认Runpod四张A40，下一步是接入、核对缓存并进行真实校准。首次GPU各执行段合计最多2小时，含模型加载、前后向、吞吐测量、保存/退出/重载和导出；模型传输和环境准备另计。启动器仅约束单次wall时间，执行者须跨调用累计预算，不能每次重试都重新获得2小时。根据实测速度再确定正式主训练时长，不提前承诺整轮训练能在某个小时数内完成。

最终运行包为 **`smith-pilot-calibration-v3`代码+`pilot-admitted-v2`数据**，每桶16个独立问题、4个optimizer steps；16K实际长度7,102–15,768，32K为16,575–32,382。[压缩包](data/gpu-bundles/smith-pilot-calibration-v3.tar.gz)为6,127,400 bytes，SHA256：`9cf66d95c05eaf4ba866d57d1caa7d4083aa0bdce88143d6336cae5a0ebbed91`；[回执](training/generated/smith-pilot-calibration-v3/build-manifest.json)记录逐trainer哈希。启动、真resume和32K命令见[GPU执行说明](training/gpu-run.md)。包已包含最终trainer、`fsdp_checkpoint.py`、实际tokenizer与工具schema，不含模型权重；恢复必须保持总步数4，跨阶段adapter初始化与真resume分开。

v3已修复父进程退出后独立session的rank可能残留的清理缺口：Linux按`/proc`的PID+启动时间跟踪自有后代，先TERM父、到截止再KILL仍存活且身份匹配的已观察进程，父退出后和finally仍清理。CPU真实进程模拟覆盖两种父退出时序并清理成功；四张A40上的容量、NF4/NCCL与恢复仍待实测。该机制不是cgroup，不能承诺捕捉未被观察且立即脱离父链的任意daemon；未退出PID会如实报告。

本地当前只有18.30GB官方QAT派生NF4缓存的manifest/config，没有完整权重。服务器需要复用此前私有Kaggle `ccjjllt/p005-cpu-nf4-cache` 的真实输出，或在用户指定已有缓存路径核对同一来源；不默认为权重已经传输。所有tokenizer和训练数据已在本地准备好，训练与推理CT基座路径分开。

用户已有 heldout/select/dev/预留题用途保持不变。只使用预先冻结的 train 身份，原件与 SHA 留存；缺完整 Git SHA 的离线示范如实记录固定 dataset revision、instance/traj和原件哈希，不假填 SHA，也不把环境重放要求变成所有离线数据的新增门槛。没有新增测试文件或套件，必要核对均为一次性手动命令。
