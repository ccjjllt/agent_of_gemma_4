# Verified Focal Agent RL 数据准备

2026-10-24。采用用户指定的研究方向：真实六工具冷启动、关键错误恢复、已执行的动作对照、预算感知策略优化。四个adapter仍为G r128五投影、S/F/R r32七投影，计算与导出BF16。**首批96题已完成逐题来源与机制审阅，并已产出本地官方工具和验证器通过的人工off-policy示范。** 已验证关键动作对、模型on-policy rollout和GPU训练仍为0。

当前实物与边界见 [首批96题说明](pilot96/README.md)、[96题目录](pilot96/catalog.md)、[实际重放结果](replay/replay-summary.json) 与 [Gemma对白导出清单](replay/gemma-sft-manifest.json)。静态任务库为G36/S20/F20/R20，43仓；S2、F2、R2、R5各4题。尚未重放的任务不算成功轨迹。本机使用官方SubprocessManager，Docker环境等价性尚未建立。

## 四个adapter如何取数据

**仓库修复机制和Agent错误类型是两个维度。** S/F/R分别强化表示与结构算法、跨层流转与状态、动态语义与兼容；错误定位、编辑恢复、利用反馈、预算管理由四个adapter共同学习。不能把一个专家训练成只会修`edit_file`错误。

先以96个独立任务检验采集产率：G主责36题，S/F/R各20题，每位专家的五个子机制各约4题。G的共同课程覆盖全部机制；共同六工具冷启动课程可供四个adapter复用。专家共同起点按r32独立训练，G按r128五投影独立训练，不能因共用数据就复制不兼容权重。

每题先取得目标协议的真实成功过程，审阅assistant监督span；再由指定adapter各采2次，共192条基线attempt。从真实失败中最多选128个pre-action checkpoint，每个起点重放1个原动作并尝试3个替代动作，最多512条continuation。32–64个完整结果对、16–32个多步恢复起点是**产率假设**，不是承诺或现有库存。此阶段不是专家相对G的配对收益实验，后者需另行安排相同任务/预算的G运行。

详细字段与成本算术见 [data-contract.md](data-contract.md)，数量及入池条件见 [collection-plan.json](collection-plan.json)。先测环境恢复率、完整结果对产率和每单位预算新增成功数，再决定是否扩到原先的1,400题。

## 四种训练材料分别准备

| 材料 | 采集方法 | 可以提供的监督 |
| --- | --- | --- |
| 六工具冷启动 | 来源过程作候选；在真实目标环境重放/重新采样，保存工具原始观察及完整验证 | 经审阅的有效assistant动作；错误历史可作条件，不能把成功轨迹每步都当正例 |
| 单动作结果对 | 同一pre-action快照，重放原动作与替代动作，由同一continuation策略继续，双方完整验证 | 同prefix的局部偏好；单次观测收益与重复seed稳定性分别保存 |
| 多步恢复段 | 从同起点执行read→edit→run等短分支，保存每步真实反馈 | 恢复策略监督；短段进展不自动等于resolved |
| 当前策略RL rollout | 实际当前Gemma+adapter采样，记录行为logprob、mask、状态组、回报和成本 | 实际实现所需的episode/state优势；旧教师轨迹不能冒充on-policy |

`state_id`必须关联完整可见prefix、文件与工具/环境状态、恢复范围及剩余预算。只有prefix hash相同不够。后台进程、缓存、外部副作用等无法恢复时，材料只能按可观察重放证据使用。分支奖励的成本保留原prefix消耗，恢复不能重置整题预算。

整题验证完成后才记`resolved=true/false`；验证未完成或环境失效记unknown。编辑成功、局部上游检查改善单独记录。金补丁、验证patch、事后评估日志留在采集侧，不能泄漏到早期policy prefix。

## 最初48条来源种子与整理器（历史采集阶段）

原始数据在本地 `artifacts/seed/raw/`，派生候选在 `artifacts/seed/prepared/`；大JSONL按项目约定不进Git。来源固定revision、选择索引、文件hash及实际计数见 [seed-inventory.json](seed-inventory.json) 和 [来源采集清单](sources/seed/manifest.json)。首次种子是来源资格的32个任务ID、48条轨迹，含32条publisher成功和16条publisher失败；跨来源修复家族去重尚未完成。整理器提取262处工具错误/重复动作候选，已验证关键错误数量仍为0。

```sh
python3 competition/vfarl/prepare_native.py \
  --raw competition/vfarl/artifacts/seed/raw/SWE-bench__SWE-smith-trajectories__raw.jsonl \
        competition/vfarl/artifacts/seed/raw/nebius__SWE-rebench-openhands-trajectories__raw.jsonl \
  --out competition/vfarl/artifacts/seed/prepared
```

整理器保留原生消息、工具ID和观察，不执行来源命令。它输出原生episode、来源成功候选、错误/重复动作审查队列和机制审查队列。错误文本只用于召回，不判定关键错误；同题的两条独立attempt也不会被配成“同状态对”。`think/task_tracker`等源工具需重新设计并实跑，不能改名后伪装成六工具观察。

最初六工具参数从当前wheel中实际wrapper签名与docstring提取，见 [six-tools.json](six-tools.json)；后续已在真实运行环境捕获 [ADK FunctionTool声明](replay/adk-six-tool-declarations.json)，并核对参数名及required。源`str_replace_editor`与目标`edit_file`的锚点、错误、路径和截断规则不能只靠改名对齐，成功示范均实际执行目标工具。

本批Smith的`row.patch`发现跨仓错配，整列隔离；真实submit回显中的diff可以另行核对，不从错误列补答案。Nebius的`model_patch`含大量辅助脚本/文件，不能直接把整份diff当生产修复正例。具体逐题机制与质量审阅在本目录审阅文件中保存，来源成功不等于本地复现。

这32个最初种子任务的主标签候选为G 5、S 16、F 8、R 3，历史记录见 [mechanism-supply.json](mechanism-supply.json)。[Smith审阅](mechanism-review.smith.json)接受5个机制锚点，其余11题保留证据缺口；[Nebius审阅](mechanism-review.nebius.json)逐题记录源码行为、因果改动与实际源观察。原生episode本身仍未改名冒充目标协议；新完成的96题课程与实际重放单独计数。

最初种子缺少S2坐标/跨度、F2引用/所有权、R2 callable/wrapper、R5导入/可用性的主标候选，R1/R3/R4也薄弱，因此后续从允许train任务定向补到了当前96题，未用包含“Unicode”“async”“type”字样的题代替机制证据。历史合并候选在本地 `artifacts/seed/prepared/mechanism-curriculum.jsonl`，不会被整理器自动当作训练正例。

复用已缓存的官方31B tokenizer作CPU字段编码，[native-token-profile.json](native-token-profile.json) 保存方法和hash。Smith的原生content+工具参数每轨迹中位37,746 tokens、Nebius中位54,568.5；共35/48条的这个字段累计量超过32,768。它没有套目标chat template、没有计完整角色/工具包装，也没有生成loss mask，因此不是最终输入长度、监督tokens或推理预算实测。长轨迹必须重采短过程，或在真实决策边界构造仅依赖此前信息的训练窗口，不能裁掉结尾后仍称完整成功。

## 论文机制与我们的实现边界

[FocalRL官方代码](https://github.com/microsoft/FocalRL/tree/bfdf5f9aee6aedb00a9c66d7217421373bb753c2)支持关键错误前恢复与短续跑，其局部rubric奖励不是整题resolved。官方命令重放实现也不足以自动证明完整状态相等，我们须额外保留声明范围内的恢复证据。[CSO](https://arxiv.org/html/2602.03412v2)在替换一个动作后继续执行并验证任务结果，不能用“工具返回ok”替代。

[GiGPO](https://arxiv.org/html/2505.10978v3)原式为`A_episode + ω A_step`；用户的α混合式是本项目待研究组合。任务均值按同任务/预算分组，分支均值按同state/剩余预算分组，避免将任务难度和局部进展混为一谈。

[Kimi K3报告](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf)§4.1.2中，agent累计输出包含推理和工具参数，超过`τ·b0(x)`时将任务奖励覆盖为−1。其完整预算估计过程和具体τ未公开；我们的方案先保存原始成功/成本信号，研究约束目标或同成功等级的成本偏好，不能声称复现其参数。仅设置推理时thinking上限不能代替训练。

本轮没有启动GPU、付费模型采样或读取heldout。来源命令仅作审阅线索；实际重放执行经审阅的recipe、记录真实六工具观察并走原版完整验证。必要核对均以一次性手动命令完成，没有新建测试套件。原始语料不随比赛submission提交。
