# 新任务小轮研究材料

实际进度（2026-10-27 15:45北京时间）：私有短作业 [Unsloth recovered reward update r1](https://www.kaggle.com/code/jiyujia/vfarl-unsloth-recovered-reward-update-r1) 已实际COMPLETE。找回的两个异奖组、4条动作在新Unsloth双GPU训练端执行了1步真实optimizer，梯度范数0.811325，LoRA变化L2为0.057921。没有重新采样；原始v1概率未改写。8K窗口完整保留5117–6495-token序列，更新计算/导出322.57秒，整作业815.60秒，消耗0.45776h配额、剩17.43349h。更新后的奖励方向指标略负，同路径重算目标loss从0.235803升至0.236187，不能宣称训练收益或泛化改善；16K反向也未验证。该SHA已训练，勿再次提交。权威原件及逐项记录见 `execution-gpu-short-r1/`。

新版训练入口为 `training/train_vfarl_unsloth.py`，保留VFARL的真实focal奖励、PPO clip＋KL、sequence TIS与原始行为概率。`--learner-backend unsloth` 选择单进程双GPU训练；原生四进程FSDP入口仍可显式选择。新采样执行身份为v2，精确前缀追加时复用KV，任务/分支与评分边界清空。不能把新版身份改写进旧批次。

`unsloth-v2-short-local/` 是未提交的4K单题计时载体，`--update-only` 在真实更新尝试后结束，不跑独立对照；仅用于后续决定短实验时审阅，不属于已运行结果。单题不保证产生异奖组，4K也不能证明16K容量。它不自动启动，不能当作12＋4正式轮。`unsloth-v2-local/` 为已被该短载体取代的本地草稿，也未提交。

2026-10-27，北京时间。本轮已实际冻结并打包 **12 道采集任务＋4 道独立研究验证任务**，共16个仓库、16个repair family和16个snapshot group。采集与研究验证在 task、family、snapshot及仓库四个层面均无交叉。全部来自既有S600 train白名单；题目正文只保留官方初始用户提示，来源、参考补丁和完整验收要求保留在裁判侧。

选题先保守排除旧VFARL v1–v4冻结清单的114个任务及114个family，再合入P015冻结train身份和现有P017实验manifest的身份投影，总计排除174个历史ID。排除的是已列明来源，不能据此宣称全库每一次旧实验都已枚举。16题此前均在S600人工off-policy示范课程内，本轮研究验证衡量新VFARL任务表现，不属于比赛官方heldout或“从未见过的预训练/SFT内容”。

本轮每个预先指定仓库取一个新family，复用已有非stock Linux环境；按历史thinking=0示范最大长度不超过6144、距4500最近、seeded SHA256打破平局选题。历史长度只用于控制小轮成本，不代表thinking=1024的新轨迹长度，也未使用新模型奖励选题。两个本地来源文件不完整的候选在采样前排除，详情保存在selection中。

材料已在本地实际生成：

- `selection.json`：冻结任务身份、来源SHA、选题规则、历史排除来源和`research_split`。
- `questions.jsonl`：16条原官方初始提示；按仓库约定不追踪大JSONL，SHA保存在selection。
- `material-v1/material-build.json`：7,657,170 bytes归档的SHA及源文件指纹。
- `material-v1/dataset/vfarl-train100.bin`：兼容现有载体的归档名，内部实际是本轮16题；私有目标metadata为`jiyujia/vfarl-newtasks-r1`。

16题全部使用原已准备的profile-01/02/05/07/08/09/14/16，无新增依赖要求。Linux CPU作业已分别完成这16题的原baseline和审阅修复官方验收，逐题原件保存在 `cpu-v4-evidence/vfarl-prevalidate/prevalidation.jsonl`。模型rollout、异奖组、RL更新和独立任务成绩只按后续实际运行记录报告。

准备命令：

```bash
python3 -m competition.vfarl.l4x4.next-round.prepare
python3 -m competition.vfarl.l4x4.kaggle.build \
  --selection-path competition/vfarl/l4x4/next-round/selection.json \
  --out competition/vfarl/l4x4/next-round/material-v1 \
  --dataset-id jiyujia/vfarl-newtasks-r1 \
  --dataset-title 'VFARL new tasks research round1'
```

构建输出目录必须是新目录；再次构建应使用新版本目录，以保留已冻结原件。模型运行的源码以各次载体独立冻结的source bundle为准。研究验证题只在同预算的更新前后做原始rollout，不参与焦点分支、奖励学习或自适应选题；独立验证题的实际执行仍由实验载体决定。

本地必要核对已实际完成：来源文件SHA、16题材料构建、归档SHA、题目计数与采集/验证四类身份交叉均核对；未新增测试文件、测试套件或门控脚本。
