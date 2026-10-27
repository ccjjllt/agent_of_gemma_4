# 新任务小轮研究材料

实际进度（2026-10-27 12:57北京时间）：用户已授权先跑12＋4，有信号再考虑100题。私有CPU作业已完成16/16题的baseline 0／参考修复1，Unsloth运行时已实际组装。前两次GPU启动在动作生成前失败，均为0采集/0更新；tokenizer哈希域和LoRA初始化BF16舍入已分别修正。当前私有采集作业 [auto-r3](https://www.kaggle.com/code/jiyujia/vfarl-newtasks-collect-auto-r3) 的两个双卡模型均已就绪，已真实完成前三轮共6次工具动作，后台继续采集。尚无完整任务奖励、异奖组或新RL收益。逐次真实证据与配额见 `execution.md`、`execution-auto-r3/results.md`。

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
