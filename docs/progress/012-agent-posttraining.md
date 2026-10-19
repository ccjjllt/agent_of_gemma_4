# P012 进展：接口冻结 + SFT v4 训练中 + 评测载体就绪待位

2026-10-19。接续 [P012 交接](../plan/012-agent-posttraining-handoff.md) §6；实质运行已开始
（新 SFT v4 在途），48 次评测载体就绪、待 GPU 空位。

## GPU 实况（2026-10-19 晚，qingyu1232 账号 30h）

- SFT：`qingyu1232/p012-12b-repair-sft-256` v4 RUNNING（256 issues / 128 steps，
  23 仓）。v3 失败根因：carrier 沿用六工具 p005 配置，与五工具 `p012-repair-v1`
  数据协议不匹配（`train_ddp.py:206`）；v4 已改指 `repair_tools_v1.json` +
  `protocol_id=p012-repair-v1`，本地双检查通过。已用 2.19h。
- 评测：数据集 v2 ready；A/B 载体（24 attempts，A=P010 frozen base vs B=新架构纯 base）
  prepare/full 双模式已生成，待会话空位后先推 prepare 冒烟、再推 full。
  当前被共享账号 2/2 会话上限挡住（SFT 占 1 + 队友作业占 1）。
  载体含 `--dir-mode zip` 解包兜底；C/D（含 adapter）待 SFT 导出后另起作业。

## 已完成（CPU，可复核）

- 修复角色输入 `repair_context_v1`、五工具 schema、候选交接与选择器独占提交已冻结，
  并保存一个源自 P009 冻结事务的真实输入样例（非新执行）。
  见 `research/012/agent/`（`repair_context_v1.md`、`repair_tools_v1.json`、
  `handoff_v1.md`、`example-repair-context-v1.json`、`architecture.md`、5 个角色 YAML、
  提示与预算配置）。
- 关键纠正：官方 YAML 无 `if/switch/condition`（`schema.py:394-412`），
  条件由 base 协调器 LLM 经 AgentTool 决策；正式包不含外部 Python controller。
- 技能包已引入 `research/012/agent/skills/`，`working_state` 候选命名空间扩展完成并冒烟验证
  （`skills/VERIFY.md`）；五项 YAML 已过官方 schema 校验（`agent/compiler-precheck.md`）。
  完整 `compile_submission` 与实跑待 GPU 阶段。
- 数据线：256 先行（需 117 新问题）、512 扩展（需 373）、32–64 行为过程缺口已核对；
  剩余 626 候选仅 26 仓、Top3 占 56.4%，不可线性外推。新协议 `p012-repair-v1` 已冻结；
  新生产器已实现并小批量试跑 4/5（含失败原文与耗时），P011 零写入；
  全量放量与行为升级待定。见 `research/012/training-data/plan.md`、`produce_p012.py`、
  `producer-trial.md`。
- 评测线：12 题元数据已按哈希顺序冻结（cookiecutter4+feedparser4+schema1+conan3，
  前三库余量 0；conan 取哈希前 3），无正文读取；71 仓并集已补 `pudo/dataset`；
  48 次计划与统计口径已冻结，任务环境与运行尚未执行。
  见 `research/012/evaluation/`。
- 候选：仍以正式 57340172（0.15，31B 无 adapter）为基线；未产生新候选，未提交。
  完成标准四项中仅接口冻结达成，其余三项待 GPU。见 `research/012/candidate/status.md`、
  `research/012/training-run/README.md`、执行记录 `research/012/README.md`。

## 下一步（余量约 20.11h，2026-10-19 07:28Z 刚刷新，执行 GPU 前重读）

1. ~~实现 `working_state` 候选命名空间与候选绑定记录；正式 compiler 预检~~已完成
   （`skills/VERIFY.md`、`agent/compiler-precheck.md`；完整 `compile_submission` 待打包阶段）。
2. 新数据放量物化/行为重放（生产器试跑已过）→ 新协议编码 → 一轮连续 SFT 与独立导出（新建 LoRA/optimizer）。
3. 剩余 10 题任务环境放量（2/12 试点已确认区分性，`evaluation/task-prep-pilot.md`）→
   执行 48 次，冻结选择后做独立副本事后裁判。
4. 按选择规则决定候选并执行正式提交，跟进 Kaggle 回执。

P011 step70 为 C 参照（`71687cf9…`），step34 只作补充；旧三题不再作为训练前置。
