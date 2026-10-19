# P012 执行记录：CPU 已完成部分与 GPU 待办（2026-10-19）

## 本轮 CPU 已交付（可复核，无 GPU、无新权重、无新评分）

1. 共同接口冻结：`agent/repair_context_v1.md`（修复输入/五工具/候选交接/提交职责）、
   `agent/repair_tools_v1.json`（五工具，canonical SHA `91ec82b1…`，文件 SHA 见落盘）、
   `agent/handoff_v1.md`、`agent/example-repair-context-v1.json`
   （源自 P009 冻结事务 `p009-apispec-tx01` 真实观察，非新执行；`task_resolved=null` 保留）。
2. Agent 骨架：`agent/architecture.md`（官方无条件分支已核实，条件由 base LLM 经 AgentTool 决策）、
   `agent.yaml`（base 协调器）/`locator.yaml`/`repair.yaml`/`acceptance.yaml`/`selector.yaml`、
   `prompts/*`、`configs/sampling.yaml`（P010 同参）/`configs/eval.yaml`（8min/28call）、
   `skills/REUSE.md`。五项 YAML 已过官方 schema 校验（`agent/compiler-precheck.md`），
   技能包已引入（`agent/skills/` 落盘，`working_state` 候选命名空间扩展完成，
   见 `agent/skills/VERIFY.md`）；完整 `compile_submission` 与实跑待 GPU 阶段。
3. 数据计划：`training-data/plan.md`（256 需 117 新问题、512 需 373；剩余 626/26 仓、
   Top3 占 56.4%；32–64 行为过程从零确认；新协议 `p012-repair-v1`）。
   新生产器 `training-data/produce_p012.py` 已实现并小批量试跑 4/5（见
   `training-data/producer-trial.md`）；全量放量与行为升级待定。
4. 评测冻结：`evaluation/dev12-freeze.json`（12 题元数据冻结，无正文读取，
   cookiecutter4+feedparser4+schema1+conan3，前三库余量 0）、
   `evaluation/train-repos-71.json`（71 仓并集，`pudo/dataset` 已补）、
   `evaluation/plan-48.md`（48 次计划与统计口径）。
5. 候选状态：`candidate/status.md`（仍以正式 0.15 为基线，未产生新候选，未提交）、
   `training-run/README.md`（新 SFT 未启动）。

## 未完成（需 GPU 授权与预算，见交接文档 §6–§8）

- ~~`working_state` 候选命名空间 + 候选绑定记录代码实现~~已完成（`agent/skills/VERIFY.md`）。
- 新数据放量物化/行为重放、新协议编码、一轮连续 SFT 与独立导出（SFT 预计 1–2h 墙钟）。
- 12 题任务环境准备（2/12 试点已确认区分性，见 `evaluation/task-prep-pilot.md`；
  剩余 10 题待放量）与 48 次完整 attempt + 事后候选裁判。
- 正式 compiler 完整预检、单服务按角色 adapter 路由实测、候选打包与正式提交。

## 配额与授权

- 2026-10-19 14:47 北京时间读数：已用 32859.216207s，剩余约 20.8724h
 （`status-20261019/quota.json`）；执行 GPU 前重读。P011 V2 名义支出约 0.706 额度小时。
- 本轮未启动 GPU，未上传 submission；后续 GPU 按会话授权与预算执行。
  长行为轨迹改变吞吐，训练前用实际长度重估时间；不把余量当必须花完的预算。

## 证据边界

- 0.15 与 12B 后训练是不同实验，不混写；P011 loss 下降不是修复收益；
  12 题净胜不能换算公榜分数。负结果与失败原件保留在对应实验目录，不另建 blocked/skipped JSON。

## 2026-10-19 晚：SFT v4 训练中 + 评测载体就绪（待 GPU 空位）

- SFT v4（`qingyu1232/p012-12b-repair-sft-256` v4）：RUNNING。v3 的 protocol
  mismatch 已修复（见 training-run/README.md），评测任务不受影响。
- 评测数据集 `qingyu1232/p012-eval-tasks` v2 就绪（ready）：tasks-selected.jsonl
  + snapshots.zip（12 快照）+ deps.zip（含 cp312 wheels；混入 2 个 macOS cp39
  wheel，Linux pip 自动忽略）+ test-patches.zip。
- 评测载体 `research/012/evaluation-run-ab/` 已生成（prepare 模式，bootstrap 注入
  `P012_EVAL_PREPARE_ONLY=1`）：24-attempt A/B（A=P010 frozen base，B=新多角色架构，
  均为纯 base 无 adapter）。新增对 `--dir-mode zip` 数据集的解包/嵌套目录兜底。
- 推送被共享账号会话上限（2/2）挡住：SFT v4 占 1 席 + 队友作业占 1 席。
  待空位后按顺序推：v1 prepare（冒烟）→ v2 full（24 attempts）。

## 2026-10-19 续：SFT 完成 + 评测 full 在途

- SFT v4 COMPLETE（qingyu1232）：128 steps / 2 epochs，256 issues / 23 repos，
  adapter `b8653b17…`（110MB），训练 wall 1961s，协议 `p012-repair-v1` 确认。
  新 adapter 数据集 `ccjjllt/p012-new-adapter-128` ready（C/D 作业用）。
- 评测 prepare v7 COMPLETE（4xL4）：A=`swe_agent`（6 tools），B=`p012_coordinator`
  （6 agent-tools），task deps 安装 RETURNCODE 0（29 wheels）。
- 评测 full v8 已推（24 attempts，A/B x 12 tasks x r1）。沿途修了 6 个 carrier
  bug：缺 carrier-manifest、variants 用 json.dumps 导致 `null` 非法、缺 L4
  machine_shape（上了 T4x2）、adapter 误报（扫到比赛自带 sample_submission）、
  deps 安装调了不存在的 run_logged、prepare 误断言 CPU 会话。

## A/B 结果（2026-10-19，ccjjllt/p012-eval-ab v8，24/24 completed）

- A（P010 frozen base）：2/12（schema-331，feedparser-334）。
- B（新多角色架构，纯 base）：1/12（feedparser-334）。
- B−A = −1/12：B 在 schema-331 上丢了 A 拿到的题；其余 10 题双方都没解出来。
- 所有 attempt 几乎都打满 28 tool calls（预算耗尽型失败为主）。
- 这是 r1 单轮；C/D（adapter 臂）待新 carrier。12 题净胜只是机制证据，
  不换算公榜分数。候选 stores 已打捞（candidate-stores），事后裁判待做。

## C/D full 在途（2026-10-19）

- C/D prepare v3 COMPLETE：双 LoRA（c=P011 step70，d=新 b8653b17）挂载验
  sha 通过，两臂 `p012_coordinator` 编译通过。修了手敲 pin 丢 `ea` 的错
  （实际=`c2af...8faea90e...`，模板里写成 `...8fa90e...`），加了 pin 格式断言。
- C/D full v4 已推（24 attempts，双 LoRA 同服，`--max-loras` 由 harness 按
  manifest+lora_modules 计数自动推导）。

## 48 次全跑完（2026-10-19，A/B v8 + C/D v4，全部 completed）

| 题 | A(base) | B(新架构base) | C(新架构+P011) | D(新架构+新adapter) |
|---|---|---|---|---|
| schema-331 | T | F | F | F |
| cc-1669/1779/2010 | F | F | F | F |
| fp-334 | T | T | T | F（0 tool calls，8min 超时，4 次 LLM 空转） |
| fp-469 | F | F | T | T |
| fp-417/344 | F | F | F | F |
| conan x4 | F | F | F | F |
| 合计/12 | 2 | 1 | 2 | 1 |

配对差：B−A=−1（丢 331）；C−B=+1（得 469，无回退）；D−C=−1（丢 334，
一次超时异常）；D−B=0（334↔469 互换）；C−A=0（331↔469 互换）。
唯一一致的 adapter 信号：C/D 均解出 469（base 两臂均未解出），n=1。
10/12 题四组全灭；几乎所有 attempt 打满 28 tool calls。
结论：机制证据（新架构+repair adapter 可解 469）成立，但无配置占优，
不能换算公榜；事后裁判与提交决策待定。基线仍是正式 0.15（31B）。
