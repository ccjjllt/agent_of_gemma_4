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

## 事后裁判范围裁定（2026-10-19/20，诚实记录）

- plan-48 §4 要求对“已生成候选”做独立副本事后裁判（区分：无正确候选 /
  选错 / 验收误判 / 恢复错版本）。该项**未能完整执行**，原因是我方 carrier
  的打捞钩子指错了目录：worker 只复制了 `SPEC['scratch']/p005_*`，而
  working_state/patch_store 实际落在 sandbox 系统临时目录（随沙箱销毁）。
  `candidate-stores` 在 48 个 attempt 输出中一个都不存在；trace 里也无
  diff（selector 交接的双候选 diff 只存在过沙箱内存）。
- 因此“未选中候选”的判定永久缺失：缺失裁判写可判定分母与未知数——此处为
  B/C/D 三臂 × 有候选的 attempt（候选存在过，有 trace 旁证：patch_store/
  snapshot/restore 词频），但内容不可恢复。
- 可判部分：已提交 patch（`patches/*.patch`，48 个）= 在线 FAIL_TO_PASS 已判。
  抽到的 B-334 patch 为真修复（`import datetime` + 冗余 `import time` +
  无害的 reproduce_issue.py），非空洞通过——harness 自带 test_patch 且剥离
  agent 侧测试文件，空洞通过路径不存在。
- 教训：候选打捞必须以 selector 交接文件为 attempt 级强制产物（落盘到
  attempt dir，不经过沙箱 temp），或钩子直接定位 sandbox_temp(toplevel)。
  下次 carrier 先修钩子再跑。

## C/D 路由修正（本地已完成，未推送，未跑 GPU）

- v4 的真实错误：`c_lora/d_lora` 被声明在根协调器 `agent.yaml`，
  `repair.yaml` 无 adapter；按官方按角色解析，修复调用实际走 base。
- 已改：根协调器去 adapter；`repair.yaml` 独自声明 `c_lora/d_lora`；
  修复角色去掉 skills，只剩五个训练约定工具；carrier 增加
  根/修复/其余角色三层断言，不再只查根 adapter。
- `research/012/evaluation-run-cd` 已用 `--job=cd --prepare-only` 本地重建；
  生成 manifest 中 `coordinator_adapter=null`，
  `repair_adapter=c_lora/d_lora`；未推送 kernel，未消耗 GPU。
- GPU 约定：后续一律用 ccjjllt 配额；当前查得 ccjjllt 剩余约 14.93h。
  GPU 命令在运行时恢复获授权账号并校验账号身份，不依赖凭据文件的持久状态。

## bcd6 小矩阵（步骤 2，进行中；ccjjllt）

- 新增 `agent-f`：修正后 base 对照（与 C/D 同架构，修复角色 base、无 skills）。
- worker 打捞钩子改为递归式、沙箱销毁前执行（`post-evaluate` + `pre-cleanup`
  两次打捞并留 manifest；此前只拷 `scratch/p005_*` 顶层是漏捞根因）。
- carrier 推广到 N 臂 + 任务子集：6 题按“每仓 sha256 顺序分层”
  （schema1+cc2+fp2+conan1）预先固定，保持冻结相对顺序；属探索性子集。
- 途中修了两个新 bug：子集任务文件与 TASK_IDS 不一致；
  supervisor 硬编码 `2/24` 导致三臂 prepare 被判 incomplete。
- bcd6 prepare v3 COMPLETE（三臂编译，c/d LoRA 就位，f 无 adapter）。
- bcd6 full v4 已推（18 attempts，f/c/d × 6 题）。跑完后再做步骤 1
  （corrected C/D full 24 attempts）。

## Corrected C/D prepare v5（2026-10-20，ccjjllt，COMPLETE）

- 推送 `ccjjllt/p012-eval-cd` v5（修正后 carrier，prepare-only），
  supervisor 状态 `prepared`，双 LoRA 挂载验 sha 通过，
  C/D 两臂编译通过；prepare 只做编译检查，不含模型调用与判分。
- 本轮 Kaggle 调用均在同一命令内恢复 ccjjllt token 并核对
  `username: ccjjllt` 后执行；不再依赖 token 文件的持久状态。
