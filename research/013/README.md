# P013 执行记录：两轮诊断完成，行为 SFT 已完成，尚无新评分

**2026-10-20 晚更新：** 已取回两轮 31B 诊断（每轮基线/候选均 0/4）、12B 行为 SFT（34 题/20 仓、72 步、独立 adapter 导出并取回）、CPU 新题预验证失败日志（缺 criteria index，尚无有效新题集）。实时正式成绩仍为 `57340172` 的 0.15；zhuating GPU 余量 30.00h。原件与下一步见 [P014 研究状态](../014/README.md)和 [P014 交接方案](../../docs/plan/014-score-first-submission-handoff.md)。

下面保留 P013 启动时的 CPU 交付记录。其“GPU 未执行/训练未启动/待推送”等描述只对应早期时点，已被上面的实际结果覆盖，不能据此重复推送训练。

## 启动时的 CPU 交付（历史记录）

1. Agent 候选：`agent/agent.yaml`（SequentialAgent repair→review）+ `agent/repair_agent.yaml`（五工具，无 submit，
   output_key=repair_handoff）+ `agent/review_agent.yaml`（六工具，include_contents:none）+
   `agent/prompts/repair.md` + `agent/prompts/review.md`（固化 P0-P1 四项：PYTHONPATH+module.__file__ 核验；
   timeout 10 单行/60-120 pytest 且 124=阻塞未完成严禁撤回；Call 12 前主体+关联分支一次读完集中写；
   收尾 git status/diff + 清理 *.tmp/repro* 且 patch_size==0 严禁重调 submit）+
   同改进单 Agent 备选 `agent/agent-single.yaml` + `agent/prompts/single.md`；
   采样与预算沿 behavior_v1（31B CT、8min/28calls/80turns、temp 0.2/top_p 0.95）。
   官方 schema 预检见 `agent/compiler-precheck.md`（四文件 OK；完整 compile_submission 待 GPU 打包阶段）。
2. 旧题机制诊断：`diagnosis/mechanism-review.md`（rich_4006 240s / rich_3180 415s+脏文件 /
   requests_7205 系统包 / requests_6757 连交空补丁 + 脏补丁；引用已有 trace 原件 + 两次一次性手动核验；
   未跑 31B，未产生新证据分数）。
3. 数据线：`behavior-data/p012-defect-fix.md`（produce_p012.py 第 277-289 行虚假 requirement_done:True 缺陷原文 +
   P013 修正 diff：False + verification_status:not_run + needs_other_route:None + 主/辅层分离统计；
   未改 P012 冻结文件；首批 32→64 条行为轨迹计划与复用资产清单）。
4. 评测线：`eval-31b-pair/plan-12pair.md`（12 新开发题冻结约束 + 24 attempts 配对载体配置 + 四维对比表模板 + 决策规则；
   冻结清单与执行待 GPU）。

## 启动时的 GPU 待办（历史记录；当前执行入口为 P014）

1. 少量旧题 31B 诊断：短超时截断、路径核验命中、Call 12 编辑、Sequential 交接 vs 切单 Agent、补丁卫生。
2. 冻结 P013 候选 ZIP → 12 新开发题 × 2 臂 24 attempts 同预算配对 → 四维对比 → 按决策规则决定是否正式提交。
3. 行为数据放量（≥32→64 条，复用 apispec/marshmallow/pydantic/tenacity/mypy/deepdiff/yamllint/dataset 等轻量仓）
   → 首批到位即启动 12B 4×L4 DDP SFT（r=8/alpha=16/gb=4/lr=2e-5，新 adapter）→ 同预算验证后再议 31B 迁移。
4. GPU 命令公约：同一命令内恢复获授权账号 token、核对 `username`、重读余量；绑定比赛获 4×L4；不读 heldout，
   不把预留评测题用于训练；长轨迹重估时间，不把余量当必须花完的预算。

## 启动时的证据边界（历史记录，不代表当前状态）

- 未产生新评分：公榜基线仍为正式 0.15（`57340172` COMPLETE）；behavior_v1 的 4/12 与 P012 的 f2/12 等均为历史数据，
  12 题净胜不换算公榜分数。
- 未完成训练：12B 行为 SFT 未启动；P012 256 条仍为 behavior_verified:false 辅助层。
- 未冻结新候选 ZIP、未选定 12 新开发题 ID、未推送任何 Kaggle 作业。
