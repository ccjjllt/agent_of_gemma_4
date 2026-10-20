# P013 31B 新题 1v1 配对评测冻结方案（CPU 冻结，执行待 GPU）

日期：2026-10-20。对照：A=`public022_v10`（31B 正式基线，公榜 0.15，ZIP SHA256 `640b8abc…`，8min/28calls/80turns，
temperature 0.2/top_p 0.95）vs B=P013 冻结新候选（两阶段 SequentialAgent 或同改进单 Agent，见 `../agent/`）。
均为纯 base、无 adapter，完全同预算。

## 冻结：12 道新开发题（训练外多仓，判据区分性待 GPU 验证）

- 选择约束：不属于本轮提示/代码修改用题（排除公共 12 题、外部 8 选择仓 gpxpy/feedparser/pandas/schema/cookiecutter/conan/modin/iniconfig、
  旧 heldout 28 题、旧预留 8 题）；全部来自 `research/012/evaluation/train-repos-71.json` 71 仓训练划分之外的仓库问题；
  覆盖 ≥3 个仓库/≥2 种问题类型；元数据冻结顺序按"每仓 sha256 排序分层"固定（沿用 bcd6 子集方法），冻结后不增删。
- 每题预验证（GPU 准备阶段）：修前失败（fail-to-pass 在基线环境真实失败）+ 参考补丁通过（pass-to-pass 同环境通过）
  均需实际跑通，否则换题并记录原因；未验证区分性的题不进入 24 attempts。
- 冻结清单：`dev12-freeze.json`（待 GPU 准备阶段生成：题 ID、仓库、sha256 顺序、修前失败/参考通过 log 指针）。
  本轮 CPU 未选定具体题 ID，不编造。

## 载体：24 attempts 配对配置（8min/28calls 同预算）

- `pair-carrier.json`（待生成）：12 tasks × 2 arms × r1 = 24 attempts；A/B 同镜像、同 harness、同裁判；
  每 attempt 记录：逐题胜负、首次生产修改调用序号（Call #）与墙钟秒、单命令阻塞超时次数（exit 124）、
  空补丁提交次数（patch_size=0）、半成品次数（仅复现脚本/只补 import）、终补丁脏文件检查（*.tmp/repro*/自建目录）。
- 复用 P012 载体经验：prepare 模式先验编译（两臂 SequentialAgent 编译通过 + 工具注册），再推 full；
  打捞钩子递归式、沙箱销毁前执行并留 manifest；候选打捞以 attempt 级产物落盘为准（P012 教训：沙箱 temp 不可追）。

## 四维对比表模板（逐题记录，不换算公榜）

| task | A 结果/首次编辑Call/阻塞/空提交/脏文件 | B 结果/首次编辑Call/阻塞/空提交/脏文件 | 胜负 | 回退/备注 |
|---|---|---|---|---|
| t01 | F/–/n/n/clean | … | … | … |
| … | | | | |
| 合计/12 | /中位Call/Σ阻塞/Σ空提交/脏数 | 同上 | Wins–Regressions= | 覆盖仓/类型数 |

另记：验证后回退检查（通过相关测试后收尾是否误删/过度改邻接行为）。

## 决策规则（研发验收，非公榜承诺）

- 达标（跨题/跨仓净增 Wins−Regressions ≥ +2，运行可靠无明显回退）：冻结 ZIP（官方 compile_submission + 补丁卫生核对）→ 正式提交 → 记录回执与评分。
- 临界（仅净胜 1 或有新增伴随回退/证据混杂）：只补一批小批新开发题确认净胜稳定性，不盲目提交。
- 未达标（0 或净回退）：拆具体失败/回退根因，针对性修正一次后复测；禁扩矩阵/换 seed 找偶然漂亮结果。
