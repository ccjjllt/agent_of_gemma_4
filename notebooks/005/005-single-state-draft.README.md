# notebooks/005/005-single-state-draft — DRAFT 说明（未冻结，不可直接推送）

状态：**DRAFT 草稿，未冻结**。本地评审用；不推送 Kaggle、不运行模型。
主文件：`005-single-state-draft.py`（首行 DRAFT 水印 + `IS_DRAFT` 护栏，
任何模型加载/安装/评测前 `raise RuntimeError` fail-fast）。

## 1. 与 002 载体（`002-joint-explore-single-cell.py`）的差异点

| 方面 | 002 载体 | 005 本草稿 |
|---|---|---|
| 臂数/变量 | 双臂 t0 vs t1，单变量=导航技能；12 attempts（6 题 x AB/BA） | 单臂 `single_state`，单变量=状态/交接增量相对 P002 冻结基线；6 attempts（占位） |
| agent 内嵌 | BUNDLE 双变体全文 + EXPECTED_CONFIG_HASHES 冻结哈希 | BUNDLE 四文件全文（agent.yaml / system.md / sampling / eval_config 占位）；三技能包只嵌清单+SHA，脚本全文不 whole-copy（打包时真实拷贝+校验，禁 symlink） |
| 新增 working_state 交接 | 无 | I2 行为约束：issue 契约 + 最多 2 假设（producer/transformer/consumer/boundary）+ `record/show`（changed_fields + next_action）；I4 handoff 往返入口（record-issue/write-handoff/read-handoff）已进 skill，本载体只用单 Agent 部分 |
| 新增 probe | 无（仅 read_file/run_command） | I3 `source_probe`：locate（1–4 queries + question）/ inspect（1–3 targets，exact/outline，截断无句柄）/ probe（单命令 + 具名 assert，exit 0 ≠ 复现/修复；trace 仅 repo 过滤 setprofile） |
| 新增 diff 快照 | 无 | I1 `patch_store`：最多 4 份，parent_patch_id + 文件哈希，显式 restore，不自动选最佳 |
| sampling | 16384 + thinking 旧语义注释（budget>0 仅开/关） | 16384 上限不变 + thinking 新转发语义注释（positive budget=真实 token 预算；4096 为实验起点非最优） |
| eval 预算 | 冻结（60s/50 calls/5min/80 turns） | 占位复用同数值，标注 TODO v30 pins，推送前重 pin |
| 指纹 | 冻结（官方包版本+sha、tasks.jsonl sha+字节数、config 哈希） | 全部留空 `TODO_V30_PINS`，不抄 002 旧指纹；推送前填 v30 值 |
| 任务名单 | 冻结 6 题 + AB/BA 顺序 | 同 6 题占位 + 明确标注“待执行前按§5预注册改名单” |
| 服务/归档 | 单次 start、逐题新 Evaluator/沙箱、attempt 独立归档、stop 前归档日志 | 相同结构（单次加载、逐题重置沙箱/会话/attempt 目录、独立归档、日志归档后停止）；实现为推送版骨架，DRAFT 护栏后不可达 |

## 2. 待确认项（不进本载体）

- `SequentialAgent` 交接编排暂不进本载体：本载体=单 Agent + 状态（handoff §3 I1–I3）。
  一次真实交接（I4 explorer→executor）另起载体（`one_handoff`，同工具同总预算），
  避免在本载体混合解释（§5：先同 sampling 比较交接本身）。
- 正式任务名单与顺序：待执行前按 §5 预注册（小探索先筛方向，候选有依据再扩大；
  不重跑已完成 0C，不以无限 preflight 阻塞）。
- thinking 4096/1024 等预算对比：本载体固定 4096 起点；预算消融另起条件，不混入状态变量。
- eval 预算与官方包 revision：待 v30 值，以推送前重查为准。

## 3. 推送前检查清单（全部通过才能去 DRAFT 化）

1. 指纹：填 `TODO_V30_PINS`（model revision + tokenizer、官方包版本+sha256、
   tasks.jsonl sha+字节数、BUNDLE 四文件冻结哈希、三技能包冻结哈希），
   与 002 旧指纹无关，全部重算。
2. BUNDLE 哈希：四文件全文与候选目录 `research/005/candidates/single_state/agent/`
   逐字节比对（`agent.yaml`/`prompts/system.md`/`configs/sampling.yaml` + 定稿
   `eval_config.yaml`），哈希写入冻结表并在载体内断言。
3. 技能包：三技能脚本从候选目录真实拷贝进打包目录，按清单 SHA 校验；
   无 symlink、无仓库外路径引用；SKILL/脚本以候选目录为准。
4. 调度：按 §5 预注册正式名单+顺序，替换 `PLACEHOLDER_TASKS`，冻结后不再改；
   已完成臂保持历史（参考 002 “只改仍排队臂”原则）。
5. manifest：`draft_experiment.json` 升级为正式 run_manifest（run_id、版本、
   GPU/tp、采样、预算、单变量声明、server_start_count=1、逐题归档路径）。
6. 配额重查：GPU 配额、模型输入挂载、wheelhouse、任务数据四项 fail-fast 复查；
   暖内核拒绝（已加载 torch/swegemma/adk/vllm 即停，转新内核）。
7. 合规复查：无 heldout 硬编码、无参考答案（`patch`/`test_patch` 仅出现在
   system 提示的“永不进入提示”规则句中，无答案正文）；`hints` 缺失不失败；
   临时脚本不出最终 diff、不碰测试。
8. 去 DRAFT 化：移除 `IS_DRAFT` 护栏后重新 `py_compile` + 本地结构 diff 复核，
   另存为 `005-single-state-single-cell.py`（本 DRAFT 文件本身永不改名推送、
   永不覆盖冻结文件）。
