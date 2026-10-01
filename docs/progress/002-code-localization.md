# 002 进展：可提交的定位与有效编辑（先校准，再比较技能）

更新：2026-10-01。当前状态：CPU预检、0B官方入口资格诊断、0C的24次GPU容量对照已完成，保留16384；T0/T1探索已冻结但未运行，未正式提交。0A的真实服务reasoning留存和compaction核验仍未完成。下文初次实现时的unknown/DRAFT/未运行描述保留为历史记录，以后续运行段落为准。TPU可行性及本次复核见[10月1日交接](../plan/002-tpu-feasibility.md)。

执行顺序以[下一阶段交接](../plan/002-next-stage-handoff.md)为准：0A/0B 校准 → 0C 容量对照（8192候选）→ T0/T1 技能主实验 → 有依据才加图 → 有稳定目标才做 LoRA。

## 已实现（本轮新增，可复核）

| 产物 | 路径 | 说明 |
|---|---|---|
| 实验冻结清单 | `research/002/experiment_manifest.json` | schema 1；T0/T1/C8192 采样与预算、0C 6题×2组×2次=24 attempts、单变量声明、heldout 无交集、晋升规则 |
| 资格表框架 | `research/002/verification_eligibility.json` | 24 题全 `unknown`；空/参考补丁对照规则冻结，结果待 `002-preflight` 在冻结环境 CPU 填写；不按候选结果挑子集 |
| T0 冻结底座 | `research/002/variants/t0/agent/` | 继承 P003-A0；`max_output_tokens 16384`，thinking 保持（4096 仅为 enable_thinking 证据见下） |
| 容量候选 | `research/002/variants/c8192/agent/` | 与 T0 唯一差异为 `max_output_tokens 8192`；prompt/budget/tools 一致 |
| T1 技能包 | `research/002/variants/t1/agent/skills/repo_navigation/` | `SKILL.md` + `scripts/{find_symbol,read_range,check_patch}.py`；stdlib only；候选≤8、输出约4000字符初值；支持 `async def`；路径禁 `..`/绝对/逃逸/symlink；只报告不自动删/提交 |
| T1 agent | `research/002/variants/t1/agent/agent.yaml` 等 | 采样/预算与 T0 一致，单变量=技能；`skills: [skills/repo_navigation]`；prompt 增加技能使用与 `old_string not found` 回读纪律 |
| 预检载体 | `notebooks/002-preflight-single-cell.py` + `002-preflight.ipynb` | 0A CPU：指纹/静态参数/转义往返/技能检查/资格框架；GPU 最小推理探针默认关闭，需独立冻结运行 |
| 容量载体 | `notebooks/002-capacity-24-single-cell.py` + `002-capacity-24.ipynb` | 0C 24 attempts 联合载体：一次模型启动、逐题重置、AB/BA 均衡、嵌入式 BUNDLE+哈希校验；由 P003 面板载体派生，采样断言已改为单变量检查 |
| 探索模板 | `notebooks/002-joint-explore-single-cell.py` + `.ipynb` | T0/T1 探索结构模板，**DRAFT 未冻结**，任务待定，不可用于结果 |
| 配对分析 | `research/002/analyze.py` | 读 run root 的 result/context/patch/trace；缺失永不补零；配对键 (task, slot)；多标签描述统计 + discordant 二项 p（仅描述） |

## CPU 验证已通过（本机，2026-09-30）

- 技能三脚本：`find_symbol` 在合成 `async def`/`def` 同名文件返回 2 候选并标注 kind；`read_range` 对 `..` 返回越界错误；`check_patch` 在干净树返回空 diff、`syntax_ok true`。
- `002-preflight-single-cell.py` 本地降级运行通过：thinking 桥接 `4096,true` 与 `1024,true` 同为 `enable_thinking:true`；5 类合成转义往返全 ok；技能包 4 文件共 16368 字节、无 symlink、可编译、有必需文件；资格框架 24 题读出。
- `002-capacity-24-single-cell.py` 通过 `py_compile`；内嵌 BUNDLE 与源文件逐字节一致、哈希匹配；调度独立复算 24 个 attempt 唯一、T0/C8192 各 12、r1/r2 镜像均衡。
- `research/002/analyze.py` 在合成单配对 run 上输出正确配对、增益与多标签计数。

## 未做 / 待额度（明确留白，截至 2026-09-30）

- 0C 已完成（t0 7/12 vs c8192 6/12，保留 16384）；资格表已填官方入口实测（9/24 eligible）。未做跨工具 reasoning 留存的服务端渲染验证（需冻结模型服务/GPU，记“转发已证、最终渲染未知”）。
- T0/T1 探索 6 题已冻结、载体已生成待推（等 GPU 额度 2026-10-03 刷新）；24 题完整配对、图增量 GPU 对照未启动。
- 未下载新图/embedding 做覆盖与检索实测（上次 429 限流后停止；见下 1A 抽查状态）；不预设图有效/无效。

## 1A 索引抽查（2026-10-01，rich_3518 快照实测，非社区数字）

- 对象：`graphs/rich_5ba9…json`（1969 节点/5624 边，全 `calls` 型）、`embeddings/rich_5ba9…npz`（1969×256 float32）、`snapshots/rich_3518.tgz`（94MB）。脚本 `research/002/graph_audit.py`（stdlib），结果 `research/002/graph_audit_rich_3518.json`。
- 图↔向量键 1:1（双向 0 缺失）；节点 text 无空；短名歧义 153 组（如 `Foo`×7）；4 个 hub 节点邻居超 50，会被 wrapper 字母截断（非相关性 top50）。
- 向量：维度正常、全有限、无零向量、无精确重复；但随机对余弦 p50=0.936、p95=0.995，基线相似度极高；`Console.print` 查询 top5 全 ≥0.999，top1 超中位数仅 0.064——区分度弱，强制相似检索组暂无依据（符合交接“覆盖不足则跳过”）。
- 源码覆盖（正确限定名口径）：快照 190 文件 2014 符号中图覆盖 1839（91.3%）；缺 175（含唯一的 1 个 `async def` 测试协程）；图多出 130（快照外符号）。
- 提交一致性：快照 git HEAD `4aa2da77` ≠ 任务 base `5ba9cb…`（且 base 对象不在快照历史内），覆盖率数字带提交偏移 caveat；运行时以 harness `setup_baseline_commit` 为准。
- 结论：单快照证据，不推广；图邻域保持“可选补充”，不进 GPU 主对照。
- 未训练 LoRA；外部教师条件许可已记录，但 P004 进入条件（adapter 加载、稳定目标、隔离训练集、端到端量化检查）均未满足。

## Kaggle 实跑记录（2026-09-30，subagent 执行）

- 推送：`ccjjllt/002-preflight-0a-cpu-forensics-2026-09-30`（id 136468579，v1，private），状态 COMPLETE，机器 TeslaT4。URL：`https://www.kaggle.com/code/ccjjllt/002-preflight-0a-cpu-forensics-2026-09-30`。本轮只推 preflight，未推 capacity-24。
- RUN_ID：`run-002-preflight-20260930T020645Z-8390627d`。日志原文关键行：thinking 桥接 `4096,true` 与 `1024,true` 同为 `enable_thinking:true`；`double-serialization outer bytes=97`；`escaping roundtrip: all ok`；`WROTE /kaggle/working/results/002/.../preflight_evidence.json`；无 traceback。
- 阻断性缺失：`input_files=[]`，三数据源均未挂载，故无 `tasks sha=`/`wheels=` 行；skill 与 eligibility 按预期报缺失（单文件推送未随包上传 repo）。配额快照：已用 71438.158s/108000s，余约 10.16h，未额外消耗。
- 结论：资格表不可填，不能进 0C。下一步：网页核对 notebook Settings→Data 挂载与竞赛条款后，有计划地做一次带挂载的 0A/0B CPU 重取证（单次，不连续重推）。

## 预检 v2（带挂载，2026-09-30 subagent 执行，已验证落盘）

- 同 slug 原位更新至 v2（CPU：`enable_gpu false`，CLI 键 `dataset/competition/model_sources`，隔离目录仅 2 文件）。RUN_ID：`run-002-preflight-20260930T022902Z-46739304`，状态 COMPLETE，无 traceback。
- 日志原文：`input_files` 非空（493+，wheelhouse 41 个）、`input resolved` 三行齐全、`tasks sha=e4b3fd60f69dbc2b bytes=1984455`、`wheels=41`、thinking 桥接 3 映射、转义往返全 ok、WROTE 产物路径。
- 回读三数组均非空，`machine_shape None`（CPU 回填），挂载故障消除。证据落盘 `research/002/preflight_evidence_run-002-preflight-20260930T022902Z-46739304.json`（28963B，指纹与 manifest 一致：tasks 1984455/`e4b3fd…`、HARNESS_README 49356/`3d6e57…`）。`skill/eligibility` 报 not found 系单文件推送无 checkout，预期内，不阻塞 0C。

## 0B 资格验证（2026-09-30 subagent 执行，方法失效、诚实保留 unknown）

- Kernel `ccjjllt/002-eligibility-0b-cpu` v1 CPU COMPLETE，RUN_ID `run-002-eligibility-20260930T024331Z-8fa486b3`，tasks/sha、snapshots 129、wheels 41、input 三行齐全，heldout 未读。
- 载体绕过官方入口（手动 tar+git apply+host pytest，未用 `extract_snapshot/apply_patch_in_container/verify_task`），`workspace_head` 全空、脏树串扰、宿主缺 `dirty_equals/requests` 伪影：24 题 0 eligible，全保留 `unknown`，不可作排除依据。0C 6 题 0/6 eligible 系方法失效，不触发替换，冻结名单不变。
- 落盘 `research/002/eligibility_raw_….json`（116577B）与 24 行实测资格表；补跑清单：全部 24 题需官方函数重跑（P0 为 0C6+requests_6592 全档）。配额余约 10.15h，CPU run 未消耗。

## 合规审查结论（2026-09-30，subagent 只审不推，后续4项已修见文末记录）

阻断项（进 Kaggle 前必须修）：B1 T1 `tools` 缺 `run_skill_script`（`agent.yaml:4-10` 与 `system.md:11`/`SKILL.md:14` 不一致，单变量会被破坏）；B2 缺 `kernel-metadata.json` 三 sources 注册；B3 `read_range`/`check_patch` 最终 stdout 上限 10000/6000 与 SKILL 约 4000 不一致；B4 三脚本 `--root` 默认放行非 `/workspace`（`ws` 变量未强制），违反路径禁区承诺。警告项：preflight 缺失 fail-fast（capacity 已有）、joint-explore 为 DRAFT 易误用、改单 arm 会变 12 attempts（analyze 会报错而非静默混算，属正确设计）。通过项：顶层 stdlib、前置逻辑无 pip/网络、双布局候选+marker、单启动逐题重置 AB/BA 均衡、超时与 manifest 一致、无 heldout 读取、不改裁判、不追榜、单 cell 可粘贴、analyze 不猜字段。最小补丁清单已收录，进 0C 前逐项关闭。

## 阶段 1 探索冻结（2026-09-30，FROZEN 未运行，待 GPU 额度 2026-10-03 刷新）

- 0C 保留 16384，故 T0/T1 均在 16384 公共底座（T1 采样与 T0 解析相等，单变量=技能）。
- 冻结 6 题（`research/002/explore_freeze.json`）：`rich_3942`（上下文故障）、`rich_3130`（迟编辑）、`fastapi_14873`（工具循环）、`fastapi_5077`（FastAPI eligible，与 0C 的 9425 不同，避免按胜负挑题）、`requests_6644`（稳定+Requests）、`rich_3518`（稳定对照）；orders AB/BA 交替，共 12 attempts。0B-official 下 4/6 eligible，双分母报告。
- 载体 `notebooks/002-joint-explore-single-cell.py` + `.ipynb`（BUNDLE t0×4/t1×8 可解析、12/12 哈希与源文件一致、调度 12 唯一，`py_compile` 通过）。此前 DRAFT 模板已替换。
- 至多选 1 个候选进入 24 题完整配对；跑之前重查配额（12 attempts 名义 1h+开销）。

## 下一步（按交接，不跳步）

1. 在 Kaggle 运行 `002-preflight`（CPU）：回传版本指纹、静态参数证据、技能包校验、资格表 60/300s 行。
2. 资格冻结后，运行 `002-capacity-24`（24 attempts，名义 Agent 额度 2h + 开销）：按冻结准则决定 8192 是否晋升公共底座；出现截断/JSON 不完整/参数不生效先查原始请求并停止扩量。
3. 晋升后冻结 T0/T1 探索 6 题并生成 frozen 探索载体；最多选 1 候选进入 24 题完整配对（按仓库与 smoke/其余 21 分报）。
4. 图邻域仅作 CPU 覆盖与 issue-only 检索质量门控后的增量；LoRA 满足六条件后再议。

本文件是结果记录，不是获奖承诺。任何新 GPU 数字必须附 run_id、manifest 版本与原始产物路径，否则不进入结论。

## 阻断项修复记录（2026-09-30，P002 本地 CPU，只改代码不推送不跑 GPU）

### B1 run_skill_script 注册判定（事实）
- 源码证据：`/tmp/current-google-adk.whl:google/adk/tools/skill_toolset.py:SkillToolset._tools=[ListSkillsTool,LoadSkillTool,LoadSkillResourceTool,RunSkillScriptTool(name="run_skill_script")]`；`adk_submission/builders/llm.py:compile_llm_agent` 在 `skills` 非空时 `tools.append(skill_toolset)`；`adk_submission/resolvers/tools.py:184-221,253-264` 经 `load_skill_from_dir` 加载并 `SkillToolset(skills=...,additional_tools=resolved_tools)`；`swegemma/tools/__init__.py:create_tools` 注册表仅 9 名（run_command/read_file/write_file/edit_file/submit_patch/get_status/get_code_neighbors/search_similar_code/get_code_subgraph），无 run_skill_script；`adk_submission/registry.py:ToolRegistry.get` 未命中抛 ToolNotFoundError。
- 判定：`run_skill_script` 由 SkillToolset 自动提供，无需也不应在 `agent.yaml:tools:` 显式声明；在 `tools:` 加该名会导致编译期 ToolNotFoundError。`SKILL.md:16` 与 `system.md:11` 引用的名字符均为 `run_skill_script`，一致。
- 改动：`research/002/variants/t1/agent/agent.yaml:4-10` 保持 6 工具不变（未加 run_skill_script）；`research/002/variants/t1/agent/prompts/system.md:11` 改为条件式“若 run_skill_script 已注册则优先用，否则用 read_file/run_command 回退（含中文同义）+ 注明经 skills 自动提供、不加到 tools:”；`experiment_manifest.json:t1.prompts/system.md` 前缀由 `1ea2f33d88f9a7a9` 同步为 `011ea0c0b99b130b`（其余三项不变，已复算匹配）。
- 验证：`grep -n run_skill_script agent.yaml/system.md/SKILL.md` 显示 yaml 无命中、md 各 1 处同名字符；`py_compile` 三脚本通过。

### B2 kernel-metadata（事实）
- 新建 `notebooks/kernel-metadata.json`（notebooks/ 下此前无该文件）：`id=ccjjllt/002-preflight-0a-cpu,title=002 Preflight 0A CPU,slug=002-preflight-0a-cpu,code_file=002-preflight.ipynb（CLI 相对 notebooks/ 目录；仓相对 notebooks/002-preflight.ipynb）,language=python,kernel_type=notebook,is_private=true,enable_gpu=true,enable_internet=false,enable_tpu=false,dataset_data_sources=[metric/gemma-4-developer-agent-wheelhouse],competition_data_sources=[gemma-4-developer-agent],model_data_sources=[google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2]`；另含 CLI 模板兼容键 `dataset_sources/competition_sources/model_sources` 同值。
- 对齐：与 `research/003/queue-migration-20260929/joint-verified.json:9-22` 及 `official-notebook-metadata.json:9-25` 的 language/kernel_type/is_private/enable_gpu/enable_internet/三 sources 逐项一致（本地 python 比对全 True）。
- CLI 形态注明：`kaggle kernels init -p` 模板字段为 `dataset_sources/competition_sources/model_sources`，MCP 回读为 `dataset_data_sources/...`；本文件双写以兼容；`code_file` 取 `.ipynb` 因 `kernel_type=notebook`，单 cell `.py`（8235 字节）内容与 ipynb 代码单元一致可粘贴但不作 code_file。
- 验证：`python json.load` 通过；`notebooks/002-preflight.ipynb` 存在且 nbformat 4/cells 2；未执行 `kaggle kernels push`。

### B3 输出上限 4000（事实）
- 改动：`scripts/read_range.py:59` 由 `min(...,10000)` 改为 `min(...,4000)`；`:84` 由 `print(blob[:10000])` 改为 `print(blob[:4000])`；`scripts/check_patch.py:91` 由 `print(blob[:6000]...)` 改为 `print(blob[:4000]...)`（`sh():23` 内管道 6000/2000 保留）；`SKILL.md:16-18` 由“约 4000 初值”改为“稳定上限 4000、截断看 truncated/is_truncated、超限用 next_start 分片接续”。
- 验证：`py_compile` 三文件通过；`find_symbol --symbol foo` 在合成 async/def 同名返回 2 候选（async def/def）ambiguous true；`read_range --path ../evil.py` 拒绝对（path must be repo-relative）；`check_patch` 干净树 `syntax_ok true,empty_source_diff true`；`--max-chars 10000` 请求仍被压到 4000（stdout 4001 字节含换行）。

### B4 workspace 禁区（事实）
- 改动：三脚本默认拒非 `/workspace` root，本地诊断加显式 `--allow-non-workspace-root`；`find_symbol.py:24-31` 的 `ws` 现用于强制比对（`p != ws and not flag` 即 error），`read_range.py:25-32`、`check_patch.py:32-39` 同逻辑；各脚本 Usage 行及 `SKILL.md:11`（工作区段）与三处 Usage 同步该 flag 说明。
- 验证：`--root /tmp` 无 flag 三脚本均 `non-workspace root rejected`；有 flag 则放行（find_symbol 返回 2 候选、read_range 返回 ok、check_patch 返回 syntax_ok）；`--root /workspace` 无 flag 不报 non-workspace（本地报 root missing，因本机无 /workspace，Kaggle 下为正常路径）。

### 联动（事实）
- `notebooks/002-capacity-24-single-cell.py` 内 BUNDLE 键仅 t0/c8192（SELECTED_VARIANTS=('t0','c8192')，EXPECTED_CONFIG_HASHES 仅二项；文件中 `t1` 仅见于允许集合断言），故未重嵌 T1。
- `research/002/experiment_manifest.json` 已同步 t1 system.md 哈希前缀（见 B1），t0/c8192 未动；capacity 载体内嵌哈希仍为 t0/c8192 未受影响。

## 0B 官方入口重跑（2026-09-30 subagent 执行，已验证落盘）

- Kernel `ccjjllt/002-eligibility-0b-official-2026-09-30` v1 CPU COMPLETE（约 20.7min），RUN_ID `run-002-eligibility-20260930T030619Z-bb8d744e`。官方 `SubprocessManager+extract_snapshot+setup+apply_patch_in_container+verify_task`，24 头非空，上轮脏树消除。
- 9/24 eligible（`fastapi_11194/14873/5077/9425/rich_2725/3518/3777/3942/4075`，60 与 300 一致）；其余 unknown（gold-fail 4、already-pass 1、setup-error 10，含 requests_6592/7505 的 fixture+timeout，与交接一致永不永久排除）。
- 0C6 为 4/6 eligible（`rich_3130` gold-fail、`requests_6644` already-pass 为任务属性）；manifest 无替换名额，不换名单，双分母报告。
- 落盘 `research/002/eligibility_official_raw_run-002-eligibility-20260930T030619Z-bb8d744e.json`（189231B，24/24）+ 资格表 24 行实测。配额余约 10.15h，CPU 未消耗。

## 0C 容量对照（2026-09-30 运行完成，结果已回收分析）

- Kernel `ccjjllt/002-capacity-0c-24-2026-09-30` v2（L4）COMPLETE，RUN_ID `run-002-cap-20260930T082350Z-6dedcd65`，24/24 返回结果（v1 T4 `bf16` infra 失败属环境非变量）。产物 `results/002/run-002-cap-…/`（git 忽略）+ 分析 `research/002/analysis_0c_run-002-cap-20260930T082350Z-6dedcd65.json`。
- 配对（12 对 = 6 题×2 重复，`analyze.py` 缺失不补零）：t0 7/12，c8192 6/12，净 -1（c 胜 `fastapi_14873@slot3`；负 `fastapi_14873@slot9`、`fastapi_9425@slot4`），discordant 二项 p=1.0（仅描述，无效力/等效断言）。
- 上下文异常：t0 侧 1 次（`rich_3942-r1-t0` 仍 `ContextWindowExceededError`，即 8192 想修的原故障在对照组复现）；c8192 侧 0 次（rich_3942 两次均为 5min 超时 `exit -1`，非上下文崩溃）。方向符合预期，但样本为单题重复。
- 生产修改：9 对双方皆有，t0-only 1 对，无 c-only；耗时均值 +4.7s（duration）/+4.6s（agent）、工具调用 -3.1（c8192 略少），均为描述性。
- 截断证据（2026-10-01原件复核更正）：已恢复23/24条trace，缺`23-rich_3518-r2-t0`。已恢复的738个Agent步骤均有`metrics.completion_tokens`，逐条求和匹配final_metrics：c8192为12条/360步、最大2093、P95 937.3；t0为11条/378步、最大2239、P95 875.1。旧文“无单步计数”不成立。全部已恢复trace均缺finish_reason字段，因此只能说已记录输出远低于上限，不能用“没有length标记”证明无截断；缺失trace不影响已回收resolved配对，但影响过程审计完整性。
- 晋升判定：**保留 16384，8192 不晋升**。冻结准则要求三者同时满足（异常减少 + 无新增截断 + 无修复退步）；异常方向虽好（1→0），但修复净 -1 且截断仅“无标记”（非证无），不满足 AND。6 题仅支持筛选，正式采用仍需扩大面板。

- 补列预先冻结的eligible子集：4题×2次，t0 5/8、c8192 4/8，净差仍-1。requests_6644空补丁已通过，双方各2次raw成功不能归因修复；保留全分母，并另报eligible子集，不更换名单。

## 2026-10-01 TPU与校准边界复核

TPU已有指定W4A16模型的新上游实现，但本项目未运行TPU；当前冻结载体依赖CUDA，迁移必须另立后端研究，不能将GPU T0与TPU T1拼接比较。见[TPU可行性与有界验证交接](../plan/002-tpu-feasibility.md)。

当前preflight的RUN_MINIMAL_GPU_PROBE只有标志与打印，没有推理执行分支；其合成bridge与JSON往返只是静态/stdlib证据，不等于完整ADK→服务渲染验收。0A仍部分完成，旧交接允许将最终渲染未知列为边界继续当前冻结公开环境比较，故本次复核不将0C作废。实际消息转换、模板渲染和compaction机制的一部分可在CPU完成，不必全部等待GPU。

## 受控正式提交（2026-10-01 subagent 执行，已 review）

- 基线包 `research/002/formal_baseline_t0_2026-09-30.zip`（2850B，SHA `c25efb23…`）：T0 四文件与 manifest prefix16 一致，已验无 T1/c8192/P003 混入；根唯一 `agent.yaml`，无 symlink/`..`，扩展名仅 yaml/md。清单 `formal_baseline_manifest_2026-09-30.json`（status 如实记 `SUBMIT ATTEMPTED ONCE -> REJECTED 400, NO RETRY`）。
- 提交 1 次被服务端拒绝：`400 Client Error: Bad Request .../CreateSubmission`，无 submission id、无公开分。`submission-limits` 仍 0/0/1，非次数耗尽。原因未明（本地校验通过），按纪律不重试；下个操作日先只读检查上传链/命名要求再定是否单次重交。
- 环境快照：GPU 已用 98497/108000s（0C 消耗后），刷新 2026-10-03T00:00:00Z；正式推理走赛事算力，不耗本额度。

## 探索运行前核对（2026-10-01 subagent 只读，已 review：全部通过）

- BUNDLE 可解析（t0×4/t1×8）、12/12 哈希与磁盘一致；code 与 .ipynb 逐字节一致（60732B），`py_compile` 通过；调度 12 唯一（T0/T1 各 6，AB/BA 与 freeze 一致）；T1 采样与 T0 解析相等、预算一致（原始字符串仅注释头不同）；6 题与 heldout 28 无交集；skill 无 symlink、无答案读取（字面 grep 命中均为防护语/工具名，语义通过）。
- 配额：GPU 余约 2.2h，名义够 Agent 部分但余量薄，判断刷新后可跑、当前不抢跑。可推送检查清单已备（push 当天重查配额/哈希/slug/L4/timeout/三源）。
