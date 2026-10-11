# 四题 T0 / single_state 同加载对照选题

2026-10-11，只读历史证据后的建议；最终选题由本轮运行者冻结。建议采用 `rich_3942`、`fastapi_14873`、`fastapi_5077`、`rich_3518`，每题两臂，共 8 次。未启动 GPU、未读取 heldout 问题/答案/轨迹、未运行测试套件。

## 允许范围与判分依据

四题均是 [task_splits.json](../../task_splits.json) 中的 `dev_panel`，issue 和 `(repo, base_commit)` 均不与 heldout 或预留八题相交。预留名单为 `fastapi_13713`、`requests_6592`、`requests_6644`、`requests_7309`、`requests_7505`、`rich_3130`、`rich_3296`、`rich_3469`。旧 single_state 六题允许列表中的 `rich_3130`、`requests_6644` 必须剔除。

四题在 [官方环境资格结果](../../002/verification_eligibility.json) 中均为 **empty FAIL / reference PASS**，60 秒冻结条件与 300 秒诊断一致；依据是官方 `SubprocessManager + extract_snapshot + setup + verify_task` 重跑，不能引用此前 manual-git-fallback 作为环境证明。资格核验不把 reference patch 或历史失败提示注入待测 Agent。本轮 v30 的实际判分仍以本轮原始结果为准，旧版本资格仅证明这些题曾能正常区分空补丁与参考修复。

| 题目 | 官方目标测试数；验证耗时 empty/reference | 既有 Agent 结果（非本轮成绩） | 本轮角色与解释边界 |
|---|---|---|---|
| `rich_3942` | 8；5.1 / 4.2 秒 | P002 T0 两次均失败：110.93 秒上下文溢出、301.25 秒 session timeout；exit 均为 -1 | 检查读取、上下文占用和收尾能否改善。exit -1 与已经判定补丁不正确分开统计；失败结束快不是有效提速。 |
| `fastapi_14873` | 10；5.1 / 4.9 秒 | P002 T0 1/2：308.04 秒超时失败、276.18 秒成功。更早 P003-B 有 43 次 FileReadError，其中同参连续重复 33 次 | 对工具参数恢复与状态推进敏感；历史结果不稳定，一对胜负仅决定是否扩大。历史错误来自旧运行链，不预设 v30 仍复现。 |
| `fastapi_5077` | 1；5.3 / 5.1 秒 | 本地可核实的 P003-A0/B 均失败：309.50 / 305.03 秒超时；没有在本地 P002 0C 原件中找到该题 T0 重复结果 | 保留原六题中预先选定的定位/编辑困难题；不能把 P003-A0 数字冒充本轮 T0。单目标测试能区分正反控制，但覆盖较窄。 |
| `rich_3518` | 20；3.6 / 3.5 秒 | P002 T0 2/2：303.59 / 113.47 秒成功；P003-A0/B 也都成功 | 作为回退哨兵，检查技能/状态开销是否破坏既有成功。result 的总时长包含外围开销，303.59 秒不应单凭数值重新标作 session timeout。 |

四个不同快照：`rich_3942=c595fa95068dd2c5095d25446b2f0571bc0b3995`；`fastapi_14873=cd31576d5704b6aef1bdadc43770238cd88aad2b`；`fastapi_5077=930b27e5fa32ce150ddcea61f0c913435ae71c83`；`rich_3518=5ba9cb56e68dcdf2db50c214d8f1412c8a50761b`。这些都是已曝光开发题，不能作为独立泛化、排行榜或金牌收益证据。

不建议本轮换题：`fastapi_9425` 虽环境合格且 T0 2/2，但 [原六题冻结记录](../../002/explore_freeze.json) 已明确用 `fastapi_5077` 代替它以避免按已知赢家选题；继续沿用该选择更清楚。`rich_3777` 和 `rich_4075` 合格且目标测试更多，可在扩大阶段加入，但分别偏预算耗尽、定位失败，没有必要取代本轮的 2 FastAPI + 2 Rich 组合。

## 本轮是整体组合对照

对照取 [P002 冻结 T0](../../002/variants/t0/agent/agent.yaml)，候选取 [single_state](../candidates/single_state/agent/agent.yaml)。已手动解析两者 YAML：采样完全相同，均为 temperature 0.2、top_p 0.95、max_output_tokens 16384、thinking_budget 4096、include_thoughts true；预算也完全相同，为 5 分钟 / 50 工具调用 / 80 turns / 单命令及最终验证 60 秒。T0 文件里的旧 thinking 语义注释不代表当前 v30 行为；两臂必须共用同一镜像、wheel、模板及服务，保存实际请求参数。

两臂都声明官方六工具，候选额外声明 `source_edit`、`working_state`、`source_probe`，由 skills 机制提供 `run_skill_script`，有效工具入口因此有差异。候选还重写了系统提示：契约与最多两个假设、决策式读取、完整 span 句柄编辑、显式快照/恢复、状态更新；这些脚本和技能说明也改变提示长度与执行开销。结论应写作 **“T0 vs single_state 提示＋技能组合”**，不能写成只测试 working_state、单独 source_edit 或纯 prompt 单变量。

执行时兼容修正：CPU实际编译发现v30/ADK默认拒绝下划线技能名，v3统一使用`source-edit`、`working-state`、`source-probe`目录、frontmatter和引用。三项机制及8个Python脚本行为保持不变；本轮按v3冻结包记录结果，不把旧命名包宣称为已运行候选。

建议四题顺序和臂顺序预先冻结：`rich_3942` T0→candidate，`fastapi_14873` candidate→T0，`fastapi_5077` T0→candidate，`rich_3518` candidate→T0。复用一次模型服务加载，但每次使用新 sandbox/会话/技能状态目录，不在两臂间共享状态、补丁或观察。不要带 adapter 或中途按成绩改参数。

主要结果是逐题新增修复/回退和净 resolved；同时保留 exit/error、是否完成判分、捕获补丁大小/生产文件变化、主动提交、耗时、工具及 LLM 调用、上下文异常。不要把“程序退出 0”“有补丁”“skill 返回 ok”当作 resolved。8 次各 5 分钟是最多 40 分钟 Agent 预算；模型启动、sandbox 设置和最终验证另计，4×L4 账单按实际配额快照确认。此处不指定或扩大 GPU 额度。

## 原件位置

- P002 精确 T0 数据：[0C 分析](../../002/analysis_0c_run-002-cap-20260930T082350Z-6dedcd65.json)，以及本地 `results/002/run-002-cap-20260930T082350Z-6dedcd65/t0/r1/attempts/{01-rich_3942-r1-t0,14-rich_3942-r2-t0,05-fastapi_14873-r1-t0,18-fastapi_14873-r2-t0,12-rich_3518-r1-t0,23-rich_3518-r2-t0}/result.json`。
- P003 曝光题原件：本地 `results/003/recovered-20260930/results/003/run-003-joint-panel-20260929T151450Z-32b18764/{a0,b}/r1/attempts/` 中对应任务的 `result.json`；机制说明见 [原件二次审计](../../003/review-20260930/README.md)。这些历史臂不等于新的 single_state。
- [原 single_state 构建器](../../../notebooks/build_005_single_state.py) 目前只有候选臂且保留旧六题列表，不能直接当作四题双臂对照运行入口。
