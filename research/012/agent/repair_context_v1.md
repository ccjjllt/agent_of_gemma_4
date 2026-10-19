# P012 repair_context_v1：修复角色输入接口（冻结 v1）

日期：2026-10-19。对应 `docs/plan/012-agent-posttraining-handoff.md §2`。
状态：已冻结，可供 Agent 线与数据线并行使用。后续改动须升版本，不原地修改。

## 1. 设计约束（已核实）

- 官方 `submit_patch` 捕获当前工作树且整题停止（`research/006/.../swegemma/tools/execution.py:76-116`），不能传候选 ID。
  故修复 A/B、定位、验收均不持有该工具；只有最终选择器恢复选中快照后调用它。
- 官方 YAML 无 `if/switch/condition/selector`（`adk_submission/schema.py:394-412` 仅 `Llm/Sequential/Parallel/Loop`；
  `SubAgentRef` 明确排除 `code` 防任意 import）。条件分支不在 YAML，由 base 协调器 LLM
  按证据决定调用哪个 `AgentTool`（修复 A → 验收 → 必要时修复 B → 选择提交）。
  不要提交外部 Python coordinator 作为正式入口；正式 compiler 不支持任意动态导入
 （`adk_submission/compiler.py:1-10,59-72,208-242`）。
- 修复角色五工具为 P005 六工具减去 `submit_patch`：
  `run_command / read_file / edit_file / write_file / get_status`。
  Schema 见 `repair_tools_v1.json`。新 SFT 必须编码此五工具 schema，不得沿用含 submit 的旧 schema
  却声称匹配新角色，也不得把 AgentTool/技能工具当成旧工具别名。
- 验收者不看修复者自信结论，只看问题、实际 diff 与执行证据；`run_command` 仍可写文件，
  故验收纪律是“不得修改生产源码”，不是技术只读。验收前后记录 diff，临时验证产物只放临时目录。
- 候选与验收共享同一任务沙箱，串行修改工作区；B 从最初 baseline 恢复，不在 A 补丁上继续改。

## 2. repair_context_v1 字段

修复角色每次接收（单次调用输入）：

| 字段 | 说明 |
|---|---|
| `candidate_id` | 本次修复的候选身份，如 `A` / `B`。与 patch 快照 ID、源码指纹、命令结果、验收结论绑定同一 ID |
| `problem_text` | 原问题全文（未经改写） |
| `repo` | 如 `marshmallow-code/apispec` |
| `base_commit` | 固定 40 位 buggy commit；B 路线同样从此恢复 |
| `baseline_identity` | baseline 快照 ID、HEAD、源码指纹 |
| `role_identity` | `base` 或指定 adapter（如 P011 step70 / 新 adapter）；C/D 仅修复角色路由不同，其余配置与 B 完全相同 |
| `localization_evidence` | 真实定位证据：文件/行号/哈希、关键分支原文（不是摘要替代未读代码） |
| `prior_observations` | 已执行命令的真实输出与退出码（含失败）；B 可继承 A 的反证，不得继承 A 补丁 |
| `current_diff` | 当前工作树相对 baseline 的 diff（B 开始时为空） |
| `hypotheses` | 明确标注的假设与反证；区分原样观察与角色推测 |
| `budget` | 剩余整题预算（8 分钟 / 28 调用为四组统一 easy 配置；各角色+交接+快照+验收全计入，LLM turns 同口径） |
| `goal` | 此次目标：完成生产修改 → 真实验证 → 正确交接结束 |

上下文只保留与下一动作有关的证据，不复制定位者全部推理历史。
`output_key` + `include_contents: none` 经文件 handle 传真实证据；核对角色确实收到原问题与必要代码。

## 3. 修复角色返回（候选交接）

```json
{
  "candidate_id": "A",
  "files": [{"path": "src/apispec/ext/marshmallow/openapi.py", "after_sha256": "<hex>"}],
  "requirement_done": true,
  "verification": {"command": "<实际命令>", "exit_code": 0, "scope": "修改后源码的同判据"},
  "unresolved": [],
  "needs_other_route": false
}
```

- 只报“完成修改 → 报告验证 → 交接结束”；不调用 `submit_patch`。
- 局部编辑未完成时如实交接 `requirement_done: false` / `needs_other_route: true`，
  不生成虚假“验证通过”。
- 选择器依据需求覆盖与实际执行证据选择，不以 patch 长度、模型自评或非空 diff 直接判优。
- 事后裁判先冻结选择与提交 patch，再在独立副本评估各候选；不反馈给在线选择器。

## 4. 与 P011 的协议差异（C 配置须保留说明）

- P011 协议 ID `p005-harness-tools-v1`，六工具，`tools_hash 4645523d…`，`protocol_hash 4d5f551…`
 （`research/011/.../dataset-manifest.json:13-18`）。
- 新协议 ID `p012-repair-v1`，五工具，修复角色无 submit；base 协调器/选择器持有 submit。
  P011 在新角色下是迁移参照，不是同协议复用。

实际完整输入样例见 `example-repair-context-v1.json`（源自 P009 冻结事务
`p009-apispec-tx01` 的真实观察，非新执行；`task_resolved=null` 边界保留）。
工具清单见 `repair_tools_v1.json`；交接与提交职责见 `handoff_v1.md`。
