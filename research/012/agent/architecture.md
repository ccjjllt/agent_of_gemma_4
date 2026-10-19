# P012 Agent 架构：完整修复流程（CPU 骨架 v0，未实跑）

日期：2026-10-19。对应交接文档 §3。状态：YAML/提示/预算已起草，**未在 GPU/正式 compiler 实跑**；
选择器独占提交、候选绑定、按候选区分的运行记录均待实现与验证。本节只交付可复核骨架，不声称新 Agent 已运行。

## 1. 流程

```text
定位(base) → 修复A(base或指定adapter) → 独立验收(base)
  ├─ 证据足够且未见反证 → 恢复/确认A → 最终提交(base)
  └─ 有具体反证且预算足够 → 从同一baseline修复B → 独立验收
                               → 依据证据选择A或B → 恢复选中快照 → 提交
```

- 同一 12B CT 服务，不为每个角色加载模型；题内角色和候选串行，题间最多 4 个隔离任务并发
 （复用 P010 `coordinator.py:46,460-481` 单服务 + 按 attempt 隔离目录模式）。
- B 不是固定必跑；A 已有充分通过证据时及时结束。默认只做 A、B 两条路线。
- 所有角色、交接、快照、候选及验收消耗计入整题 8 分钟 / 28 调用；LLM turns 同口径。
  阶段时间（定位约 1 分钟、预留选择/恢复/提交约 45 秒）是调度策略，不是每角色硬限。

## 2. 官方约束下的编排选择（已核实，不虚构条件分支）

- 官方 YAML 仅支持 `Llm/Sequential/Parallel/Loop`，无 `if/switch/condition/selector`
 （`schema.py:394-412`）；`SubAgentRef` 仅 `config_path`，明确排除 `code`
  （防任意 import）；回调仅字符串名。**不存在合法 YAML 条件编排。**
- 因此 P012 不写“YAML 条件流程”；条件由 base 协调器 LLM 按证据决定调用哪个 `AgentTool`。
  静态 YAML 只声明可用子 Agent 清单，运行时分支是 LLM 工具选择。这与
  “base 协调器负责调用与决策；不要把外部 Python coordinator 当正式提交入口”一致：
  正式包内只有 YAML + 提示 + 技能 + adapter，没有外部 Python controller。
- `output_key` + `include_contents: none` 经文件 handle 传真实证据（复用 one_handoff 模式）；
  定位、修复、验收使用独立消息上下文；只追加状态卡不能称为上下文重置。
- 技能包复用 `single_state` 合法 kebab-case（`working-state/source-edit/source-probe`），
  不复用 `one_handoff` 含下划线的技能名（已有官方编译拒绝先例）。

## 3. 角色与工具持有

| 角色 | 模型路由 | 工具 | 技能 | 说明 |
|---|---|---|---|---|
| base 协调器 | base | `get_status` + 5 个 AgentTool（定位/修复A/验收/修复B/选择器） | `working-state`、`source-edit`（快照管理） | 不直接编辑源码；按证据决定是否进 B；独占提交经选择器 |
| 定位 | base | `run_command/read_file/get_status` | `working-state`、`source-probe` | 只读；输出定位证据 |
| 修复 A/B | base 或指定 adapter | 五工具（无 submit） | `working-state`、`source-edit`、`source-probe` | A/B 同配置，仅 `candidate_id` 与 adapter 不同；B 从 baseline 恢复 |
| 验收 | base | `run_command/read_file/get_status` | `working-state` | 不修改生产源码；前后记录 diff；临时产物放临时目录 |
| 选择器 | base | `run_command/read_file/get_status/submit_patch` | `working-state`、`source-edit` | 恢复选中快照后唯一可调 `submit_patch` 的角色 |

- 验收三工具仍含 `run_command`（可写文件），纪律靠 prompt + 前后 diff 约束，不是技术只读。
- 修复角色返回候选交接（见 `handoff_v1.md`）；选择器依据需求覆盖与执行证据选择，
  不以长度/自评/非空 diff 判优。
- `working_state` 已按 `candidate_id + stage` 命名空间扩展（`--candidate/--stage`，
  缺省走 legacy 单份文件；见 `skills/REUSE.md`、`skills/VERIFY.md`）；
  `patch_store` 4 快照/8 文件/200KiB/512KiB 上限保留、存储未改，
  候选绑定由调用方 `parent_patch_id` 链保证，HEAD 变化按实际失败处理。

## 4. 本目录产物

- `agent.yaml`：base 协调器（LlmAgent + AgentTool 清单）。
- `locator.yaml / repair.yaml / acceptance.yaml / selector.yaml`：各角色。
- `prompts/*.md`：各角色提示（含提交语义、验收纪律、选择规则）。
- `configs/sampling.yaml`：temperature 0.2、top_p 0.95、thinking 1024、max_output 4096（沿 P010）。
- `configs/eval.yaml`：timeout 240、max_tool_calls 28、max_time_minutes 8、max_turns 80。
- 适配器路由：A/B 全 base；B/C/D 架构相同，C 修复用 P011 step70，D 修复用新 adapter
  （`adapter: trained_adapter` 别名，复用 P010 `attempt_worker.py:73-97` 模式）。

## 5. 待实现与验证（GPU/正式 compiler 前置）

1. ~~`working_state` 候选命名空间 + 候选绑定记录~~已完成（`skills/VERIFY.md`）。
2. 选择器恢复-提交原子性（恢复失败按实际失败处理，不静默删文件）。
3. 正式 compiler 完整 `compile_submission` 预检（YAML schema 五项已过，见 `compiler-precheck.md`；
   技能包已引入，待跑全量编译）。
4. 单服务按角色 adapter 路由实测（同一服务不同 `adapter` 别名请求）。
5. 12 题 × 4 组 48 次完整 attempt（见 `../evaluation/`），含事后候选裁判。

A 组为当前单 Agent 冻结 P010 12B base 配置重跑基线，不拿历史三题代替。
