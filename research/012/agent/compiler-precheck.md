# P012 Agent YAML 官方 schema 预检（CPU，已执行）

日期：2026-10-19。方法：一次性手动命令，非测试套件。

## 方法

- 用官方 `research/006/official-data-20261013/wheels-source/adk_submission/` 的
  `yaml_loader.load_yaml` + `schema.SandboxedAgentConfig`（`!include` 沙箱解析 + pydantic 校验），
  在隔离临时目录加载（官方包 `__init__` 依赖 `google.adk`，故仅复制
  `errors/paths/limits/schema/yaml_loader` 五个纯依赖模块做 schema 级校验，不动仓库）。
- 依赖：pydantic + pyyaml 装入临时 target（homebrew python3.11），仓库外，不进 git。
  技能包已引入（`skills/` 落盘，见 `skills/REUSE.md`）；完整 `compile_submission`
  （工具/模型注册表）仍待 GPU 打包阶段。

## 结果

| 文件 | 结论 |
|---|---|
| agent.yaml（base 协调器，Llm + 4 AgentTool + get_status） | OK SandboxedLlmAgentConfig |
| locator.yaml | OK SandboxedLlmAgentConfig |
| repair.yaml（五工具，无 submit） | OK SandboxedLlmAgentConfig |
| acceptance.yaml（三工具） | OK SandboxedLlmAgentConfig |
| selector.yaml（含 submit_patch） | OK SandboxedLlmAgentConfig |

- `!include` 全部在 `research/012/agent/` 根内解析，无越界。
- `agent_tool` 均只用 `config_path`，无 `code`；无条件分支键（符合官方无 if/switch 的约束）。

## 未覆盖（待技能包引入与 GPU 作业）

- 完整 `compile_submission`（工具/模型注册表、技能目录 `SKILL.md` 落盘、adapter 发现）
  尚未运行：`research/012/agent/skills/` 当前仅 REUSE.md，三个技能包正在引入中。
- 单服务按角色 adapter 路由实测、正式包体积与扩展名检查，待 GPU/打包阶段。
