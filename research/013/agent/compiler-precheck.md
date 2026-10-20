# P013 Agent YAML 官方 schema 预检（CPU，已执行）

日期：2026-10-20。方法：一次性手动命令，非测试套件。

## 方法

- 用官方 `research/006/official-data-20261013/wheels-source/adk_submission/` 的
  `yaml_loader.load_yaml(path, root_dir)` + `schema.SandboxedSequentialAgentConfig` /
  `SandboxedLlmAgentConfig`（`!include` 沙箱解析 + pydantic 校验），`root_dir=research/013/agent`。
- 五个纯依赖模块（errors/paths/limits/schema/yaml_loader）复制到隔离临时包做 schema 级校验；
  依赖 pydantic + pyyaml 装入隔离 target（homebrew python3.11），仓库外，不进 git。

## 结果

| 文件 | 结论 |
|---|---|
| agent.yaml（SequentialAgent → repair_agent.yaml → review_agent.yaml） | OK SandboxedSequentialAgentConfig |
| repair_agent.yaml（五工具，无 submit_patch，output_key=repair_handoff） | OK SandboxedLlmAgentConfig |
| review_agent.yaml（六工具含 submit_patch，include_contents=none） | OK SandboxedLlmAgentConfig |
| agent-single.yaml（同改进单 Agent，六工具） | OK SandboxedLlmAgentConfig |

- `!include` 全部在 `research/013/agent/` 根内解析，无越界。
- 工具条目均为字符串工具名，无 `code`；根为 SequentialAgent + `config_path` 子引用，无条件分支键。

## 未覆盖（待 GPU 打包阶段）

- 完整 `compile_submission`（工具/模型注册表、adapter 发现、包体积与扩展名检查）尚未运行。
- 单服务 31B 下两阶段真实 turns/调用分布（repair 是否按时交接）待 GPU 旧题诊断确认；若 repair 耗尽预算，启用同改进单 Agent。
