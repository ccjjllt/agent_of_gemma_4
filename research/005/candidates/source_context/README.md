# P005 source_context：共用源码 / 状态 / 执行路径组件（实现完成，尚未跑模型）

`scripts/` 下三个命令脚本与共享 `workspace_paths.py`（stdlib only）。打包时将辅助模块复制到每个技能的scripts目录，无symlink或仓库外import。

2026-10-10接线更新：默认 `/workspace` 在Docker中直接使用，在官方subprocess中通过匹配的 `PYTHONPATH=<task>/workspace` 与 `VIRTUAL_ENV=<task>/venv` 识别实际路径；不依赖每次技能释放时的cwd。record-issue接受本题临时文件，`/tmp/...`按任务临时目录解析，无需诊断flag。补丁restore已恢复整个快照工作树状态，包括快照后修改的其他tracked文件及新文件；HEAD变更拒绝恢复。

| 脚本 | 入口 | 复用来源 |
|---|---|---|
| `working_state.py` | `record` / `show`（I2 schema）+ `record-issue` / `write-handoff` / `read-handoff`（I4） | T3 四问合同→`issue_contract`；frontier E2A §B 双假设状态；状态目录绑定复用 T2 `span_store` 思想（显式 `--root`、禁 symlink root、沙箱系统临时目录、模型不可自定状态路径） |
| `source_probe.py` | `locate`（queries 1–4 + question）/ `inspect`（targets 1–3，`exact`/`outline`）/ `probe`（command + question + timeout） | T4 `layered_lookup` 的 outline / locate / AST 完整读取与输出预算思想；`probe` 借鉴公开 v19 的 `sys.setprofile` / `threading.setprofile` 过滤任务仓库 |
| `patch_store.py` | `snapshot` / `list` / `restore`（I1 diff 快照，最多 4 份） | T2 `source_edit` 的 `inspect`/`apply` + 哈希校验思想；状态放沙箱临时目录绑定工作区 |

各 agent 打包时把需要的文件**真实拷贝**进自己的 `skills/*/scripts/`，
不依赖本目录相对路径、不使用 symlink（交接 §2 打包规则）。

## 小补丁说明：完整 JSON 预算（代替 T2 复制导航的直接截断）

冻结 `research/002/variants/t1/agent/skills/repo_navigation/scripts/read_range.py`
末尾是 `print(blob[:4000])`：先完整序列化再**切断 JSON 字符串**，
可能返回半截 JSON。本候选不改任何冻结文件（T1/T2 保持原样），新脚本统一用
**完整 JSON 预算**：先缩文本字段（snippet/claim/tail，设 `truncated` 标记），
绝不切 JSON 半串；实在放不下返回最小合法错误 JSON 要求缩小范围。
`source_probe.py::fit_json`、`working_state.py::emit`、
`patch_store.py::emit` 均为此实现。若将来复用 `read_range.py`，
必须在外层 wrapper 做同样的“先缩字段、再判长度”处理，不得直接切片。

## 其它实现要点（对应交接 §3）

- **I1**：可编辑句柄只由 `source_edit inspect` 对完整展示签发（T2 原逻辑，
  本目录不重复实现）；`source_probe inspect` 永不签发 `span_id`。
  截断片段不能编辑。所有大文本参数走单 `run_command` 字符串 + 稳定 skill
  入口，避免多层转义；`working_state record` 用 `|` 分隔的结构化 flag
  也是为此（claim 含 `|` 时改用 `--facts-json` / `--hyps-json`）。
- **I2**：`record` 只记录不推断根因；`hypotheses` 最多 2 个，
  `layer` 限 `producer|transformer|consumer|boundary`；`show` 只展示
  `changed_fields` + 当前 `next_action`，不做长表输出。
- **I3**：`locate`/`inspect` 的关系边只标 `textual_reference`，
  实际运行捕获才标 `observed_call`（仅 `probe --trace` 输出）。
  `probe` 退出 0 ≠ 复现/修复，只有具名 `--assert` 算观察；
  脚本放系统临时目录，不收集宿主文件，不建全调用图，不强制每题运行。
- **I4**：`record-issue` 的问题文本由模型传入（`--issue-file`），
  skill 不读宿主 tasks、不宣称 host 自动复制；`write-handoff` 的 issue
  来自上步、代码重读自已登记路径、diff 采自当前工作树；
  普通 skill 不写 ADK `session.state`，最终文本 handle
  由 `output_key: repair_context` 保存（见 `one_handoff`）。
- 快照（`patch_store`）：`parent_patch_id` + 文件哈希显式记录；
  恢复是明确动作，不自动选“最佳”；超 4 份直接失败并列出现有 id。
