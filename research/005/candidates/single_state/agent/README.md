# P005 single_state agent（四题探索对照已完成）

当前配对研究入口为 [005-t0-single-state-pair.ipynb](../../../../../notebooks/005-t0-single-state-pair.ipynb)，使用说明见 notebooks 目录的 005-t0-single-state-pair.README.md。旧 005-single-state.ipynb 保留为历史入口，未重生成。本包已有显式 eval_config；workspace_paths 支持官方 Docker/subprocess 工作区。2026-10-11官方CT v2四题对照为本候选2/4、T0 1/4，但四题均未实际调用新增技能，不能声称state/handle技能获益；同补丁的rich_3518还多耗约139秒。[原始结果与解释](../../../t0-single-state-pair-20261011/results.md)。

2026-10-11 按 v30 / Google ADK 1.36.1 的默认命名规则，将三项技能统一为 `source-edit`、`working-state`、`source-probe`，同步目录、frontmatter 和提示引用；不启用 snake_case 功能开关。全部 8 个 Python 脚本字节保持不变，脚本文件名、内部模块、CLI 参数及行为不变。冻结 T0、one_handoff 和历史载体均不迁移。

单 Agent + 行为约束 + 当前状态版。与 P002 冻结变体（`research/002/...`）无
任何覆盖；历史结果不动；不读 heldout。

## 与 single_state 一起交付的内容

- `agent.yaml`：6 官方工具（`run_skill_script` 不进 `tools:`，由 `skills:`
  声明自动提供），`skills:` 声明三个自包含 skill，打包均为真实脚本拷贝，
  无 symlink，不引用仓库外路径（`source_context` 只作开发期来源，见下）。
- `configs/sampling.yaml`：继承 16384 输出上限 + thinking 新语义注释；
  4096（本文件）/1024 为新转发语义下的实验起点，非最优值。
- `prompts/system.md`：短段落行为约束（T2 句柄 → I1 事实保管 → issue 契约 →
  最多 2 假设 → 区分动作 → 推翻更新）。
- `skills/source-edit/`：T2 原型复刻（`source_edit.py` + `span_store.py`
  打包时字节一致拷贝，冻结原件不动）+ I1 快照扩展 `patch_store.py`
 （最多 4 份，`parent_patch_id` + 文件哈希，显式恢复，不自动选最佳）。
- `skills/working-state/`：`record`/`show`（I2 schema）+ handoff 往返入口。
- `skills/source-probe/`：`locate`/`inspect`/`probe` 三入口雏形。

## source_context 引用说明

`../../source_context/` 是开发期共用来源（复用映射与 JSON 预算补丁说明见
`../../source_context/README.md`）。本 agent 目录已把所需脚本完整拷贝进
`skills/*/scripts/`，提交包不依赖 `source_context` 路径。后续若改共用脚本，
重新拷贝并记录版本，不得改 P002 冻结文件。

## 手动验证（无测试文件，按任务要求全部手跑）

- `python3 -m py_compile` 三脚本通过；`--help` 正常。
- 合成小仓库手跑：`record`（2 假设）→ 第 3 假设被拒 → `drop+replace` 一次
  调用成功 → `show` 显示 `changed_fields` + `next_action`。
- `locate`（textual_reference）/ `inspect exact`（complete）/ `outline`
  （签名）/ `probe --trace`（有序函数 + 计数 + 短栈 + 具名断言，
  exit 0 注释）/ 异常分类修正。
- `patch_store`：snapshot → parent 链 → restore 恢复字节一致 →
  第 5 份被拒并列出现有 id。
- `handoff.json` 往返：`record-issue` → `write-handoff`（1996 chars，
  ~0.5K tokens）→ `read-handoff` 字段齐全。
