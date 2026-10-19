# P012 skills 说明：已引入（不再是复用引用）

日期：2026-10-19（更新；初版 2026-10-19 为“未复制避免分叉”）。

- 已把 `research/005/candidates/single_state/agent/skills/` 下三个合法 kebab-case
  技能包完整引入本目录：`working-state`、`source-edit`、`source-probe`
  （保持连字符名，无下划线版）。
- 仍不引入 `research/005/candidates/one_handoff/agent/skills/source_edit`
  等含下划线包（已有官方编译拒绝先例）。
- `research/005` 原件未动；复制时刻 11 个文件 SHA 与 source 逐一一致，
  见 `VERIFY.md` §1 全值表。
- 本轮唯一改动：`working-state/scripts/working_state.py`
  `4393d21201f53583756a4989ba71c52f799778f225259824b7e4ef2af2` →
  `b22e601b7dd1afa2e3b0edeecd299988f13243db2f699f3fd3ecc29ebefca24c`；
  其余 10 个文件与 source 字节一致。

## 1. 新增参数（`working_state.py` P012 候选命名空间，后向兼容）

`record` / `show` / `record-issue` / `write-handoff` 新增：

- `--candidate ID`：候选 id（`A`/`B`；缺省 `""` = legacy 单份文件行为）。
- `--stage STAGE`：阶段（`repair`/`acceptance`/`locator`/`selector` 等；
  缺省 `""` = legacy 行为）。

文件名规则（同 state 目录下按后缀区分，缺省候选时行为不变）：

- 均缺省：`state.json` / `state.prev.json` / `issue.json` / `handoff.json`。
- 仅候选：`state_A.json` / `state_A.prev.json` / `issue_A.json` / `handoff_A.json`。
- 候选 + 阶段：`state_A_repair.json` / `state_A_repair.prev.json` /
  `issue_A_repair.json` / `handoff_A_repair.json`。
- 命名空间 token 与 state id 同字符集（1–32 位 `[A-Za-z0-9_-]`），非法直接报错；
  ok 输出新增 `candidate` / `stage` 回显字段。
- `read-handoff` 保持按 `--handle` 读取，无需命名空间参数。

效果：修复 A、验收 A、修复 B 的运行记录互不覆盖（验证见 `VERIFY.md` §3–§4）。

## 2. patch_store 候选绑定（存储未改，调用方保证）

`source-edit/scripts/patch_store.py` 原样复用（SHA
`ddc96d8c9dcb4940193b9f959ef7e2255bc4199d6f4ef552273b756f79e0cb36`，
4 快照 / 8 文件 / 单文件 200KiB / 合计 512KiB 上限与 HEAD 硬拒保持不变）。
候选绑定由调用方传 `parent_patch_id` 链保证：同一候选的后继快照以该候选
快照 id 为 `--parent`（A 链 / B 链），并把所用 `patch_id` 经
`record --candidate <ID> --stage <STAGE> --current-patch-id <p_xxx>`
记入对应命名空间 state；选择器按候选验收结论恢复指定 `patch_id`。
冒烟见 `VERIFY.md` §5。
