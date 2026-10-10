# P005 one_handoff agent（实现完成，尚未跑模型、无成绩）

2026-10-10：工作区和本题临时issue文件接线已修正；显式eval_config限制两阶段共享总预算。文件级handoff在模拟官方subprocess环境的真实CLI中通过；ADK模型消息历史替换仍未实跑，本包不混入single_state单臂载体。

同工具、同总预算的交接版。对照 `single_state` 回答：去掉旧探索历史、只给
一次真实交接，是否改善后续动作。不覆盖 P002 冻结变体/历史结果，不读 heldout。

## 三段职责（交接 §I4）

1. **入口保存问题**（`explorer.md` + `working_state record-issue`）。
   探索者在原始任务上下文中读到可见问题文本，第一个动作即用
   `--issue-file` 把该文本 + 预算字段传给 skill。绝不使用 `{problem_statement}`
   等未确认变量，不读宿主 tasks；不宣称 host 自动逐字节复制，长问题不无声
   缩成一句话（原文逐字存）。
2. **工具生成工作状态文件**（`working_state write-handoff`）。
   skill 在本题沙箱临时目录写 `handoff.json`：issue 来自上一步存储，
   代码从已登记路径/范围重读原文，diff 从当前工作树采集，假设/反证/下一
   动作由探索者显式传入。普通 skill 不写 ADK `session.state`。
   探索者最终文本只输出 handle 路径一行，由 `output_key: repair_context`
   保存 —— 存的是该最终文本，不是自动执行的 JSON 合并器。
3. **执行者使用新上下文**（`executor.md`，`include_contents: none`）。
   首动作即用同一 skill `read-handoff` 读 handle，取得原 issue、真实源码
   和 diff 后执行自己的 `next_action`；此后新工具调用/回包正常进入历史。
   最多一次交接，不回流、不辩论。文件在共享题目沙箱，不在每次调用即回收
   的 skill 临时目录。

`agent.yaml` 为 `SequentialAgent`（explorer → executor），两 sub-agent 同
6 工具、同 `configs/sampling.yaml`。软时间目标：探索约 1/3、执行其余，
只写在 prompt 里，无 host 强制。

## 状态卡 vs 真实历史替换（必须如实区分）

- 本设计目标是**真实替换**：executor `include_contents: none`，旧搜索历史
  不再传入，只有 `repair_context` 中的 handle 可达。
- 若实际消息核查发现原 issue/执行期新观察并未正确传递、实际只相当于在原
  历史后追加一张卡，则研究报告必须如实写“状态卡”，不得称“上下文重构”。
  不为此重写 runner。
- `agent_tool.skip_summarization` 声明：顺序主路径不需要它；若把 explorer
  改作 agent-as-tool 被动调用，须照官方 schema 这样声明以保留原始输出：

  ```yaml
  tools:
    - agent_tool:
        config_path: explorer.yaml
        skip_summarization: true
  ```

## handoff 大小建议

目标约 2–4K tokens（`write-handoff` 报告 `token_estimate`）。超限时保代码、
减叙述：先缩 `unknown`/`purpose`/claims，保留 issue 全文与关键函数原文；
仍超则调用失败并要求缩小登记范围，不得把关键分支截掉冒充完整。

## 与 single_state 的比较控制

先用同一 sampling 比较交接本身；再单变量研究探索 4096 / 机械执行 1024 等
差异。记录 task、环境/协议版本、adapter 指纹、最终 patch、原始工具动作、
实际预算与 resolved；解释指标看假设是否被新证据更新、首动作是否回应
`next_action`。
