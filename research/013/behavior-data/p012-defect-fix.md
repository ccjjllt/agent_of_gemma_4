# P012 produce_p012.py 虚假完成标签缺陷 → P013 修正说明（CPU diff，未放量）

日期：2026-10-20。未重跑生产器放量、未覆盖 `research/012/training-data/encoded-train-12b.jsonl`
（P012 frozen 策略：该文件已冻结，见 `research/012/training-data/dataset-manifest.json` `frozen_policy`）。

## 缺陷原文（已核实，`research/012/training-data/produce_p012.py` 第 273-290 行，核心赋值在 277/286）

```python
handover = {
    'candidate_id': 'p012-' + row['instance_id'],
    ...
    'requirement_done': True,   # ← 缺陷：本会话仅局部编辑+get_status，无行为验证
    'requirement_scope': 'local production edits applied and syntactically parsed; full-task validation NOT performed in this session',
    'verification': {
        'observation': 'get_status',
        ...
        'scope': 'post-edit local patch status of the same workspace; no repository test suite or whole-issue oracle executed',
    },
    ...
    'needs_other_route': False,
}
```

同一生产器 `observed_outcome`（第 325-328 行）诚实记录 `behavior_verified: False, task_resolved: None`，
`dataset-manifest.json` 第 44-48 行 `layers.p012_repair_local_edit` 亦标 `behavior_verified: false, task_resolved: null`；
但被监督的尾部 assistant 交接消息（第 288-290 行 `rec.messages.append` + 第 301-306 行纳入 `target_assistant_steps`）
硬编码 `'requirement_done': True` + `'needs_other_route': False`，模型被监督学习"无验证却报告完成"。

## P013 修正（patch 形式，待新生产器 produce_p013.py 采用；本次只落盘说明，不改 P012 冻结文件）

```diff
-            'requirement_done': True,
+            'requirement_done': False,
+            'verification_status': 'not_run',
             'requirement_scope': 'local production edits applied and syntactically parsed; full-task validation NOT performed in this session',
             ...
-            'needs_other_route': False,
+            'needs_other_route': None,  # unknown: no whole-issue oracle executed
```

配套（同文件）：

1. `observed_outcome` 保留 `behavior_verified: False, task_resolved: None`（已诚实，无需改）。
2. 监督目标：局部编辑样本仅监督 `edit_file` 动作步；尾部 handover 若保留，必须以修正后的
   `requirement_done: False / verification_status: not_run` 形式出现，或整步移出 `target_assistant_steps`。
   禁止监督任何"验证通过/已完成"表述。
3. 主/辅层统计分离：`dataset-manifest.json` 新增层（如 `p013_behavior_recovery`）单独计数行数、
   `behavior_verified: true` 比例、修前失败→修后通过闭环数；旧 `p012_repair_local_edit` 层标记为
   辅助层并控制其每 epoch 采样占比与有效监督 token 权重，防止 256 条静态局部补丁淹没行为信号。
   `task_resolved` 仅在真实同判据通过后方可为 true；`behavior_verified: false` 的样本永不记入 32-64 条行为配额。

## 首批行为轨迹生产计划（CPU 计划，执行待 GPU/环境）

- 目标：前 12 小时 ≥32 条不同训练问题真实轨迹（读取→修前失败→完整修改→报错恢复→同判据通过→准确交接），24 小时向 64 条推进；不等待 512 题。
- 复用资产（已冻结清单，免重建）：`research/012/evaluation/train-repos-71.json` 71 仓并集；
  优先 P009 已验证轻量仓 `marshmallow-code/apispec`、`marshmallow-code/marshmallow`、`pydantic/pydantic`、
  `jd/tenacity`、`python/mypy`、`seperman/deepdiff`、`adrienverge/yamllint`、`pudo/dataset`。
  协议与编码沿用 `gemma_tool_chat.py` 零基 assistant 序号校验（`--thinking on --thought-loss mask`）。
- 真实性铁律：验证命令一律 `PYTHONPATH=/workspace/src:/workspace` + 打印模块路径；修前必须有具体失败、
  修后同一判据通过；锚点/参数/分支错误必须在真实工具执行中发生，严禁注入拼写错误、拼接他题报错或伪造回包；
  未验证不标完成；开发题（公共 12、外部 8 仓、heldout 28、旧预留 8）正文/补丁/测试永不进训练集。
- SFT 启动：首批 32+ 条合格行为数据到位即启动 12B 4×L4 DDP（r=8, alpha=16, global_batch=4, lr=2e-5，
  新建 adapter+optimizer，不 `--resume` 旧协议 checkpoint）；12B 见同预算修复收益后再评估向 31B 迁移。
- 配额：执行前同命令内校验账号身份 + 重读余量；本轮 CPU 未消耗 GPU。
