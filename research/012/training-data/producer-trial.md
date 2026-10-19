# P012 生产器小批量试跑记录（producer-trial.md）

日期：2026-10-19。生产器：`research/012/training-data/produce_p012.py`
（`producer_revision=p012-local-1`，协议 `p012-repair-v1`，五工具，
canonical tools SHA `91ec82b1…` 已核对）。

## 1. 试跑命令与结果

```bash
python3 research/012/training-data/produce_p012.py --limit 5 --workers 2
```

处理对象：`data-usage-p012.json selected[0:5]`（即 P011 `selected[210:215]`，顺序逐项核对一致）。

| instance_id | repo | 状态 | 备注 |
|---|---|---|---|
| encode__starlette.db5063c2.pr_2577 | encode/starlette | local_edits_applied | transaction.json 已产出，无 submit 工具调用 |
| facebookresearch__hydra.0f03eb60.pr_2934 | facebookresearch/hydra | local_edits_applied | 同上 |
| getmoto__moto.694ce1f4.pr_5620 | getmoto/moto | local_edits_applied | 同上 |
| graphql-python__graphene.82903263.pr_1506 | graphql-python/graphene | replay_failed | 失败原因原文：`Selected read did not expose the complete exact edit context` |
| iterative__dvc.1d6ea681.pr_9089 | iterative/dvc | local_edits_applied | targets 8（含 6 处生产修改 + get_status + 交接消息） |

汇总：`attempted=5, applied=4`。链路确认：parquet 原子读取 ✓、
`git ls-remote`（swesmith 镜像） ✓、`raw.githubusercontent.com` 下载 ✓（含 LICENSE）、
本地回放（read/edit/get_status 真实官方函数） ✓、transaction.json 落盘 ✓。

## 2. 耗时（实测原文）

- 全新 5 行（含下载与回放，workers=2）：`seconds=6.445375542000001`
- 复跑 5 行（4 复用 + graphene 重新失败，workers=2）：`seconds=5.3334283749999996`
- 参照：P011 全量 `attempted=210, applied=149, seconds=75.57937070901971`
 （`research/011/data-scaling-20261018/materialize-batch.log` 末行原文）。

## 3. 中途插曲（无数据影响，如实记录）

试跑中共遇到 2 个生产器自身 bug，均在写事务前/打印时触发，已修复并重跑验证：

1. `importlib.util.module_from_spec` 误传 2 个位置参数（应为 `spec_from_file_location` +
   `module_from_spec`，与 P011 写法对齐）。触发于 import 阶段，无任何写操作。
2. 结果打印行把 `flush=True` 误写进 `json.dumps()` 内。该轮 5 行事务实际已全部写盘，
   仅首行打印时崩溃；修复后重跑，4 成功行经 provenance+revision 双重核对命中复用，
   dvc 行重新完整回放一次（targets 一致），graphene 行两次运行失败原因原文完全相同
   （确定性锚点失败，非网络抖动）。

## 4. 失败原因说明（不放宽）

graphene `pr_1506` 的 `Selected read did not expose the complete exact edit context`
是 P011 已有失败类别（P011 210 行中有 43 例 `replay_failed`，多为此类锚点读取失败），
不是模型工具失败恢复。本阶段不伪造、不重试改写，只记录。

## 5. 仓作用域说明（已核对的事实，不回避矛盾）

`selected[210:]` 的 26 个仓库经过程序核对 **626/626 全部落在 71 仓并集内**。
这是训练扩量的预期形态：交接要求“沿已冻结 train 来源与固定 commit 继续批量物化”，
256 档缺的 117 是**独立新问题数**（同仓新问题计入），不是新仓库数。
71 仓 + 外部 8 主选仓的硬排除作用于评测 dev 选题，不作用于训练物化；
本生产器沿用 P011 的 15 仓排除表（逐项核对一致），未引入训练外新仓，
未读取 heldout/预留/dev 新题正文。

## 6. 全量 117 外推与下一步建议

- 转换率：P011 先验 149/210 = **71.0%**；本次 4/5 = 80%（n=5，仅作一致性参考，不作估计）。
- 按 71% 计，117 个成功需处理约 **165 行**；候选池 626 行充足。
  若 512 档需 373 个成功，按 71% 需约 525 行——池内刚好覆盖，无余量，
  且后段 moto/pydantic/dvc 占比 56.4%，失败结构可能与前段不同，**不能线性外推**。
- 耗时：按本次 pace（~1.3 秒/行，workers=2）165 行约 **3–4 分钟墙钟**；
  按 P011 pace（~0.36 秒/行）约 1 分钟。均为 CPU 分钟级，**建议直接放量**：
  先跑 `selected[0:190]`（约 190 行，期望 ~135 成功，覆盖 117 + 缓冲），
  保留失败原文后再决定是否继续向后。
- 行为升级：当前 4 条均为局部编辑事务（`behavior=local_edit`，
  `task_resolved=null`，`behavior_verified=false`），**不计入 32–64 条完整修复/
  错误恢复过程**。下一步需另起行为重放步骤（真实失败→修正→同判据通过），
  不得用更多局部样本宣称行为缺口已补。
- 编码注意：事务尾部交接消息为无 tool_calls 的 assistant 文本 turn
 （`tool_calls: []`），新协议编码时需先用 `training/encoding/gemma_tool_chat.py`
  验证该形状可正常编码、监督 token 仅落在 edits + get_status + 交接三类序号上，
  然后再做 256 档组装。

## 7. 追加修复（2026-10-19）：targets 口径 bug → p012-local-2

试验编码时发现 `target_assistant_steps` 存的是消息索引（如 `[1,3,4,12]`），
而 `gemma_tool_chat.encode_transaction` 要求零基 assistant 序号，编码拒收
（`target_assistant_steps needs zero-based assistant ordinals`）。
已修复：`produce_p012.py` 改为收尾时按最终 messages 计算 assistant 序号
（成功 `edit_file` turn + `get_status` turn + 尾部交接 turn），
`PRODUCER_REVISION` 升为 `p012-local-2`，旧 revision 输出不复用。
4 条重跑后 targets 如 `[1,2,3]`，`--thinking on --thought-loss mask` 下
3/4 成功编码（1 条 8778 tokens 整条跳过，无截断），监督 tokens 正常。
