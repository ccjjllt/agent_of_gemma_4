# P016 六工具目标协议：复用六题的真实动作恢复

2026-10-23，北京时间。只使用 CPU、既有 tokenizer 和 P015 冻结 train 原件；新增 GPU、模型采样、网络下载、比赛提交均为零。未读取 dev/select/heldout 正文，未新增测试文件或套件。

实际产物在 [`../data/target-protocol/`](../data/target-protocol/)。可用于下一阶段的小规模目标协议适配候选；**仍只是六个独立问题的精选动作 RFT，不能称为完整自主任务 SFT，也不足以证明协议适配完成。**

## 本次真正增加了什么

| 指标 | P015 冻结首轮 | 本次实际产物 |
|---|---:|---:|
| 独立 train 问题 | 3 | 6 |
| 监督动作 | 14 | 70 |
| 编码序列 | 14 个单动作窗口 | 9 个同前缀多目标片段 |
| 监督 tokens | 1,014 | 6,230 |
| 生产源码编辑 | 3 | 16 |
| 提交动作 | 1 | 5 |
| 上游已有验证动作 | 0 | 0 |
| 可训真实 final reasoning | 0 | 0 |
| 最长序列 | 8,165 | 15,273 |

本次将长度上限放到 32,768 后，实际九条全部落在 **16,384 内**，不需要为了这些目标协议片段先运行 32K 训练。dask、graphene、oauthlib 的生产编辑资格恢复；这些是原有成功 draw 的再利用，不是新解决了三道题。

同样的 70 个目标若分别编码，其完整因果窗口合计 644,775 input tokens；将严格一致的前缀合并后为 **110,996**，减少 533,779（82.79%）。目标 token 数保持 6,230，监督密度由 0.966% 变为 **5.613%**。这是静态 token 工作量减少；GPU wall time 和真正提分均未测，不能把 82.79% 写成训练加速或能力增益。

| 冻结问题 / draw | 动作 | 监督 tokens | 生产 edit | 片段长度 |
|---|---:|---:|---:|---|
| dask-9212 / r1 | 8 | 353 | 2 | 3,248；14,399 |
| graphene-1478 / r1 | 7 | 286 | 1 | 13,923 |
| tldextract-258 / r2 | 5 | 199 | 1 | 12,607 |
| boltons-298 / 首轮 | 12 | 387 | 1 | 11,560 |
| marshmallow-2715 / r2 | 22 | 4,289 | 10 | 13,161；15,273 |
| oauthlib-771 / r1 | 16 | 716 | 1 | 12,278；14,547 |

全部动作分布：13 定位、32 源码读取、16 生产编辑、4 patch 复查、5 提交。其中 9 个动作带明确的错误恢复标记（恢复源码路径、失败 edit 后读取/重试、错误回退后重建有效修改）。工具 schema 是实际比赛六工具；正向目标只覆盖 `run_command` 19、`read_file` 32、`edit_file` 14、`submit_patch` 5，两个真实 shell 编辑算在 `run_command` 内。没有伪造 `write_file`、`get_status` 或验证动作来补齐覆盖。

## 来源和筛选依据

入口先读取六个已选 attempt 的 `result.json`、`agent-task.json`，核对 P015 `train-identity-freeze.json` 中的 issue/repo/base_commit，再要求 `split=train`、`resolved=true`、`judge_valid=true`、`agent_error` 为空；完成全部身份核对后才打开轨迹。保留此前按严格成功 draw 的工具数选出的六个 draw，不因为窗口长度换 draw，oauthlib 超时的 r2 仍拒绝。

复用 `on-policy-rft-first-v1/complete-action-windows/` 中已有的真实请求、当前动作、下一请求中的真实观察；核对原始 wire/model/ADK 哈希、prepared 事务哈希及原始仓库快照哈希。原始 `training_trace_usable` 与 consumer-close 审计独立保留，并未把整条 capture flag 改成成功。逐动作完整性依据仍是 P015 原有交叉核对，本次由 v4 重新验证序列化和逐目标因果前缀。

以下具体改进比单纯扩大条数更重要：

- dask 的 `edit_file` 连续缺少参数失败，因此不监督这些动作；监督真实执行成功的 `sed` 和 inline Python 生产编辑，全部失败历史仍保留。
- graphene 的 call 10 编辑被 call 14 撤回，再由 call 16 恢复。只监督最后保留的 call 16，不将三次动作算成三个正向编辑。
- tldextract 的 call 9 提交夹带三个 cache 文件；它原来因落在 8K 内入选，本次明确排除。call 15 checkout 错误撤回修复，call 22 重新修复；只监督后者，call 5 重复编辑保留为历史。
- tldextract 最后 call 25 虽是真实提交，但同时包含 tool call 和 visible content。固定 CT 模板会把该 content 放在未来 tool observation 之后，v4 因果核对实际拒绝。没有删 content 强行编码，也没有用早先污染提交替代。因此只有五道题有提交监督。
- marshmallow 保留十个实际有效编辑，包括 compaction 后失败 replacement 的读取及成功恢复；不因每题六条上限丢掉大部分修复。
- 新写 repro/helper 文件、运行或清理这些脚本、失败调用、重复查询、回退错误均不监督。它们若是选中动作的真实先前历史，则原样保留并 mask；没有事后改写为理想化短轨迹。

这六条轨迹没有可确认的上游已有测试运行，不能把运行新写 repro.py 当上游验证。所有普通 final 缺少可靠原始生成边界或为未审计的中断；没有拼接 parsed reasoning/content。

## 合并规则和实际核对

仅当早先窗口的完整 normalized messages 是较晚实际窗口的严格前缀，才把其真实 assistant ordinal 加为同序列目标。v4 对每个目标分别执行 generation-prefix 检查；额外逐 token 核对合并前目标处的完整前缀与标签不变，并要求合并前后监督 tokens 总数相同。dask、marshmallow、oauthlib 的真实 compaction 边界分成两段，不反向拼接丢失历史；其他问题各一段。

一次性手动 Python 核对结果：16 个生产编辑的实际 diff 新增行（dask 两条 shell 编辑对应原始 final patch 行）均出现在原 attempt 的最终 patch；九条产物均使用真实 tokenizer/v4、非 synthetic、无截断、全部 train；所有监督 span 均为 tool-call，历史、工具观察和 reasoning mask；数据文件哈希与 manifest 一致。核对没有执行原始 shell 命令或重放仓库，因此不声称本次重新验证了修复成功。原 host judge 成功只是已保存的既有证据。

重建命令，复用当前可用依赖：

```bash
PYTHONPATH=/private/tmp/p012libs311 /opt/homebrew/bin/python3.11 \
  research/016/target-protocol/build_target_protocol.py
```

`dataset-manifest.json` 保存源哈希、每个动作的入选/排除理由、合并关系、实际计数、脚本和编码器身份。`merged-transactions.jsonl` 是选中动作的完整可读状态；`encoded-data.jsonl` 是训练候选。旧 P015 冻结数据未改动。首个命令使用旧 `/private/tmp/p009-replay-env/bin/python` 时因其当前缺 transformers 失败，随后改用已缓存的 Python 3.11 依赖成功；没有安装新环境。

## 对接主训练的边界

这批数据只有六题，marshmallow 单题占监督 tokens **68.84%**；不能仅凭 70 个动作或 6,230 tokens 放大成充分的适配语料。九段仍含大量实际错误探索历史，正向验证和两个工具没有覆盖。合理用途是后续六工具适配的种子或受限短阶段，需与固定 Agent 下的真实权重 A/B 一起判断，而不是在准备期反复过采样到与数百题 Smith 等权。

主训练仍按当前 Smith 原生协议进行。切换到六工具时从 adapter 开始独立阶段，显式换协议/数据；不要把它接成相同 optimizer/data cursor 的断点续训。GPU 开启后，可先用这些真实 3.2K–15.3K 样本测吞吐与适配可用性，再决定是否值得扩大同协议独立题覆盖。
