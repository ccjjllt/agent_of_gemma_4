# 提交后 sampling 异常的真实结果保全

2026-10-28。生产 `run.py` SHA `9fd1afd97180084a4a7f3aa9e616facbc5d40070b6a87b6f339fe88c18344ea5`。只改该文件：正常尾部和异常路径共用 `persist_and_verify_episode_patch`。普通 sampling Exception 仍包装为 fatal `PolicyProtocolError`；已有成功的非空提交时，先保存 submitted/evaluated patch、patch-capture，并尝试原始完整验收。第二次 finalization/verification 或 cache-cleanup 错误独立记录，不替换最初 sampling cause。异常路径不返回正常 episode，也不创建 focal pair。

此次只做一条 raw-preserved CPU 演示，复用正常证明的固定 train index26 Tenacity wait_fixed、原 source74文件／模式、当前 prompt／预算、原正确 patch 和官方 CPU runtime。真正的 `TaskEnvironment`、`SwegemmaContext` 和六工具 API 依次完成 get_status、read_file、edit_file、run_command、submit_patch。第6次 facade request 观察到 `ctx.patch_submitted=True` 后，人为抛出明确的 RuntimeError；该次没有返回任何 final-response token。这是 Codex 明示的受控异常，不是 Gemma 故障、模型采样或收益实验。

| 原始完整验收 | Cases | Failures | Errors/skips | 实际结果 |
| --- | ---: | ---: | --- | --- |
| Baseline | 124 | 13 | 0 / 0 | 原0 |
| 提交后异常的诊断验收 | 124 | 0 | 0 / 0 | 真实 diagnostic1 |

生产 episode 确实仍抛出 `PolicyProtocolError('Current-policy sampling failed')`，`__cause__` 保留第6次 RuntimeError。submitted/evaluated patch 与原正确 teacher patch 逐字一致；完整 XML、官方结果、原始异常栈和 diagnostic JSON 均保留。只有5个已完成 action／tool event、6个请求，无新 terminal 输出、nudge 或 focal 文件。没有正常 episode summary。顶层 collect 没有被此次驱动调用；其保留 fatal 和诊断指针的源路径已读审，不能把此次结果说成一个实际可训练 collect pair。

正常证明 [post-submit-cpu-episode-r1](../post-submit-cpu-episode-r1/README.md) 保持原 `bde8…` 源版本及结果不变。新加载源码原字节在 `loaded-source/`，pre/post/end 文件哈希一致；实际导入的 episode code object 与这些字节的独立编译结构相等。最初一个过严 marshal 字节比较在输出目录创建和任何工具／验收之前停止；实际 code object 及关键字段相同，随后只修正这一预执行比较。该准备失败没有被记作奖励0。

shared helper 的正常路径只做窄重构，未重跑正常 case：机械源码对照确认 `final_patch`、`TaskEnvironment.verify/_verify` 函数原语逐字不变；默认 label 仍为 `out.name`，同一 evaluated patch 送入同一 `env.verify`，返回 verification 路径仍为 `env.output/(out.name+'-result.json')`。cache release 与 stop_agent_session 仍在正常 finalization 之前。唯一顺序差异是 `submitted.patch` 现在先于 `final_patch` 写出，因此后续持久化失败时也尽早保留原始提交；没有替换或重新归属旧正常结果。正常证明属于 `bde8…`，此次异常证明属于 `9fd1…`，不能把前者标为新源码实跑。

此次 secondary verification 实际通过，未伪造第二个验收失败。若未来原验收失败或未知，其实际结果也只能作为诊断保存；不能借此吞掉策略身份／缓存故障，或把 fatal attempt 准入训练。Cancellation 等 BaseException 控制流不在此次普通 Exception 恢复范围。

主要结果：[result.json](result.json)，SHA `c45f7b662aac5cd4400fe6ad4a8dbeb1668b666ea263ddfdad0b4829752f7069`。

源差异与逐项证据：[narrow-production-change.patch](narrow-production-change.patch)、[artifact-bindings.json](artifact-bindings.json)（SHA `72a2a2a4d4d494537a42640c71f207ba788e76044aeb20e7969a132faead52f6`）。实际一次性输入原文在 `executed-inline.txt`，命令来源在 `execution-command.json`；它们是研究执行记录，不是新增测试套件。

没有 GPU、模型调用、下载、SFT 导出或正式训练；当前100主数据、全部99 terminal、原 oracle 和旧奖励保持不变。
