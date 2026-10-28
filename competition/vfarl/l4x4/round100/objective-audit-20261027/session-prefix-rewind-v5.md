# 显式v5前缀rewind缓存

2026-10-28。原v4仅在新输入严格延伸已缓存raw prefix时复用。CT结构化历史会规范化当前assistant的换行和参数顺序，令raw suffix不同；把整个已知pre-action P也重新prefill是可避免的工作。本轮实现独立opt-in v5，不改默认raw历史或v3/v4选择。

## 机制与身份

每次action decode前，保存精确P的token tuple、policy/epoch/层metadata，并只clone各sliding层实际尾部KV。完整注意力KV留在decode后cache中。行停止时先按原机制抽出独占B1 cache，防其他行dummy更新驱逐该行sliding状态。下次先走原strict-extension命中；仅当新prompt严格延伸同一P时，full层crop回P、sliding层恢复所备尾部，再prefill新的canonical suffix。早期prefix变化、较短prompt或policy/epoch变化均fresh miss，不虚报命中。

full crop只是view，会保留原backing storage直到后续update/cat，不能称立刻释放原raw尾部显存。备份有额外存储成本，实际anchor tensor bytes单独统计；logical token budget不是整个CUDA显存上限。LRU、release、clear、异常及已dispatch请求被取消时都会释放或退休anchor，晚到put不能复活旧epoch。恢复操作仅用于B1独占或停止行抽取后，不回卷仍在变长batch中更新的共享状态。

代码在[training/vfarl_batched.py](/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/training/vfarl_batched.py)，API为configure_batched_policy/BatchedPolicyService的session_prefix_rewind=False。开启须有正数session cache，使用独立REWIND_SESSION_BATCHED_EXECUTION_PROFILE，version5，profile digest `5ab1c9b7dc64491a0a307fb311e1bcd27ca0bc4a48d92558be71f903375ead88`。actor_learner_binding另识别该profile，未改变loss。False保留原v3/v4 profile及runtime键/值；sampler源码SHA随修改更新，因此不伪称整个运行时身份相同。

## 可用入口与当前边界

builder的`--resident-session-prefix-rewind`要求resident模式和`--resident-session-cache-tokens`正数，沿runner/resident/bootstrap/worker传入真实service；直接resident CLI也提供session开关。默认False不新增旧bootstrap/settings字段。参数接线的实际CPU构造范围见[说明](../optimization-overnight/session-prefix-rewind-plumbing-r1/README.md)，使用惰性模型/进程替身，不当作硬件采样。

开关本身不会把rawappend改为结构化history；输入从未变化时只增加备份成本，不承诺已有raw运行更快。独立history_format=canonical-ct-v1的新实验另有[参数接线](../optimization-overnight/history-format-plumbing-r1/README.md)，只允许live pair，并可在cache0/rewind关闭时fresh-prefill。collector已通过[真实CPU工具循环及原完整oracle](../optimization-overnight/canonical-history-cpu-episode-r1/README.md)，覆盖连续call ID和坏调用后的raw回退；其加载源码边界单列，未包含Snapshot恢复或真实模型KV，不代表官方宽松parser或完整HTTP服务已复现。

## 真实CPU证据

[原型](real-gemma4-sliding-prefix-backup-20261028.json)及[实际v5 API](session-prefix-rewind-api-20261028.json)运行真实Transformers Gemma4小配置、FP32 eager，window4/P7。两请求实际采样循环完成停止行抽取、存储与rewind，再prefill不同canonical suffix，selected logprob对完整前缀最大差2.38e-7；随后变长packed decode的hidden误差小于1e-6。错误地仅重置sliding长度的控制出现0.13量级logprob差，说明所需尾部不能凭空裁回。

小配置每action的sliding备份实际192B，对照全KV clone为640B；这些是toy tensor存储数，不外推31B显存或吞吐。完整sample_batch在CPU仍受CUDA显存metadata汇总限制；已执行的真实请求与KV结果保留，异常未靠替换CUDA kernel/计数掩盖。最终service取消/异常释放证据另标最终源码版本，先前数学证据与随后仅取消处理修订的版本分界明确。没有模型下载、GPU、训练、optimizer或旧概率重算。

冻结零更新载体r4不含后续v5变更，也没有因此重写；它用于原pending pair的head/backbone数值探针，不能当v5硬件验证。当前仍缺真实31B/BF16/CUDA下的数值、容量和耗时证据。
