# P016 FSDP 真续训实现与 CPU 实跑记录

2026-10-22，北京时间。本次只完成 CPU 实现与一次性手动实验，未启动 GPU、未安装依赖、未读取 heldout，也未新增测试文件或套件。这里的成功不代表 31B / NF4 / CUDA 续训已经实测。

## 实现边界

`training/train_fsdp.py` 现在在 optimizer step 边界保存完整的训练恢复点：

- 标准独立 PEFT LoRA adapter，仍沿用原有导出 key 与配置格式。
- 每个 rank 自己的 AdamW 状态。FSDP1 已将 trainable FlatParameter 分片；保存的是各 rank 实际持有的优化器状态，不把 rank0 状态冒充完整状态。
- scheduler、global step、下一 epoch / batch 游标、当前 epoch 批次顺序 SHA256、累计有效输入与监督 token。
- 每 rank 的 Python、NumPy、Torch CPU 和该 rank CUDA device RNG；保存操作结束后恢复同一 RNG，使保存本身不改变下一次随机前向。
- 已验证 base/config/data/protocol 的整体 signature，以及 world size、依赖版本、设备类型、distributed backend。恢复时还逐参数核对优化器顺序、原始 FQN / 形状及 shard 布局。

当前只支持 **相同 world size、相同运行版本与包装布局** 的续训，不支持跨拓扑 optimizer reshard。保持这个限制可以直接复用原生 `optimizer.state_dict()`，不导出冻结 NF4 基座，也不引入另一套 checkpoint 框架。基座、tokenizer、工具及训练数据仍按原代码的来源指纹重新确认。

`training/fsdp_checkpoint.py` 是生产续训功能模块。原有 Notebook builder 若逐项列出打包文件，后续生成 P016 运行载体时必须将它加入训练包；只更新旧 `train_fsdp.py` 而漏掉此模块会导入失败。历史已经冻结的 P015 carrier 不应改写。

## 保存与恢复

所有 rank 在同一个临时 `.checkpoint-*-incomplete-*` 目录写入各自状态。rank0 核对并记录各文件的大小与 SHA256，写出 `complete.json`，完成 fsync 后通过同盘目录 rename 一次提交为 `checkpoint-NNNNNN`。旧完整 checkpoint 保留；半写目录没有被当作可恢复记录。

常规文件写入、版本核对和布局失败会传播到全部 rank；rank0 adapter 文件导出错误也会广播，不再让其他 rank 只等待原先末尾 barrier。

恢复命令仍沿用现有 CLI：

```sh
torchrun --standalone --nproc-per-node=4 -m training.train_fsdp \
  --config /path/to/frozen-config.yaml --data /path/to/frozen-encoded.jsonl \
  --out /persistent/run --resume /persistent/run/checkpoint-000100
```

这只是后续运行方式，没有在本次执行。`--resume` 必须指向 `--out` 内最新完整 checkpoint。参数、总训练步数、数据和采样顺序不允许在同一续训中悄悄改变；扩充数据或修改配方应另起明确阶段。旧 adapter-only checkpoint 会被拒绝。

原 `run_config.json` 和历史 `metrics.jsonl` 保留。恢复进程写独立 `resume-config-*.json` 与 `metrics-resume-*.jsonl`，训练 manifest 指明本次日志文件，避免崩溃前已经打印、但未提交 checkpoint 的步骤混成一次连续有效训练。有效 token 计数从 checkpoint 恢复；没有把此前已完成部分重置为零。

若在最后一次更新的完整 checkpoint 提交后、最终 adapter/manifest 导出前中断，再次恢复只补最终导出，不重复更新，也不做伪造前向。累计时间仅用于恢复后的训练运行诊断：包含已记录的运行区间，不包括中断后丢弃的工作、排队或完整平台账单；因此不能把它用作 GPU 实际费用。

## 一次性 CPU 实验结果

复用 `/private/tmp/p009-replay-env/bin/python`：Python 3.11、Torch 2.8.0、Transformers 5.13.1、PEFT 0.21.2。实验使用两个独立解释器进程、真实 Gloo / FSDP1、单层随机初始化 Llama、q/v LoRA r2。它是可重建的机制诊断，没有冒充真实训练语料。注意力 dropout 0.1、LoRA dropout 0.15，以实际随机前向覆盖 RNG 恢复。

五个不同长度样本、每 rank microbatch 1、累积 2、两 epochs，共四次更新；每个 epoch 的尾批只有一个真实样本，另一 rank 零权重参加 collectives。

| 手动运行 | 实际结果 |
|---|---|
| 两 rank 连续完成四步 | 两进程 exit 0 |
| 第一步 checkpoint 完成后两进程立即 `os._exit(17)` | 两进程均真实 exit 17；checkpoint 完整存在 |
| 启动两个新进程、加载第一步 checkpoint 后继续至第四步 | 两进程 exit 0；从 epoch 0 / next_batch 2 继续，跨 epoch 与尾批均完成 |
| 与连续路径比较最终 adapter | 四个 LoRA tensor 全部逐元素精确相同 |
| 比较两 rank 最终 optimizer / scheduler / Python、NumPy、Torch RNG | 全部精确相同 |
| 比较最终有效计数 | 均为 step 4、epoch 2 / next_batch 0、100 输入 token、70 监督 token |
| 从最终第四步 checkpoint 再恢复 | 两进程 exit 0，仅导出，新增监督 token 为 0 |
| 不兼容或损坏的恢复输入 | world size、版本、数据 signature、参数顺序/布局、破损 rank payload、缺少完成标记均实际拒绝 |

原始一次性日志、微型 checkpoint 和比较结果位于：

`/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/p016-fsdp-resume-6j760ucn/`

其中 `comparison.json` 保存连续/恢复比较，`resume-final-rank-0.json` 和 `resume-final-rank-1.json` 保存最终导出恢复与实际拒绝信息。执行源直接由 Shell/Python here-doc 传给子解释器，没有保存专用测试脚本。

首次沙箱内 Gloo 被本机 `uv_bind: operation not permitted` 阻止，两进程 exit -6，没有得到训练证据；随后经一次性本机回环通信审批成功完成上述实验。该失败未被当成模型或训练实现失败。

## GPU 到位后仍需确认

在用户提供的实际 GPU 机器和冻结版本上，以少量真实 train 数据复做保存后进程退出与继续更新，确认 NF4 bytes / CUDA RNG / NCCL / 显存与持久化路径；CPU 小模型结果不能替代这一步。还需据真实 checkpoint 写盘开销选择保存频率。上述实现未自动调大 batch、rank 或序列长度，也未改变正式 CT 部署要求。

## 新协议阶段的显式 adapter 初始化

随后已实现 `--init-run /path/to/completed-run`，与 `--resume` 互斥。它读取已完成 FSDP run 的真实 `training_manifest.json` 和独立 adapter，核对来源签名、官方 NF4 缓存身份、权重 SHA 以及 LoRA rank/alpha/dropout/模块，然后在新输出目录开始一个明确的新阶段。新阶段允许改变数据、工具协议及学习率；optimizer、scheduler、随机状态初始化、数据游标和累计 token 重新开始，不伪装成旧训练的连续恢复。来源 manifest/adapter 的指纹绑定新阶段签名，后续该阶段 resume 保留来源链。

CPU 双 rank 手动实验确认：新阶段初始 LoRA 参数与前阶段最终 adapter 逐项相同；前阶段 AdamW 已到 step4，新阶段首次保存为 step1、scheduler step1；新阶段只累计24输入/18监督token，没有带入前阶段100/70。原始实验材料在 `/var/folders/9s/sdk1x32s58g8dycwh6c74bqc0000gp/T/p016-warm-start-x7zy4zzl/`。另外使用本地已有真实 P015 官方来源 adapter/manifest 完成只读身份核对，改变协议/学习率可作为新阶段、alpha不匹配被拒绝；这不意味着该诊断权重具有修复收益。

当前初始化入口接受已完成run，不声称支持任意外部adapter或不同拓扑/基座迁移；GPU warm-start同样尚未实测。
