# P011 连续SFT训练结果（V2四卡L4，70/70步完成）

复核日期2026-10-19。V2作业平台状态 `KernelWorkerStatus.COMPLETE`，本地已取回全部小文件证据、导出adapter（110MiB，SHA已核对）与最终完整checkpoint的
`training_state.pt`（220MiB）及四rank RNG；`checkpoint-000070` 内嵌adapter权重仍在取回中，不影响以下结论。
run为 `run-011-continuous-sft-20261018T110137Z-383c69c3`；本地原件根目录为 `gpu-output-v2/results/011/<run>/`（约440MiB：小文件证据、
导出adapter、最终`training_state.pt`、四rank RNG与末checkpoint内嵌adapter，SHA均已核对）。
V1（2×T4、0步）保持失败归档，不计入结果。

## 结论

**连续两轮SFT真正跑完：139个独立问题、58个仓库、global batch 4、70个optimizer steps全部执行，导出独立adapter。**
这仍是批量LoRA微调（r8/alpha16、LR2e-5线性衰减），不是全参数预训练；训练loss下降只是拟合证据，
**没有证明修复收益**——诊断（P010三题载体复用）与正式submission都尚未运行，`repair_gain=null`。

## 完成证据

- supervisor报告 `status=completed`、`timed_out=false`、exit 0，总 **1254.94秒（约20.9分钟）**；训练进程1179.28秒，
  optimizer循环984.05秒（metrics求和980.9秒，口径差约3秒为rank落盘开销）。
- 5个完整checkpoint：`checkpoint-000017/000034/000051/000068/000070`，各含`training_state.pt`、adapter与四rank RNG；
  最终progress为epoch 2、global_step 70。`phase.json`、`stage-summary.json`、`run-summary.json`三处adapter SHA一致。
- 导出adapter：`adapter_model.safetensors` 115,020,728字节，
  SHA256 `71687cf933c3570284521485466cb089bf44b7a01458809386eef6716e6530e2`（本地已复算一致）；
  `trainable_sha256`为`025eaedebc363c2cf871382bb232b778429fbbdde85027de760afe79b4fb9648`。
- 数据身份：输入SHA `ebfa8d1357c413e467fc71295c371f178f08932b10fb179b95a2014b1c8638b0` 与冻结清单一致；
  两轮合计903,948输入tokens、203,646监督tokens（恰为单轮451,974/101,823的两倍）。
- 硬件：4×NVIDIA L4，每卡23034MiB（`hardware.json`与运行时nvidia-smi一致）；峰值分配13.83–14.63GiB/rank，余量充足。

## 训练曲线（仅训练集loss，不作泛化声明）

70步完整记录在 `training/train-run/metrics.jsonl`。step1 loss 0.2927，末步0.1025；
epoch0均值0.2425（范围0.0425–0.5108），epoch1均值0.1335（范围0.0440–0.2545）。
学习率2e-5线性衰减至0。loss下降只说明该批数据可拟合；局部动作数据本身 `behavior_verified=false`，
不能把loss下降读成已验证修复。

实测约14.0秒/step（optimizer循环），快于按P009旧步时（25.09秒/step）的点估计17.6分钟纯循环。
差异来源未做归因：长度分布、监督密度、长度分组都不同，**不声称长度分组带来加速**。
stage总耗时1185秒，在5400秒上限内；平台7200秒未触发。

## 时间与配额

- V2前共享账号已用29389.877秒；完成后CLI读数已用8.87h（名义31932秒，CLI舍入到0.01h），剩余21.13h。
  名义差值约2542秒（约0.706额度小时），与2倍账单下的supervisor墙钟（2×1254.94=2509.9秒）相符，
  在舍入与平台开销范围内。这是账号共享累计差值，不是独占计费。原件为`quota-after-v2.json`。
- 下次刷新为2026-10-24T00:00:00（CLI返回）。

## 未完成与下一步

- `checkpoint-000070/adapter/adapter_model.safetensors`（110MiB）已取回并归档，SHA与导出adapter相同（`71687cf9…`），
  说明导出保留了原字节；本地归档共约440MiB，中间checkpoint（000017/34/51/68）的权重文件有意不归档，
  只保留其progress与adapter config。
- 下一步是复用 `training/diagnostic-handoff.json` 指定的P010载体做三题诊断（base对照），
  再决定是否正式submission；在此之前不声称排行榜收益。
- 512问题扩量仍是新数据集的新运行，不静默续训当前optimizer；是否启动取决于诊断结果与剩余额度（约21.13h）。
