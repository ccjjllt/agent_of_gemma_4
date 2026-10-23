# P016 Runpod 4×A40：真实校准记录

2026-10-23（Asia/Shanghai），Pod `0wietnrru530l9`。校准已结束：**16K桶完成4次更新、完整checkpoint恢复和最终adapter导出；32K桶只完成2次更新，最长32,382样本在backward阶段OOM。** 数据为冻结的 `pilot-admitted-v2` 原生三工具轨迹。这些结果证明当前机器的训练与恢复路径可用，不证明真实修复能力提高；没有新增比赛评测，最好分数仍为0.15。

## 已完成的执行段

| 执行段 | 实际墙钟秒 | 结果及证据 |
|---|---:|---|
| v3首次16K启动 | 156.585064 | 首次NCCL collective阻塞，操作员停止；未加载模型、未更新。 |
| v3 16K，关闭NCCL P2P | 164.733812 | step 1完成并提交checkpoint，启动器按完整checkpoint停止。 |
| v3从checkpoint 1恢复 | 117.243505 | FSDP包装前报 `Adapter initialization differs across ranks`；没有新增更新。 |
| 两卡加载同一个真实adapter | 64.636118 | GPU0的600个参数全部匹配保存值；GPU1的300个LoRA_B不匹配。没有optimizer更新。 |
| 单独定位safetensors设备选择 | 0.997097 | 当前CUDA设备为1，请求裸 `cuda` 却得到 `cuda:0` 上的tensor。没有optimizer更新。 |
| v4恢复16K | 522.541461 | 从checkpoint 1接续完成step 2–4，导出adapter，训练进程退出码0。 |
| v4首次32K | 755.412977 | step 1完成；step 2 backward在rank 2 OOM，操作员停止挂住的分布式清理。 |
| 32K启用expandable segments后恢复 | 1,026.472678 | 从checkpoint 1接续完成step 2；step 3 backward在rank 3 OOM，操作员停止清理。 |

上述执行合计 **2,808.622712秒（46.81分钟）**，未超过校准7,200秒预算。精确值由实际启动日志和两份诊断计时重新求和，见 [actual-progress.json](actual-progress.json)；模型下载、环境准备、空闲时间及之后正式主训练不计入该校准小计。Pod的实际计费时间另算。

原始日志已回传至本目录，包括 [16K恢复日志](calibration-16k-resume-v4.log)、[32K首次日志](calibration-32k-v4.log)、[32K重试日志](calibration-32k-expand.log) 和对应GPU采样CSV。首轮完整checkpoint触发停止时，外层日志记录 `explicit_committed_checkpoint_stop`、`child_returncode=1`；该停止不能当作自然完成或训练失败。

## 16K：四步完成，真实恢复成功

| 更新 | Input tokens | 监督tokens | 步耗时/秒 | Loss | 当步最长序列 |
|---|---:|---:|---:|---:|---:|
| 1 | 46,005 | 8,070 | 138.326640 | 0.617812472 | 12,349 |
| 2 | 54,371 | 7,615 | 185.115960 | 0.629700261 | 14,556 |
| 3 | 61,767 | 9,693 | 216.988225 | 0.558685645 | 15,768 |
| 4 | 32,811 | 5,919 | 86.357447 | 0.769489200 | 9,380 |

来源为 [首步metrics](calibration-16k/metrics.jsonl) 和 [恢复后metrics](calibration-16k/metrics-resume-000001-31baa64e85c2.jsonl)。共处理16条完整轨迹、194,954 input / 31,297监督tokens；四步耗时626.788272秒，对应311.04 input tokens/s和49.93监督tokens/s。这里的分母只包括训练步，不能直接当作含加载、checkpoint和导出的端到端吞吐。

[最终training manifest](calibration-16k/training_manifest.json) 为 `training_completed`，optimizer_steps=4，四个rank结果一致，最后checkpoint为 `checkpoint-000004`。rank记录中的最大allocated为29,495,986,688 bytes，最大reserved为46,802,141,184 bytes；两个峰值不一定同时发生，不能相减当作碎片大小。

恢复前完整checkpoint已回传至 [checkpoint-000001](../../data/runpod-20261023/checkpoint-000001/complete.json)，包含step、optimizer、RNG和数据游标。最终adapter和manifest已回传至 [calibration-16k-final](../../data/runpod-20261023/calibration-16k-final/training_manifest.json)。adapter-only导出不等于完整恢复点。manifest仍记录 `runtime_compatibility_verified=false`，比赛推理端加载和真实修复效果尚未验证。

逐参数诊断见 [GPU0](adapter-load-diagnostic-gpu0.json)、[GPU1](adapter-load-diagnostic-gpu1.json)、[诊断计时](adapter-load-diagnostic-summary.json) 和 [裸cuda设备记录](safetensors-bare-cuda-device.json)。v4在PEFT adapter恢复时显式传 `torch_device="cpu"`，先反序列化到CPU，再复制到各rank已绑定的模型设备。真实运行通过adapter一致性和optimizer恢复，继续step 2–4，没有重跑step 1。v3历史包与证据保留；[v4构建回执](../generated/smith-pilot-calibration-v4/build-manifest.json) 记录包SHA256 `6c9cc2dc84adf09af00643d7ed7a2f44afdc1a5617548c855ddb4215da4e0b2c`。

## 32K：分配器调整救回25K一步，最长样本仍不可行

首次step 1完成：79,466 input / 13,470监督tokens，363.485774秒，loss 0.443612，最长20,873。step 2的rank 2序列25,135在backward OOM：请求516 MiB，当时剩余263.75 MiB，PyTorch allocated 38.11 GiB，reserved但未allocated 5.62 GiB。

从真实checkpoint 1恢复时，只新增 `PYTORCH_ALLOC_CONF=expandable_segments:True`。step 2随后完成并提交checkpoint：96,189 input / 16,613监督tokens，522.734289秒，loss 0.475018427，最长25,450。证据见 [首步metrics](calibration-32k-v4/metrics.jsonl) 和 [恢复后metrics](calibration-32k-v4/metrics-resume-000001-5bc5771751fe.jsonl)。

step 3的rank 3序列32,382仍在backward OOM：请求1.30 GiB，当时剩余791.75 MiB，allocated 40.17 GiB，reserved但未allocated 3.04 GiB。该桶没有完成4步，也没有最终完成manifest。已成功的25,450样本不能推导所有25K轨迹都安全；显存还随监督tokens等因素变化。

本轮停止继续调整长上下文，优先独立冻结的537题、最长16K正式主语料，以更短时间取得可用于后续修复评测的训练结果。正式一轮是新实验，从官方NF4基座新建adapter；不接着校准adapter训练。正式预算和执行状态见 [主训练计划](../main-short-run-plan.md)。

该实例必须显式设置 `NCCL_P2P_DISABLE=1`；CUDA报告peer access可用不能替代实际NCCL collective结果。正式启动还保留expandable segments，并采用v5异常退出修复，避免OOM后挂在NCCL清理；v5异常路径并未在本轮校准中另行触发验证。
