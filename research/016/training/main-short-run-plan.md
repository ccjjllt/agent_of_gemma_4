# P016 首轮主训练执行记录（完整一轮，最多8小时）

2026-10-23，北京时间。用户已选择完整一轮、最多8小时、原估算约18.88美元，并随后明确要求不暂停Pod、不建立定时任务。主训练于 **11:23:32** 启动，已于 **17:40:24** 正常退出，完成全部68次更新，未触及19:23:32的8小时上限；Pod保持运行。启动回执见 [launch.json](main-short-training/launch.json)，最终结果见 [final-result-summary.json](main-short-training/final-result-summary.json)。用户后续授权服务器在训练结束后直接备份到Drive，上传完成后由用户自行关闭服务器。

- 数据：`main-sft-short-v1`，537个独立完整train问题、68仓，7,119,739输入/1,291,819监督token，全部≤16K、无截断。159 PR身份、375命名变异、3个旧pilot数字身份；来源偏差保留在统计中，不宣称真实问题占多数。
- 计算：现有Runpod `0wietnrru530l9` / 4×A40；读取费率2.36美元/小时。原基座全新LoRA初始化，不把校准adapter当主训练起点。
- 配方：r8/alpha16、学习率2e-5、4 ranks、每卡micro batch 1、积累2、1 epoch、68次更新、每10步保存完整checkpoint。总计划不随单次wall上限改变。
- 已实测16K：4次更新全部完成，最长15,768；真实保存/退出/重载/继续/导出成功。20K及25K批次也已完成更新；32,382在反向OOM，首轮主训练暂不继续这一容量分支。
- 启动时预计完整一轮约6–7小时；实际启动至退出为6小时16分52.59秒。长短轨迹问题分布不同，速度优势不等于已证明提分。

## 实际训练完成结果

2026-10-23只读回收最终manifest、四rank结果和日志结束事件，未把模型、训练数据或checkpoint下载到本机；本机仅保存小型摘要和文件哈希。`training_manifest.json` 的状态为 `training_completed`，`main_sft_termination` 记录 `reason=process_exit`、`child_returncode=0`，退出时间为2026-10-23 17:40:24.848226北京时间。启动器wall为22,612.5918秒；训练内计时22,583.3199秒，二者计时边界不同。

- 四rank一致完成68次更新、7,119,739输入和1,291,819监督tokens；68条metrics相加也与之相符。完整checkpoint保存在第10、20、30、40、50、60、68步，第68步记录epoch=1、data_rows=537、next_batch=0。
- 按训练内wall计算输入吞吐315.27 tokens/s；按各更新seconds之和计算315.46 tokens/s。最后一次更新的loss为0.1805935651，只代表该batch训练损失，不是验证loss、整轮平均或比赛分数。
- 最终adapter为212,261,592 bytes，SHA256为 `123a5a622decb6413a0cf9c4cf4c197604d7eba00cb32c3ad27a80eef979a701`；adapter_config为16,883 bytes，SHA256为 `e7960e62165a97bf985aac94701d6af87060b884746b5da40e9d29d542339608`。服务器只读重新计算两文件哈希，与第68步完整checkpoint记录一致。
- `runtime_compatibility_verified=false` 仍如实保留：最终adapter尚未在Kaggle官方运行时加载、验证实际生效或执行固定Agent权重对照。没有新增正式评测或分数，不将训练完成等同于修复能力提升。

云端备份成功须另以上传程序的校验回执确认；这里仅记录训练完成。用户将自行关闭服务器，本记录不授权自动Stop或Terminate。

## 已确认预算与运行方式

执行完整一轮537题、68次更新；服务器启动器硬限制28800秒，提前完成即退出，不因时限改写学习率计划，不自动开启下一轮。最初API费率2.36美元/小时，启动前控制台显示2.37美元/小时，8小时约18.96美元。Pod闲置和环境准备同样计费；训练进程退出不会停止Pod。

用户后续指令覆盖先前“备份后暂停Pod”选项：**保持Pod运行，不停止、重启或删除。** 曾创建的15分钟回查任务 `p016` 已按用户要求删除；没有Codex后台定时回查。用户又明确要求结果直接上传Drive，不回传本机，故采用下文的一次性服务器备份程序。本机仅回收轻量摘要与校验回执，云端备份完成后由用户自行关服务器。训练本身已脱离SSH会话，启动器设置8小时进程上限。

服务器SSH：`ssh -i /Users/zhaohuaxiang/.ssh/id_ed25519_ccjjllt -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -p 22078 root@194.68.245.193`。主启动器PID为2596，torchrun PID为2597；诊断进程身份时仍以命令与启动时间为准。日志 `/workspace/p016/runs/main-sft-short-v1.log`，输出 `/workspace/p016/runs/main-sft-short-v1`。启动观察已确认 **step 1/68完成** 并进入step 2：首步85,603输入/16,264监督tokens，273.0498秒，loss 0.56796105，监督吞吐59.5642 tokens/s；首次更新处于学习率warmup，实际使用1e-5。原始快照见 [initial-metrics.jsonl](main-short-training/initial-metrics.jsonl)。还不据此宣称整轮完成或比赛提分。

## 已备齐的实际入口

服务器代码 `/workspace/p016/smith-pilot-calibration-v5`（校准包名称仅作冻结代码容器），主数据 `/workspace/p016/main-sft-short-v1`，入口 `/workspace/p016/run_main_short.py`，环境 `/workspace/p016/venv/bin/python`，NF4缓存 `/workspace/models/p005-cpu-nf4-cache/qat-nf4`。入口SHA256为 `ab1715613178c2f461054f6ff4225b40304112b86c1cc53fe14b4e30aabdffa6`。

v5只在已GPU验证的CPU adapter加载修复上补充异常快速退出：先flush错误再以非零状态退出，让torchrun及时终止peer，不等NCCL清理超时。没有改训练数学目标。运行保留 `NCCL_P2P_DISABLE=1` 和 `PYTORCH_ALLOC_CONF=expandable_segments:True`；后者已把25K的真实OOM修复为完成更新。

实际已脱离SSH会话执行以下命令；不要重复启动。

```sh
/workspace/p016/venv/bin/python /workspace/p016/run_main_short.py \
  --bundle /workspace/p016/smith-pilot-calibration-v5 \
  --corpus /workspace/p016/main-sft-short-v1 \
  --model /workspace/models/p005-cpu-nf4-cache/qat-nf4 \
  --tokenizer /workspace/p016/smith-pilot-calibration-v5/assets/tokenizer \
  --out /workspace/p016/runs/main-sft-short-v1 \
  --wall-seconds 28800
```

68步规划、源数据及配置均冻结。本次8小时主训练获得独立授权，与此前最多2小时GPU校准分开记账。完整训练后仍需固定Agent权重对照才知道真实修复收益；本次没有新增正式评测或提交。

## 第30步云端备份

2026-10-23 15:25北京时间完成从Runpod直接上传的第30步恢复包，793,876,480 bytes；Drive大小与MD5核对一致。包包含完整四rank checkpoint、代码/tokenizer/数据及日志快照，基座和Python环境需按既有来源恢复。这是中途恢复点，不是最终训练结果。服务器临时Drive授权已删除；未保留本机备份包、未建立定时任务、未暂停Pod。[Google Drive备份目录](https://drive.google.com/drive/folders/1javVw9J_ihm4O7BlviXmS30DVzCbawyR)，[校验回执](drive-backup-step30.json)。

## 训练退出后自动直传 Drive

用户后续明确要求设置服务器程序，在本轮训练结束后自动备份。2026-10-23 16:48:09北京时间已启动一次性后台程序PID6116，代码为 [backup_main_on_exit.py](backup_main_on_exit.py)，入口与目录见 [部署回执](final-backup-launch.json)。它每30秒只观察本轮进程；只有launcher及四rank/torchrun全部退出，才打包上传，不提前传当前数据，不创建Codex定时任务，不停止Pod。

正常68步完成时保存全部已提交checkpoint、最终adapter/manifest、训练记录、冻结代码/tokenizer/数据、环境和基座元数据；异常退出时保留可恢复checkpoint和错误记录并标注recovery。排除18.3GB可重下载基座、venv和凭证。上传至 [本轮Drive目录](https://drive.google.com/drive/folders/1hK6u-GIwRHmZ2Tk3a7QzK2cHyialKWI1) 下的唯一final/recovery子目录，云端大小和MD5一致才写success回执，随后清除服务器临时Drive配置。需要保持本次新客户端密钥有效直至完成。

结果路径 `/workspace/p016/backups/main-sft-short-on-exit-20261023/result.json`；进行状态为同目录 `status.json`。19:33:32仍有训练进程则退出备份程序并记录原因，不打包、不上传、不碰GPU。上传阶段有两小时总上限和三次有界重试；失败保留服务器归档与日志，不会伪称云备份成功。

### 最终 Drive 备份实际完成

2026-10-23 17:51:29北京时间，4,843,816,960 bytes最终tar、backup-manifest.json和RESTORE.txt均已完成上传并逐文件通过云端size/MD5比较。程序最后读取目录ID时因rclone stat结果缺少ID写了failed；原始失败回执保留。随后通过Google Drive界面确认准确目录及三个文件，并重算服务器文件哈希确认部署代码身份，补记[恢复后的完整回执](final-backup-result.json)。[最终Drive目录](https://drive.google.com/drive/folders/1JvQjxGl3vTE5Ss1pGQfvybEmQ4X5I61X)。

包括checkpoint 10/20/30/40/50/60/68、最终adapter、训练记录、冻结代码/tokenizer/数据和环境/基座元数据；不包含基座权重和venv。备份进程已退出、临时Drive授权不存在，GPU均已空闲。用户可停止Pod；助手没有执行停止或删除。没有将归档或模型回传本机。
