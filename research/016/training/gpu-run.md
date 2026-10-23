# P016 GPU 容量校准：v4恢复入口与Runpod实际结果

此包用于测真实完整轨迹的16K/32K训练容量、吞吐、保存和恢复。2026-10-23已在Runpod完成一个真实更新并保存完整checkpoint；v4恢复后已确认step 2完成并提交checkpoint，尚未确认整桶完成。没有新增修复评测或比赛得分。生成入口 `prepare_gpu_run.py build` **不会调用torchrun**。只有显式 `run` 子命令会启动四卡训练。

当前实例为 **Runpod 4×A40**，Pod `0wietnrru530l9`，环境、SSH和完整NF4缓存已就绪。v3实际首个更新用138.326640秒处理46,005 input / 8,070监督tokens，已提交checkpoint 1；首次NCCL阻塞通过显式`NCCL_P2P_DISABLE=1`绕开，随后v3恢复暴露adapter跨rank不一致。v4修复PEFT adapter加载的设备选择，显式先在CPU反序列化，再复制到当前rank模型；已恢复完成step 2并进入step 3 forward，后续结果待实记。[完整执行记录](runpod-calibration/README.md)。

当前使用 **`smith-pilot-calibration-v4`代码包，数据仍为`pilot-admitted-v2`**。冻结产物：[smith-pilot-calibration-v4.tar.gz](../data/gpu-bundles/smith-pilot-calibration-v4.tar.gz)，**6,127,456 bytes**。压缩包SHA256为`6c9cc2dc84adf09af00643d7ed7a2f44afdc1a5617548c855ddb4215da4e0b2c`；包内`bundle-manifest.json` SHA256为`b14a435e85aa50008183f4a297f914b5f06d149353d72f2b06836aeda3daf7cc`。[构建回执及训练源码哈希](generated/smith-pilot-calibration-v4/build-manifest.json)。

v3包和真实失败记录保留；其压缩包SHA256仍为`9cf66d95c05eaf4ba866d57d1caa7d4083aa0bdce88143d6336cae5a0ebbed91`。v4继承v3的worker清理修复，不覆盖旧包或修改旧包哈希。

## 数据与冻结内容

输入必须已经`training_eligible=true`、`group.split=train`，且所有tokenizer/template/tools/protocol指纹一致。生成器使用实际训练loader读取输入，并匹配显式`--admission-manifest`中的分片SHA/字节数/记录数；拒绝混合thought模式、重复问题或裁剪轨迹。本轮只用修复后已admit Smith原生三工具的thought-include分片，不混六工具协议。

16K桶只含≤16,384的完整轨迹；32K桶只含(16,384,32,768]的完整轨迹。各取最多16个独立问题，按长度分位和仓库覆盖选取，并包含真实最长样本；不足时保留实际数量，不复制凑满。每个桶四rank、每rank micro batch=1、accumulation=1、1 epoch、每步checkpoint。总步数在生成时冻结为`ceil(实际样本数/4)`，当前pilot每桶16题、4步。

当前v4包的16K实际长度为 **7,102–15,768**，32K为 **16,575–32,382**；每桶16条、4个optimizer steps。包已冻结最终trainer、`fsdp_checkpoint.py`及支持`--init-run`的新阶段初始化入口。修改源码后必须用新名字重新构建，不得改包内代码或配置继续沿用旧哈希。

包包含训练源码（明确含`training/fsdp_checkpoint.py`）、固定CT tokenizer/template、原生bash/editor/submit schema、真实编码训练行和配置。**不含模型权重、adapter权重、凭据、依赖安装或自动启动任务。** 模型NF4缓存由用户服务器路径显式绑定。

## 在项目工作区重新生成

v4已生成，当前无需再次构建。以下仅供源码需要更新时使用；`--name`必须取未使用的新名字，数据继续指向已admit的v2分片。

```sh
PYTHONPATH=/private/tmp/p012libs311 /opt/homebrew/bin/python3.11 research/016/training/prepare_gpu_run.py build \
  --name smith-pilot-calibration-next \
  --data research/016/data/encoded-native/pilot-admitted-v2/swe-smith-native-le16k-thought-include.jsonl \
  --data research/016/data/encoded-native/pilot-admitted-v2/swe-smith-native-16k-to-32k-thought-include.jsonl \
  --tools research/016/data/encoded-native/pilot-admitted-v2/swe-smith-native-tools.json \
  --admission-manifest research/016/data/encoded-native/pilot-admitted-v2/swe-smith-native-manifest.json
```

生成目录和tar.gz位于`research/016/data/gpu-bundles/<name>`；配置及带SHA256的产物回执位于`research/016/training/generated/<name>/`。已有名字会被拒绝，重新冻结应换名字。后续若更换已admit的同模式分片，生成器仍只抽最多16条/桶；正式大规模训练不应重复使用这个容量校准小样本包。

## 服务器依赖与模型绑定

Runpod隔离环境`/workspace/p016/venv`已安装并导入以下版本，与P015成功31B作业一致；包本身不自动安装或升级依赖：

| 库 | 实际成功作业版本 |
|---|---|
| torch | 2.10.0+cu128 |
| transformers | 5.13.1 |
| peft | 0.21.2 |
| bitsandbytes | 0.50.2 |
| accelerate | 1.15.0 |
| safetensors | 0.8.0 |

来源为`research/015/capacity/output-fsdp-metadata/p015-fsdp/run/training_manifest.json`，摘录与原件SHA保存在包内`reference-runtime.json`。P015旧作业只证明最长6,846token；P016已完成的step 1实际最长12,349token，恢复后的step 2最长14,556token。完整16K桶、32K及恢复完成情况以本次真实日志为准，不能从已有短样本外推。[实际环境记录](runpod-environment/README.md)。

启动器复用成功P015的关键环境：`FSDP_TRANSFORMER_CLS_TO_WRAP=Gemma4TextDecoderLayer`、`TOKENIZERS_PARALLELISM=false`、`PYTHONDONTWRITEBYTECODE=1`、`OMP_NUM_THREADS=1`、`NCCL_DEBUG=WARN`、`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`及HF/Transformers离线模式；torchrun明确`--max-restarts=0`。不会用自动重启消耗额外GPU预算。**该实例额外设置`NCCL_P2P_DISABLE=1`**：默认首次collective实际阻塞；peer access能力查询为true并不保证NCCL运行成功。此设置在调用时传入，不改冻结包。

`--model`当前绑定`/workspace/models/p005-cpu-nf4-cache/qat-nf4`，来源为私有Kaggle `ccjjllt/p005-cpu-nf4-cache`。下载已完成并逐文件核对SHA/size：15个载荷文件共18,295,563,073 bytes，源manifest SHA256为`3a9661d56f0eec51474741b01d7ddb2932fb4836d7f776051d8ece7c333750cf`。完成记录见`research/016/data/runpod-20261023/cache-download-retry1.log`。首次SSH断连留下的原download.log不是最终状态。CT tokenizer随包提供；CT评分权重不能代替NF4训练缓存。

## 四张A40的首次预算

本次仅做容量、吞吐与恢复校准，**所有GPU执行段合计最多2小时（7,200秒）**，含各段模型加载、训练、保存与清理；模型传输和环境准备单独记录。单次执行通常给1,800秒，但必须受剩余累计预算约束；失败诊断和必要补跑同样计入，不能自动循环重试。每次启动前按已消耗时间缩短剩余`--wall-seconds`；启动器只约束单次调用，不会自动跨调用累计2小时预算。

v4启动前已消耗**504.195596秒**，包括三次训练启动及两段GPU诊断；正在执行的v4和后续运行还需继续累计。16K整桶及恢复完成后才进入32K；若OOM或耗时不合算，记录真实长度、错误和GPU时间后调整，不把较短轨迹成功当作32K通过。

## 当前16K恢复入口

v3已完成`checkpoint-000001`。以下是实际启动v4恢复的命令，正在运行时不得再次并行启动。若更晚checkpoint已经完整提交，应改为最新一个；trainer拒绝回退覆盖新进度。

```sh
NCCL_P2P_DISABLE=1 /workspace/p016/venv/bin/python /workspace/p016/smith-pilot-calibration-v4/prepare_gpu_run.py run \
  --bundle /workspace/p016/smith-pilot-calibration-v4 \
  --bucket 16k \
  --model /workspace/models/p005-cpu-nf4-cache/qat-nf4 \
  --tokenizer /workspace/p016/smith-pilot-calibration-v4/assets/tokenizer \
  --out /workspace/p016/runs/calibration-16k \
  --wall-seconds 1800 \
  --resume /workspace/p016/runs/calibration-16k/checkpoint-000001
```

v4于2026-10-23 10:27:16北京时间启动，约16.09秒完成FSDP包装并通过adapter/hash及optimizer恢复，随后完成step 2并提交checkpoint 2：185.115960秒、54,371 input / 7,615监督tokens，loss 0.629700261。当前已进入step 3 forward，含最长15,768；此处尚未确认其更新或整桶完成。运行入口脚本保存在服务器`/workspace/p016/incoming/p016-run-16k-resume-v4.sh`。

恢复保持基座、资产内容、数据、配置、四rank和依赖版本一致；本NF4分支将model/tokenizer/template/tools路径归一到内容指纹，因此v3到v4的目录变化不会单独改变恢复签名。总`max_steps=4`不变，不能调小或调大步数配合恢复。`--resume`恢复optimizer/scheduler/RNG与数据游标；adapter-only加载不等于resume。若最后一步已提交、仅最终导出丢失，trainer可按已完成状态恢复导出。

以后需要重新做首段中断实验时，必须选新的空输出目录、删除`--resume`并加`--stop-after-checkpoint 1`，不要覆写本次历史。该参数只在完整checkpoint标记出现后停止，退出码75表示约定中断，并不把4步作业改成1步完成。

## 32K与跨阶段初始化

16K完成后再单独运行32K桶，使用新的`--out /workspace/p016/runs/calibration-32k`，其余绑定保持一致。32K第一次可从基座独立初始化以单独测容量；不要让两个校准桶同时争用四张GPU。

```sh
NCCL_P2P_DISABLE=1 /workspace/p016/venv/bin/python /workspace/p016/smith-pilot-calibration-v4/prepare_gpu_run.py run \
  --bundle /workspace/p016/smith-pilot-calibration-v4 \
  --bucket 32k \
  --model /workspace/models/p005-cpu-nf4-cache/qat-nf4 \
  --tokenizer /workspace/p016/smith-pilot-calibration-v4/assets/tokenizer \
  --out /workspace/p016/runs/calibration-32k \
  --wall-seconds 1800
```

以后若要从已完成16K阶段adapter开始一个新的32K阶段，可显式加 `--init-run /workspace/p016/runs/calibration-16k`，由包内trainer读取并核实前阶段。**这会重置optimizer/scheduler/RNG并形成新的训练阶段，和`--resume`互斥**；不得称为同一轮优化器状态连续恢复。

启动器不接受`--max-steps`覆盖，只转发冻结配置。v4沿用v3的清理实现，在Linux读取`/proc`记录本次torchrun及已观察后代的PID、启动时间与进程组，持续识别独立session的worker；发送信号前复核启动标识。到wall上限前预留15秒先向torchrun发SIGTERM，截止时对仍匹配身份的已观察进程发SIGKILL；父提前退出及`finally`也执行清理。wall停止返回124，checkpoint主动中断返回75；正常退出保留实际子进程退出码。轮询有约0.5秒间隔，发信号和回收还受系统调度影响。

CPU一次性真实进程模拟已验证“独立session、忽略TERM的worker”在父收到TERM退出和父已提前退出两种情况下均被清理；这不是GPU/NCCL实测。该跟踪不是cgroup，无法保证捕捉快照间立即脱离且未被观察的任意daemon，也无法保证系统不可中断进程瞬间退出；若发信号后仍未退出会输出其PID，执行者需核实并计入预算。已提交checkpoint保留，未提交的中间写入不作为恢复点。

运行后记录真实input/监督tokens每秒、单步和加载耗时、显存峰值、checkpoint保存耗时及恢复结果。OOM也应按真实长度与错误记录；不要把较短样本成功外推成32K通过。本轮已有step 1及恢复后step 2和对应完整checkpoint证据；余下步骤、最终导出和32K结果需继续归档，不能预先写成成功。
