# P016 GPU 容量校准包：v3代码已冻结，待Runpod四张A40实测

此包仅用于测真实完整轨迹的16K/32K训练容量、吞吐、保存和恢复；不代表已完成GPU训练或修复能力提升。生成入口 `prepare_gpu_run.py build` **不会调用torchrun**。只有显式 `run` 子命令会启动四卡训练。

用户已确认 **Runpod 4×A40（每卡48GB）**，并授权通过Runpod官方Agent入口接入。当前交付是 **`smith-pilot-calibration-v3`代码包，使用`pilot-admitted-v2`数据**；v3修复了torchrun父进程退出后独立session的worker可能残留的问题。连接、缓存传输和GPU实测尚需分别记录，CPU冻结不代表服务器已运行。

冻结产物：[smith-pilot-calibration-v3.tar.gz](../data/gpu-bundles/smith-pilot-calibration-v3.tar.gz)，**6,127,400 bytes**。压缩包SHA256为`9cf66d95c05eaf4ba866d57d1caa7d4083aa0bdce88143d6336cae5a0ebbed91`；包内`bundle-manifest.json` SHA256为`1cda6ada1113a99cddc1672d45b899985bdc1dd8d22464569687a5a4f1cdb584`。训练源码哈希及构建回执见[build-manifest.json](generated/smith-pilot-calibration-v3/build-manifest.json)，全包逐文件哈希在包内manifest。旧draft-v1和代码v2保留作过程证据，本次使用v3。

## 数据与冻结内容

输入必须已经`training_eligible=true`、`group.split=train`，且所有tokenizer/template/tools/protocol指纹一致。生成器使用实际训练loader读取输入，并匹配显式`--admission-manifest`中的分片SHA/字节数/记录数；拒绝混合thought模式、重复问题或裁剪轨迹。本轮只用修复后已admit Smith原生三工具的thought-include分片，不混六工具协议。

16K桶只含≤16,384的完整轨迹；32K桶只含(16,384,32,768]的完整轨迹。各取最多16个独立问题，按长度分位和仓库覆盖选取，并包含真实最长样本；不足时保留实际数量，不复制凑满。每个桶四rank、每rank micro batch=1、accumulation=1、1 epoch、每步checkpoint。总步数在生成时冻结为`ceil(实际样本数/4)`，当前pilot每桶16题、4步。

当前v3包的16K实际长度为 **7,102–15,768**，32K为 **16,575–32,382**；每桶16条、4个optimizer steps。包已冻结最终trainer、`fsdp_checkpoint.py`及支持`--init-run`的新阶段初始化入口。修改源码后必须用新名字重新构建，不得改包内代码或配置继续沿用旧哈希。

包包含训练源码（明确含`training/fsdp_checkpoint.py`）、固定CT tokenizer/template、原生bash/editor/submit schema、真实编码训练行和配置。**不含模型权重、adapter权重、凭据、依赖安装或自动启动任务。** 模型NF4缓存由用户服务器路径显式绑定。

## 在项目工作区重新生成

v3已生成，无需再次构建即可上传。以下仅供源码需要更新时使用；`--name`必须取未使用的新名字，数据继续指向已admit的v2分片。

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

复用P015真实成功31B作业记录的环境，不在本包安装或升级依赖：

| 库 | 实际成功作业版本 |
|---|---|
| torch | 2.10.0+cu128 |
| transformers | 5.13.1 |
| peft | 0.21.2 |
| bitsandbytes | 0.50.2 |
| accelerate | 1.15.0 |
| safetensors | 0.8.0 |

来源为`research/015/capacity/output-fsdp-metadata/p015-fsdp/run/training_manifest.json`，摘录与原件SHA保存在包内`reference-runtime.json`。已有作业只证明最长6,846token的真实更新；本包16K/32K尚待GPU实测。新版checkpoint/warm-start也不能继承原作业的实跑结论。

启动器复用成功P015的关键环境：`FSDP_TRANSFORMER_CLS_TO_WRAP=Gemma4TextDecoderLayer`、`TOKENIZERS_PARALLELISM=false`、`PYTHONDONTWRITEBYTECODE=1`、`OMP_NUM_THREADS=1`、`NCCL_DEBUG=WARN`、`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`及HF/Transformers离线模式；torchrun明确`--max-restarts=0`。不会用自动重启消耗额外GPU预算。

`--model`应指向带来源manifest的**官方QAT派生NF4训练缓存**，例如`/srv/models/gemma4-31b-qat-nf4`；不能用评分CT权重目录替代。**约18.30GB完整NF4权重目前未在本地备齐，本地只有manifest/config**；优先复用此前私有Kaggle `ccjjllt/p005-cpu-nf4-cache` 的真实输出或用户服务器已有的同源缓存。路径存在不等于权重完整，也不代表传输已完成。`--tokenizer`绑定包内固定CT tokenizer。

## 四张A40的首次预算

本次仅做容量、吞吐与恢复校准，**所有GPU执行段合计最多2小时（7,200秒）**，含各段模型加载、训练、保存与清理；模型传输和环境准备单独记录。以下16K首次运行、16K恢复和32K各给1,800秒，合计上限5,400秒，余下1,800秒只用于有明确原因的必要补跑，不能自动循环重试。每次启动前按已消耗时间缩短剩余`--wall-seconds`；启动器只约束单次调用，不会自动跨调用累计2小时预算。

四张A40上的16K/32K、NF4/NCCL恢复均待实测。16K成功后才进入32K；若OOM或耗时不合算，记录真实长度、错误和GPU时间后调整，不把较短轨迹成功当作32K通过。

## 显式启动16K并验证真恢复

以下都是待执行命令，不是本次已完成操作。`/srv/p016/smith-pilot-calibration-v3`、模型路径和输出路径是示例，连接后替换为服务器实际目录。须先核对上传压缩包SHA256及完整NF4缓存，再显式启动；输出目录初次必须不存在或为空。

```sh
python /srv/p016/smith-pilot-calibration-v3/prepare_gpu_run.py run \
  --bundle /srv/p016/smith-pilot-calibration-v3 \
  --bucket 16k \
  --model /srv/models/gemma4-31b-qat-nf4 \
  --tokenizer /srv/p016/smith-pilot-calibration-v3/assets/tokenizer \
  --out /srv/p016/runs/calibration-16k \
  --wall-seconds 1800 \
  --stop-after-checkpoint 1
```

`--stop-after-checkpoint 1`只在完整`checkpoint-000001/complete.json`出现后停止本次训练及已观察的worker，退出码75表示显式中断；**总max_steps仍是4**。它不把这段作业伪装成已完成主训练。若进程在观察到checkpoint前异常退出，保留真实退出码和日志。

从最新已提交checkpoint恢复，使用相同包、数据、配置、模型路径、输出目录、四rank和依赖版本：

```sh
python /srv/p016/smith-pilot-calibration-v3/prepare_gpu_run.py run \
  --bundle /srv/p016/smith-pilot-calibration-v3 \
  --bucket 16k \
  --model /srv/models/gemma4-31b-qat-nf4 \
  --tokenizer /srv/p016/smith-pilot-calibration-v3/assets/tokenizer \
  --out /srv/p016/runs/calibration-16k \
  --wall-seconds 1800 \
  --resume /srv/p016/runs/calibration-16k/checkpoint-000001
```

若受中断时序影响已经存在更晚的完整checkpoint，应恢复最新一个，trainer会拒绝恢复旧状态覆盖新进度。恢复保存optimizer/scheduler/RNG与数据游标；不能以adapter-only导出冒充resume，也不能调小或调大max_steps“配合”恢复。若最后一步已提交、仅最终导出丢失，trainer会按已完成状态恢复导出。

## 32K与跨阶段初始化

16K完成后再单独运行32K桶，使用新的`--out /srv/p016/runs/calibration-32k`，其余绑定保持一致。32K第一次可从基座独立初始化以单独测容量；不要让两个校准桶同时争用四张GPU。

```sh
python /srv/p016/smith-pilot-calibration-v3/prepare_gpu_run.py run \
  --bundle /srv/p016/smith-pilot-calibration-v3 \
  --bucket 32k \
  --model /srv/models/gemma4-31b-qat-nf4 \
  --tokenizer /srv/p016/smith-pilot-calibration-v3/assets/tokenizer \
  --out /srv/p016/runs/calibration-32k \
  --wall-seconds 1800
```

以后若要从已完成16K阶段adapter开始一个新的32K阶段，可显式加 `--init-run /srv/p016/runs/calibration-16k`，由包内trainer读取并核实前阶段。**这会重置optimizer/scheduler/RNG并形成新的训练阶段，和`--resume`互斥**；不得称为同一轮优化器状态连续恢复。

启动器不接受`--max-steps`覆盖，只转发冻结配置。v3在Linux读取`/proc`记录本次torchrun及已观察后代的PID、启动时间与进程组，持续识别独立session的worker；发送信号前复核启动标识。到wall上限前预留15秒先向torchrun发SIGTERM，截止时对仍匹配身份的已观察进程发SIGKILL；父提前退出及`finally`也执行清理。wall停止返回124，checkpoint主动中断返回75；正常退出保留实际子进程退出码。轮询有约0.5秒间隔，发信号和回收还受系统调度影响。

CPU一次性真实进程模拟已验证“独立session、忽略TERM的worker”在父收到TERM退出和父已提前退出两种情况下均被清理；这不是GPU/NCCL实测。该跟踪不是cgroup，无法保证捕捉快照间立即脱离且未被观察的任意daemon，也无法保证系统不可中断进程瞬间退出；若发信号后仍未退出会输出其PID，执行者需核实并计入预算。已提交checkpoint保留，未提交的中间写入不作为恢复点。

运行后记录真实input/监督tokens每秒、单步和加载耗时、显存峰值、checkpoint保存耗时及恢复结果。OOM也应按真实长度与错误记录；不要把较短样本成功外推成32K通过。本轮只完成CPU打包与真实输入loader核对，无GPU结果。
