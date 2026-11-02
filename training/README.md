# 双 A800 训练与运行

研究方案、已有结果和下一步见[研究说明](../research/rl-adapter-feasibility.md)。当前使用 Unsloth BF16基座＋r128/alpha256 LoRA＋TRL GRPO；每题8条完整轨迹，同批最多更新2次。已实跑的[首题组配置](../results/grpo-20261102/004-config.toml)将每批有效题组数和批次数设为1；[默认配置](../configs/a800_2x80.toml)为每批4题、最多32批。

新实验从当前默认配置调整题组数；004配置保留作历史证据，其中旧的2 GiB缓存上限不再推荐沿用。前缀缓存每条轨迹保留一个prompt、相同prompt共享，每卡最多使用15%显存，并动态扣除活动/重组KV、复制峰值、梯度/Adam预留及2 GiB工作余量；可用`prefix_cache_bytes`再设总上限，0表示关闭。轨迹结束释放其快照，进入训练前清空生成缓存。

`sliding_cache_enabled=true`使用镜像环形KV，只双写新增token，额外镜像存储计入显存预算。成员退出时连续行与共同左padding使用view裁剪，非连续行只复制survivors；真实位置与完整上下文保持。日志中的`sliding_*`和`batch_row_*`记录分配/复制量，开启现有profiler时可单独查看`cache_membership`区域；这些计数不等于GPU耗时收益。

## 当前路径

| 内容 | 服务器路径 |
| --- | --- |
| 项目代码 | `/root/gemma4/project` |
| 学习器 Python | `/root/gemma4/envs/learner/bin/python` |
| 官方公开数据 | `/root/gemma4/official/competition` |
| QAT未量化31B模型 | `/root/autodl-fs/models/gemma4-31b-qat` |
| 裁判与工具环境 | `/opt/gemma-runtime/bin/python`、`/opt/gemma-sandbox-baseline/bin/python` |
| 运行输出 | `/root/autodl-tmp/gemma4/runs` |
| 沙箱与临时目录 | `/root/autodl-tmp/gemma4/sandboxes`、`/root/autodl-tmp/gemma4/tmp` |

训练环境、代码和官方数据位于系统盘；辅助运行库实际在数据盘，模型在共享盘。subprocess后端需要root，由不同低权限UID运行agent工具与裁判；agent接收官方题面，参考补丁和验收资产只由裁判侧读取。数据和模型手动放到服务器，训练直接读取原目录，不在本地复制训练数据，不读heldout。

## 启动与续跑

将所选配置放到服务器项目的`configs/first-group.toml`后执行；每次新运行使用新的输出目录：

```bash
cd /root/gemma4/project
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TMPDIR=/root/autodl-tmp/gemma4/tmp \
/root/gemma4/envs/learner/bin/python -m training.run \
  --config configs/first-group.toml \
  --data-root /root/gemma4/official/competition \
  --model-path /root/autodl-fs/models/gemma4-31b-qat \
  --run-dir /root/autodl-tmp/gemma4/runs/grpo-next
```

模型调用窗口32K，4个生成槽与8个CPU worker。配置中的7200秒是单次run墙钟上限，包含模型加载；不保证完成全部批次。启动记录已有来源声明与廉价文件元数据，不扫描完整权重SHA256。

仅在同一run保留完整`recovery/`状态时续跑：

```bash
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TMPDIR=/root/autodl-tmp/gemma4/tmp \
/root/gemma4/envs/learner/bin/python -m training.run \
  --resume --run-dir /root/autodl-tmp/gemma4/runs/grpo-next
```

续跑恢复轨迹、采样器、LoRA、Adam、scheduler、RNG与更新进度，不重置该run预算；追加额度用`--additional-walltime-seconds N`显式记录。只有adapter不能完整续跑，更改LoRA rank须新建run。默认训练会话时钟扣除已确认的纯GPU排队时间，并保留原墙钟；这与官方墙钟评测口径不同。

## 结果与adapter

- `metrics.jsonl`记录真实采集和更新；`reward_groups.jsonl`、`reward_summary.json`记录完整候选题组、奖励与选组结果。
- `training_summary.json`从已提交checkpoint汇总有效组比例、采集/更新耗时、实际使用LR与非零LR步数、KL及adapter身份；恢复不重复计数。采样成功率明确属于更新前。
- `generation_metrics.jsonl`与调用数值摘要用于判断吞吐、排队、prefill和KV成本；不把速度或训练集成绩称为独立泛化收益。
- `recovery/`保留训练精度的完整续跑状态；`adapters/batch-*/`保存更新后的独立BF16 adapter。

提交仅取所选adapter的`adapter_config.json`与`adapter_model.safetensors`，按官方YAML引用。初始0-step adapter不是训练成果；最终必须在官方CT基座上验证挂载与修复收益。

原始调用、轨迹、补丁与JUnit只留服务器。数值日志可用[sync_experiment_logs.py](../scripts/sync_experiment_logs.py)同步；本地精简保留的性能证据和实际配置见[results/grpo-20261102](../results/grpo-20261102/)。主线优先完成完整题组和GRPO更新，吞吐重放不作为训练前置。

## 收益与单变量比较

更新结束用`post_update_kl_max_calls=4`在均匀选取的固定完整调用上补测KL；不截断上下文、不重新生成。结果同时给出基座reference与同批old的before/after/delta、采样范围和耗时；这是behavior样本上的近似诊断。设为0可关闭；诊断失败保留未知，不阻止已完成更新的adapter导出。

本地只准备一个对照配置，不启动GPU：

```bash
python3.11 -m training.ablation prepare --baseline configs/a800_2x80.toml \
  --output-dir tmp/reuse-one --factor reuse --value 1
```

同一目录保存`parent.toml`、`candidate.toml`与语义差异。支持`rank`、`reward.<单项weight>`、`sampling=uniform`、`reuse`。rank保持alpha/r；复用对照显式固定`grpo.scheduler_total_steps`，避免同时改变warmup及LR曲线。分别从同一初始基座新建run；不能在原run续跑时修改因素。

拿到adapter后，在服务器显式执行公开任务配对评测（`TASK_ID`替换为选定的公开题ID，可重复`--task-id`）：

```bash
python -m training.evaluation --config configs/first-group.toml \
  --data-root /root/gemma4/official/competition \
  --model-path /root/autodl-fs/models/gemma4-31b-qat \
  --adapter /root/autodl-tmp/gemma4/runs/grpo-next/adapters/batch-0001 \
  --training-run-config /root/autodl-tmp/gemma4/runs/grpo-next/run_config.json \
  --task-id TASK_ID --seed 3407 --seed 3408 --walltime-seconds 7200 \
  --output-dir /root/autodl-tmp/gemma4/runs/eval-next \
  --scratch-dir /root/autodl-tmp/gemma4/tmp/eval-next
```

此入口顺序比较禁用adapter的基座与所选adapter，每臂重置随机状态与缓存，使用相同墙钟预算、官方题面和完整验收。`paired-summary.json`给出配对修复/退化、repair/regression、证据覆盖率与尝试成本；它属于BF16后端，不能代替官方CT结果。默认按训练题成绩解释；不读取隐藏评测数据。仅有训练摘要时收益保持未知。

`python3.11 -m training.ablation compare --help`列出摘要与配对评测输入；比较时同时读取实际run配置与adapter身份，只在同一因素、同一评测题目/seed/流程下计算差异。已有数值记录可随上述同步脚本取回本地。
