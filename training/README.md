# AutoDL 双 A800 80GB：先准备，再运行真实 RL

当前实现为Unsloth＋TRL GRPO、复合奖励、动态筛题和多次轨迹复用。双A800服务器已完成学习器与官方运行库安装、真实导入和两卡识别；32K实际容量、GRPO更新与CT适配效果以实验记录为准。数据与模型在服务器手动获取，本地不保存题目或权重。

## 当前 GRPO 训练

- 对应官方 QAT 未量化31B，BF16冻结基座；Unsloth单进程跨两张A800分层，LoRA r32/alpha64/dropout0。保留400个文本投影，暂排除10个共享K/V的full-attention k_proj。
- 实例化固定TRL 1.15的 `GRPOTrainer` 子类，用其真实裁剪损失、KL和 `Trainer.train()` 优化器生命周期。为适配官方多轮工具环境，外部完成采样和验收，Trainer逐次模型调用重算完整前缀并累积梯度。
- 每题8条完整轨迹，目标每批4个有效题组，同批最多更新2次。更新前固定优势、旧learner概率和关闭初始零LoRA所得reference概率；Adam状态及学习率调度跨批保留，不跨策略批次回放旧轨迹。
- 新旧概率比裁剪区间为0.8～1.28，reference KL系数0.01；采用固定32K常数归一。组内优势按标准差归一，分母至少0.05。行为概率与旧learner概率有差异时，用上限2的固定重要性权重修正；这是有截断的近似校正，不宣称无偏。
- 每次模型调用总窗口32,768；只对实际生成token计损失。工具反馈保留为条件，不detach前缀KV，不缩短任务或验收。思考受原输出预算约束，无额外强制思考截断；采样为temperature1/top_p1/top_k0。
- 默认复合奖励：`完整修复 + 0.3×修复率 − 0.5×回归率 − 0.02×成功轨迹token成本`。修复/回归统计来自同一次完整官方JUnit报告；成功奖励始终高于失败最高分，效率成本有上限并只在成功后计算。
- 任一候选缺少可信测试分项时，整组停用这些分项，保留原始缺失状态；官方二元成功率与复合奖励分别记录。未完成或无法可靠归因的验收异常不冒充模型失败。
- 动态采题先覆盖未访问题，再按奖励方差、中等成功率、新颖度调整优先级，保留20%均匀探索并兼顾仓库多样性。完整8条结束后才筛掉低方差组；每批最多尝试12个不同题组补位，不无限重采。
- 不足4个有效组时按实际组数更新，全部无信号则记录后进入下一批。记录实际选题概率和全部候选/入选组各自成功率；筛组与自适应采样会改变训练分布。

工具和验收在独立CPU进程运行。GPU生成现改为逐步连续批处理：HTTP请求进入队列，单个生成线程每步让多条活动序列共享一次decode forward，新请求可在步间加入，已完成请求单独返回。默认最多4条；`device_map="balanced"`仍只是权重分层，不是张量并行。此改动已做极小真实Gemma4的CPU正确性核对，31B＋Unsloth的双卡兼容性和性能收益仍待实测。GRPO实现位于[grpo_trainer.py](grpo_trainer.py)，生成位于[generation.py](generation.py)，奖励位于[rewards.py](rewards.py)，采题位于[curriculum.py](curriculum.py)。

## 生成调度与性能测量

- `generation.max_batch_size`限制活动序列数；`max_batch_tokens`按最长完整请求预算乘批大小预留容量，不通过截短上下文、输出、思考或验收凑批。默认最多4×32K；显存是否足够仍由31B实际运行确认。
- 普通decode步持续使用同一批KV；只有请求加入或退出时拆合缓存。各层保留原生full/sliding KV的不同头形状、设备和滑窗计数，逐行使用真实position IDs。新请求的prefill目前在decode步间逐个处理，长prefill仍会延迟其他请求；尚未实现chunked prefill、paged attention或CUDA graphs。
- 同一策略版本内用有条目数及字节上限的LRU复用完全一致的token前缀。精确命中复用最后位置logits，扩展命中只计算新增后缀；不回滚已丢弃的滑窗历史。结束采集会排空请求并清除KV，之后才计算完整前缀训练概率和更新LoRA。推理缓存不会进入梯度路径。
- 题组采集共用CPU池，正式配置最多预取2个候选组，轨迹槽位完成后立即补位。完整组按选题顺序计算奖励和选入更新，避免偏向先完成的快题。达到有效组目标时，已预取的余量仍完成全部8条及验收，明确标为`prefetch_after_target`，计入全部候选统计；首批配置预取1组以控制成本。
- `generation.profiling`可对少量decode迭代采集CPU/CUDA算子、形状和显存。首批默认等待16步、warmup2步、记录8步；正式配置默认关闭。输出在`profiles/<collection_id>/`，不同采集批不会覆盖。profiler额外同步只发生在窗口边界；另在整批采集关闭时同步两卡，确认KV操作排空后才允许训练。profile对吞吐的扰动必须单独报告。
- 连续批处理的各请求耗时会重叠，不能再把它们相加作为总生成时间。`generation_metrics.jsonl`记录不重叠的引擎活动耗时、token数、实际最大批大小、缓存命中和排除profile窗口后的吞吐；端到端耗时还应计入排队、工具和验收。原来的`first_scores_seconds`只存在于旧单请求日志，新调度记录实际`first_token_seconds`及排队时间。

2026-10-30手动核对使用Transformers5.17、CPU Torch2.14.1及随机极小Gemma4，覆盖混合full/sliding、异构KV头、K=V、共享KV、滑窗跨界、请求中途加入/退出、前缀复用、EOS和取消。实际采样概率对独立完整前缀forward最大差约2.4e-6；HTTP调度及完整题组顺序用一次性模拟核对。数值记录与CPU profiler摘要保存在`results/local-batching/2026-10-30/`。这些结果不代表31B吞吐、显存或修复收益。

后续有可用GPU入口及本次预算时，可用`training.profile_generation`直接读取服务器保留的完整`model_calls.jsonl`，对同一组完整prompt和completion进行旧串行`generate`及新batch 1/2/4重放。默认关闭前缀缓存以隔离批处理效应，`--prefix-cache`用于另一次缓存对照；`--profile`另跑独立采样窗口，不将profile轮次混入吞吐。输出包含源文件指纹、实际decode批大小、吞吐、显存与逐token概率差汇总，不导出原始token。它强制使用记录中的输出路径，不能称为新轨迹、官方验收或模型收益，采样开销也不完全等同自由生成。

例如，确认模型及以下调用记录仍在服务器后，在本次明确授权的时间预算内执行（这里仅提供命令，不表示已经运行）：

```bash
CUDA_VISIBLE_DEVICES=0,1 /root/autodl-tmp/gemma4/envs/learner/bin/python -m training.profile_generation \
  --config configs/first_batch.toml \
  --model-path /dev/shm/gemma4-models/gemma4-31b-qat \
  --calls /root/autodl-tmp/gemma4/runs/first-batch-008/batch-0000-group-000-rollout-000/model_calls.jsonl \
  --calls /root/autodl-tmp/gemma4/runs/first-batch-008/batch-0000-group-000-rollout-001/model_calls.jsonl \
  --max-calls 4 --batch-sizes 1,2,4 --profile \
  --output-dir /root/autodl-tmp/gemma4/runs/generation-batching-001 \
  --walltime-seconds 1800
```

实际额度由本次会话确定；父进程的总时限包含模型加载。用户于10月30日明确要求“没有入口不实验”，本次未执行此GPU命令。

## 手动准备数据与模型

到 AutoDL 后按实际网络和存储情况手动获取官方数据、模型及运行库，不使用项目下载脚本。服务器可直接调用 Kaggle/Hugging Face 的官方工具，数据不需要经过本地电脑；取得的官方原始数据作为训练的只读输入。

当前服务器的Kaggle访问已打通，首批所需公开资产已直接下载到服务器。其他实例仍需账号比赛访问权、接受规则及可用网络。通过标准Kaggle认证配置令牌，不把密钥写入代码、聊天、命令参数或项目Git。

需要准备的内容：

| 来源 | 内容 |
| --- | --- |
| 比赛官方公开目录 | 原始 `tasks.jsonl`、`snapshots/`、`graphs/`、`embeddings/`、任务 `wheels/`、`sandbox/`、`docker/imp.py`、`docker/telnetlib.py` 和 `sample_submission/`，保留原目录层级 |
| `metric/gemma-4-developer-agent-wheelhouse/31` | 当前环境安装脚本使用的官方运行库 wheels，与任务目录中的 `wheels/` 用途不同 |
| Google QAT 未量化模型 | 全部权重分片及索引、模型配置、generation 配置、完整 tokenizer 文件与官方 chat/response 模板 |

只获取本次训练需要的官方公开资产，保持原文件内容，不读取隐藏或预留评测内容。参考补丁和验收资产仅供裁判侧使用。训练只需通过 `--data-root` 和 `--model-path` 指定实际目录，不要求下载脚本生成的清单或回执文件。

## 服务器准备命令

下面命令均在**AutoDL终端**执行。将本项目源代码放到例如`/root/autodl-tmp/agent_of_gemma_4`；模型、数据、环境放到代码目录之外的`/root/autodl-tmp/gemma4`。不要将原有开发目录的`.git`、`.venv`、凭证或旧研究一起打包上传。

手动准备完成后，安装分离环境；先确保系统已有官方Dockerfile要求的 `git`、`patch`、`protoc` 和 `pigz`。`--wheelhouse` 指向官方运行库，`--data-root` 指向公开任务资产。示例使用本机现有的Miniconda解释器引导安装：

```bash
cd /root/autodl-tmp/agent_of_gemma_4
bash scripts/prepare_server.sh runtime --bootstrap-python /root/miniconda3/bin/python --wheelhouse /root/autodl-tmp/gemma4/official/wheelhouse-v31
bash scripts/prepare_server.sh sandbox --bootstrap-python /root/miniconda3/bin/python --data-root /root/autodl-tmp/gemma4/official/competition
bash scripts/prepare_server.sh learner --bootstrap-python /root/miniconda3/bin/python
```

如需独立的 Kaggle/Hugging Face CLI 工具环境，可选运行 `bash scripts/prepare_server.sh data`；这一步仅安装工具，不获取数据或模型。认证按[Kaggle官方说明](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md)及 Hugging Face 标准方式处理。

- 学习器：Python3.12，CUDA12.8 Torch2.9.1、Unsloth/zoo2026.10.10、Transformers5.17、TRL1.15、PEFT0.21.2；版本固定在`requirements/learner.lock.txt`，已在服务器安装并识别两张A800。
- 官方运行库：Python3.13，`/opt/gemma-runtime` 使用v31原始官方wheels及CPU Torch，已完成普通和低权限进程导入。
- 工具基础环境：`/opt/gemma-sandbox-baseline` 以官方Dockerfile的测试、构建工具及兼容源文件为基础，并按官方wheel依赖声明补齐公开包中缺失的依赖；因此不宣称与原Docker镜像完全一致。环境由root拥有、低权限进程只读，优先使用官方任务wheels，补充依赖固定版本；每条轨迹的项目依赖另行隔离。各环境保存安装freeze。
- 原图的PyTorch2.8镜像只是基础操作环境，学习器使用自己的虚拟环境。

[AutoDL无卡模式](https://www.autodl.com/docs/save_money/)适合认证、传代码和下载；文档列出的CPU/RAM资源较小，不保证能在无卡模式完成全部编译或安装，更不能用它验证模型加载。无卡模式会释放GPU，之后能否立即取得同一双卡取决于库存。

## 模型工具与裁判分开

AutoDL普通容器不支持Docker。此处采用官方subprocess后端，并增加独立低权限UID：官方准备工作和裁判保持root，模型执行的工具命令降权。官方数据及结果根目录为root-owned 0700，模型不能读原始tasks.jsonl中的裁判资产；只得到官方构造的题面和自己的工作区。

这种方式不等于正式Docker评分环境的网络和资源隔离。最终adapter仍需在官方CT评分栈验证。程序明确禁用官方Evaluator自动向父目录寻找secret数据的逻辑；不读取隐藏评测资产。

## GPU开启后运行

先看有效配置，这条命令不读取题目、不加载CUDA：

```bash
/root/autodl-tmp/gemma4/envs/learner/bin/python -m training.run --describe
```

下面是首批GPU训练命令：`first_batch.toml` 只采集1个批次，最多2次更新，关闭warmup，并受2小时进程上限约束。可在 `data.task_ids` / `exclude_task_ids` 或命令参数中限定题目；这些上限不是耗时预测，中断和基础设施故障不会被当成零奖励。当前数据盘只有50GB，约58.25GiB的BF16权重暂存于 `/dev/shm`，已核对官方SHA256；关机后需重新获取权重，结果保留在数据盘。

```bash
CUDA_VISIBLE_DEVICES=0,1 /root/autodl-tmp/gemma4/envs/learner/bin/python -m training.run \
  --config configs/first_batch.toml \
  --data-root /root/autodl-tmp/gemma4/official/competition \
  --model-path /dev/shm/gemma4-models/gemma4-31b-qat \
  --run-dir /root/autodl-tmp/gemma4/runs/first-batch-001
```

输出包括官方结果与奖励分项、完整日志/trace/patch、真实调用token/行为概率、课程状态、旧策略偏移、KL/裁剪统计、adapter指纹和显存峰值。新运行使用新目录；`adapters/initial`与`adapters/batch-0001`等保存独立LoRA。筛掉的组仍保留真实采样/验收记录，不能只看入选组成功率。

正式提交只取所选adapter目录的`adapter_config.json`与`adapter_model.safetensors`，按官方YAML引用。不要提交训练基座或合并权重。当前代码没有实现自动比赛提交，也没有在官方CT上取得适配效果证据；训练题上的改善不能称为独立泛化成绩。

## 取回分析日志

在本机复用已认证的SSH会话执行，`--run` 可重复指定多次：

```bash
python3 scripts/sync_experiment_logs.py \
  --host root@connect.nma1.seetacloud.com --port 50248 \
  --control-path /private/tmp/gemma4-autodl-ssh.sock \
  --known-hosts /private/tmp/gemma4-autodl-known_hosts \
  --run first-batch-008
```

默认写入已被Git忽略的 `results/autodl/`，包括学习器日志、配置、奖励与结果摘要、依赖版本、`generation_metrics.jsonl`、profiler算子表与元数据，以及服务器端计算的调用token数、排队/生成耗时和行为logprob统计。原始调用token、worker日志、官方trace、补丁、JUnit、官方题目及权重不取回；较大的原始profiler trace也留在服务器。文件逐个原子替换；连接失败保留已取得的日志，只有完整同步后才更新带时间与哈希的 `sync_metadata.json`，不将其当作训练完成证明。
