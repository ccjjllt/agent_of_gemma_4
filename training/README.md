# AutoDL 双 A800 80GB：先准备，再运行真实 RL

当前实现为Unsloth＋TRL GRPO、复合奖励、动态筛题和多次轨迹复用。数据与模型在服务器手动获取；本地没有下载题目或权重，没有启动GPU。依赖元数据与必要代码核对已完成，实际软件兼容、32K容量和CT效果在接入机器后验证。

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

工具和验收在独立CPU进程运行；GPU生成仍串行，连续批处理与跨工具KV复用尚未实现。GRPO实现位于[grpo_trainer.py](grpo_trainer.py)，分项奖励位于[rewards.py](rewards.py)，动态采题位于[curriculum.py](curriculum.py)。本次不以GPU运行结果作为上述实现的前置条件，也不声称已经取得模型收益。

## 手动准备数据与模型

到 AutoDL 后按实际网络和存储情况手动获取官方数据、模型及运行库，不使用项目下载脚本。服务器可直接调用 Kaggle/Hugging Face 的官方工具，数据不需要经过本地电脑；取得的官方原始数据作为训练的只读输入。

前提是账号有比赛访问权、已在Kaggle接受规则，且实例可以访问Kaggle及其文件存储域名。具体机房网络还未实测。通过标准Kaggle认证配置令牌，不把密钥写入代码、聊天、命令参数或项目Git。

需要准备的内容：

| 来源 | 内容 |
| --- | --- |
| 比赛官方公开目录 | 原始 `tasks.jsonl`、`snapshots/`、`graphs/`、`embeddings/`、任务 `wheels/`、`sandbox/` 和 `sample_submission/`，保留原目录层级 |
| `metric/gemma-4-developer-agent-wheelhouse/31` | 当前环境安装脚本使用的官方运行库 wheels，与任务目录中的 `wheels/` 用途不同 |
| Google QAT 未量化模型 | 全部权重分片及索引、模型配置、generation 配置、完整 tokenizer 文件与官方 chat/response 模板 |

只获取本次训练需要的官方公开资产，保持原文件内容，不读取隐藏或预留评测内容。参考补丁和验收资产仅供裁判侧使用。训练只需通过 `--data-root` 和 `--model-path` 指定实际目录，不要求下载脚本生成的清单或回执文件。

## 服务器准备命令

下面命令均在**AutoDL终端**执行。将本项目源代码放到例如`/root/autodl-tmp/agent_of_gemma_4`；模型、数据、环境放到代码目录之外的`/root/autodl-tmp/gemma4`。不要将原有开发目录的`.git`、`.venv`、凭证或旧研究一起打包上传。

手动准备完成后，安装分离环境。下面目录只是示例，可按实际存放位置调整；`--wheelhouse` 指向手动取得的官方运行库目录。

```bash
cd /root/autodl-tmp/agent_of_gemma_4
bash scripts/prepare_server.sh runtime --wheelhouse /root/autodl-tmp/gemma4/official/wheelhouse-v31
bash scripts/prepare_server.sh learner
```

如需独立的 Kaggle/Hugging Face CLI 工具环境，可选运行 `bash scripts/prepare_server.sh data`；这一步仅安装工具，不获取数据或模型。认证按[Kaggle官方说明](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md)及 Hugging Face 标准方式处理。

- 学习器：Python3.12，CUDA12.8 Torch2.9.1、Unsloth/zoo2026.10.10、Transformers5.17、TRL1.15、PEFT0.21.2；101包的Linux依赖解析已通过，固定在`requirements/learner.lock.txt`。没有声称实际安装或GPU验证通过。
- 官方任务环境：Python3.13，在可供低权限工具进程读取的`/opt/gemma-runtime`；使用v31原始官方wheels及CPU Torch。其余依赖安装后记录freeze。没有给CPU工具环境安装vLLM。
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

下面是实际GPU训练命令。默认遍历挂载的官方公开题目，按动态策略采集；可在配置的 `data.task_ids` / `exclude_task_ids` 指定范围，或用 `--task-id` / `--task-index` 限定单题。默认最多32个采集批次、每批最多2次优化器更新，仍受2小时总进程上限约束；这些是可修改的运行上限，不代表训练耗时预测。中断和基础设施故障不会被当成零奖励。

```bash
CUDA_VISIBLE_DEVICES=0,1 /root/autodl-tmp/gemma4/envs/learner/bin/python -m training.run \
  --config configs/a800_2x80.toml \
  --data-root /root/autodl-tmp/gemma4/official/competition \
  --model-path /root/autodl-tmp/gemma4/models/gemma4-31b-qat \
  --run-dir /root/autodl-tmp/gemma4/runs/grpo-run-001
```

输出包括官方结果与奖励分项、完整日志/trace/patch、真实调用token/行为概率、课程状态、旧策略偏移、KL/裁剪统计、adapter指纹和显存峰值。新运行使用新目录；`adapters/initial`与`adapters/batch-0001`等保存独立LoRA。筛掉的组仍保留真实采样/验收记录，不能只看入选组成功率。

正式提交只取所选adapter目录的`adapter_config.json`与`adapter_model.safetensors`，按官方YAML引用。不要提交训练基座或合并权重。当前代码没有实现自动比赛提交，也没有在官方CT上取得适配效果证据；训练题上的改善不能称为独立泛化成绩。
