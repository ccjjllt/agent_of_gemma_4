# P011：按独立问题数量运行完整两轮 SFT

此目录的真实连续训练载体已构建、冻结。最终数据为 **139个独立issue、58个仓库**：128条新local-edit加11条旧行为事务，451,974输入token、101,823监督token，长度中位2847、P95 6824、最大8080，完整两轮为 **70个optimizer steps**。这些是已冻结训练输入，不是已完成训练的结果。

**当前是V2重跑，平台RUNNING，已从运行时日志确认4×NVIDIA L4、每卡23034MiB。** [实际作业](https://www.kaggle.com/code/ccjjllt/p011-12b-continuous-sft-139-issues)、[V2回执与硬件证据](job-request-v2.json)。V1虽然请求NvidiaL4，但载体错误地删掉比赛关联，实际得到2×T4，在75.15秒后退出，训练0步、无checkpoint；[故障说明](failure-v1.md)保存了原始证据与官方资源条件。V2恢复比赛关联，显卡记录与检查移到安装依赖前，数据和训练配方不变；目前优化进度仍待日志确认。

data SHA256为 `ebfa8d1357c413e467fc71295c371f178f08932b10fb179b95a2014b1c8638b0`。V1原载体已归档到 `submitted-v1/`；不得将V1的RUNNING请求记录当作训练完成或四L4运行证明。

## 入口与冻结条件

```sh
.venv/bin/python research/011/training-run-20261018/build_carrier.py \
  --dataset-manifest research/011/data-scaling-20261018/dataset-manifest.json \
  --platform-wall-seconds 7200 --training-stage-seconds 5400
```

构建只读已冻结 train 数据并打包，不启动远端作业。默认生成现有私有 kernel `ccjjllt/p011-12b-continuous-sft-139-issues` 的 `kernel/train.py` 与 metadata；根代理统一执行实际 push与账单记录。L4要求比赛关联、关闭网络，并明确NvidiaL4资源参数；保留比赛挂载不意味着读取比赛任务内容。最终资源以运行时nvidia-smi为准。

要求数据 manifest 为 `status=frozen, split=train`，包含 `encoded.path`（仓库相对路径）及 SHA256、rows、independent_issues、independent_repos、max_sequence_length、provenance。载体核对实际编码数据、全train、同一协议指纹、**至少128个独立issue且每issue一行**。不以旧8条数据填充，不重复采样伪造规模，不截断历史，不读竞赛 tasks 或 heldout。

provenance 延用已验证12B CT v2 tokenizer/template与完整工具schema；新数据 max length 上限8192，运行 config 取实际最大长度向128对齐。8192是载体接受上限，不是已有最大长度实测容量结论；P009实跑最大长度为6089。

## 训练与资产保留

- 复用 `ccjjllt/p007-12b-cpu-nf4-cache` 的12B QAT→NF4冻结基座；缓存manifest SHA为 `e7cde8010895cf3fc4928701c9346ac442c05be97e34c97478b9d6e97eeff4ca`。不重新量化或下载模型。
- 使用原Linux CPython3.12镜像、v31 wheelhouse、四rank DDP；每rank microbatch1、梯度累积1，global batch=4。r8/alpha16、LR2e-5、两轮、线性衰减延续P009配方。
- **实际N决定steps：`ceil(N/4) * 2`**。128为64步；512为256步。尾批由既有DDP用零权重空rank参与，不重复样本进入目标。
- 开启 `ddp_length_grouped_batches=true`：原随机顺序在64条窗口按长度组织全局batch后再打乱batch；每条每epoch恰一次，不改变监督或截断。其GPU吞吐收益尚未实测。
- checkpoint间隔为 `min(32,max(8,steps//4))`：128条每16步，512条每32步，并强制保留最后一步。checkpoint含adapter、optimizer、scheduler、scaler、四rank RNG、epoch/next_batch与训练指纹；冻结基座继续引用缓存，不重复保存基座权重。
- 完成后保存 `training/train-run/` 完整checkpoint和独立 `training/adapter/`，附export manifest、逐步loss/实际LR/每rank token长度、实测GPU峰值与耗时。
- 只有实际optimizer_steps达到计划值且导出完成才写completed；超时保留已完成checkpoint和失败日志，不声称完成两轮。

`P011_RESUME_CHECKPOINT` 可指向同一次冻结数据/配置/四rank的完整checkpoint，复制到新输出后由既有训练指纹核对并恢复。128扩512属于新数据集，不能静默恢复旧optimizer；同一builder入口支持新的完整运行。默认从QAT/NF4基座初始化新LoRA，不把P009八条四步adapter自动续作大样本结果。

## 预算估计与本轮时限

P009真实训练四步计算合计100.37秒，约25.09秒/step；包含模型加载的训练进程约365秒，其中训练循环约101秒。为排队/安装/加载/导出留出宽裕规划，以约6分钟加载和12分钟总固定开销估计：

| 数据规模 | 两轮steps | 按旧分布的计算时间 | 1×–2×旧step时间加12min规划 | 若四L4按2×墙钟计额度 |
|---|---:|---:|---:|---:|
| **本轮139独立issue** | **70** | **29.3min** | **41.3–70.5min** | **约1.38–2.35额度小时** |
| 128独立issue | 64 | 26.8min | 38.8–65.5min | 约1.29–2.18额度小时 |
| 512独立issue | 256 | 107.1min | 119.1–226.1min | 约3.97–7.54额度小时 |

以上是基于真实旧吞吐的**假设范围，不是新数据测量或上界**。实际长度分布、每步最慢rank、attention计算、监督label密度、checkpoint写入均影响耗时；不能仅按监督token总数估算。builder把真实输入长度min/mean/P50/P95/max、输入及监督token总量写入run manifest，运行后以新metrics替换估计。

首轮采用平台7200秒、worker7080秒、训练stage最多5400秒；stage另为导出保留180秒。该预算预期容纳139条两轮，若实际密度高导致未完成，保留checkpoint恢复。512不能沿用90分钟stage；例如显式设置平台14400秒、stage13200秒，需由根代理结合实际长度和剩余额度确定。builder会拒绝明显低于旧step率点估计加固定开销的时限。

## 诊断与当前完成边界

载体默认**只训练和导出**，不在尾部重复旧三题，也不提交。输出 `diagnostic-handoff.json` 提供adapter身份，可在完整训练之后交给既有P010固定诊断载体，是否运行及正式submission由后续真实结果决定。

已完成139行真实数据冻结、实际build，以及V1失败定位与V2资源配置修复。V2服务器回读的源码SHA与本地一致，competition_sources与NvidiaL4正确保存，且运行时日志实际确认四张L4。没有执行额外模型容量探针或新增测试文件。待确认的是实际优化进度与训练结果，不能把平台RUNNING写成导出已经完成。

V2 `kernel/train.py` SHA256为 `94fec02e238e739b5ebe009794948aec4bc4c6379b0f2cb93c235bb52df9e75a`。V1旧SHA与carrier manifest保存在 `submitted-v1/`。README不在嵌入payload内。
