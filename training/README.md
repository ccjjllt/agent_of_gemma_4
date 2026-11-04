# 双 A800 训练与运行

当前采用 Unsloth BF16 LoRA＋TRL GRPO。**已有31B双卡采集记录，尚无完整有效题组、GRPO更新或已训练adapter。** 方案、官方依据、结果与未解问题见[研究说明](../research/rl-adapter-feasibility.md)，数值记录见[2026-11-02结果](../results/grpo-20261102/)。

训练基座为 `google/gemma-4-31B-it-qat-q4_0-unquantized`，固定revision与参数见[主配置](../configs/a800_2x80.toml)。目标是在官方 `gemma-4-31b-it-qat-w4a16-ct` Version 2 上挂PEFT adapter；基座与格式有官方依据，非零adapter的CT兼容与收益仍待验证。

## 服务器路径与数据边界

| 内容 | 当前服务器路径 |
| --- | --- |
| 项目代码 | `/root/gemma4/project` |
| 学习器 Python | `/root/gemma4/envs/learner/bin/python` |
| 官方公开数据 | `/root/gemma4/official/competition` |
| QAT未量化31B模型 | `/root/autodl-fs/models/gemma4-31b-qat` |
| 裁判、工具基线 Python | `/opt/gemma-runtime/bin/python`、`/opt/gemma-sandbox-baseline/bin/python` |
| 运行输出 | `/root/autodl-tmp/gemma4/runs` |
| 沙箱、临时目录 | `/root/autodl-tmp/gemma4/sandboxes`、`/root/autodl-tmp/gemma4/tmp` |

数据和模型在服务器手动获取，程序直接读取官方原目录；不维护下载脚本，不在本地下载、复制或打包训练题。共享盘仅放模型权重与官方题目，环境、日志、checkpoint放系统盘或数据盘。

subprocess后端需root启动，再由不同低权限UID运行agent工具与裁判。官方题面、base_commit、参考补丁和test_patch保持原样；agent只接收题面，参考补丁与验收资产仅供裁判侧读取。不得读取heldout，也不得把预留评测题加入训练。启动沿用已有来源声明，记录廉价文件元数据，不重复扫描完整模型权重SHA256。

验收按责任归因：官方明确拒绝模型补丁的内容或路径时，记录为可学习的终局失败；pytest收集错误、conftest导入错误及异常JUnit拒绝，仅在独立裁判沙箱的同题空补丁验收能正常执行测试、且候选没有明确环境故障诊断时归为模型失败。对照只在这些异常后执行一次，耗时计入run预算，不使用参考补丁，不替换候选奖励或轨迹。权限、磁盘、工具缺失、超时、信号、官方test_patch错误和无法归因的异常仍保留重试。`verification_complete`表示终局裁决完成，`pytest_ran`单独说明是否实际执行pytest；未观察到的修复率和回归率仍为未知区间，不能补成零。

奖励目标为`wallclock_composite_intervals_v3`：所有有效轨迹都扣`efficiency_weight × delivery_wall_seconds / walltime_scale_seconds`，默认每3600秒扣0.02，不按成功与否开关、不封顶。`delivery_timing`从官方任务执行入口计到裁判入口，包含setup、prefill、生成、工具、补丁提取、代理清理及排队，排除裁判setup、验收、归因对照和worker初始化；它独立于会话超时的queue-excluded时钟。计时缺失时，已测worker总时长只能提供成本上界，不能替代实际交付耗时；连有限上界也缺失则记录未知并筛掉该组，不触发基础设施重试。旧token目标的配置、采集及更新checkpoint不混入新run；历史结果保持原语义。该线性项是时间与修复的权衡，权重仍需实测，不能保证任意慢的成功始终优于失败。配对评测单列交付墙钟、覆盖率和差值。

## 启动与同run续跑

以下命令只在已授权的Linux GPU服务器执行。从主配置准备本次预算内的 `configs/first-group.toml`；默认为每批4个有效组、最多32批，首个短程可设为单批单组。历史[004配置](../results/grpo-20261102/004-config.toml)仅作证据，不复用其旧设置。新run使用全新目录。

在同一终端先设置环境，再执行所需命令：

```bash
cd /root/gemma4/project
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=/root/autodl-tmp/gemma4/tmp
```

```bash
/root/gemma4/envs/learner/bin/python -m training.run \
  --config configs/first-group.toml \
  --data-root /root/gemma4/official/competition \
  --model-path /root/autodl-fs/models/gemma4-31b-qat \
  --run-dir /root/autodl-tmp/gemma4/runs/grpo-next
```

默认窗口32K、4路生成、8个CPU worker；7200秒run预算包含模型加载。训练会话时钟扣除已确认的纯GPU排队时间，同时记录原墙钟，与官方评测口径不同。

采集在总预算截止前预留`run.update_reserve_seconds`（默认900秒，须为正有限值），用于停止在途任务、保存状态、GRPO更新、adapter导出和最终checkpoint。达到采集截止时交付已有完整有效组，无须凑满4组；未完整组的预算取消不计基础设施故障、不补零分。已有有效组会先完成更新及落盘，再以75退出等待续跑（已达最大批数则正常完成）；没有有效组时保留采集checkpoint，以75退出且不推进批次。900秒是初始配置，尚不是31B完整更新耗时的实测上界，总预算仍为硬截止。

每题目标8条；单个槽位耗尽基础设施重试后只剔除该槽位，其他轨迹继续完整运行和验收。全部槽位终结后，至少2条有效轨迹即可计算组内优势，保留其中所有成功和模型失败；不足2条不形成学习组，基础设施故障不补零分。ledger记录计划、有效和耗尽槽数，组大小改变按实际轨迹数计算的损失权重；这种筛除有幸存者偏差，不能称无偏恢复完整8条分布。

任务运行健康状态单独持久化，不写入模型成功率或奖励。含耗尽槽位的题默认退避60秒，连续故障指数增长、上限1800秒；策略更新及续跑不清除退避。覆盖预算只给从未调度的题；已调度但没有能力反馈的题，退避结束后仅通过均匀探索重试。全部可选题均退避时保存状态并退出75，不忙等、不消耗空批次；纯预算取消不记健康失败。

仅在同一run保存完整 `recovery/` 状态时续跑：

```bash
/root/gemma4/envs/learner/bin/python -m training.run \
  --resume --run-dir /root/autodl-tmp/gemma4/runs/grpo-next
```

续跑使用保存的配置与来源，恢复轨迹、采样器、LoRA、Adam、scheduler、RNG及更新进度，只消费原预算剩余量。追加额度按会话授权与预算执行，在续跑命令加 `--additional-walltime-seconds N`；重复启动不自动续额度。只有adapter不能完整恢复。改变rank、奖励、采题、摘要模式或其他训练参数须新建run；旧调度、旧状态版本与不匹配的来源/配置不可混用。

当前采用首步正值的[线性调度](lr_schedule.py)，skipped不推进调度；原始optimizer调用、非零LR调用和权重变化分别记录，零LR、跳步与初始权重不算有效训练。

## 前缀复用的运行注意事项

生成前缀缓存动态上限为每卡15%，预留活动KV、梯度/Adam与工作空间；`prefix_cache_bytes=0`关闭，训练前清空。镜像sliding KV的额外存储纳入预算。实现见[生成引擎](generation.py)、[缓存预算](cache_budget.py)、[KV存储](kv_cache.py)。

训练默认 `training_prefix_reuse="tree"`，合并相同token路径，保留原调用上下文、每次目标损失及完整前缀梯度。全局和滑窗注意力都沿祖先路径分块计算，默认每块最多256个查询；祖先KV、矩形mask和GQA扩展在非重入checkpoint内部重建，不保存全树N×N mask。组上限4个调用、32768个唯一节点；每设备512 MiB约束临时mask和索引空间，预算不足先缩查询块，再回退原独立后端。**此预算不包含Q/K/V、attention分数、激活或优化器，31B的512维头走math SDPA时单块峰值仍需实测。**

注意力布局版本为`segmented_tree_sdpa_v1`，要求Transformers 5.17 native Gemma4/SDPA，共享前向作用域覆盖对应反向；独立调用不再切换native后端。旧稠密布局checkpoint不能混入新布局续跑。见[规划器](prefix_tree.py)、[backend](tree_backend.py)、[policy](policy.py)。一次性CPU核对中，9K/30K共同前缀分别形成9004/30004节点共享树，Q256时mask/index预算估算约17.8/59.3 MiB；长输入attention前反向及小型Gemma4层输出/梯度已对照独立计算。当前31B双卡兼容性、总峰值和端到端收益未测，这些机制证据不等于GPU提速。

## 工具状态摘要修复

用户暂不启用修复，训练和评测保持 `event_summarizer="official"`。ADK默认摘要可能遗漏结构化工具状态；[本地实现](tool_state_summary.py)保留备用，比赛提交接口不能安装宿主summarizer。本地模式不能称比赛侧已修复，详见[研究说明](../research/rl-adapter-feasibility.md)。

## 产物、同步与关机

- `run_config.json`：实际配置与来源身份；与对应run一同保留。
- `recovery/`：完整训练精度续跑状态；不能用BF16导出adapter代替。
- `adapters/batch-*/`：每批导出的独立BF16 adapter；`adapters/initial/`不是训练成果。
- `training_summary.json`：已提交批次的更新、LR、KL、奖励覆盖与adapter身份；`metrics.jsonl`、`generation_metrics.jsonl`记录运行和性能，`reward_summary.json`记录奖励与筛组证据。

提交取所选adapter的 `adapter_config.json` 与 `adapter_model.safetensors`，按官方YAML引用，仍需验证官方CT加载与修复收益。

原始调用、轨迹、补丁和JUnit留服务器；用[日志同步入口](../scripts/sync_experiment_logs.py)经SSH取回数值摘要、重要日志与配置，不同步题目、权重或凭据。关机前正常停止实验，等待进程退出及checkpoint落盘，完成关键记录和代码同步；不擅自重启。续跑状态与有用adapter留服务器非共享盘，本地不保留重复初始状态和临时流水。

## 收益与单变量比较

实际更新后再做BF16配对评测，沿用上述环境；`TASK_ID` 替换为公开题ID，可重复 `--task-id`。输出和scratch使用不同的新目录：

```bash
/root/gemma4/envs/learner/bin/python -m training.evaluation \
  --config configs/first-group.toml \
  --data-root /root/gemma4/official/competition \
  --model-path /root/autodl-fs/models/gemma4-31b-qat \
  --adapter /root/autodl-tmp/gemma4/runs/grpo-next/adapters/batch-0001 \
  --training-run-config /root/autodl-tmp/gemma4/runs/grpo-next/run_config.json \
  --task-id TASK_ID --seed 3407 --seed 3408 --walltime-seconds 7200 \
  --output-dir /root/autodl-tmp/gemma4/runs/eval-next \
  --scratch-dir /root/autodl-tmp/gemma4/tmp/eval-next
```

入口顺序比较两臂，每臂重置随机状态和缓存，使用同题、同seed、同墙钟规则及完整官方验收；7200秒是整个评测run的总额度。`paired-summary.json`报告修复/退化、repair/regression、证据覆盖和成本。它是BF16结果，默认按训练题成绩解释，不能替代官方CT或独立泛化结果。训练摘要、KL或吞吐本身不证明能力收益。

单变量配置准备与结果比较由[ablation入口](ablation.py)完成，不会自动启动GPU：

```bash
/root/gemma4/envs/learner/bin/python -m training.ablation prepare \
  --baseline configs/first-group.toml --output-dir tmp/reuse-one \
  --factor reuse --value 1
/root/gemma4/envs/learner/bin/python -m training.ablation compare --help
```

各臂从同一初始基座新建run，提供实际配置、训练摘要和绑定adapter的配对结果。reuse固定LR调度，更新次数与算力仍不同；被评测adapter须有完整非零LR历史及独立权重变化证据，缺失保持未知，零更新不算收益。其他因素见CLI；完整训练会改变轨迹与动态采题，不能当作固定同批因果对照。当前尚无对照的更新后收益。
