# 双 A800 训练与运行

当前实现为 Unsloth LoRA＋TRL GRPO，算法与研究状态见[方案概览](../research/rl-adapter-feasibility.md)。已做 CPU 机制核对；Linux 权限隔离实测、31B 双卡吞吐、完整题组训练和官方 CT 效果尚未验证。**没有可用入口及本次 GPU 预算就不实验。**

## 准备环境与资产

在 Linux 服务器手动获取官方公开资产和模型，训练直接读取原目录；不在本地复制训练集，不读取 heldout，不使用自建课程数据。题面、base_commit 和官方资产保持原样，参考补丁与验收资产仅供裁判侧使用。

| 目录 | 必需内容 |
| --- | --- |
| 官方公开任务 | `tasks.jsonl`、`snapshots/`、`graphs/`、`embeddings/`、`wheels/`、`sandbox/`、`docker/imp.py`、`docker/telnetlib.py`、`sample_submission/` |
| 官方运行库 | 安装脚本使用的 metric wheelhouse v31，与任务目录中的 `wheels/` 分开 |
| QAT 未量化 31B 模型 | 全部 BF16 权重分片、索引、配置、tokenizer 和官方模板；核对配置中的 revision 与 `response_template_path` |

代码、资产和环境分目录存放。先备好 `git`、`patch`、`protoc`、`pigz` 及支持 venv 的 Python；以下命令在服务器项目根目录执行，路径按实际位置修改：

```bash
gemma_root=/root/autodl-tmp/gemma4
gemma_python="$gemma_root/envs/learner/bin/python"
gemma_model=/path/to/gemma4-31b-qat

bash scripts/prepare_server.sh runtime --base "$gemma_root" --wheelhouse "$gemma_root/official/wheelhouse-v31"
bash scripts/prepare_server.sh sandbox --base "$gemma_root" --data-root "$gemma_root/official/competition"
bash scripts/prepare_server.sh learner --base "$gemma_root"
```

学习器依赖固定在 [learner.lock.txt](../requirements/learner.lock.txt)；官方运行库与工具基础环境分别安装到 `/opt/gemma-runtime`、`/opt/gemma-sandbox-baseline`。安装脚本不下载训练资产、不启动训练。若用 `/dev/shm` 存权重，关机后需重新准备。

当前 subprocess 后端需要 root：模型工具和验收仓库代码分别使用不同的低权限 UID；数据、裁判结果及每条轨迹的私密 API 凭据由 root 隔离。这不等同正式 Docker 评分环境，最终 adapter 仍需在官方 CT 栈验证。

## 启动训练

统一使用 [a800_2x80.toml](../configs/a800_2x80.toml)：每题 8 条轨迹，每批目标 4 个有效题组、最多尝试 12 组，最多运行 32 批。

LoRA 为 `r=128 / alpha=256`。基座按 BF16 加载，评分使用 BF16 autocast；LoRA 训练参数可保留 FP32。adapter 默认导出 BF16 副本并核对文件内实际精度，完整续跑检查点保留训练精度。更改 rank 后须新建 run，不能用 rank 32 的检查点继续训练。

同批最多更新 2 次、8 个 CPU worker、4 个 GPU 活动请求；单次调用窗口 32K，run 初始额度 7200 秒并包含模型加载。学习率按最多 64 次更新设置 7 步预热；时间额度不保证完成全部批次。候选组须完成全部轨迹及验收，基础设施失败不算零奖励。

```bash
# 只显示配置，不读取题目或加载 CUDA
"$gemma_python" -m training.run --config configs/a800_2x80.toml --describe

# 获得本次 GPU 授权与预算后执行；新运行使用新目录
CUDA_VISIBLE_DEVICES=0,1 "$gemma_python" -m training.run \
  --config configs/a800_2x80.toml \
  --data-root "$gemma_root/official/competition" \
  --model-path "$gemma_model" \
  --run-dir "$gemma_root/runs/grpo-001"
```

可用 `data.task_ids`、`exclude_task_ids` 或 `--task-id` 限定公开任务。首次运行前核对配置中的实际路径；程序会记录来源指纹并检查运行库兼容性。

## 会话预算与续跑

默认 `runtime.session_clock="queue_excluded"`：60 分钟会话只扣除已确认的纯 GPU 排队区间，生成和工具执行仍计时，原墙钟同时记录。这是训练口径，不等同官方墙钟评测；设为 `wall` 可恢复原口径。单命令与完整验收各 300 秒、50 回合、100 次工具及整个 run 的墙钟上限保持生效。

```bash
CUDA_VISIBLE_DEVICES=0,1 "$gemma_python" -m training.run \
  --resume --run-dir "$gemma_root/runs/grpo-001"
```

续跑沿用已保存的配置和路径，核对来源并独占运行目录；恢复已持久化轨迹、采样器、LoRA、Adam、scheduler、RNG、冻结概率与更新进度。未完成会话用新身份重跑；只有 adapter 的旧运行不能完整续跑。

预算不会重置；非正常中断的失联时间保守扣入旧额度。仅在另获授权后使用 `--additional-walltime-seconds N` 增加额度。基础设施故障默认额外重试一次；单次采集达到 3 个失败组后收尾，若仍有计划批次则保存下一批状态并退出 75，供显式续跑。

## 吞吐测量

生成端已实现连续批处理、分块与共享 prefill、KV 扩容及异步持久化。`device_map="balanced"` 是跨卡分层，不是张量并行；实际速度与显存容量待 31B 实测。

用服务器保留的 `model_calls.jsonl` 重放至少 4 个完整调用，`--calls` 可重复指定。先查看含暖机的 token 预算，不加载模型、不创建输出目录：

```bash
"$gemma_python" -m training.profile_generation \
  --config configs/a800_2x80.toml --model-path "$gemma_model" \
  --calls /path/to/model_calls.jsonl --max-calls 4 --batch-sizes 1,2,4 \
  --output-dir "$gemma_root/runs/generation-profile-001" \
  --walltime-seconds 1800 --plan-only
```

获准实测后去掉 `--plan-only`；`--profile` 添加独立分析轮次，`--prefix-cache` 启用 LRU 对照。重放按所有调用同时就绪测量引擎，报告吞吐、显存及概率差，不能代表真实 agent 排队或模型收益。完整流程还需计入 8 条轨迹、验收、GRPO 和检查点的总耗时。

## 输出与日志

| 位置 | 用途 |
| --- | --- |
| `run_config.json`、`metrics.jsonl` | 配置、来源及运行事件 |
| `reward_groups.jsonl`、`reward_summary.json` | 全部完整候选组的奖励、缺失原因、筛选与更新统计 |
| `generation_metrics.jsonl`、`profiles/` | 生成与缓存指标、性能分析 |
| `adapters/`、`recovery/` | 独立 LoRA 与完整续跑状态 |

只有消费确认且完成官方验收的轨迹才能训练。原始 token、trace、补丁和 JUnit 留在服务器；提交时仅取所选 adapter 的 `adapter_config.json` 与 `adapter_model.safetensors`，按官方 YAML 引用。

在本机用可用 SSH 地址取回数值日志，具体参数见 [sync_experiment_logs.py](../scripts/sync_experiment_logs.py) 的 `--help`：

```bash
python3 scripts/sync_experiment_logs.py --host root@example.com --port 22 --run grpo-001
```

将示例地址和端口替换为实际入口。同步默认写入 `results/autodl/`，不取回题目、原始调用、补丁或权重；日志取得成功不代表训练完成。
