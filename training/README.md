# 双 A800 训练与运行（Single-stream SPO）

固定路线：Gemma 4 31B IT QAT BF16 基座＋LoRA Rank 128 / Alpha 256 / Dropout 0＋单流 SPO＋异步多 worker。SGLang 负责生成，同机双 A800 同时承担推理与训练（分时租约，非双引擎并行）。首轮只跑在线 SPO。

训练基座为 `google/gemma-4-31B-it-qat-q4_0-unquantized`，固定 revision 与参数见[主配置](../configs/spo_a800_2x80.toml)。目标是在官方 `gemma-4-31b-it-qat-w4a16-ct` 上挂 PEFT adapter；基座与格式有官方依据，非零 adapter 的 CT 兼容与收益仍待验证。

## 服务器路径与数据边界

| 内容 | 当前服务器路径 |
| --- | --- |
| 项目代码 | `/root/gemma4/project` |
| 学习器 Python | `/root/gemma4/envs/learner/bin/python` |
| 官方公开数据 | `/root/gemma4/official/competition` |
| QAT未量化31B模型 | `/root/autodl-fs/models/gemma4-31b-qat` |
| SGLang 服务 | `http://127.0.0.1:8000`（另行部署，本程序只做 health 握手） |
| 运行输出 | `/root/autodl-tmp/gemma4/runs` |
| 沙箱、临时目录 | `/root/autodl-tmp/gemma4/sandboxes`、`/root/autodl-tmp/gemma4/tmp` |

数据和模型在服务器手动获取，程序直接读取官方原目录；不维护下载脚本，不在本地下载、复制或打包训练题。共享盘仅放模型权重与官方题目，环境、日志、checkpoint 放系统盘或数据盘。

subprocess 后端需 root 启动，再由不同低权限 UID 运行 agent 工具与裁判。官方题面、base_commit、参考补丁和 test_patch 保持原样；agent 只接收题面，参考补丁与验收资产仅供裁判侧读取。不得读取 heldout，也不得把预留评测题加入训练。启动沿用已有来源声明，记录廉价文件元数据，不重复扫描完整模型权重 SHA256。

## 奖励与价值

以官方验收为绝对核心：候选补丁与空补丁基线在同一需求、同一环境下做节点级配对（P=修复进展，G=回归），基线缓存带内容哈希校验，不一致直接判 ineligible。基础奖励为有界复合奖励，效率项只在成功任务中启用小权重。过程评价只认执行证据，不认模型自述。价值跟踪按任务持久化（连续奖励 EMA＋独立成功概率），派发时冻结基线，共享更新按派发序号有序应用。

## 启动与续跑

以下命令只在已授权的 Linux GPU 服务器执行。SGLang 须先用基座模型、双卡并行、`--enable-lora` 部署好；本程序不拉起服务端，只做启动握手与部署探针。

在同一终端先设置环境，再执行所需命令：

```bash
cd /root/gemma4/project
export CUDA_VISIBLE_DEVICES=0,1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=/root/autodl-tmp/gemma4/tmp
```

```bash
/root/gemma4/envs/learner/bin/python -m training.run_spo \
  --config configs/spo_a800_2x80.toml \
  --data-root /root/gemma4/official/competition \
  --model-path /root/autodl-fs/models/gemma4-31b-qat \
  --run-dir /root/autodl-tmp/gemma4/runs/spo-next \
  --backend-url http://127.0.0.1:8000
```

启动前只有纯 CPU 预检（`--check-only`）：配置、模型目录结构、tokenizer、SGLang 可达性。非空 run_dir 直接拒绝启动；继续训练必须用 `--resume`，且恢复时校验 serving 版本、adapter SHA、trainer checkpoint 步数与 ledger（含 ACTIVATED 中断态的来源证明），对不上就停。

更新事务按阶段推进：RESERVED → LEARNING → OPTIMIZED → EXPORTED → REGISTERED → ACTIVATED → CONSUMED；训练租约内只做优化、导出与注册，发布与提交在训练租约之外完成，避免 GPU 租约死锁。退出码：0 跑满批次，75 预算耗尽（可续跑），2 配置或运行失败。

预算覆盖整个 GPU 会话（含初始化）：`run.walltime_seconds` 默认 7200，`update_reserve_seconds` 默认 900。续跑只消费剩余量，不自动续额度；损坏或缺失的 `budget.json` 在续跑时直接失败，不重置。追加额度走 `--additional-walltime-seconds N` 显式授权。

## 产物、同步与关机

- `model_identity.json`：模型路径、revision、config/tokenizer 哈希、LoRA 配置。
- `spo_stream.db`：轨迹、奖励、价值、buffer、更新事务账本。
- `spo_trainer_state.pt`：FP32 LoRA＋AdamW＋scheduler＋RNG 检查点。
- `adapters/versions/`：不可变版本注册表；`metrics.jsonl`、`budget.json`：运行事件与预算。
- 提交取所选 adapter 的 `adapter_config.json` 与 `adapter_model.safetensors`，仍需验证官方 CT 加载与修复收益。

原始调用、轨迹、补丁和 JUnit 留服务器；用[日志同步入口](../scripts/sync_experiment_logs.py)经 SSH 取回数值摘要、重要日志与配置，不同步题目、权重或凭据。关机前正常停止实验，等待进程退出及 checkpoint 落盘，完成关键记录和代码同步；不擅自重启。续跑状态与有用 adapter 留服务器非共享盘，本地不保留重复初始状态和临时流水。

## 尚未验证

双 A800 80GB 能否同时容纳 31B BF16、Rank 128、32K 上下文与训练激活，须经受控首次 GPU 启动实测（模型加载→Genesis adapter→单次完整更新），确认显存峰值、真实吞吐与权重变化后再进入正式训练预算。本地 CPU/Mock 验证不代替该结论。
