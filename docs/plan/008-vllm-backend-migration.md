# 008 vLLM 推理后端迁移

状态：本地协议、lease 顺序和 PEFT target 审计已完成；A800 真实加载、LoRA 生效与 Level-2 sleep 待授权实测。
日期：2026-11-12。
范围：只替换推理后端及其部署/依赖；不改 SPO 目标、奖励公式、Rank 128、checkpoint 事务和 training prefix tree。

## 0. 结论

活动推理路径已从 SGLang 切换为独立 vLLM 实现：

- `training/rollout/vllm_backend.py`：独立 vLLM OpenAI 协议，不再继承 SGLang 的 `/generate`、`/cancel`、`/release_memory_occupation` 语义。
- `training/run_spo.py`：按 `inference.engine` 严格分派；`vllm` 创建 `VLLMBackend`，`sglang` 只作为历史兼容入口，未知引擎直接失败。
- `training/adapter/manager.py`：增加 vLLM LoRA 名称、serving binding 和 Gemma 4 target 审计。
- `training/remote_serving.py`：原 SGLang serving engine 的通用化版本；`sglang_serving.py` 仅保留导入 shim。
- `configs/spo_a800_2x80.toml`：`engine = "vllm"`、`max_loras = 2`、`sleep_level = 2`。
- `requirements/inference-vllm.lock.txt`：独立推理环境，vLLM 0.32.0 / torch 2.13.0；不并入 learner 的 torch 2.9.1 环境。

本地证据见 `results/vllm-migration-local-20261112.json`。

## 1. 为什么不能只改 TOML

原 `VLLMBackend` 只是 `HTTPInferenceBackend` 的空子类，仍使用 SGLang 的 `/generate`、`/cancel`、`/drain`、`/release_memory_occupation`、`/resume_memory_occupation`。标准 vLLM 不提供这套协议，因此只改 `engine = "vllm"` 会在第一次提交或第一次训练租约时失败。

新实现使用 vLLM 原生接口：

| 训练需求 | vLLM 接口 |
| --- | --- |
| 健康检查 | `GET /health` |
| 部署能力 | `GET /server_info?config_format=json`、`GET /v1/models`、`GET /version` |
| 生成 | `POST /v1/completions` |
| 取消 | `POST /abort_requests` |
| 暂停/恢复 | `POST /pause?mode=wait`、`POST /resume` |
| 训练期释放 GPU | `POST /sleep?level=2&mode=abort` |
| 训练后恢复 | `POST /wake_up?tags=weights` → `POST /collective_rpc {"method":"reload_weights"}` → `POST /wake_up?tags=kv_cache` |
| 动态 LoRA | `POST /v1/load_lora_adapter`、`POST /v1/unload_lora_adapter` |
| 负载指标 | `GET /metrics`（Prometheus 文本） |

Level-2 sleep 需要服务端 `VLLM_SERVER_DEV_MODE=1 --enable-sleep-mode`；动态 LoRA 需要 `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`。`start()` 在首次 rollout 前检查 `max_lora_rank >= 128`、`max_loras >= 2`、`enable_sleep_mode`、`/is_sleeping` 和 `/is_paused`，不满足就失败，不做静默降级。控制端点若返回空 200，客户端会继续用 `/is_paused` / `/is_sleeping` 轮询确认，而不是把 HTTP 200 当作成功。`generation_timeout_seconds` 控制整条 completion 的 HTTP 等待，不能沿用 30 秒健康检查默认值。

## 2. Token 与采样契约

`/v1/completions` 请求使用：

- `prompt`: 原始 `input_ids` 列表；`add_special_tokens=false`，避免 vLLM 插入 BOS 改变 TITO 对齐。
- `return_token_ids=true`：响应必须带 `prompt_token_ids` 和 `token_ids`。
- `logprobs=1`：响应必须带 `choices[0].logprobs.token_logprobs`，逐项与 `token_ids` 等长且有限。
- `model=<lora_name>`：vLLM 用该名称选择 adapter；响应 `model` 必须等于该名称。
- `request_id`：响应 `id` 必须绑定为同一 ID（vLLM 形式为 `cmpl-<request_id>`）。
- `cache_salt=adapter_sha256`：仅在 `inference.enable_prefix_cache=true` 时发送，给 block hash 加盐；首轮配置保守关闭 prefix cache，直到 A800 实测跨 adapter 隔离。

后端拒绝以下响应：缺少 prompt token IDs、token IDs 与 logprobs 不齐、logprob 非有限值、`model` 不匹配、`id` 不匹配、`finish_reason` 缺失。不会用文本重新 tokenize 或重算概率。

## 3. LoRA 名称与 Gemma 4 兼容性

### 3.1 名称映射

vLLM server 名由不可变 identity 派生：

```text
spo-v{version:08d}-{adapter_sha256[:16]}
```

`AdapterManager.serving_binding()` 返回 manifest、name、path、SHA；`VLLMBackend` 在 `load_adapter` 后调用 `/v1/models` 确认该名称真的注册，卸载后确认名称消失。HTTP 200 不作为唯一证据。若 trainer 进程崩溃但 vLLM 仍持有同名 adapter，`load_adapter` 会在本地 SHA/path 校验通过后接管该注册，避免 `--resume` 被重复名称 400 拒绝。

每个 server 至少注册两个 LoRA（`--max-loras 2`），供 N/N+1 版本切换；后端本地再限制 `max_resident_adapters=2`，不会静默加载第三份。

### 3.2 k_eq_v 风险

Gemma 4 31B 有 10 个 `full_attention` 层（5, 11, 17, 23, 29, 35, 41, 47, 53, 59）使用 `attention_k_eq_v`：checkpoint 只有 `k_proj`，V 复用 K。HF 中 `k_proj` 的 LoRA 同时改变 K 和 V。

vLLM PR #61267 修复了 packed `qkv_proj` 的 K→V LoRA 打包，但该 PR 于 2026-11-02 合并，**晚于 0.32.0**。因此：

- 当前 `policy.py` 故意省略这 10 层的 `k_proj`（400 个 target 而不是 410）。
- `AdapterManager.audit_lora_compatibility()` 对 adapter regex 与模型 `layer_types` 逐层匹配：
  - `target_count` 必须为 400；
  - `full_attention` 层出现 `k_proj` 时，除非显式 `allow_k_eq_v_lora=True`，否则拒绝（reason `k_eq_v_k_proj_requires_vllm_pr_61267`）；
  - `full_attention` 层出现 `v_proj` 直接拒绝；
  - `num_kv_shared_layers` 覆盖的层只允许 q_proj，出现 k/v target 直接拒绝。
- 该审计在 `VLLMBackend.load_adapter()` 前执行，失败即不加载。

本机对现有 `adapters/initial/adapter_config.json` 的审计结果：`target_count=400`，`k_eq_v_targets=[]`，`compatible=true`。人工把 `layers.5.self_attn.k_proj` 加进目标后，审计返回 `compatible=false`，`target_count=401`，reason 为 `k_eq_v_k_proj_requires_vllm_pr_61267`。

如果以后要训练这 10 个 `k_proj`，必须换到包含 #61267 的 vLLM wheel/commit，并重新做 vLLM↔Transformers 逐 token logprob 对照；不能只凭加载成功判断。

### 3.3 LoRA alias / 在线生效

PR #39816 仍 Open，针对 Gemma4 `self_decoder`/`cross_decoder` alias 导致 LoRA 被 reset 的问题。vLLM 0.32.0 的 `LoRAModelManager._create_lora_modules()` 已使用 `named_modules(remove_duplicate=False)` 并按物理模块去重（代码注释明确提到 alias reset），但这不是本项目 adapter 的端到端证明。

首次 A800 必须验证：

1. 同一 prompt/tokenizer，Transformers+peft 与 vLLM 的逐 token logprob 差；
2. 非零 adapter 的 B 权重是否真的改变输出（与 base 对比）；
3. adapter 加载/卸载/重载后输出是否一致；
4. Level-2 wake + `reload_weights` 后 base 与 adapter 是否都恢复。

## 4. 服务端部署

独立环境（已装在数据盘，避免占用系统盘；`/root/gemma4/envs/inference-vllm` 是指向它的软链）：

```bash
python3.12 -m venv /root/autodl-tmp/gemma4/envs/inference-vllm
ln -sfn /root/autodl-tmp/gemma4/envs/inference-vllm /root/gemma4/envs/inference-vllm
/root/gemma4/envs/inference-vllm/bin/python -m pip install -r requirements/inference-vllm.lock.txt
```

2026-11-12 无卡模式实测：PyPI 官方源单连接被限速到约 20–50 KB/s，必须换国内镜像并提高并发（`--index-url https://mirrors.aliyun.com/pypi/simple/`、`UV_CONCURRENT_DOWNLOADS=24`）；32 个连接聚合可达 50 MB/s，整套 8.0 GB 环境约 11 分钟装完。`torch==2.13.0` 在 PyPI 上就是 CUDA 13.0 构建（`cuda-toolkit==13.0.3`、`nvidia-*-cu13`），不需要 `download.pytorch.org` 的 cu130 索引。`numpy` 同时被 `numba==0.65.0`（`<2.5`）和 Python 3.12 上的 `mistral-common[image]`（`<2.4`）限制，锁定为 2.3.5。

启动：

```bash
export VLLM_SERVER_DEV_MODE=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
/root/gemma4/envs/inference-vllm/bin/vllm serve /autodl-fs/data \
  --served-model-name google/gemma-4-31B-it-qat-q4_0-unquantized \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --enable-lora \
  --max-loras 2 \
  --max-cpu-loras 2 \
  --max-lora-rank 128 \
  --enable-sleep-mode \
  --no-enable-prefix-caching \
  --host 127.0.0.1 --port 8000
```

`--max-lora-rank 128` 必须与训练 adapter 的 Rank 128 对齐；不要用默认 16。`--max-cpu-loras 2` 允许两个版本在 CPU 侧注册。`--no-enable-prefix-caching` 是首轮安全默认；后端仍发送 `cache_salt`。

## 5. 本地核对结果

环境：本机 `.venv` 只有 CPU torch；没有安装 vLLM，也没有连接 A800。核对的是协议接线和 PEFT target，不是 GPU 运行。

```text
python3 -m compileall -q training -> 0
import training.run_spo, training.rollout.backend,
       training.rollout.vllm_backend, training.adapter.manager,
       training.remote_serving -> imports_ok
```

一次性 mock vLLM HTTP server 覆盖：`/health`、`/version`、`/v1/models`、`/server_info`、`/is_sleeping`、`/is_paused`、`/v1/load_lora_adapter`、`/v1/unload_lora_adapter`、`/v1/completions`、`/abort_requests`、`/pause`、`/resume`、`/sleep`、`/wake_up`、`/collective_rpc`、`/reset_prefix_cache`、`/metrics`。

结果：

- `start()` 解析出 `max_lora_rank=128`、`max_loras=2`、`max_cpu_loras=2`、`sleep_enabled=true`。
- 加载 adapter 后 `lora_name=spo-v00000000-a8112ebd13ba298f`，`/v1/models` 确认存在。
- 提交 `[11,12,13]` 返回 `input_ids` 回显一致、`output_ids=[9001,9002]`、`logprobs=[-0.25,-0.5]`、`finish_reason=stop`。
- `GpuLease.training()` 进入 `pause?mode=wait` → `drain` → 卸载 adapter → `sleep?level=2`；退出时执行 `wake_up?tags=weights` → `collective_rpc reload_weights` → `reset_prefix_cache` → `wake_up?tags=kv_cache`，重新加载 adapter 并 `resume`；随后再次提交成功。
- `abort_requests` 返回 `status=aborted`；mock 共记录 32 次 HTTP 调用，全部通过。
- 空 200 控制响应变体：pause/resume/sleep/wake/abort/reload 全部返回空 200，`GpuLease` 仍通过 `/is_paused` / `/is_sleeping` 轮询完成 sleep→wake→adapter reload。
- 重启接管变体：同一 mock server 已注册同名 adapter 时，新建的 `VLLMBackend` 调用 `load_adapter` 返回 `already_loaded=true, server_resident=true`，不重复 POST。

`--check-only` 对 mock 返回 `rc=0`，输出 `engine=vllm`、`deployment_reachable=true`、`metrics_available=true`，并验证了 `max_lora_rank=128`、`max_loras=2`、`sleep_enabled=true`。

## 6. A800 首次实测（2026-11-13，2×A800 80GB）

启动（`--enable-lora --max-loras 2 --max-cpu-loras 2 --max-lora-rank 128 --enable-sleep-mode --enable-prefix-caching`，TP=2）：

```text
Resolved architecture: Gemma4ForConditionalGeneration
权重从 /autodl-fs（FUSE.AutoFS）加载：Model loading 235s；每卡 32.78 GiB
GPU KV cache size: 148,633 tokens（32K 请求 4.54× 并发）
health 200 后每卡 74.9 GiB 常驻
```

实测中修掉的问题：

1. **`ninja` 不在 PATH**：torch.compile 子进程需要 `ninja`，只给绝对路径的 `vllm` 启动会 `FileNotFoundError: ninja`，engine 初始化失败。启动脚本必须把推理环境的 `bin` 放进 PATH。
2. **sleep level 2 实测**：每卡 74,901 → 3,151 MiB（释放 143.5 GB，95.9%），wake 后回到 74,899 MiB；`reload_weights` 后 logprob 与 sleep 前逐位一致（max|diff|=0.0）。
3. **prefix caching 是 agent loop 的必需项**：关闭时每轮重算 32K 前缀，单次调用约 25s，8 分钟会话预算只够约 20 次调用，全部轨迹超时被排除。开启后实测 `Prefix cache hit rate: 88.7%`、`Running: 7–8 reqs`、生成 131–172 tok/s（关闭时 1 reqs、20–51 tok/s）。`cache_salt` 字段被 vLLM 0.32 接受（HTTP 200）。

真实 rollout 链路修复的四个缺陷（此前没有任何一条官方轨迹能完成）：

| 缺陷 | 症状 | 修复 |
| --- | --- | --- |
| `PYTHONSAFEPATH=1` + 脚本方式启动 `official_runtime.py` | `ModuleNotFoundError: recovery`（还有 `model_transport`、`session_clock`） | 模块加载时把脚本目录加回 `sys.path` |
| empty-patch probe 不传 `--endpoint` | 基线永远 `baseline_infrastructure_error` | probe 传合法占位 endpoint（不调模型） |
| attempt 命名空间与 trajectory 绑定不一致 | `KeyError: 'trajectory has no pinned adapter'` | serving 层把 `<traj>-aNNNNNN` 还原为父 trajectory |
| `build_node_report` 只用空的 `FAIL_TO_PASS` | `expected_universes` 为空，奖励永远 ineligible | 与 `junit_reward_features` 一致，使用 test_patch fallback |

另有一个运维教训：vLLM 对已注册 LoRA 是**懒加载**，删除 adapter 目录而不先 unload 会让下一个请求把 engine 打挂（`LoRAAdapterNotFoundError` → `EngineDeadError`）。

首条有效的官方轨迹（`fastapi` 类任务，agent 未修复，但验收有效）：

```text
status=BUFFERED   action_tokens=6996
P=0.0  G=0.0  total_reward=0.15  resolved=False  eligible=True
```

## 7. 仍需 A800 实测

1. 一次完整 SPO 优化步：forward/backward/optimizer/checkpoint 秒数、训练态显存峰值。
2. 真实 Unsloth Rank 128 adapter 的逐 token logprob 与 Transformers+peft 对照（当前只有 sleep/wake 前后自一致）。
3. `attention_k_eq_v` 层（当前 adapter 不覆盖 k_proj）在 vLLM 0.32.0 下是否与 HF 一致；若未来覆盖 k_proj，必须换 #61267 后的版本。
4. `pause?mode=wait` 在长轨迹下的排水时间，是否超过 `control_timeout_seconds=600`。
5. 两个 adapter 注册/切换时 `/v1/models`、`model` 参数和 prefix cache salt 的行为。
6. CT runtime 加载非零 adapter 的端到端结果。


以上任何一项未通过，不能宣称 vLLM 迁移完成或 SPO 可启动。

## 7. 首次 A800 pilot 结论（2026-11-13）

服务侧全部通过，训练侧未能在本次会话内完成一次优化步。

已验证：

| 项目 | 证据 |
| --- | --- |
| vLLM TP=2 加载 31B | `Gemma4ForConditionalGeneration`，权重 235s，每卡 32.78 GiB，health 200，常驻 74.9 GiB/卡 |
| Genesis Rank 128 adapter | `spo-v00000000-b70fe5a72f418f8c` 载入 `/v1/models` |
| Level-2 sleep / wake | 每卡 74,901 → 3,151 → 74,899 MiB；释放 143.5 GB（95.9%） |
| logprob 一致性 | 同请求重复与 wake 后均 `max|diff| = 0.0` |
| prefix caching | 命中率 88.7%，7–8 并发，生成 131–172 tok/s（关闭时 1 并发、20–51 tok/s） |

未完成：**没有 SPO 优化步，没有 v1 adapter，没有 adapter 权重差异，没有训练态显存峰值**（76,875 MiB 是服务态采样）。

原因（`logs/pilot-g-evidence/` 与 `results/a800-vllm-pilot-20261113/`）：

```text
441 次官方 rollout：440 EXCLUDED，1 BUFFERED
排除原因：438 candidate_verification_incomplete，2 baseline_verification_incomplete
抽样 result.json：Agent exceeded session timeout (8.0 min) / pytest_ran=true
唯一有效轨迹：rich_4070，verdict_valid=true，verdict_kind=agent_pytest_collection_failure
              reward=0.0，action_tokens=6,996（目标 32,768）
              其 candidate/baseline expected_universes 均为空 → 零信号
```

两个叠加的结构性问题：

1. **8 分钟 / 28 次工具调用的官方会话预算与 31B 模型的节奏不匹配**：`pytest_ran=true` 说明补丁和裁判都跑完了，但整条 episode 超时，被判无效。合格率约 0.2%（1/441），无法在 90 分钟内凑满 32,768 token 的批次。
2. **奖励信号为真空**：本数据集没有 `FAIL_TO_PASS`/`PASS_TO_PASS`，只能从 test_patch 提取测试函数；对 `rich_4070` 这类任务提取为空，`expected_universes` 两侧皆空，P/G 恒为 0，即使凑满批次也只会得到全零优势（代码会扩批或降级成 NOOP，不产生真实梯度）。

需要用户决策才能继续（都涉及研究口径，不能自动改）：放宽 `[runtime.evaluation]` 的会话预算、改用更容易的任务子集、还是接受更长会话以收集足够轨迹。修改 `target_action_tokens` 会改变 SPO 批次语义，未获授权前不动。

## 8. 回滚

- 配置把 `engine` 改回 `"sglang"` 只会走历史 SGLangBackend；不保证新 vLLM 路径的 sleep/LoRA 语义可复用。
- vLLM 路径的 adapter 名称、`/v1/models` 校验和 target 审计是强制的；不要通过关闭校验来绕过加载失败。
- 本迁移没有修改 SPO 目标、奖励、Rank、checkpoint 或 training prefix tree；这些组件的既有验证仍有效。
