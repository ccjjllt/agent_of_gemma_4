# 008 vLLM 推理后端迁移

状态：2026-11-15 已完成首次真实 SPO 更新、v1 激活、权重差和新 adapter 推理核对。非零 adapter 的 logprob 重复性存在小幅波动；尚无修复收益证据。CPU 根因修复见第 9 节，GPU 结果见第 10 节。
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

首条入缓冲的轨迹（2026-11-15 账本复核为 `rich_4070`，agent 未修复，但验收有效）：

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

服务加载与 sleep/wake 有实测，但生成调度器在本次会话中发生故障；训练侧未完成一次优化步。

已验证：

| 项目 | 证据 |
| --- | --- |
| vLLM TP=2 加载 31B | `Gemma4ForConditionalGeneration`，权重 235s，每卡 32.78 GiB，health 200，常驻 74.9 GiB/卡 |
| Genesis Rank 128 adapter | `spo-v00000000-b70fe5a72f418f8c` 载入 `/v1/models` |
| Level-2 sleep / wake | 每卡 74,901 → 3,151 → 74,899 MiB；释放 143.5 GB（95.9%） |
| logprob 一致性 | 同请求重复与 wake 后均 `max|diff| = 0.0` |
| prefix caching | 命中率 88.7%，7–8 并发，生成 131–172 tok/s（关闭时 1 并发、20–51 tok/s） |

未完成：**没有 SPO 优化步，没有 v1 adapter，没有 adapter 权重差异，没有训练态显存峰值**（76,875 MiB 是服务态采样）。

2026-11-15 直接只读查询服务器 SQLite 后修正的结果：

```text
441 条 trajectory：440 EXCLUDED，1 BUFFERED；updates 表为 0。
437 条 EXCLUDED 没有封存调用且 action_token_count=0；其余 3 条 EXCLUDED 共 9,571 tokens。
workspaces 中共 11 份正式 result.json，3 份 verdict_valid=true。
rich_4070：官方 reward=0；SPO total_reward=0.15；V_pre=0.5；raw_advantage=-0.35；6,996 tokens。
fastapi_14258、rich_4076：官方验收有效，各有 1 个 repair requirement，仅因 baseline_unavailable 被排除。
```

此前“441 次完整官方 rollout，438 次因 8 分钟超时失效”的归因撤回。441 是轨迹账本计数；绝大部分记录没有可训练的封存调用。`pytest_ran=true` 仅说明观察到 pytest 调用，必须同时看 exit code、JUnit 和交付确认。`fastapi_14258` 的 active_timeout_reached=true，但 pytest exit=1、verdict_valid=true，直接反证“会话超时必然让验收无效”。

此前“空 expected_universes 意味着整个训练奖励全零”的结论也撤回。`rich_4070` 缺的是节点级 P/G 观测；合法性与偏置使复合奖励为 0.15。单条轨迹或同 raw advantage 的批次中心化后归零，不等于任意多条轨迹必然归零。不能靠缩小 token 阈值制造梯度，亦不能从这条样本推断整个公开集没有修复节点。

8 分钟、28 工具调用及 32,768 action-token 阈值本次保持不变；先修框架，再在另行授权的真实训练中测有效轨迹产率和首次更新。

## 8. 回滚

- 配置把 `engine` 改回 `"sglang"` 只会走历史 SGLangBackend；不保证新 vLLM 路径的 sleep/LoRA 语义可复用。
- vLLM 路径的 adapter 名称、`/v1/models` 校验和 target 审计是强制的；不要通过关闭校验来绕过加载失败。
- 本迁移没有修改 SPO 目标、奖励、Rank、checkpoint 或 training prefix tree；这些组件的既有验证仍有效。


## 9. 无 GPU 根因审计与修复（2026-11-15）

证据：`results/a800-vllm-pilot-20261113/cpu-audit-20261115.json`。源为服务器 `/root/autodl-tmp/gemma4/runs/vllm-pilot-20261113g/` 的只读 SQLite、已有结果和日志；没有复制题目、参考补丁或权重，也没有读取 heldout。

### 9.1 可用 CPU 修复的连接缺陷

- **基线正常执行仍返回进程码 2**：`_empty_patch_probe_result` 原先只有嵌套 baseline_report，没有顶层 verdict_valid；main 因而始终按无效退出，runner 无法读取正常报告。现在顶层状态取自实际空补丁验收，正常断言失败仍是有效基线。
- **基线重试复用目录**：原路径仅含 TaskKey，第二次必撞已有 runtime_submission，并覆盖首次结果。现在每个执行 attempt 独立，缓存身份仍由 TaskKey 决定。
- **基线报告缺缓存必需的来源字段，且可被强行标为有效**：补入真实 JUnit SHA；仅正常执行、无 collection/setup/JUnit 故障的空补丁控制可作基线。参考补丁不用于造奖励。
- **取消请求时生成调度计数不一致**：运行中的 HTTP Future 无法由 Future.cancel 立即终止，但 endpoint 已移除其 in-flight 身份，engine 却仍把它计为 active，触发一致性异常。现在逻辑 active 剔除已取消请求，并保留 Future 供清理。
- **生成线程故障未传给训练派发循环**：现在及时抛出首个故障、记录原始原因并停止派发；全局服务故障不再降低具体任务的健康记录。
- **整段 HTTP 生成被空步忙轮询**：历史摘要为 521.6 秒内 44,113,881 个 scheduler steps。现在无新请求时对未取消 Future 有界等待；纯 CPU 对照中约 0.16 秒只需 3 步。此为调度 CPU 负载证据，不是 GPU 吞吐收益。
- **全任务健康冷却抛裸 IndexError**：现在走已有 NoDispatchableTask 等待路径，健康恢复后可再次派发。

历史日志的 generation_scheduler_summary.failed=true 证明生成线程确实失败；旧日志没有保留首次异常，取消计数缺陷是 CPU 可复现的触发路径，不能声称已从历史日志恢复了唯一触发原因。

### 9.2 证据边界与后续研究

一次性 CPU 核对覆盖：取消后的活跃计数、下一请求完成、故障后停止派发、等待中的异常传播、健康冷却恢复、基线失败后独立重试、真实 main 返回码和来源缓存归一化。未新增测试文件。

本次不修改 SPO 目标、优势中心化、奖励权重、官方资产和任务预算。空 universe 的任务仍只有稀疏成功/合法性信号；如需增强节点覆盖，应在裁判侧用公开任务的 CPU 空补丁验收建立固定且可重复的节点集合，再与候选配对，不能从候选报告临时挑有利节点。当前未宣称该扩展已实现，也未把同奖励下价值估计差异形成的梯度称为修复收益。

queue_excluded 目前仅扣有证据的本地队列等待，不扣 vLLM 内部排队；没有逐请求服务时间证据时不能把整段 HTTP 等待抵扣。CPU 修复可以消除已发现的框架失败，但 31B 在预算内的完成率、训练显存、首次有效更新、非零 adapter 的 CT 兼容和修复收益仍需后续真实 GPU 实验。


### 9.3 两个真实官方 CPU 基线与既有候选回放

在服务器独立解释器中载入本地修复函数，CUDA_VISIBLE_DEVICES 为空，probe 不创建模型会话，SDK 占位 endpoint 为本地不可达端口；正式部署文件及旧训练账本保持原样。官方公开任务、test_patch 和验收流程未修改。结果摘要与 JUnit SHA 已同步至上述 cpu-audit JSON；裁判明细仍留服务器。

| 任务 | 空补丁进程码 / pytest exit | 修复节点 | 旧候选配对后 eligible | P | SPO reward |
| --- | --- | --- | --- | --- | --- |
| fastapi_14258 | 0 / 1 | 1 | true | 0/1 | 0.15 |
| rich_4076 | 0 / 1 | 1 | true | 0/1 | 0.15 |

候选与新基线的 TaskKey 完全一致，基线 normalize 与配对成功。这里只证明两份旧候选的奖励资格可恢复；没有把旧轨迹重新入批，也未声称所有 token/ACK 可以恢复。两者都是未修复结果；与 rich_4070 一样奖励0.15，在原始 V_pre=0.5 下同优势仍会被中心化为零，不能据此宣称已有有效梯度或能力提升。

真实 CPU 验证还发现并修复了直接 --empty-patch-probe 无凭据启动却强制创建 SessionClock 的连接缺陷。probe 现不读取模型凭据、不绑定模型计时/ACK逻辑；普通 rollout 的对应边界保留。两次探针的最初手工目录/无凭据失败仅在摘要中说明，不将临时失败流水加入研究结论。


## 10. 首次真实 SPO 更新（2026-11-15）

用户明确开启 GPU 并授权实验。北京时间 01:31:52–02:47:47，共 75.90 分钟，低于 90 分钟会话上限；结束时 learner、vLLM、worker 和本轮后验进程均已退出，两卡显存为 0 MiB。实例未关机。

使用独立代码快照，直接读取服务器原有公开题目和固定 revision 模型。未改 8 分钟、28 工具调用、Rank 128、奖励公式及 32,768 action-token 目标；仅把本次累计优化步上限设为 1。实际配置、代码指纹、账本数值摘要、关键 metrics 与显存采样见 `results/spo-pilot-20261115/`，汇总为 `runtime_evidence.json`。

### 10.1 更新与产物

| 项目 | 真实证据 |
| --- | --- |
| 有效轨迹 | 8 条，覆盖 4 个公开任务，均未 resolved；其中 5 条用于本次更新，3 条仍缓冲 |
| 实际训练批次 | 35,964 action tokens、135 calls；prompt 原始总量 1,190,858；最长单次序列 18,003 tokens |
| 前缀计划 | 40 组：32 tree、8 independent（dense_attention_work_limit）；计划输入 1,226,687 tokens、unique 410,696、saved 815,991；最大 packed tree 23,784 |
| 学习信号 | 五条奖励均为 0.15；四条 V_pre=0.5、raw A=-0.35，一条 V_pre=0.325、raw A=-0.175；归一化前 std=0.0575291 |
| 真实优化 | AdamW step=1；实际 LR=5e-6；global_grad_norm=1.19194；nonzero_gradient=true |
| checkpoint | schema=2、checkpoint_phase=optimizer_step、completed_optimizer_steps=1；11,640,389,949 bytes |
| 发布闭环 | 02:41:06 v1 published/activated；ledger `upd-0001=CONSUMED`、cursor=1、registry active_version=1；learner退出码0 |
| 权重变化 | 800张量中400张量改变，全部为LoRA B；538,704,713个元素改变；max abs=5.00679e-6，L2=0.114536；所有值有限 |
| KL | pre→post action-token K3=0.0003924332，816 token、4 calls；不是全词表精确KL |

v1 SHA256：`809308049e90c4e7bc254ce5241bcf1460e110a7e95a98b98fa8fc376c0cfb80`。

服务器保留目录：`/root/gemma4/runs/spo-pilot-20261115-0134/`。adapter 为 `adapters/versions/00000001-809308049e90c4e7/` 下的 PEFT 两文件；训练状态为 `spo_trainer_state.pt`。日志与部署快照位于 `/root/autodl-tmp/gemma4/experiments/spo-pilot-20261115-0134/`。未把题目、轨迹内容或大权重复制到本地或共享盘。

本次中途发现同值优势浮点舍入可能绕过严格零信号判断。已在任何优化步之前正常中断，保留4条/31,531 tokens，修正严格同值时mean/std精确性后按同run、同预算恢复。由这次主动中断造成的8条零token排除不计为模型失败；另有4条自然验收排除，未转成零奖励训练。

### 10.2 耗时与显存

- 首个vLLM冷加载约268秒，learner Genesis加载59.6秒，更新阶段learner加载30.7秒。
- 从冻结批次到获取训练租约等待约770.5秒（12.84分钟），主要等待在途任务及重试结束。
- forward 431.27秒、backward 1,242.96秒、optimizer 0.104秒、checkpoint 22.00秒、KL 11.74秒。
- learner加载至导出结束的显存窗口为1,755.83秒（29.26分钟）；从冻结批次至发布约42.65分钟。原15分钟更新预留不足以覆盖本批实际完整更新路径。
- PyTorch learner峰值 allocated：48.70 / 51.07 GiB；reserved：50.11 / 53.46 GiB。该指标不包含vLLM和驱动分配。
- 同一训练窗口内，2秒一次的 `nvidia-smi` 整卡采样峰值为55,001 / 58,419 MiB（53.71 / 57.05 GiB）。这不是服务态峰值，亦不能据此声称完整32K单序列训练已验证。

前缀复用确实生效，未整体退化；当前learner按层切分、分组串行，tree路径使用native Gemma4/SDPA及分块与逐层checkpoint。代码没有逐组日志，不能从长时间缺少optimizer日志推断死锁，也不能仅凭saved tokens推断实际加速。此次主要耗时在反向，尚未取得kernel级profile。

### 10.3 新 adapter 推理与尚未解决的数值问题

只使用人工短算术输入，固定输入token IDs、贪心生成32 tokens，不读取评测题。

- v1重复生成输出token IDs相同；max logprob差=0.00062810。
- v1经Level-2 sleep/wake、reload_weights和adapter重载后，输出token IDs仍相同；max logprob差=0.00310435。
- 显式在每次冷请求前清空prefix cache后，上述差异仍可复现；不能简单归因为缓存冷热。
- 单resident adapter的v0→v1→v0对照：三次输出32 token相同；v0重复logprob差=0，v0/v1最大差=0.01473001，支持非零adapter在该输入上影响服务概率。
- sleep前/期间/wake后整卡约74,913 → 3,163 → 74,911 MiB；后验结束服务已停止，两卡归零。

精确逐位logprob一致性未通过，差异来源待定位；没有做HF/learner与vLLM逐token等价性或官方CT runtime加载对照。不得用输出token相同替代概率完全一致。

本轮证明了真实梯度更新、持久化、v1发布与后续生成的闭环。五条训练奖励相同，优势差异来自价值历史而非修复进展；所有有效轨迹均未修复，因此没有能力提升或独立泛化结论，不应直接把本轮视作正式长训收益。
