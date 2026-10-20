# P013 首轮 12B 4×L4 DDP 连续 SFT 执行清单（behavior 主层）

日期：2026-10-20（CPU 编码完成，GPU 待推送）。数据：`research/013/behavior-sft/`。
本轮只训主层 34 行；P012 辅助层引用不混入（§5）。效果以后续评测为准，不声称修复收益。

## 1. 冻结数据（已就绪，可训练行数与 token 统计）

- `encoded-train-12b-behavior.jsonl`：**34 行 / 34 独立 issue / 20 仓库**，
  SHA256 见 `dataset-manifest.json::encoded.sha256`（大文件不进 git，只记 sha）。
- 输入 token **152,711**，监督 token **69,587**（trainer 复算 `supervised_tokens_after_shift` 一致）。
- 长度分布：min 2847 / median 4267 / p90 6222 / **max 7929**，le6144=29，le8192=34。
- 超长：**4 条**（boltons-358: 8637，md-161: 9069，mypy-12943: 9548，sqlparse-746: 10560）
  整体拒绝、未截断工具块，单列 `layers.p013_behavior_oversize_deferred`，留待后续长上下文训练。
  **不是 38 全可编码**——任务预期 38 全 ≤8K 与实测不符，以实测为准。
- Provenance：`protocol_id=p013-behavior-v1`；tokenizer / chat_template / tools 指纹
  与 P011/P012 部署资产完全一致（`d0017181…` / `36e3a42b…` / `91ec82b1…`），仅 protocol 不同。
- 门控（`assemble_p013.py` + trainer 复检通过）：38 条全部 train split、denylist 零重叠、
  handover `requirement_done=true` + `verification_status=pass`、pre 非零 int / post 0、
  pre/post 同命令、零基正典弧 targets、`task_resolved=null`、无 synthetic、无截断工具块。
- 编码环境说明：本机系统 transformers 4.57.6 无法解析 CT v2 tokenizer 配置
 （`extra_special_tokens` list 形式），改用隔离 venv + 仓库钉选 **transformers==5.13.1**
 （`training/requirements.txt`）编码；所得 `tokenizer_hash=d0017181…` 与 P011/P012 一致，
  训练侧 `tokenizer_fingerprint` 校验可通过。venv 为一次性产物，不在仓库内。

## 2. 训练配置（新建文件，不复用旧协议配置）

新建 `training/configs/sft_p013_12b_behavior_ddp.yaml`（内容模板，主智能体落盘）：

```yaml
model_name_or_path: google/gemma-4-12B-it-qat-q4_0-unquantized
model_revision: b6ed86275a6a5735884e208bfed95b445a2c684
tokenizer_name_or_path: training/artifacts/tokenizers/gemma4-12b-ct-v2
tokenizer_revision: null
model_source_manifest: training/configs/gemma4_12b_sources.json
chat_template_path: training/configs/chat_templates/gemma4_12b_ct_v2.jinja
tool_schema_path: research/012/agent/repair_tools_v1.json
base_family: gemma-4-12b-it-qat-q4_0-unquantized
source_version_note: HF QAT fixed commit, deployment Kaggle 12B CT version 2
is_fallback_reference: false
local_files_only: true
allow_cached_nf4: true
quant_backend: nf4
compute_dtype: bfloat16
device: cuda
attn_implementation: sdpa
target_text_modules: [q_proj, o_proj, gate_proj, up_proj, down_proj]
rank: 8
alpha: 16
dropout: 0.0
bias: none
learning_rate: 0.00002
max_sequence_length: 8192
micro_batch_size: 1
gradient_accumulation_steps: 1
ddp_length_grouped_batches: true
epochs: 8
warmup_ratio: 0.0
max_grad_norm: 1.0
gradient_checkpointing: true
weight_decay: 0.0
loss: selected_assistant_tokens
loss_backend: selected_chunked
loss_chunk_size: 32
kbit_prepare: norms_fp32
protocol_id: p013-behavior-v1
checkpoint_every_n_steps: 18
max_steps: 0
seed: 20261020
```

- 步数：`ceil(34/4)=9` batch/epoch × 8 epochs = **72 optimizer steps**（≈70 步级，对齐 P011 的 70 步量级）。
- checkpoint 间隔：`min(32, max(8, 72//4)) = 18`，强制保留最后一步。
- **新建 LoRA + optimizer，不 resume 任何旧协议 checkpoint**（P012 计划 §4：换数据/协议不得 `--resume` step70；P011 adapter 保持独立参照）。
- 协议一致性（P012 v3 故障教训）：config `protocol_id` 必须 = `p013-behavior-v1`，
  `tool_schema_path` 必须指向包内 `research/012/agent/repair_tools_v1.json`（五工具）。

## 3. 70 步级估时与配额预算

- 实测参照：P011 V2 同 recipe（r8/alpha16、global batch 4、LR 2e-5、长度分组）70 steps，
  4×L4 supervisor 墙钟 **1254.94s（≈20.9min）**。
- 本轮 72 steps、输入 token 约 P011 的 1/3，但按步成本看齐旧分布：估计训练墙钟 **20–35min**，
  加加载/导出按 P011 例留 12min 固定开销，stage 时限沿用 **5400s**、平台 **7200s**。
- L4 按 2 倍墙钟计额度：预计 **0.9–1.6 额度小时**。账号 **`qingyu1232`**（30h 额度；
  P012 于 2026-10-19 已用 2.19h）。推送前必须重拍 `quota-before-push.json` 并核对余量；
  若排队遇到 `Maximum batch GPU session count of 2` 则等待重试，不换账号、不读比赛题。

## 4. 推送命令模板（主智能体执行）

```sh
# 0) 干跑校验（CPU 可做）：协议一致 + 数据门
python3 training/train_sft.py --config training/configs/sft_p013_12b_behavior_ddp.yaml \
  --data research/013/behavior-sft/encoded-train-12b-behavior.jsonl --dry-run

# 1) 构建载体（参照 research/011/training-run-20261018/build_carrier.py 模式；
#    单文件 ~2MB 超过 Kaggle ~1MB 单文件上限，必须用 P012 v4 模式：小 bootstrap + 数据走 dataset 挂载）
#    载体 manifest 须核对：实际编码数据、全量 34 行、同一 p013-behavior-v1 指纹、checkpoint 计划、
#    复用 ccjjllt/p007-12b-cpu-nf4-cache（缓存 manifest e7cde801… 推送前重验）

# 2) 推送（示例名，主智能体按实际 kernel 回执记录）
#    kernel: qingyu1232/p013-12b-behavior-sft-34
#    kaggle kernels push -p <carrier-dir>   # 或既有 dataset+kernel 两步，以实际回执为准

# 3) GPU 内执行（4×L4，比赛关联，不断网外其余同 P011 V2）
torchrun --standalone --nproc-per-node=4 -m training.train_ddp \
  --config training/configs/sft_p013_12b_behavior_ddp.yaml \
  --data <mount>/encoded-train-12b-behavior.jsonl --out <outdir>

# 4) 完成后取回：完整 checkpoint + 独立 adapter + export manifest + metrics.jsonl（逐步 loss/实际 LR/每 rank 长度/峰值显存/耗时）
#    只有 optimizer_steps 达到 72 且导出完成才记 completed；否则保留 checkpoint 与失败日志，不声称完成。
```

## 5. P012 辅助层（引用，不混入本轮）

- `research/012/training-data/encoded-train-12b.jsonl`（256 行，`02d239f1…a877`，已本地核对一致）。
- 本轮不拼接：trainer 拒绝混合协议文件（`p012-repair-v1` ≠ `p013-behavior-v1`）。
- 后续 schedule：stage A 训本 34 行文件；stage B 在同 recipe 下单独训 P012 文件（新 optimizer state）。
  若未来构造单协议混合，以 step 为单位按 `primary:auxiliary = 1:1` 交替 global batch，不做行复制。

## 6. 推送就绪确认

- [x] 数据冻结：34 行可训练，token/长度统计见 §1 与 `dataset-manifest.json`；超长 4 条单列后续
- [x] 协议一致：config 模板 `protocol_id` + 五工具路径与数据指纹对应（干跑待主智能体执行）
- [x] 新建 LoRA + optimizer，不 resume 旧 checkpoint；P011 adapter 保持独立参照
- [x] 70 步级估时与配额预算见 §3；账号 `qingyu1232`
- [x] 大 jsonl 不进 git（ignore 已验证）；`assemble_p013.py` + `dataset-manifest.json` +
      `admitted-source-manifest.json` + 本计划进 git
- [ ] 主智能体：落盘 config → 干跑 → 重拍配额 → 构建载体 → 推送 → 记录 kernel 回执
