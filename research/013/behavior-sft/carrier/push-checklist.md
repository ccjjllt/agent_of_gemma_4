# P013 首轮 12B 行为 SFT 推送清单（主智能体执行，不读 heldout）

账号：`ccjjllt`（余 7.22h，需覆盖 0.9–1.6h SFT + 2h 配对；qingyu1232 正跑预验证，
本轮不用 qingyu1232）。载体目录：`research/013/behavior-sft/carrier/`。

## 0) 前置（已完成，可复核）

- [x] `training/configs/sft_p013_12b_behavior_ddp.yaml` 已落盘（§2 模板；
  唯一修正：`model_revision` 取钉选值 `…445a684ca2`，计划模板 `…445a2c684` 为笔误）
- [x] CPU 干跑通过：协议 `p013-behavior-v1` 一致，数据门通过
  （34 行 / sha `f0354c13…` / token 152711 / 监督 69587 与 manifest 一致）
- [x] 载体已构建：`carrier_sha256=d49e9612…`，72 steps，bundle 61777B（<1MB 上限）
- [x] 缓存重验：`e7cde801…`（见 `cache-check.md`）

## 1) 建 SFT 数据 dataset

按 `carrier/dataset-packaging.md` 打包 3 文件并创建私有 dataset：

```sh
kaggle datasets create -p /tmp/p013-behavior-data-34 \
  --slug p013-behavior-data-34 -t "P013 12B behavior SFT 34 (p013-behavior-v1)"
```

记录回执 slug，用实际 `ccjjllt/p013-behavior-data-34` 替换载体中两处
`SFT_OWNER` 占位后**重跑** `build_carrier.py`（`--dataset-slug` /
`--kernel-id` 实参），使 `run-manifest.json` 与 `kernel-metadata.json`
记录真实 slug：

```sh
python3 research/013/behavior-sft/carrier/build_carrier.py \
  --kernel-id ccjjllt/p013-12b-behavior-sft-34 \
  --dataset-slug ccjjllt/p013-behavior-data-34
```

## 2) 推 kernel

```sh
# 推送前重拍配额
# (按既有方式记录 quota-before-push.json，确认 ccjjllt 余量 ≥ 3h)
kaggle kernels push -p research/013/behavior-sft/carrier/kernel
```

kernel 引用（见 `kernel/kernel-metadata.json`）：private、GPU、比赛关联
`gemma-4-developer-agent`、模型 `google/gemma-4/Other/gemma-4-12b-it-qat-w4a16-ct/2`、
`kernel_sources: ccjjllt/p007-12b-cpu-nf4-cache`、dataset 为上一步的
`ccjjllt/p013-behavior-data-34`。记录 kernel URL 与回执。

## 3) GPU 内执行（载体自动，主智能体只跟进）

`torchrun --standalone --nproc-per-node=4 -m training.train_ddp
--config <bundle>/training/configs/sft_p013_12b_behavior_ddp.yaml
--data <mount>/encoded-train-12b-behavior.jsonl --out <outdir>`；
新建 LoRA + optimizer，**不 `--resume`** 任何旧 checkpoint。
注意 P012 v3 教训：`tool_schema_path` 必须指向包内五工具
`research/012/agent/repair_tools_v1.json`（载体已固定），
`protocol_id` 必须为 `p013-behavior-v1`。

## 4) 取回 adapter + metrics（72 步完成判定）

- 取回：完整 checkpoint（含 `training_state.pt` + 4 rank rng）+ 独立 adapter +
  `export-manifest.json` + `metrics.jsonl`（逐步 loss / 实际 LR / 每 rank 长度 /
  峰值显存 / 耗时）。
- 完成判定：**`optimizer_steps == 72` 且导出完成** 才记 completed；
  否则保留 checkpoint 与失败日志，不声称完成。
- P012 256 行文件不混入；超长 4 条（`layers.p013_behavior_oversize_deferred`）
  留待后续长上下文训练。
