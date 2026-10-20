# P013 SFT 数据 dataset 打包说明（`SFT_OWNER/p013-behavior-data-34`，主智能体执行）

`encoded-train-12b-behavior.jsonl` 为 2,040,479 字节（~2MB），超过 Kaggle 单文件
~1MB 推送上限，不随 kernel 代码推送。采用 P012 v4 模式：小 bootstrap +
数据走 dataset 挂载，运行时按 SHA 验一致。

## Dataset 内容（3 个文件，全部小文件除外 jsonl 本体）

| 文件 | SHA256 | 说明 |
|---|---|---|
| `encoded-train-12b-behavior.jsonl` | `f0354c13c62fb60bd8e8b650d580887489943d317db49cf25ee22fd4c2c27ccd` | 34 行 / 34 独立 issue / 20 仓库，protocol `p013-behavior-v1` |
| `dataset-manifest.json` | 见 `research/013/behavior-sft/dataset-manifest.json`（`source_manifest.sha256=9935ff22…`） | 冻结清单：行数、指纹、长度分布、超长 4 条单列 |
| `p013-data-marker.json` | 创建见下 | 供 GPU 侧 `runner.py` 定位挂载目录 |

`p013-data-marker.json` 内容模板（主智能体按实际回执填写 slug 后落盘进 dataset）：

```json
{
  "dataset_slug": "SFT_OWNER/p013-behavior-data-34",
  "dataset_manifest_sha256": "<dataset-manifest.json 的 SHA256>",
  "encoded_data_sha256": "f0354c13c62fb60bd8e8b650d580887489943d317db49cf25ee22fd4c2c27ccd",
  "protocol_id": "p013-behavior-v1",
  "rows": 34
}
```

## 打包命令（示例，主智能体按实际账号替换 `SFT_OWNER`）

```sh
mkdir -p /tmp/p013-behavior-data-34
cp research/013/behavior-sft/encoded-train-12b-behavior.jsonl /tmp/p013-behavior-data-34/
cp research/013/behavior-sft/dataset-manifest.json /tmp/p013-behavior-data-34/
# 按模板写入 /tmp/p013-behavior-data-34/p013-data-marker.json
kaggle datasets create -p /tmp/p013-behavior-data-34 \
  --slug p013-behavior-data-34 -t "P013 12B behavior SFT 34 (p013-behavior-v1)"
```

## GPU 侧校验（`runner.py` 自动执行，主智能体只看日志）

1. 按 marker `p013-data-marker.json` 定位挂载目录（`resolve_input` 要求全 `/kaggle/input` 唯一）。
2. `load_frozen_data`：manifest `status=split`、encoded SHA、34 行全 train、无截断、
   `protocol_id=p013-behavior-v1`、provenance 四指纹一致。
3. 与 `run-manifest.json` 核对 `dataset_manifest_sha256` + `planned_optimizer_steps=72`，
   不一致即停（`Build/runtime training step plans differ`）。

约束：P012 的 256 行文件（`protocol_id=p012-repair-v1`）不进本 dataset；
trainer 拒绝混合协议文件。大 jsonl 不进 git（`.gitignore` 已覆盖 `*.jsonl`）。
