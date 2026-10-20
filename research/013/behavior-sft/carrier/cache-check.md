# P013 缓存复用核对（CPU 侧已重验，主智能体推送前再拍一次）

复用 `ccjjllt/p007-12b-cpu-nf4-cache`（12B QAT→NF4 冻结基座），不重新量化或下载模型。

## 重验结果（2026-10-20，CPU）

- `carrier/runtime-pins.json` 记录 `nf4_cache_manifest_sha256` =
  `e7cde8010895cf3fc4928701c9346ac442c05be97e34c97478b9d6e97eeff4ca`，
  与 P012 `training-run/runtime-pins.json` 字节一致（`sha256sum` 均为
  `2603f89b…`）。
- 同一值在 P007 CPU 制备记录与 P011 GPU 运行产物中多处一致（全仓 grep 确认，
  含 `research/007/.../summary-report.json` 与 P011 `runtime-pins.json`）。
- `build_carrier.py` 内置断言：pins 值偏离 `e7cde801…` 即拒绝构建。
- GPU 侧 `runner.py` 运行时再验：挂载 `cache-manifest.json` 的 SHA 必须等于
  `runtime-pins.json` 中的 `nf4_cache_manifest_sha256`，否则停机
 （`P007 immutable NF4 cache manifest changed`）。

## 推送前待办（主智能体）

```sh
# 1) 确认远端缓存版本未变（kernel_sources 引用解析后核对 manifest SHA）
# 2) 重拍 quota-before-push.json 并核对 ccjjllt 余量（需覆盖 0.9–1.6 额度小时）
```

tokenizer 指纹 `d0017181…` 由训练侧 `validate_tokenizer_protocol` 在权重加载前
自动校验（encoded tokenizer hash vs 挂载 CT tokenizer），不通过不加载权重；
CT 三文件（tokenizer.json / tokenizer_config.json / config.json）由
`runner.py` 按 `runtime-pins.json` 的 `model_pin` 逐文件验 SHA。
