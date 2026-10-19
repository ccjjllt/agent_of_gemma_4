# P012 训练运行占位（未启动 GPU）

日期：2026-10-19。对应交接文档 §4.3 与 §7。

- 新协议 `p012-repair-v1`、五工具 `91ec82b1…`、256 先行（含 32+ 行为）已冻结计划，
  见 `../training-data/plan.md`。新编码、连续 SFT、独立导出尚未启动。
- P011 参照：139 题、70/70 更新、监督 token 加权 epoch loss 0.232977→0.135908、
  导出 adapter `71687cf9…`（C 配置固定），step34 `f16a8940…` 补充。
- 下一 GPU 作业需：新建 LoRA 与 optimizer（P011 保持独立参照）、固定数据版本、
  保存逐步指标/完整 checkpoint/独立 adapter、D 预先固定最终 checkpoint。
- 执行前重读配额与四卡 L4 确认；本目录待写入真实 supervisor/训练/导出原件后更新。

## GPU 推送状态（2026-10-19 16:4x UTC+8 → UTC 待确认回执）

- 载体已就绪：256 issues / 23 repos / 128 steps，`encoded-train-12b.jsonl`
  SHA `02d239f1…a877`，五工具 `91ec82b1…`，tokenizer/CT 指纹已冻结。
- 推送发现 Kaggle 单文件 ~1MB 上限（900KB 通过 / 1100KB 拒收，已探针确认），
  载体改为 `train.py`（917B）+ `payload_part1/2.py`（各 692KB）多文件推送，
  重组 SHA `8443d155…` 已本地核对一致。
- 当前阻塞：共享账号 GPU 会话占满（2/2，`Maximum batch GPU session count of 2 reached`），
  非本包问题；配额余量约 18.9h。待会话释放后重试推送，届时记录 kernel URL/回执。
  推送前配额原件：`quota-before-push.json`（used 11.10h）。

## GPU 推送（2026-10-19，qingyu1232 账号）

- ccjjllt 会话占满（2/2）且余量被共享消耗（9.89→11.10h），用户授权 teammateb
 （qingyu1232，全新 30h）执行 P012 GPU。已核对：比赛已参加、可读 wheelhouse
  与 `ccjjllt/p007-12b-cpu-nf4-cache`、L4 走比赛关联。
- Kernel `qingyu1232/p012-12b-repair-sft-256` v1 已推送（256 issues / 128 steps，
  载体 zip SHA `59485d37…`，重组核对一致），状态 QUEUED，待分配后跟进。
  推送前配额原件：`quota-before-push.json`（ccjjllt used 11.10h）。

## SFT v3 失败根因 + v4 修复（2026-10-19）

- v3 在 DDP 启动后 ~85s 内失败：`training/train_ddp.py:206`
  `TrainingError('Data protocol differs from config')`。四个 rank 同错，非环境抖动。
- 根因：carrier 的 `training_stage.py` 把 `tool_schema_path` 写死为六工具
  `training/configs/harness_tools.json`（`protocol_id: p005-harness-tools-v1`），
  而 P012 数据是新的五工具 repair-role 协议（`protocol_id: p012-repair-v1`，
  `research/012/agent/repair_tools_v1.json`，tools_hash `91ec82…` 与数据一致）。
  这是按设计的新协议与旧配置不匹配，不是数据污染。
- 修复（`training_stage.py`）：`tool_schema_path` 指向 bundle 内的
  `research/012/agent/repair_tools_v1.json`（已在 carrier-manifest 中），并增补
  `'protocol_id': 'p012-repair-v1'`。本地复刻两道检查通过：
  data.protocol_id == cfg.protocol_id；sha256_json(5 tools) == 数据 tools_hash。
- v4 已推送（carrier `2695893e…`，单文件 bootstrap ~82KB + 数据走 dataset 挂载），
  状态 RUNNING。配额：qingyu1232 共 30h，当前已用 2.19h。
