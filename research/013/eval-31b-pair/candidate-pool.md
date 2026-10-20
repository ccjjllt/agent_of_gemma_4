# P013 新开发 12 题候选池（CPU 元数据冻结，执行待 GPU）

日期：2026-10-20。状态：**CPU 冻结完成，GPU 预验证未执行**。本文件只含元数据统计与 ID 冻结顺序；
`dev12-freeze.json`（修前失败 / 参考通过 log 指针）待 GPU 预验证后另行生成，本轮不编造。

## 1. 方法与纪律（元数据只读）

- 只投影 `instance_id / resolved / model / traj_id` 四列 + 行号；**从未读取 `messages / patch`，
  从未读取任何问题正文、参考补丁、test_patch**。
- `repo` 由 `instance_id` 前缀派生（`__`→`/`，小写归一，`pdfminer/pdfminer(.six)` 统一），属标识符元数据。
- heldout / 旧预留题**只计数、不列 ID**：heldout 28（`research/task_splits.json`：129 记录 =
  dev_panel 24 + dev_future 76 + heldout 28 + explore_singleton 1）；旧预留 8（P007 报告计数）。
  本文件不复现任何 heldout / 预留题 ID。
- 一次性手动命令核验，未建测试文件，未推送 GPU。
- `tasks.jsonl` 不在库内（比赛数据），本轮计数以 `research/task_splits.json`（来源 SHA
  `e4b3fd60…3f8c8ad`）为准；不从该文件提取任务内容。

## 2. 数据源与总量

- 候选源：SWE-smith 轨迹分片 8 文件（`research/011/data-scaling-20261018/shards/tool-0000{0,1,2,3,5,6,7}-of-00008.parquet`
  + `training/artifacts/external/swe-smith/tool-00004-of-00008.parquet`，来源 revision `08e109b4`，P012 先例一致）。
- 总量：24100 行；`resolved=true` 9427 行；其中真实 PR 身份（含 `.pr_`）3302 行，
  覆盖 82 个仓库（行级计数，非 82 次修复）。

## 3. 仓库级排除与分布

排除（按仓，整仓剔除；行级计数）：

| 排除类 | 依据 | 剔除行数 | 备注 |
|---|---|---|---|
| 71 训练仓 | `research/012/evaluation/train-repos-71.json` | 2457 | P009+P011 并集；`pdfminer/pdfminer` 归一命中 |
| 外部 8 选择仓 | `external-pr8-manifest.json` | 780 | gpxpy / feedparser / pandas / schema / cookiecutter / conan / modin / iniconfig（含 P012 dev12 的 4 仓，按仓排除更稳） |
| 比赛/公共 12 题所用仓 | `public-dev12-manifest.json` + `external-pr8-manifest.json` | 0（本源中无行） | fastapi / requests / rich / httpx；按构造强制排除，ID 不复现 |
| heldout 28 / 旧预留 8 | `task_splits.json` / P007 报告 | —（不同源，按仓已隔离） | 只计数，不列 ID；其 repo（fastapi/requests/rich）已整仓排除 |

排除后剩余：**3 仓 / 65 行 / 16 个不同 issue**（上游 `resolved=true` 为可解性代理标签，P007 已承认其选择偏差）：

| 仓库 | resolved 行数 | 不同 issue 数 | teacher 来源（元数据列） |
|---|---|---|---|
| pallets/click | 11 | 2 | claude-3-5 + claude-3-7 |
| pallets/markupsafe | 17 | 5 | claude-3-5 + claude-3-7 |
| pydicom/pydicom | 37 | 9 | claude-3-5 + claude-3-7 |

## 4. Issue 级排除（内容或梯度已被项目触碰，P012“prior 排除”先例）

| instance_id | 触碰事实 | 处理 |
|---|---|---|
| `pallets__click.fde47b4b.pr_2555` | P006 训练容量样本；P007 V1–V6 各完成一个 AdamW 步（`research/008/review-20261018/training-review.md:9`），**梯度已流过** | 排除 |
| `pallets__markupsafe.620c06c9.pr_469` | P006 候选语义诊断 + 团队已写出语义修复路径（`research/006/data-replay-20261013/data-usage.json`），正文已读，无梯度 | 排除 |
| `pydicom__pydicom.7d361b3d.pr_2079` | P006 select 诊断（`pinned-instance-manifest.json:25`），目标文件已提取，正文已读，无梯度 | 排除 |

排除后可用池：**3 仓 / 13 个 issue**。

## 5. 冻结选择规则（每仓 sha256 排序分层，P012 方法延续）

- 规则：仓内按 `sha256(instance_id)` 升序取前 K 个（K = 配额）；一 issue 一行，不选 teacher traj
  （评测用官方 loader 物化，不消费教师轨迹）；种子字符串 `p013-dev12` 仅用于文档化仓顺序
  （`sha256('p013-dev12'+repo)`：markupsafe `b0cfa6…` < click `d1b8da…` < pydicom `e8acd7…`），不改变仓内哈希序。
- 配额（合计 12，满足 ≥3 仓）：click 1 + markupsafe 4 + pydicom 7；pydicom rank7 为全池唯一浮动替补。
  问题类型按包域划分（元数据级，非正文 bug 分类）：click=CLI 框架 / markupsafe=模板转义 / pydicom=DICOM 影像 IO，
  3 域 ≥ 2 类型要求；正文级 bug 形态多样性 CPU 阶段不可验证，GPU 阶段如实记录、不据此换题。
- 槽位顺序（块序，确定性）：t01=click rank0；t02–t05=markupsafe rank0–3；t06–t12=pydicom rank0–6。

## 6. 冻结 12+1 组合

| 槽位 | 仓库 | instance_id | 仓内哈希 rank / sha256 前 16 | resolved traj 数（3-5/3-7） |
|---|---|---|---|---|
| t01 | pallets/click | `pallets__click.fde47b4b.pr_2604` | rank0 `bd4d6faeaa3721a8` | 2（3-7） |
| t02 | pallets/markupsafe | `pallets__markupsafe.620c06c9.pr_402` | rank0 `3eff07e0ae4bbfb4` | 1（3-7） |
| t03 | pallets/markupsafe | `pallets__markupsafe.620c06c9.pr_418` | rank1 `43929120a8c952e4` | 8（双） |
| t04 | pallets/markupsafe | `pallets__markupsafe.620c06c9.pr_379` | rank2 `514d8660aa8807a0` | 1（3-7） |
| t05 | pallets/markupsafe | `pallets__markupsafe.620c06c9.pr_400` | rank3 `cdc07f0062821eb8` | 1（3-7） |
| t06 | pydicom/pydicom | `pydicom__pydicom.7d361b3d.pr_1938` | rank0 `1799dc4b50293cca` | 3（双） |
| t07 | pydicom/pydicom | `pydicom__pydicom.7d361b3d.pr_2069` | rank1 `1cf4dad182a95bb9` | 3（双） |
| t08 | pydicom/pydicom | `pydicom__pydicom.7d361b3d.pr_1867` | rank2 `44d8358933f9c08b` | 2（3-7） |
| t09 | pydicom/pydicom | `pydicom__pydicom.7d361b3d.pr_2163` | rank3 `4bf520de1fd4eb2a` | 9（双） |
| t10 | pydicom/pydicom | `pydicom__pydicom.7d361b3d.pr_1859` | rank4 `4ebe922ef77c5026` | 1（3-7） |
| t11 | pydicom/pydicom | `pydicom__pydicom.7d361b3d.pr_1995` | rank5 `83fd1a730733751b` | 9（双） |
| t12 | pydicom/pydicom | `pydicom__pydicom.7d361b3d.pr_1987` | rank6 `a11fbe936e43f847` | 4（双） |
| s01（浮动替补） | pydicom/pydicom | `pydicom__pydicom.7d361b3d.pr_1920` | rank7 `d53ea788402e88da` | 2（3-7） |

完整 sha256 见第 5 节方法可复算（`sha256(instance_id)` 十六进制全串，命令已一次执行留痕于本会话）；
行级来源：各 issue 的 resolved traj 分布于 8 分片（以 `tool-00006` 最密），GPU 阶段以官方 loader 按
`instance_id` 物化为准（`instance_id` 内短 commit 是 profile 身份，**不是** buggy commit，P007 教训）。

## 7. 替补规则（含环境成本过高的预先替补顺序）

- R1（同仓优先）：某题预验证失败 → 取同仓下一哈希 rank 未用 issue（当前仅 pydicom s01 可用）。
- R2（跨仓浮动）：同仓耗尽 → 动用 s01（pydicom pr_1920），配额变为 1/3/8 或 1/4/7，仍计 12 题；
  s01 已用后再次失败 → 冻结失败闭合（见 R4），不得静默缩减。
- R3（click 单点）：t01（pr_2604）同仓无替补；若其预验证失败，**12 题冻结即告无效**：
  不得重召 pr_2555（梯度污染），不得仓外加题；唯一出口是如实记录失败原因后重走冻结（新种子 + 新文件）。
- R4（整仓环境不可物化）：任一仓整体失败 → 剩余题 < 12 或仓数 < 3 → 冻结无效，同 R3 出口。
- 永远禁止：按模型成败换题；重召第 4 节 3 个触碰 issue（除非 R3 式重冻结 + 显式污染声明）；
  跳过哈希顺序挑题；列 heldout / 预留题 ID。

## 8. 每题预验证要求（GPU 阶段，逐题，未验证不进入 24 attempts）

- V1 物化：官方 SWE-smith loader 在固定 buggy commit 建环境，记录 snapshot / 构建系统 / 许可证；
  环境/源码原因失败 → 走第 7 节替补并留原因（与模型失败分别保存）。
- V2 修前失败：FAIL_TO_PASS 在 buggy 环境真实失败（log 指针入 `dev12-freeze.json`）。
- V3 参考通过：参考补丁同环境通过（F2P 通过 + P2P 同环境通过，log 指针）。
- V4 门控：V2+V3 均实际跑通才可进入 24 attempts；任一不满足即换题（R1/R2）或冻结无效（R3/R4）。
- 判据区分性（A/B 拉开差距）**不是**预验证门槛：只要求任务本身有效（失败→通过），不按“两臂是否分胜负”筛选。

## 9. 诚实声明

- 本文件只含元数据统计与 ID 冻结顺序；无任何题目正文/补丁/测试内容；无评分、无训练、无 GPU 执行。
- 题 ID 一经冻结即为开发集：后续任何提示、采样、预算、模型选择若参考本集表现做调整，
  必须在报告中明确承认为**适配行为**，12 题净胜数不得换算公榜分数（沿 `plan-12pair.md` 决策规则）。
- 上游 `resolved=true` 只是教师轨迹标签，不是本项目修复成功；P007 已承认其可解性选择偏差，本集同理。

## 10. 主要风险

1. **池深 13 / 槽位 12，余量仅 1**：任一非 pydicom 题预验证失败即触发 R2 配额变更；click t01 失败则冻结整体无效（R3）。
   这是本冻结的最大单点风险，源于训练划分外有 resolved 候选的仓库只剩 3 个（§3）。
2. **环境成本未知**：CPU 阶段未做任何 `ls-remote` / 构建探针（避免触碰正文）；pydicom 9 个 issue 的
   构建与测试耗时、click/markupsafe 的依赖闭合均待 GPU 实测，整仓失败路径已在 R4 预置。
3. ** teacher- resolved 偏差**：全池以 `resolved=true` 筛选，可解性被高估；V2/V3 门控是唯一的真实校准，
   不得用“看起来难/简单”做二次筛选。
4. **域类型≠bug 类型**：≥2 类型在包域层面满足；若 GPU 阶段发现多题同 bug 形态，只记录、不换题。
