# P017 C：16题有效材料，7仓

这里是三个完成CPU阶段中真实通过原前后判据的 **16题**，不是原定24题全部成功。C1 已使用本页最终材料完成官方 CT v2 Base／Step-68 同期对照：2026-10-24 15:01:04 BJT 提交 `jiyujia/p017-step68-confirm16-c1-v1` v1，17:40:47 保存的最终官方状态为 COMPLETE。实际4×NVIDIA L4、单worker、禁网，内部14400／平台15000秒；coordinator实际9180.484秒（约2小时33分）。32次尝试全部返回且16对有效，Base14/16、Step-68 9/16，W0/L5，净减5题。完整原始结果与配额见[回收记录](../../inference/evidence-c1-v1/recovery-receipt.json)及[原始summary](../../inference/evidence-c1-v1/weight-ab-c-20261024T070519Z-05d256b1/summary.json)。

**只使用以下最终文件。** 较早的 `p017-confirmation-valid16-material.bin`、`selection-draft.json` 和 `material/` 含完整host审计记录，仅留本地，不得上传为GPU输入。

| 文件 | 大小/SHA256 |
|---|---|
| `p017-confirmation-valid16-gpu-material.bin` | 187,408,219 bytes；`7abe6fac3d9f9a3981fac339d5b37ca8ac9016c8c4c28cdcf661116fa083feb2` |
| `selection-final.json` | `d7c253b3300dcd5f1983368945f45cf1a866ed75f4e1ea4f0d13c9ae62aafaf1` |

selection兼容当前 `research/017/inference/build_carrier.py --phase C`：包含 `dev_task_ids`、空 `train_task_ids`、材料名/大小/SHA。其 `model_results_seen=false` **是冻结时只针对选定C题的历史事实**，同时记录 `no_C_model_results_seen=true`、`D_first_result_seen=true` 和选择未基于D输赢；不宣称全局盲评，也不代表现在仍未取得C结果。冻结selection及上述SHA保持原样。

| 仓库 | 有效题数 | 来源 |
|---|---:|---|
| networkx/networkx | 3 | 首轮CPU |
| fatal1ty/mashumaro | 3 | 首轮CPU |
| reata/sqllineage | 3 | 首轮CPU |
| rdflib/rdflib | 2 | 首轮1456；CPU4候补1117 |
| hynek/structlog | 2 | 首轮198；CPU4原题586 |
| facelessuser/soupsieve | 2 | CPU3原题119；CPU4原题263 |
| hgrecco/pint | 1 | CPU3原题2089，使用真实collection核准的node mapping |

13题有实际P2P两态通过计数；`networkx6908`、`rdflib1456`、`rdflib1117` 来源无P2P，报告F2P-only/null。所有16题reference F2P都有真实passed计数，没有以skip冒充通过。`host-criteria.json` 保留各题成功轮次的实际环境：soupsieve119、structlog586、soupsieve263为精确源码导入，其他题维持实际成功的安装方式。

GPU包包含16份原issue说明/buggy快照、判分必需的来源criterion test patch/host-criteria、各题依赖与199个wheel，以及**精简**CPU结果/来源哈希。成员核对确认没有 `host-source/`、`reference-patches/`、原始 `cpu-evidence/`；JSON provenance无 `tail/stdout/stderr` 或参考执行trace。sqllineage19的原问题描述本身带Traceback，属于原输入；16份problem_statement均逐一与冻结来源精确核对，没有混入参考运行文本。

本地host审计保留全部3轮原始CPU JSON、完整日志、40次实际尝试、33个不同候选、17个未有效身份和9条固定替补关系：

- `material/cpu-evidence/`：合并前的完整审计副本与 `candidate-history.json`。
- `../cpu-preparation/downloaded-output/`、`../cpu-preparation/cpu3-output/`、`../cpu-preparation/cpu4-output/`：各轮原始产物与日志。
- `../confirmation-identity-freeze.json`、`../replacement-identity-freeze-v2.json`：原24槽位与固定同仓候补关系，均保留不改。

最终GPU包的 `source-provenance.json`（本地可直接读 [gpu-provenance.json](gpu-provenance.json)）仅带逐题执行出处、源JSON/材料哈希、本地审计路径、布尔/exitcode/pass计数和替补关系摘要。原11有效、CPU3新增2、CPU4新增3分别绑定原执行，绝不算作本轮重新执行。

原24题仍有8个未接纳槽位，jsonargparse仓没有有效题；采用16题意味着确认范围缩小，并保留上游教师resolved筛选与环境可复现筛选偏差。不得将这些环境失败计作模型修复失败，也不把16题结果外推成完整比赛分布。

C1 保留全16题主结果，同时按**冻结selection中的既有覆盖字段**分列，未根据模型结果改变子集：

| 范围 | 有效配对 | Base / Step-68 | W / L | 共同成功 / 共同失败 |
|---|---:|---:|---:|---:|
| 全16题主结果 | 16 | 14 / 9 | 0 / 5 | 9 / 2 |
| P2P覆盖题 | 13 | 12 / 9 | 0 / 3 | 9 / 1 |
| F2P-only题 | 3 | 2 / 0 | 0 / 2 | 0 / 1 |

全部32次尝试的判据、实际源码来源、请求/响应权重路由均有效；F2P-only的P2P仍为null，不能称回归通过。Base有2次session timeout；Step-68有11次session timeout和1次40-turn预算耗尽。其中Base1次、Step-68 5次超时后的回收补丁修复通过，超时和修复结果分别保留。这里依据原始 `analysis-rows.json.agent_error`，不能将summary中外层 `base_error/adapter_error=null` 误读为Agent没有预算错误。

唯一结果ZIP为 `../../inference/evidence-c1-v1/p017-weight-ab-output.zip`，13,912,723 bytes，SHA256 `4732a49c99dee59ccd515e7b706ab204d7d01d89952d102fd76c858d8d2e8bb4`；798个成员CRC完整且全部解包保留。C1前后账户GPU余额28.60h→23.47h，按CLI四舍五入读数减少约5.13h，低于本次平台上限对应的8.33h配额预算；这不是精确的逐作业计费数。C1与四worker的D分开报告。

**C题、解答和轨迹只用于评估，不得回流训练。** worker中的 `training_trace_usable` 仅描述采集在技术上是否完整，不构成训练授权。
