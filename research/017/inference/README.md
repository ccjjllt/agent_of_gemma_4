# P017 Step-68 同期权重对照载体

本目录的builder只构建运行包，不自行启动GPU、上传或提交；实际授权作业另由执行者启动。复用 P015 的官方依赖 bootstrap、四份冻结 runtime-v2 Agent 文件及原 `attempt_worker.py`；A/B 仅 `agent.yaml` 的官方 `adapter` 字段不同。旧 P015 文件与历史产物不改写。 真实Step-68 manifest及权重现已恢复；D已完成旧12题24 attempts（Base6/12、Step-68 6/12、W1/L1、净0、全部配对有效）。C1已按[冻结16题/7仓](../panel/merged-valid16/selection-final.json)完成32 attempts，单活动Agent，内部14400/平台15000秒；Base14/16、Step-68 9/16，W0/L5、净−5，全部有效。最终决策见[结果与下一步](results-and-next-step.md)。原24–32题是目标规模，不能写成已执行范围。尚未取得实际最终 manifest 时不能生成 carrier；不得用最终摘要或校准 manifest 替代。

`build_carrier.py` 要求真实 P016 Step-68 `training_manifest.json`，按已保存最终摘要的 SHA/大小核对它及两件 PEFT 文件身份。默认 `--adapter-layout nested` 的输入布局为：

```
training_manifest.json
adapter/adapter_config.json
adapter/adapter_model.safetensors
```

若上传数据集的三个文件均在顶层，显式传 `--adapter-layout flat`，即 `training_manifest.json`、`adapter_config.json`、`adapter_model.safetensors` 为兄弟文件。layout写入冻结pin，运行时只读取指定布局，不猜测、不回退；两种布局使用完全相同的SHA核对。

支持 `--adapter-dataset owner/slug` 或 `--adapter-kernel owner/slug`，并显式记录 `--adapter-source-version`。Kaggle metadata 使用来源 slug；版本参数记录上传来源，真正挂载内容由完整 SHA256 强制核对。因此上传后不要用同 slug 的不同权重替换运行输入。载体不会联网从 Drive 下载，也不挂 optimizer/checkpoint。

以下是builder调用格式示例，不是已运行D的逐字命令；实际D冻结在`carrier-d-v2/`，内部5400/平台6000秒，结果见[执行记录](../README.md)：

```
python3 research/017/inference/build_carrier.py \
  --training-manifest /absolute/path/to/training_manifest.json \
  --adapter-dataset jiyujia/ACTUAL_ADAPTER_DATASET \
  --adapter-source-version 1 \
  --kernel-id jiyujia/ACTUAL_KERNEL_SLUG \
  --phase D --workers 4 --wall-seconds 4800 \
  --output-dir /absolute/path/to/new-kernel-dir \
  --manifest-path /absolute/path/to/new-carrier-manifest.json
```

D 默认旧12题及既有 CPU material。材料来源可选互斥的 `--material-kernel` / `--material-dataset`，可用 `--material-source-version` 记录实际版本。D可以迁移到新私有dataset，旧题序、材料SHA/大小不变；顶层文件名从selection的 `material_archive_name` 读取，因此可用opaque `.bin` 避免Kaggle自动解压。C 用 `--phase C --selection <冻结新清单>` 及实际材料来源，默认单 worker；selection 保留 `dev_task_ids`、`train_task_ids`、`material_sha256`、`material_bytes`，可选 `material_archive_name`，并明确 `model_results_seen=false`仅指选定C题；最终清单同时记载`D_first_result_seen=true`，不宣称D/C整体盲评。这只是入口，不代替实际仓库/家族去重、训练排除与 CPU 前后判据证据。`--task-start`、`--task-count` 只截取预冻顺序中的完整题对，保留全清单 SHA 与全局 AB/BA 交替顺序。D/C及不同并发分开解释；C同时换用训练外新题并减少活动Agent，不能把D/C差异全部归因于并发。

服务仍固定官方31B CT v2、TP4、32768上下文、BF16、memory0.90、LoRA开启、一个r8 adapter、`max-num-seqs=4`。`--workers 1` 仅减少活动 Agent，不暗改服务参数。单题仍5分钟/28工具/40 turns/60秒命令，thinking4096；无旧的1024额外探索臂。启动记录真实 `/v1/models` 注册信息，每题仍从 raw wire 核查请求model、响应model和thinking参数；加载失败不回退base。

并发方式作了必要且显式的调整：四worker以至多两对题为一个wave，同时运行每对的A/B；下一wave等本wave全部结束。单worker逐题AB/BA串行。在接纳整wave之前预留全部所需时间：四worker取最大单次worker上限，单worker取两臂上限之和，再加180秒收尾。单次上限至少720秒，且不低于Agent300秒＋该题F2P/P2P各自timeout＋180秒准备余量。不会在仅剩150秒时新开半对；极端故障导致的强杀、无效及缺失仍完整记录。这改变了旧动态补位策略，故不能用旧5/12作同期对照。

`--wall-seconds` 包含服务启动；实际平台上限是它加600秒，根执行者需应用。完整面板能否完成取决于真实启动、Agent及判分时长，不因默认4800秒而保证。按账户实际余额和当次授权选择分片与墙钟；本builder不消费或推断配额。若余额不足接纳下一整wave，summary保留未运行身份及真实原因，不产生额外 blocked/skipped 文件。

每次生成包冻结builder/coordinator/worker/Agent SHA，输出拒绝覆盖。运行输出为 `p017-weight-ab-output.zip`，含实际服务版本、输入路径、schedule、pair admission、逐题raw记录、patch、F2P/P2P与路由结果。summary直接报告W、L、W−L、有效配对数、配对修复率差、共同成功/失败、无效/缺失及P2P未观测数。Agent会话超时可与resolved并存；无P2P不是回归通过。 C16中的3题为F2P-only/null，需要与13题有P2P的结果分列。D的Faker1741按冻结判据计W，但已有严格越界的原题验证输出，语义未完整修复；保留主指标并单列这一缺口。性能用context计费工具数225/220，实际结构化调用232/236另见[语义报告](d-semantic-analysis.md)，不能混算。[运行性能分析](d-runtime-analysis.md)和[语义诊断](d-semantic-analysis.md)已基于D原始结果完成；后续不确定性分析仍从原始结果进行，不为此增加独立测试体系。

[主训练监督证据](training-supervision-evidence.md)已独立确认全部537题普通final监督缺席、末尾来源submit无回包；该终局格式差异与D结束问题相关，但尚未证明因果。验证/恢复链的全量覆盖仍未知，不能将6题种子的0验证推广到主训练，也不能以此擅改当前冻结Agent或权重。

本入口仅接受Step-68。条件性Step-30或后续适配权重须另冻真实物料与同期比较，不能改最终manifest冒充该产物。
