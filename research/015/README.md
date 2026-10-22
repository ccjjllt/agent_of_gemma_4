# P015 执行记录

启动日期：2026-10-22，北京时间。用户已要求开始执行 [P015 计划](../../docs/plan/015-31b-posttraining-and-score-handoff.md)。起点 `ce1dc4c`，已有正式最好 0.15；本轮总 GPU 使用按计划最多 14 配额小时，并受实时余额限制。

当前并行任务为：真实训练素材及长度/重放；新训练外开发面板与已有判据；31B 容量诊断与分片训练；根代理负责统一 GPU 排程、运行载体及结果选择。子任务不独立启动 GPU。继续遵守不读 heldout、不将 select/预留题转训练、不新增测试套件的项目约定。

2026-10-22 12:12 左右，逐进程绑定既有凭据，API 核实实际身份后，官方 CLI 的配额读数如下。刷新时间均为北京时间 10 月 24 日 08:00。

| 账号 | 已用 | 剩余 | 本轮安排 |
|---|---:|---:|---|
| zhuating | 20.71h | 9.29h | 主要 GPU 路线；每次推送前再读余额 |
| ccjjllt | 30.10h | 0.00h | CPU 材料工作，不分配 GPU |
| qingyu1232 | 29.63h | 0.37h | 不依赖该小余额完成训练/评测 |

因此，14h 是授权计划上限，当前不能当作已可用配额。先按 zhuating 余额推进，容量路线最多约 2h，保留至少约 4h 给修复对照和部署条件确认，再按实测速度分配训练。账户是共享资源，不据余额差额推算本项目消耗。

初次 SDK 的 `to_dict()` 将 duration 序列化为不含整天的秒数，错误呈现 6h 总额度；CLI 使用 `timedelta.total_seconds()` 后为 30h。原始查询保存在 `status/quota-start-*.json`，正确余额依据 `status/quota-cli-start-*.json`。这次差异没有被当作账号配额政策变化，也没有据此启动额外作业。

用户随后明确要求子代理使用 **GPT-6.1 Sol**。原默认继承未显式设 model，显示为 Astra；已让原子代理保存进度并结束，三个接续任务均显式指定 `gpt-6.1-sol`。全局配置未被改写。

## 已取得的新证据

- 官方 31B CT v2 tokenizer、模板与 config 已通过零 GPU CPU 作业取回，五份文件已核 SHA256，来源见 `tokenizer-cpu/tokenizer-source-manifest.json`；固定基线和工具协议见 `protocol.json`。
- SWE-Lego Real Data 固定 revision 的 32 个 train 来源完整 teacher 轨迹实际为 36,570–114,541 tokens，均超过 8K。21 条可容纳的重新执行交易只有局部编辑/action 证据，已明确限制为容量诊断素材，不能当作完整任务 SFT 成果。见 [数据准备](data-preparation/README.md)。
- 为取得同一提交协议的完整轨迹，另备 xmltodict81、boltons338、boltons298 三个 train 问题；CPU 已确认修前 F2P 失败、P2P 通过，参考修后均通过。Agent 输入为原始真实 Git base；新增来源判据与参考修复均留在 host 裁判侧。
- 新开发材料已准备四仓 12 个真实 PR 问题及预先固定的 faker 三题替补。本机前后验证已完成，Linux Python 3.12 的实际兼容性由独立 CPU 作业确认；不将本机结果直接替代正式运行条件。与三题 train 合并为私有材料载体，18 题保留用途字段。

## 实际作业

| 作业 | 实际状态/意义 | 预算口径 |
|---|---|---|
| `zhuating/p015-31b-tokenizer-cpu/1` | COMPLETE；只复制官方小文件 | 无 GPU |
| `zhuating/p015-attention-capacity/1` | COMPLETE；两臂真实前后向成功 | 平台上限 840s；按 2× 为最多约 0.467 配额小时；日志可观察区间 71.13s，不等于账单 |
| `zhuating/p015-fsdp-capacity/1` | COMPLETE；三次真实优化器更新，末步有效样本长 6,846 tokens；GPU 侧独立导出成功，600 张量/53,043,200 参数；本地完整载荷 SHA 与有限性核对通过 | 平台上限 2400s，按 2× 为最多约 1.333 配额小时 |
| `zhuating/p015-natural-pr-panel-cpu/1` | v1 ERROR：平台递归展开内层快照，未执行判据；v2 COMPLETE；18 题实际前后执行，12 个有效 dev+3train，3个旧 Voluptuous 因收集失败按预序整仓替补为 Faker | 无 GPU |
| `zhuating/p015-train-next-sampling/1` | COMPLETE；8仓各2draw，真实裁判8/16 resolved，覆盖5个独立问题；其中1次resolved仍有session timeout，不作为合格成功训练draw | 平台上限4200s；实际实验1674.53s，按2×估算0.930配额小时，非账单 |

attention 单算子结果：[实际日志摘要](capacity/attention-summary.json)。HF 原路径反向峰值 allocated 为 2,087,360,512 bytes（约 1.944 GiB），grouped 为 1,391,103,488 bytes（约 1.296 GiB），降低约 0.648 GiB；均实际使用 Cutlass efficient attention。输出与 K/V 梯度完全相同，Q 梯度 relative L2 为 1.77e-5。该数字只属于独立算子，不是完整 31B 训练的峰值节省或容量承诺。

FSDP 工程已实现保持 NF4 字节的浮点容器、冻结/可训练参数分开包装和完整独立 LoRA 导出；CPU 双 rank 更新/重载核对不代替 31B GPU 成果。编码 v3 补普通 assistant 的实际结束边界，tool continuation 不强加 EOS；只有明确 train 划分可进入参数更新。容量作业使用真实 train 局部交易，先两步含 4,203-token 样本，再尝试 6,846-token 完整交易；与任务修复收益分开。

私有材料 v1 的真实平台报错证明 Kaggle 递归展开了内层 `snapshots/*.tar.gz`，CPU 尚未运行题目。现将原包完全相同的 81,303,711 bytes（SHA256 `eeebb65c…6ff8cc`）以 opaque `.bin` 上传同一私有 dataset v2；CPU 显式校验整包与 229 个输入文件后解包，避免平台自动展开。题目、划分和原始 Git 快照不变。临时 venv 和工作树放 `/tmp`，输出只保留真实日志、汇总和一个后续 GPU 可用材料归档。

真实 FSDP 作业已确认三次参数更新：有效监督 token 合计 8,649，批次最长序列依次为 4,203 / 2,903 / 6,846，单步 42.62 / 28.07 / 74.73 秒；worker 正常 exit 0，日志区间约 469.44 秒（按 2× 估算约 0.261 配额小时，非精确账单）。末批仅 rank0 有监督，其余 rank 零权重占位，不重复计入样本；不同 batch loss 不作为收敛趋势。此成果只证明真实 train 局部交易容量和训练工程可用，尚无自主完整任务 SFT 修复收益，也无新正式提交。真实输出已确认最终 adapter 共 600 个 F32 张量、53,043,200 参数；逐 key 形状匹配训练原参数和官方架构。最终文件 212,261,592 bytes，SHA256 `f1d7cfe7…3173563`；GPU 保存前有限性核对通过，本地已完整下载并逐张量扫描，600 个张量全有限且均非零，文件 SHA 匹配。训练/导出环峰值 allocated 16.381 GiB、reserved 21.176 GiB（在 reset 后统计，非冷加载峰值）。见 [真实输出摘要](capacity/fsdp-result-summary.json)。CT 挂载、请求路由和修复收益尚未确认。

开发面板已在任何模型结果出现前冻结为四仓各三题（sqlglot/arrow/pyupgrade/faker），与三道 train 分开统计。见 [最终身份与 CPU 选择证据](evaluation/final-selection.json)。GPU 推送前最新官方余额为 8.23h；本次 base+train 采样平台上限 4,200s，即最多约 2.333 配额小时，仍预留至少 4h。另在同一服务挂载容量 adapter 做一次独立 hello 请求以确认 CT 路由；该请求不进入任务成绩。若挂载导致启动失败，记录错误后同作业回退基座服务，后续 A/B 必须匹配实际服务条件。

北京时间 13:03，已按上限4,200s推送 `zhuating/p015-base-and-train-sampling/1`，源码 SHA `156fc5d4…4fecd16`；实际结果待回。最终容量 adapter 本地完整文件核对已完成，见 `capacity/adapter-local-inspection.json`。

首轮同协议基座实际完成（`zhuating/p015-base-and-train-sampling/1` COMPLETE）：15/15 次均有裁判结论；开发12题 **5/12**，独立train采样 **1/3**。总实验1495.95s，其中模型启动413.79s，按2×估算约0.831配额小时（非精确账单）。完整日志见 ignored `inference/output-v1.zip`，输出 SHA `ffc8db47…077fe48`；可追溯摘要见 [真实基线证据](inference/evidence-v1/summary.json)。该开发比例不是官方分数，也不是权重收益。

同服务实际CT诊断已通过：最终capacity adapter文件SHA核实后挂载，单次hello HTTP200，response.model确为`p015_capacity_diag`，耗时4.11s；base修复请求仍走无adapter基座，服务启LoRA/挂载rank8作为下一次A/B必须匹配的条件。见 [路由原始证据](inference/evidence-v1/capacity-adapter-diagnostic.json)。至此已证实31B真实更新→完整独立导出→CT请求路由，但尚无任务SFT修复收益或新正式分数。

新增八仓train材料（仍在原32身份内）已于13:29推送零GPU `zhuating/p015-train-next-cpu/1`。本机Py3.12完整修前后通过，不冒称Linux已通过；输入69,711,476B、SHA `3d7f97a3…cfff240`，来源24F2P/570P2P。后续仅在真实Linux判据有效且实时余额允许时采样。

新增8题Linux CPU实际全部valid_prepost=True（8仓，24F2P/570P2P），包括tldextract离线editable及Dask原参数，见 `data-preparation/train-next-cpu/output/p015-train-next-prevalidation/summary.json`。按7.27h实时余额准备每题两次同基座采样（8独立问题、16draw，不当成16题），平台上限仍最多2.333配额小时，之后保留约4.94h。先按整题正确/完整响应过滤，每题优先更少工具调用的成功draw，不按dev表现选训练题。选择在其模型运行前冻结于 `data-preparation/train-next-selection.json`。

13:38推送的 `zhuating/p015-train-next-sampling/1` 已COMPLETE，8题各2draw、全部为train；推送前余额7.17h，源码SHA `01f9e4ed…6127cf`。16次真实裁判均有效，8次resolved，覆盖graphene1478、oauthlib771、dask9212、marshmallow2715、tldextract258五题。oauthlib第二draw虽判据通过，却有session timeout，且工具数23少于第一draw26；成功训练draw资格先排除该不完整终局，再选更少工具的合格成功draw，不能反过来因工具数少而选它。原始结果/错误保留，终局consumer-close也不改写原episode标志。见 [新采样真实摘要](inference/evidence-train-next/summary.json)。14:08账号共享余额5.72h，不能把7.17到5.72的全部差额归给本实验。

实测成功boltons298原轨迹不能直接整段用于8K训练：完整模板约11.8K且会丢普通final reasoning，v3明确拒绝该final。按真实wire请求构造的完整因果状态，≤6846有10窗口但不含生产edit；≤8192有14窗口，唯一生产edit为8165 tokens，修后验证8286、submit11560均超8K。原件完整保留；这不是整任务SFT。首轮将按真实可见状态做on-policy action RFT，仅当前动作loss、保留全部历史和实执行观察；新repro脚本写/执行不作正向目标，超长窗口整条排除，且每issue须有可训成功生产edit。见 `data-preparation/boltons298-v1-action-rft-manifest.json`。

计时机制证据：12dev原始元数据投影中，6个timeout的1800.10 Agent秒，LLM API区间union占99.7756%–99.7906%，run_command合计2.946秒、其它工具0.115秒；API包含排队/prefill/transport，非纯thinking时间。217个已返回响应观测到66,958 completion tokens，取消请求usage未知；未把字符当token或缺失当0。见 `inference/evidence-v1/timing-analysis.json`。因此保持A/B仅权重不同、4096 thinking预算，同时预备单独低thinking1024机制臂，在全部AB结束后独立执行，不混入权重收益。

通用action筛选入口已用boltons298实际运行：6条（5次必要源码读、1次8165-token生产edit），thought-mask，共246监督tokens；源schema是实际ADK wire的`d1300338…`，与旧容量素材schema分开。先按正确性、无agent_error和完整当前响应选更少工具的成功draw，再保留最多6个合规目标；每题必须有8K内生产edit。见 [筛选规则与入口](data-preparation/on-policy-action-curation.md)。

17个输入draw（新16+旧成功boltons）最终形成3个独立问题、14条动作、1014监督tokens，最长8165；10次必要源码读取、3次生产edit、1次submit。tldextract258和marshmallow2715加入；后者使用原Agent实际compaction后的完整可见请求，不重建旧历史。其余成功题目因生产edit超长整题排除：graphene11029+、oauthlib8351、dask生产shell改13307/13859。没有因长度改选其它draw、截历史或混入dev。见 [最终筛选证据](data-preparation/on-policy-rft-first-v1-manifest.json)。这远小于原计划16–32问题，仅是小规模action RFT。

14:13已推送 `zhuating/p015-31b-task-rft/1`，源码SHA `4081c196…6146564`，数据SHA `106ee30b…efee8d3`。真实31B CT tokenizer、模板与actual tools在CPU核对一致；r8/alpha16、lr2e-5、1epoch、每rank1条/四rank、累积2，预计仅2次更新（16是上限），不承诺中间阶段趋势。平台最多1800s（2×最多1配额小时），worker1500s，推送前余额5.62h，预留至少4h用于对照/部署。8165是否真正更新、导出是否完成仍待结果。A/B尚未启动。
