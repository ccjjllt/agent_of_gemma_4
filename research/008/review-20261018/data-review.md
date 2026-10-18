# 外部8仓库与真实训练数据复核

日期：2026-10-18。复核HEAD `be9b27b`，对照 `5a979ef` 冻结方案。本轮只读清单、实现和执行汇总；对6个已有tar.gz仅计算文件字节hash，没有解包或读取源码/题目。未读heldout、预留题、外部选择题正文或gold补丁，未运行测试、模型、GPU、教师或提交，未修改实现。

**结论：8个外部仓库仍是隔离的选择候选，已归档6个固定源码快照、5个目标源码导入成功、0个完成任务裁判区分验证、0次外部模型修复尝试。现有训练分组未发现与8仓库重叠；V6确实只用click一条做参数更新。不能将这批CPU准备算作新增训练数据或跨repo修复收益。**

## 1. 当前可复核的层级

| 层级 | 已有证据 | 不能据此推出 |
| --- | --- | --- |
| 冻结身份 | 8个repo各1个PR；从184条/183PR/36repo的上游resolved池按固定hash选出 | 任意私有仓库随机样本，或本项目183次修复 |
| 完整源码SHA | 本地tarball清单与V5实际执行记录有6个40位commit | 八个完整buggy commit都已可追溯复核 |
| 源码物化 | 6个tar.gz；本次重新算hash，均与本地manifest及V5记录一致 | 已取得完整benchmark任务/验证合同 |
| 导入 | V5有5个来自对应buggy源码树的成功import；feedparser失败 | 任务能运行、错误可复现、参考修复通过 |
| 测试收集 | 3个零用例；2个部分收集有错误；1个导入/收集失败 | 任一任务具有有效PASS/FAIL裁判 |
| 空/参考补丁 | 本次查看范围内没有执行记录，工具实现也没有相应步骤 | 已确认空补丁失败且参考补丁通过 |
| 跨repo模型评测 | 没有；readiness的ready_for_gpu_pairing为空 | 机制或SFT跨repo增益 |

身份与选择依据：[external-pr8-manifest.json:15](../../007/evaluation-design-20261017/external-pr8-manifest.json#L15)、[冻结说明:21](../../007/evaluation-design-20261017/report.md#L21)。后续状态：[external-env-readiness.json:132](../../007/evaluation-design-20261017/external-env-readiness.json#L132)。

## 2. 八个仓库实际准备程度

| repo | 归档的buggy commit | 最新本地V5证据 | 仍缺 |
| --- | --- | --- | --- |
| gpxpy | ec00e7d412aeb199b84671427787d7570bf26448 | 目标源码import成功；collect exit5，0用例 | 任务/实际验证入口及区分性 |
| feedparser | c4d1e1fab6173b9c574a0f91682d889254ccb460 | import缺sgmllib；收集缺responses | 依赖、正确导入与任务验证 |
| pandas | 未在所读清单/执行汇总找到完整SHA | 仅build-system/许可说明，未列入6repo CPU执行 | 来源证据归档、实际环境；不能把未尝试写成已测失败 |
| schema | b61adba2b2a32ca7ee70396e1d339148bfed91fd | 目标源码import成功；collect exit5，0用例 | 任务/实际验证入口及区分性 |
| cookiecutter | c9e241194b47675f2541582a0273bea8c7169835 | import成功；默认addopts缺cov插件；清空addopts后209收集、20错误，缺binaryornot/freezegun | 依赖与任务执行，不是209个通过 |
| conan | 27dfabad9ab1f0c96c8ca5ee0d60a303cbae6bb2 | import成功；301收集、394错误；原因未完整留在尾部 | 实际错误原因及任务合同；不是301个通过 |
| modin | 未在所读清单/执行汇总找到完整SHA | 仅分布式依赖/许可说明，未列入6repo CPU执行 | 来源证据归档、实际环境；“cannot install offline”尚非实测结论 |
| iniconfig | 64990c99dab1368d4908639cee0724a4969ce1dd | V4误导入系统包；V5加入src/后已导入目标源码；仍collect exit5，0用例 | 任务/实际验证入口及区分性 |

6repo来源完整SHA：[tarball-manifest.json:1](../../007/external-pr8-20261018/tarballs/tarball-manifest.json#L1)。V5原始摘要分别从[report.json:6](../../007/external-pr8-20261018/cpu-output-v5/ext-pr8-verify/report.json#L6)、:28、:52、:74、:102、:127开始；V5总45.34秒，V4总44.82秒。两个结果都是Python3.12.13 CPU环境。

READY_CANDIDATE仅表示值得继续准备，当前3个这样的repo全部零用例，**不能合计为3个可评测repo**。这里无需另建“过四个才放行”检查；先补出一个真实、可区分的任务合同比继续扩大import数量更有研究价值。源码镜像里没有现成用例时，递归搜目录也不保证得到SWE-smith原任务的验证合同。

## 3. 需要纠正的证据表述

**A. “8个精确buggy commit均已核实”缺两项持久证据。** [进展文档:31](../../../docs/progress/007-public022-and-transfer.md#L31)如此表述，但冻结清单只给profile短commit/instance身份，readiness没有完整buggy_commit字段，tarball和执行记录仅6repo。不能凭profile短SHA补出pandas/modin的buggy SHA。可能先前确实做过ls-remote，但现有材料未保存对应branch→完整SHA结果；本次结论是“证据未归档”，不是判定来源必错。许可目前同样主要是清单断言，未逐文件重新核验。

**B. sha_ok字段没有执行其名称所暗示的比较。** [verify_ext.py:112](../../007/external-pr8-20261018/kernel/verify_ext.py#L112)明确取消预期hash比较，:117无条件设sha_ok=True，:125只核顶层目录含完整SHA。文件头仍称“verifies SHA256 against embedded manifest”，与实际不一致。静态解析发现gpxpy嵌入digest仅62字符，和真实64位manifest不一致（:34）；其余5个嵌入hash正确。**但本轮手动重算证明6个本地tar、manifest和V5记录实际逐字节hash一致**，所以没有发现当前数据损坏；应修正字段/说明，不能用这个布尔值声称自动核验已完成，也没有证据据此归因下载差异为时间戳。

**C. “收集数量”不是通过数量，环境判断有实测/未尝试两类。** [readiness:134](../../007/evaluation-design-20261017/external-env-readiness.json#L134)中的“209 ok/20 err、301 ok/394 err”实际是collected/errors；建议表述为“209/301已收集，未执行”。pandas/modin的不能离线准备是当前依赖成本判断，不能与feedparser已实际失败混作同一种结果。

**D. V4/V5/V6状态不同步。** 进展文档:32仍停在V4结果、V5已推送；readiness已记录V5完成，又提V6待探查；当前kernel源码已有递归/Conan子目录探查，但目录中只见V4/V5输出。不能把当前源码里将做的V6动作计入已有结果。readiness的ready_for_gpu_pairing=[]与实际证据一致，应该保留这种清楚边界。

现有冻结方案已明确“不称跨仓库泛化”，本轮未发现报告凭外部8repo宣布模型收益。需要收紧的是环境就绪和来源完整度表述，不是撤销已经真实完成的下载或导入工作。

## 4. 训练与选择隔离

1. [外部清单:340](../../007/evaluation-design-20261017/external-pr8-manifest.json#L340)明确8个repo整仓保留选择、allow_parameter_update=false。替补只能因环境/来源原因按固定顺序启用，替补repo也要整仓排除训练；当前replacements_used为空。上游resolved筛选偏差已写明，不能再按本项目成败换题。
2. [P006用途:5](../../006/data-replay-20261013/data-usage.json#L5)：click=train_capacity_only；pydicom=select_diagnostic_only；MarkupSafe两个阶段为candidate且不允许参数更新。正负两个MarkupSafe片段仍来自同一个issue，不是两个独立问题。
3. V6 [training_manifest.json:3716](../../007/12b-qat-cache-and-train-20261017/gpu-output-v6/p007-12b-probe/train-run/training_manifest.json#L3716)记录1行、3368输入/519监督token、SHA `efd43a...00c6`、1个optimizer step；与用途清单click文件一致。[构建实现:74](../../007/12b-qat-cache-and-train-20261017/build_12b_gpu_probe.py#L74)区分train/select，:94仅传click给训练；:179在torch.no_grad下算select loss。本次没有发现pydicom或外部8repo参与梯度。
4. 只按旧split registry的元数据统计：repair-sft-v1训练为Rich5题/15片段，选择为FastAPI4题/12片段；v2训练为Rich4题/11片段，选择仍FastAPI4题/12片段；local-diagnostic训练Rich2题、选择FastAPI3题。它们都启用whole_repo，所有train repo与外部8repo交集为空。不同版本是重叠语料，不能相加。
5. 通用[split_registry.py:72](../../../training/data/split_registry.py#L72)默认whole_repo=False，:93才按整仓连组，且它不会自动读取external-pr8清单。**当前隔离成立，但未来新增外部数据不会自动继承这份保留名单。** 下次实际物化时直接按完整repo排除清单并记录输出来源即可，不需要新增专用check/gate脚本。

选择题的NLL变化或live行为可作为诊断，但pydicom已被反复用于调试，不能再称最终未见确认集。其当前用途保持select，不能回流训练来放大样本数。

## 5. 小SFT究竟缺多少数据

工程上的“能做真实训练”已经用click单步回答，不需要再把同一条反复跑通当新研究。新的短SFT要回答修复收益，数据规模应按独立issue/repo和有效监督统计：

| 资产 | 实际规模 | 本轮可计入多少新的跨repo效果训练数据 |
| --- | --- | --- |
| click真实事务 | 1 issue / 1 repo，3368/519 token，已单步训练 | 当前仅容量用途；未经用途调整不能自动升级为效果训练集 |
| pydicom | 1 select诊断 | 0 |
| MarkupSafe正负阶段 | 1 issue / 1 repo，未分配candidate | 0；不能把局部语义通过当whole issue resolved |
| 外部8repo | 8个选择候选，0个已执行模型修复 | 0，全部继续隔离 |
| 外部来源池 | 184记录/183PR/36repo；余28repo只是候选范围 | 0条已确认完成工具重放/编码的新增训练样本 |
| 历史本地语料 | 39 traces/214片段来自9个已暴露问题 | 不能按214当独立题；v2当前8K编码为0行，v1为3行；16K旧诊断4行来自原窄repo集合 |

历史计数来自 `training/artifacts/repair-sft-v2/manifest.json:252` 和各encoded-train.report.json，仅读取聚合报告/分组，没有读取轨迹内容。旧编码的可用长度与12B CT协议仍须按真实资产处理；单条3368容量成功不能直接外推16K训练容量。

P007的[32个独立issue/8个训练repo](../../../docs/plan/007-public022-and-transfer.md#L63)是准备目标，不是科学上保证有效或必须达到的硬门槛。按当前冻结用途，**新的跨repo效果SFT集合还没有正式分配，距该目标仍需32个合格issue/8个repo；若明确把现有click纳入该集合，至少还需31个新issue和7个新repo**。即便把旧Rich诊断重新整理纳入，也不能使用选择题补数量，且只增加一个repo；不能把183个上游成功PR直接视为缺口已补齐。

最有价值的下一步是从不在选择名单的来源repo中形成一小批完整、真实可执行的修复/恢复事务：固定正确buggy实例、原始许可与教师出处；保留观察和失败历史；只监督有证据的后续动作；按实际CT模板记录长度；区分局部合同成功与完整issue成功。先让这些样本能支撑一次同12B base/adapter的行为和修复比较，再扩大规模。当前主要缺的是这批有效数据与外部任务裁判，已不再只是训练入口或模型加载。

