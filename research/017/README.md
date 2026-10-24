# P017 同环境权重对照与后训练诊断

2026-10-24，北京时间。jiyujia两轮GPU作业均已COMPLETE、完整输出已回收，最终Step-68已在官方CT v2真实加载并逐请求验证。**当前保留Base：D旧12题双方6/12、W1/L1/净0；C1训练外16题Base14/16、Step-68 9/16、W0/L5/净−5。没有新的正式提交或提分。**

[结果与下一步](inference/results-and-next-step.md)汇总全部结果、五个回退、覆盖范围和不确定性。[D原始摘要](inference/evidence-d-v1/weight-ab-d-20261024T061136Z-7cbb50ca/summary.json)、[C1原始摘要](inference/evidence-c1-v1/weight-ab-c-20261024T070519Z-05d256b1/summary.json)。D的Faker1741仅冻结判据新增通过，已有严格越界的原题验证输出；保留主W/L并单列语义缺口。D/C题目与并发同时不同，分别解释。

C1实际为16题/7个本项目训练外仓库、32次尝试，小于原定24–32题；13道有P2P的题为Base12/13、Step-68 9/13、W0/L3，3道F2P-only为Base2/3、Step-68 0/3、W0/L2。全部32次judge/来源/路由有效，无缺失或无效环境排除。Base2次timeout；Step-68 11次timeout及1次40-turns错误。原33候选/40次CPU执行及全部失败留在host侧，评估题与轨迹不回流训练；[最终材料说明](panel/merged-valid16/README.md)。

C1于15:01:04提交，17:39:30读取COMPLETE；单活动Agent，保持冻结runtime-v2，实验墙钟9180.48秒。[真实作业](https://www.kaggle.com/code/jiyujia/p017-step68-confirm16-c1-v1)、[完整回收回执](inference/evidence-c1-v1/recovery-receipt.json)。最终账户GPU已用6.53h、剩23.47h（D差额1.40h，C1差额5.13h）；[最终配额读取](execution/quota-after-c1-jiyujia.json)。两轮已结束，本地日志reader也自然退出。



启动前已核实jiyujia身份，GPU余额当时为 **30.00h**，记录的刷新时间为10月31日08:00北京时间；[账号/余额原始读取](execution/quota-start-jiyujia.json)。现有近期作业已结束，不重复启动它们。Kaggle查询使用该账号已有凭证，未修改全局默认账号，也不在研究文件内保存凭证。

已完成的D阶段固定P015 runtime-v2和旧12个dev身份，同一官方31B CT v2服务比较Base／P016 Step-68，共24 attempts，四worker按完整题对分wave执行。只改变adapter选择。计划实验上限5400秒，平台上限6000秒，按当前2倍计额最多约 **3.33配额小时**；实际以作业与账户读取记录为准。C实际以已通过CPU判据的16题/32 attempts执行，范围缩小如实保留；不把整个30h起始余额当作单次作业预算。

## 当前交付

- [完整结果与决策](inference/results-and-next-step.md)、[C1语义诊断](inference/c1-semantic-analysis.md)、[C1运行分析](inference/c1-runtime-analysis.md)：保留Base；五个回退主要发生在定位、有效编辑和反馈利用阶段，不能统归结束监督缺口。

- [D运行性能与长度](inference/d-runtime-analysis.md)、[D工具与语义诊断](inference/d-semantic-analysis.md)：Base与Step-68的来源三工具名调用均为0次；主要可见问题是恢复、重复与结束行为，不能把全部失分归于工具名迁移。实际结构化工具调用为Base 232/Step-68 236，context计费工具数为225/220，两种口径分别保留。Step68已返回completion较少，但更多长pending没有usage；API区间不等于纯解码，D也未证明>16K输入退化。
- [537题结束/验证监督证据](inference/training-supervision-evidence.md)：哈希匹配的producer与全量转换元数据证明普通assistant final监督为0/537，537条末尾来源submit均无回包。这是主训练的独立全量结构事实，不是从6题种子外推；验证/恢复链的全量覆盖仍未知，不能说主训练没有验证。终局协议差异已证实，是否导致D循环仍未建立因果。
- [GPU载体](inference/README.md)：支持真实Step-68 manifest、adapter/material的私有dataset来源、明确输入布局、面板分片、1/4 workers及完整题对接纳。仍复用P015 worker和冻结Agent。
- [D阶段选择](execution/d-selection.json)：原12题与原材料SHA不变，离线挂载文件改为opaque `.bin`，未混入来源包中的train问题。
- [jiyujia私有材料](https://www.kaggle.com/datasets/jiyujia/p017-dev12-material)：v1已上传并ready，81,303,711-byte原材料包；[状态](execution/material-dataset-status.json)、[文件清单](execution/material-dataset-files.json)。
- [C阶段原候选材料](panel/README.md)：最初8个候选训练外仓库、24份任务/原始判据及精确base快照，浅快照共39,115,030 bytes；这是原候选身份，不是24道有效评估题。三轮CPU已完成，最终GPU输入只用[16题/7仓冻结清单](panel/merged-valid16/selection-final.json)，13题有P2P、3题F2P-only/null，保留未覆盖与筛选偏差。
- **首轮CPU准备已完成（历史阶段）**：[jiyujia/p017-confirmation-cpu-preparation](https://www.kaggle.com/code/jiyujia/p017-confirmation-cpu-preparation) v1，已COMPLETE，24/24题已实际执行，目前11题通过前后区分（其中2题F2P-only）、7题有准备异常，其余有判据/依赖失败；本地冻结载体为`panel/cpu-preparation/carrier-v2/`，源码SHA `983afa3617e01827a6c325149967f20f358c4ffcb86b5d827d7367706b9b80a8`。回读源码/挂载元数据一致，GPU/TPU均关闭，内部7200秒、平台上限7500秒；[启动回执](execution/confirmation-cpu-launch.json)。Kaggle按标题规范化了实际slug，没有重复启动请求名下的作业。[实际摘要](panel/cpu-preparation/summary-output/p017-confirmation-preparation/summary.json)、[逐题结果](panel/cpu-preparation/summary-output/p017-confirmation-preparation/prevalidation-results.json)。完整日志及201,086,917-byte材料已取回且SHA吻合；[CPU诊断](panel/cpu-preparation/results-analysis.md)。该轮11题不冒充24题确认集；后续CPU3/CPU4完成后最终为16题。
- **最终adapter已恢复并核对**：用户下载目录中的完整归档为4,843,816,960 bytes，SHA256 `32a4fd3b236a1f708048495bfae47982d854356ad79570a73ce2ad139837e165`；最终adapter/config/完整manifest的三个SHA均与P016最终摘要一致，训练状态为68次更新完成。仅提取这三个部署文件，原备份保留；[恢复回执](artifacts/main-sft-short-v1/recovery-receipt.json)。
- **首发D载体已冻结**：[carrier-d-v2](inference/carrier-d-v2/kernel-metadata.json)、[源码与身份manifest](execution/carrier-d-v2-manifest.json)。实际作业 `jiyujia/p017-step68-dev12-v1` 已COMPLETE，官方CT v2、4×L4、四worker、旧12题24次尝试；内部5400秒/平台6000秒。源码SHA `7c5f18791fc387baf6bba276af0bdb4088792e32c656c7b420831853ff62acd3`。已于14:06:03北京时间成功push为v1，初读状态QUEUED，14:10:40已转RUNNING；[实际启动回执](execution/d-gpu-launch.json)。远端源码SHA、三个dataset、CT v2模型、私有/禁网标志及`machine_shape=NvidiaL4`均已回读匹配，服务已发出`WEIGHT_AB_SERVER`，通过最终文件SHA、服务启动、Base/LoRA注册检查；[真实服务标记](execution/d-gpu-server-observed.json)。后续24次运行的逐请求路由全部通过，12/12配对有效；最终结果见文首。
- **CPU3五题环境修复已完成（历史阶段）**：[jiyujia/p017-confirmation-cpu-repair-v3-d7d6753a](https://www.kaggle.com/code/jiyujia/p017-confirmation-cpu-repair-v3-d7d6753a) v1，于14:11:04北京时间成功push；平台3000秒、内部2700秒，GPU/TPU/Internet全关闭。复用已有wheel，仅重跑5个尚未有效原题，原11有效记录不重复计算；实际已COMPLETE，5题全部执行，新增Pint2089及soupsieve119两题有效，该阶段合并13题/7仓（11有P2P、2为F2P-only；后续CPU4增至16题）。[CPU3真实诊断](panel/cpu-preparation/cpu3-results-analysis.md)；[实际启动回执](execution/confirmation-cpu3-push.json)、[冻结准备说明](panel/cpu-preparation/repair-v3-handoff.md)。
- **CPU4已COMPLETE**：[jiyujia/p017-confirmation-cpu-repair-v4-ac51dacf](https://www.kaggle.com/code/jiyujia/p017-confirmation-cpu-repair-v4-ac51dacf) v1，于14:38:04北京时间提交，14:39:18回读RUNNING、14:44:52回读COMPLETE。11次执行仅含2个原题环境重试及9个在原121池中固定的同仓候补；原13有效题不重跑。复用199个wheel、仅新增isodate0.6.1；GPU/TPU/Internet全false，内部5400/平台5700秒。源码SHA `f70bd541dc77290fd5c17c3a1ac2d018c771e0947a17f8410d8c77adea10f7df` 远端回读一致；[实际提交](execution/confirmation-cpu4-push.json)、[准备说明](panel/cpu-preparation/repair-v4-handoff.md)。候补身份冻结早于D首个模型结果，沿原映射不根据D输赢改选；CPU4新增structlog586、soupsieve263、rdflib1117三道有效题，最终合并16题/7仓，不能声称原24题均有效。最终C选择先于任何C模型结果，但晚于D首批结果；C1已完成，结果见文首。
- **最终权重私有数据集已ready**：[jiyujia/p017-sft-step68-123a5a622dec](https://www.kaggle.com/datasets/jiyujia/p017-sft-step68-123a5a622dec) v1。实际回读`is_private=true`，三个文件共212,446,939 bytes且各自大小吻合；[权限/版本](execution/adapter-content-dataset-readback.json)、[实际文件清单](execution/adapter-content-dataset-files.json)。原名被Kaggle拒绝并返回标题占用；用户已明确批准私有上传，以及仅在jiyujia账号内为解决冲突调整名称。真实SDK创建成功，载体只改变数据集挂载名，权重和Agent不变。

D24次及C1 32次均已完成加载、注册和逐请求路由核对。加载失败保留实际失败，不回退base冒充SFT。新面板只用于评估，参考答案/判据不进入Agent或训练。

按[交接方案](../../docs/plan/017-kaggle-weight-transfer-and-score-handoff.md)保存所有配对结果、回退、超时、P2P覆盖及实际配额。实现核对使用一次性手动命令，未新增测试文件或门控套件。正式提交尚未发生。
