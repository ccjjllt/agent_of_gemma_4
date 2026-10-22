# 003补充：减少GPU排队的实验组织

> 目录整理（2026-10-22）：本页历史载体路径保留当时记录；当前文件位置和重建入口见 [Notebook 目录导航](../../notebooks/README.md)。

日期：2026-09-29。本计划承接[003主计划](003-budget-and-context.md)与[运行修正](003-runtime-reliability.md)，针对长时间L4排队调整执行组织。联合载体已实现并提交；本页保留设计依据，实际迁移状态见[003进展](../progress/003-budget-and-context.md)。尚无联合运行结果。

## 参考材料的可用部分

已读取Osama Abdelaal的[task anatomy + survival kit](https://www.kaggle.com/code/osama133/gemma-4-dev-agent-task-anatomy-survival-kit)版本3全部44个单元格与运行日志。元数据表明该Notebook使用CPU、关闭网络，没有运行Gemma推理。

可借鉴的部分：在CPU完成任务分布分析、检查挂载、整理日志、准备提交包；固定任务和预算，分阶段解释失败；图检索缺失时回退源码搜索。

“without the GPU queue”只指生成submission.zip不需要申请GPU。它不提供Gemma推理或正式评分绕过GPU队列的方法；此次作者日志中的PAYLOAD为空，ZIP单元格只是示例，不能作为真实提交已验证的证据。

不要直接复制以下细节：

- 递归搜索后取第一个tasks/config/wheel，只适合提示位置；本项目仍需匹配指定来源和指纹，多候选时不能静默取首个。
- 根目录前10个JSON文件的Base64导出，会漏掉本项目嵌套的patch、trace、日志和配置。优先正常下载已保存的输出；应急文本导出必须明确文件清单、大小、完整哈希与分块，不能把全部大文件塞进日志。
- “最多2个排队/运行GPU批次”属于作者经验，不当作所有账号和时间的固定规则；以实际API和错误信息为准。
- 文中的补丁文件/函数统计为近似方法。代码并没有严格筛选所有非测试Python文件，函数名来自diff上下文，字符串匹配可能误判；百分比需要在研究开发集合按明确口径复核。不能据此直接设20行硬限制或认定定位是当前最大瓶颈。

## 当前载体造成的额外等待

目前`003-a0-explore-single-cell.py`、`003-b-explore-single-cell.py`、`003-c-explore-single-cell.py`分别固定一个VARIANT，各自安装依赖、启动和停止模型。若分别提交，需申请三次GPU资源并重复加载模型。

多单元格`003-budget-and-context.ipynb`虽然分开了服务启动与逐题执行，但仍只选择一个VARIANT；manifest中六种ABC顺序目前只是说明，没有进入实际运行循环。因此不能声称三份独立脚本已经完成逐题平衡对照。

已完成A0 smoke的远端日志确认：模型启动344.4秒、逐题评估14.3分钟、ROWS=3且resolved=1。合并会话可减少重复启动，更重要的是减少独立排队次数；它不会缩短平台分配第一次GPU的等待时间。

## 本地、CPU和GPU的分工

| 工作 | 运行位置 | 约束 |
| --- | --- | --- |
| 任务清单、配置、源码与轨迹分析 | 本地M5或CPU Notebook | 不调用模型，不逐题读取保留集答案用于选策略 |
| 冻结Agent补丁、空补丁、参考补丁的独立验证 | 版本记录完整的Linux CPU环境 | 沿用官方verify_task；M5不自动等同Kaggle Linux依赖环境 |
| 制作与核对submission.zip | 本地或CPU Notebook | 以已确定的配置快照打包，保留完整SHA-256；正式评分仍另行使用GPU |
| A0/B/C生成补丁并收集轨迹 | 一次L4会话 | 一次加载模型，顺序执行预先冻结的18个attempt，不并行模型请求 |
| 配对比较、日志分类和报告 | 本地或CPU Notebook | 使用原始产物，保留环境异常和全部执行分母 |

第一阶段仍复用当前evaluate_task的完整生成＋验证流程，不为省CPU时间而调用未经核实的私有函数拆分Phase1/Phase2。移出GPU的首先是独立诊断、数据分析和打包；以后是否拆分流水线作为单独实现问题。

## 需要实现的运行组织

1. 新增`notebooks/003-joint-explore.ipynb`与内容相同的`003-joint-explore-single-cell.py`，保留原单组载体。启动模型前同时冻结a0/b/c三份配置，核对共同预算、采样、模型和工具集合；仅system增量不同。
2. 从`research/003/experiment_manifest.json`读取六题和每题顺序，展开为明确的18行调度表。禁止只打印“ABC轮换”后仍运行单组。
3. 明确映射A→a0、B→b、C→c。六题分别采用ABC、ACB、BAC、BCA、CAB、CBA；保存完整表，含task_id、variant、ordinal、配置指纹、输出目录和attempt_id。
4. 仅创建一个VllmServer和相同模型registry。每个attempt使用所选变体自己的EvalConfig、submission_dir和独立results_dir，并创建新的Agent会话、SwegemmaContext和干净仓库。
5. 只复用模型权重与服务，不复用其他组的工作区、补丁、会话历史、摘要或裁判输出。服务级缓存保持同一设置，记录冷/热状态并按调度表平衡先后，不以“同一个服务”声称缓存完全等价。
6. 每个attempt完成即持久化结果、patch、trace、验证输出与状态。每题三组完成后即可做一块配对分析，不必等全部18次完成才保存。
7. 分别维护各variant/repeat的running_results与summary；不能把三个变体混入同一个累计列表。跨组汇总另存研究分析表。
8. 中断或单题异常保留已有产物；重试使用新实际目录。恢复时按任务、配置指纹和产物完整性决定还缺哪些attempt，不自动重跑已完整完成的任务，也不选择性替换不理想结果。
9. 在可用会话时限内给准备、验证、归档和服务清理留余量。18×5分钟只是Agent预算合计，不是总墙钟上界。因时间不足未运行的项明确记录，必要时按预定块另开下一批。
10. 批次结束先归档服务日志和研究产物，再停止本批服务。CLI需要单文件载体时，由同一权威配置/调度表生成一个联合载体，避免长期维护三个复制版本。

该组织只减少资源申请和模型冷启动次数，不新增多Agent或并行推理，也不改变B/C研究问题。

## 已知Requests诊断如何处理

003进展记录显示，`requests_6592`的A0补丁和参考补丁在300秒验证设置下都出现185个httpbin fixture错误。这说明该任务在记录的subprocess环境中暂不适合用resolved判断补丁能力，不证明Requests全部任务都相同。

本轮仍保留六题的原始执行与裁判结果，另列“可有效验证任务”分析；环境异常的具体任务、依据和分母必须公开记录，对所有变体使用同一规则。不能根据某候选失败后临时删题。`requests_7315`等其他任务需要各自证据。

如果采用新的依赖环境或改变timeout_seconds，应另立环境版本，所有比较组一致；后者还影响Agent命令，不仅改变裁判。已有旧结果作为历史资料保留，不与新环境直接合并宣称增益。

## 当前已排队任务与后续结果

用户已取消仍在排队的`003-a0-explore`与`003-b-explore`；接口确认两者为`CANCEL_ACKNOWLEDGED`，原版本与源码备份保留。`003-c-explore`没有已提交运行。本轮完整六题×三组改由[003 Joint Explore A0 B C](https://www.kaggle.com/code/ccjjllt/003-joint-explore-a0-b-c) v2执行，2026-09-29 12:01（北京时间）核对为QUEUED。已完成A0 smoke仍单独保留，不混入联合配对样本。

等待GPU期间优先导出并分析已产生的A0轨迹：定位到首次生产修改的过程、无效短循环、get_status使用、提交后编辑和临时文件进入diff。作者的数据统计只能辅助提出假设，下一步仍以本项目这些真实行为证据决定。

实现与运行情况统一回填`docs/progress/003-budget-and-context.md`，记录本批资源申请次数、服务启动次数、实际执行表、跳过/重试依据及配对结果。联合调度已通过静态源码与官方接口核对；实际服务启动次数和18次运行产物待远端完成后核验。
