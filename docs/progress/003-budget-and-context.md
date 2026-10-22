# 003 进展：主实验完成，面板无净增；原始轨迹复核

> 目录整理（2026-10-22）：本页历史载体路径保留当时记录；当前文件位置和重建入口见 [Notebook 目录导航](../../notebooks/README.md)。

更新：2026-09-30。状态：A0 smoke、18次联合探索、48次面板及冻结诊断已完成；面板A0/B均3/24，C回退，下一轮优先002。9月30日恢复原始轨迹后的归因更正见文末；实际执行以[新交接](../plan/002-next-stage-handoff.md)为准。下文各阶段“待运行/未知”保留当时记录，不代表当前主实验状态。

## 1. 本次运行事实

- Kernel：`ccjjllt/003-a0-smoke-r-fixed-carrier-common-baseline`（CLI `kernels push`，v1，private），L4 + 比赛/wheelhouse/模型三数据源（拉回元数据核对）。
- 路径：`results/003/run-003-a0-smoke/a0/r1/`；配置执行快照与 `research/003/variants/a0/agent` 四文件逐字节一致，manifest 指纹链（tasks/agent/system）全通。
- 环境：4×L4（各22GiB，tp=4），`swegemma 0.2.7 / adk_submission 0.2.11 / adk-eval-core 0.1.0 / google-adk 1.36.1 / vllm 0.19.1 / torch 2.10.0+cu128`，`tasks.jsonl` 指纹与计划一致，129 题。
- 耗时：装包约1分钟、vLLM 启动 344.4s、`EVAL_WALL_MIN=14.3`；GPU 额度本次约2644s，累计用6576s（刷新2026-10-03T00:00Z）。
- 产物：`task_results.jsonl` ROWS=3（001为0）、`summary.json`、三分patch/test_output/trace/log、`attempt_status`全completed、`run_manifest.json`、服务日志归档+停止、`submission-003-a0.zip`（3036B，6文件，sha256 `12a75450…e77759`）。`results/`等按`.gitignore`不进仓库。

## 2. A0 结果（1/3，与001互换；001仅历史参照）

| 任务 | resolved | exit | agent耗时 | 提交 | 行为摘要 |
| --- | --- | --- | --- | --- | --- |
| fastapi_11194 | **True** | 0 | 300.6s（顶满） | 未显式提交，自动回收 | 复现脚本仅2次（001约25次）；scoped grep+ expanding reads；结尾1次edit |
| requests_6592 | False | **124** | 173s | 已提交 | 清理命令在提交后执行；验证超时依旧 |
| rich_2725 | False | 1 | 300.0s（顶满） | 未提交 | `repro.py`×4、`ls rich/box.py`×2、同文件步进重读、零生产修改 |

- `get_status`：fastapi 0次、requests 1次、rich 0次；共同观察节奏基本未被遵守。
- Token：单题prompt约26–29万（含压缩）、completion约6–9k。
- 补丁污染（A0指令未被遵守）：fastapi补丁含`repro.py`+`reproduce_issue.py`；requests补丁含`check_codes.py`+`reproduce_issue.py`+新建的`tests/test_status_codes.py`（并自建用例自测）；rich补丁仅`repro.py`。
- R15实例：requests先`submit_patch`后`rm`临时/测试文件，已提交补丁冻结不变。
- R12实例：requests出现一次带反引号的错误read键（`src/requests/__init__.py`, start_line:151`），符合ADK过滤后退回默认范围的预测；其余带范围读取均正常传递（未复现001的范围无视现象）。
- `total_llm_calls`在ctx上取空（TaskResult上有值35/27/28）；`logs/*.log`官方为空文件；`test_outputs/rich` 25KB为超时错误详情。

## 3. R逐项记录（代码位置 / 版本 / 证据 / 未知点）

- R01：001 notebook c-0、003 c-0、单cell §1。标准库+metadata预检、`CLEAN_KERNEL`+模块路径打印、安装后重启要求。运行证据：CLI fresh kernel无预加载警告。未知：脏kernel重跑路径未演练。
- R02：三载体安装段。版本化staging、41 wheel文件名/大小/SHA-256、同distribution多版本断言、执行阶段`importlib.metadata`。运行证据：41包安装成功，版本行与文档指纹一致。未知：官方container解包缓存未动（只读风险）。
- R03：清单段。指纹比对、ID唯一/存在、repo-commit一致、同组不跨集合、scope枚举、冻结顺序；单cell/003探索集顺序已对齐splits。运行证据：`SELECTED=[fastapi,requests,rich]`冻结一致。未知：heldout互斥未在代码内全量校验。
- R04：`results/003/<run_id>/<variant>/<r1>/`、执行目录隔离与祖先检查、先写manifest、attempt状态机。运行证据：目录结构与manifest/快照一致。未知：跨repeat恢复未演练（设计上不覆盖、换新目录）。
- R05：CUDA存在断言、4×L4记录、别名显式比对、模型挂载stat+tokenizer/config存在性、tp说明。运行证据见§1。未知：驱动版本未单独记录（以cuda_build代理）。
- R06：`adapters.adapters`显式dict检查；003三组采样/预算一致性校验；四段内嵌与权威变体逐字节一致（本地+执行快照双验证）。未知：超出官方示例的全配置图校验口径（未发明API）。
- R07：run专属日志归档→停止顺序、启动异常output保存、单cell finally、001/003人工结束格。运行证据：归档60KB日志后停止成功。未知：启动超时分支的薄封装、PID/进程退出观察。
- R08：`running_results.append`→`append_task_result`顺序、`agent_patch_size`+patch文件核对、trace补存、`ContainerSetupError`按异常记、中断保留现场。运行证据：ROWS=3、summary生成、零坏尾行。
- R09：坏尾行保留上报、无数据不造0分汇总。运行证据：`bad_lines=0`。未知：attempt产物清单哈希与恢复跳过（无重试故未触发）。
- R10：include存在+越界检查、symlink禁令、官方声明验证、单变体冻结打包、zip指纹。运行证据：6文件3036B校验通过。
- R11：提示词以注册schema为准、图目录≠工具可用；litellm caveat。缺口：初始prompt+工具schema的运行时取证（需harness内部检查，未实现）。
- R12：提示词要求保存原始参数、重叠重读、不依赖返回end_line。运行证据：范围读取正常；捕获1例错误键名回退。shell备用策略未触发。
- R13：每题新建`SwegemmaContext`（docker/task/graph/embeddings/budget/harness）、外层perf_counter、全量ctx时间戳打印、duration原值保留。运行证据：三题agent耗时首次实测（300.6/173.2/300.0s）。缺口：ctx上`total_llm_calls`字段名不存在（用TaskResult值代替）。
- R14：C提示词载明20s为候选非硬截止、免费工具可被拒绝。本次A0无C行为；余量组成待B/C实测。
- R15：提示词提交后停止、状态与裁判分离。运行证据：requests提交后rm误删无效（已捕获补丁不变）；成功resolved时error清空、失败时携带Agent错误（rich）——已分别保存。
- R16：003 notebook `c-diag`为框架（步骤与接口已写明），未执行。001原补丁仍缺失；现可用本轮A0新补丁另名诊断（待GPU）。
- R17：同上框架；未修改/参考对照未执行，结果不覆盖001。
- R18：sandbox=subprocess记入manifest；ensurepip警告观察延续（未阻止启动≠依赖一致）。缺口：解释器路径/包`__file__`/缓存状态的逐题记录。
- R19：输入边界靠官方harness构造；A0提示词载明答案不入prompt；污染标记与重跑规则已定，未触发。

## 4. 与001的对照结论

001只作历史参照，不作严格配对。A0在共同修正后仍得到1/3，但 resolve 任务互换、fastapi重复动作从约25次降至2次——提示词修正有行为效应，但收尾（两次顶满未提交）与补丁卫生（临时文件/自建测试入diff）未解决，正好是B（无进展识别）与C（软收尾）要对照的量。temperature 0.2随机性+小样本，不宣称显著。

## 6. R16/R17 冻结诊断结果（2026-09-29，两次CPU kernel）

- `run-003-diag-requests`（4臂）：A(A0补丁@60s)=124复现；B(A0补丁@300s)=exit 1（测试跑完，197KB输出）；C(空补丁@60s)=124；D(参考462字@60s)=124。结论：60s下金补丁同样超时，本地验证预算不足成立，但不能到此为止。
- `run-003-diag-requests2`（2臂持久化输出）：B2(A0@300s)与D2(参考@300s)同为“135 passed, 1 skipped, 1 xfailed, **185 errors**”，输出除沙箱临时路径外一致。185个ERROR全是`test_requests.py`既有 Related Cases在setup期报错：`recursive dependency involving fixture 'httpbin'`（repo `conftest.py`的`def httpbin(httpbin)`包装与插件冲突，pytest新版判递归），与补丁无关。
- 源码核对（`swegemma 0.2.7`，指纹与计划一致）：`resolved = test_res.exit_code == 0`（`verification.py:505`）。185个预存ERROR使exit恒为1，金补丁亦然——**requests_6592在此subprocess环境不可解**，A0的124不能判补丁好坏（R17：参考失败则不判模型能力）。
- 60s超时另有独立证据：完整suite约65s+准备，60s对该题验证恒不够（A/C/D臂63/61/61s顶满）。
- 缺口：B容器内pytest/pytest-httpbin确切版本未记录；`test_status_code_425`是否在135 passed内未按名确认（计数一致，名字只在失败/错误摘要出现）。
- 001的requests_6592原件仍缺失，归因保留未知；本轮A0新补丁诊断另名存档，不冒充原件。

## 7. 下一步：24题面板配对（B vs A0）→ 终局判断

B/C探索已完成（见§8）：B为唯一候选，C回退。面板见§9。

## 9. 面板配对结果（2026-09-29，joint-panel 48/48 completed）

- 运行：`run-003-joint-panel-20260929T151450Z-32b18764`，24题×A0/B，AB/BA轮换（A先B先各12、仓库内均衡）。
- 总分：A0 3/24（`requests_6644`、`rich_2725`、`rich_3518`），B 3/24（`fastapi_9425`、`requests_6644`、`rich_3518`）；B对A0为+1（`fastapi_9425`）-1（`rich_2725`），净零。
- `rich_2725`的A0 smoke为0，探索A0/B/C为0/1/1，面板A0/B为1/0；001历史运行记录为1。存在跨运行波动，但不同策略混合结果不能当成同配置方差估计（9月30日据CSV纠正旧序列表述）。
- `requests_6592`两组均为124（已知环境失真，不计入能力比较）；exit=2（用法/收集错误）与-1（5分钟顶满）分布两组相近，属预算/环境类失败。
## 10. 终局判断（2026-09-29，终审后记录）

- 面板48次失败分类（`research/003/analysis_003_panel.csv`，已逐行抽查核对原始产物）：resolved 6、localization 11、edit 8、budget 14、env_verify 9、submit 0。
- B在面板未复现探索优势（3比3，+1/-1），不能宣称B提高解决率；C已回退。B是否采用待定，不随本轮冻结为最终方案。
- 可归因模型能力的19次中localization占11/19且绝对数≥8：下一轮优先002（代码定位与图检索），输入为本轮归档的定位失败案例与完整轨迹。
- budget 14为最大单类，但系5分钟固定预算与环境超时的设计内约束，不作研究目标；env_verify 9（含requests_6592已知失真）不计入能力比较。
- 004暂缓：B行为未获确认，LoRA训练目标需待002给出定位层面结论后再定。
- heldout 28题（26快照组）全程未动，留待最终方案冻结后一次性评估；dev_panel/explore均已暴露，不再称独立保留集。

## 8. 联合探索结果（2026-09-29，v5一次加载跑18次）

- 运行：`run-003-joint-explore-20260929T124817Z-0d571a9b`，18/18 completed，墙钟1.57h（含启动657.8s），4×L4。`input_dirs`证实混合布局（data/wheelhouse扁平、model嵌套）——硬编码任一布局必挂其一。
- 逐题（resolved/exit/calls）：`fastapi_11194` A0 0/1/30、B 1/0/41、C 0/1/38；`requests_6592` 三组0/124；`rich_2725` A0 0/1/39、B 1/0/26、C 1/0/43；`fastapi_15280` 三组0/-1（5分钟顶满，calls 45~49，任务级预算耗尽）；`rich_3905` 三组1/0（B/C仅13/12 calls，效率减半）；`requests_7315` 三组0/1。
- 配对：B对A0为+2/-0（`fastapi_11194`、`rich_2725`翻转，无回退）；C对B为+0/-1（`fastapi_11194`回退）。按计划回退C，保留B为唯一候选。
- 注意：A0在本批`fastapi_11194`=0而smoke批=1（temperature 0.2随机性），小样本不宣称显著；requests_6592的124属已证环境失真，不计入能力比较。

已证实的环境问题范围是`requests_6592`在本次subprocess条件下的httpbin fixture错误，不直接推广到其他Requests任务。所有六题保留原始运行与裁判结果，验证超时/环境错误另列，并以各任务证据定义三组共用的有效验证子集，不能事后删题抬高成功率。

后续组织见[003减少排队交接](../plan/003-queue-efficient-execution.md)：合并GPU资源申请并复用模型服务，独立CPU诊断与本地分析可在等待期间继续。

## 8. 联合载体实现与队列迁移（2026-09-29）

- 新增`notebooks/003-joint-explore-single-cell.py`和同代码的`notebooks/003-joint-explore.ipynb`。完整内嵌三组12份原配置与研究manifest，逐字节及SHA-256核对一致；保留模型、六题、预算60/50/5/80及采样设置。没有新增研究条件。
- 六题顺序依次为`fastapi_11194`/ABC、`requests_6592`/ACB、`rich_2725`/BAC、`fastapi_15280`/BCA、`rich_3905`/CAB、`requests_7315`/CBA；A映射a0。默认18次，顺序调用，同一服务启动一次。已有A0 smoke不替代本批a0结果。
- 每次新建Evaluator、SwegemmaContext和沙箱，独立attempt目录；分组维护官方结果列表与summary。逐次保存patch、trace、验证输出、上下文用量、异常、状态和产物哈希。服务停止前归档日志，包括启动失败内部清理路径；不自动重试。
- 修正联合载体的ctx字段为`llm_calls_used`，调用前检查`evaluate_task`支持context参数，不再用捕获任意TypeError后重新执行的方式兼容接口。
- 提交前仅作静态语法、配置原文及官方包接口核对；未运行本地评测、未新增测试文件。以上为实现事实，尚未证明远端18次全部成功运行。

迁移时原A0/B版本1均为QUEUED且无输出，C查询为`No runs found for this kernel`。用户随后手动取消A0/B，接口确认两者为`CANCEL_ACKNOWLEDGED`；没有取消已完成smoke或CPU诊断。三份原源码及元数据保存到`research/003/queue-migration-20260929/`。

合并Notebook：[003 Joint Explore A0 B C](https://www.kaggle.com/code/ccjjllt/003-joint-explore-a0-b-c)，kernel ID `136342691`。v1为QuickSave：MCP保存源码成功，但回读发现三个输入列表均丢失，因此没有作为实验运行。改用官方SDK完整设置来源并执行SaveAndRunAll，成功生成**v2**。MCP于2026-09-29 12:01（北京时间）回读确认：private、NvidiaL4、enable_gpu=true、enable_internet=false、版本2；比赛、wheelhouse、固定模型v2三个挂载齐全，代码单元与本地一致，状态为QUEUED。

- Notebook SHA-256：`4f469acf05a0564dbdf2ab4e10501c173718b2d06993f415ff13437f4918e2ab`。
- 远端回读代码SHA-256：`45dfd29d4b40dff489db569238dc9dbb04316b87aa1bff2430ce81282a6d41d7`。Kaggle规范化Notebook JSON，整文件不直接作远端逐字节比较。
- 核对凭据：`research/003/queue-migration-20260929/joint-verified.json`；迁移记录见同目录`migration.json`。不含认证信息。
- 当前是一条联合GPU运行申请；模型实际启动次数、完整结果和墙钟耗时待运行后读取。合并减少后续排队与重复启动，不能保证缩短本次平台分配GPU的等待。

分析时以`joint_summary.json`中各组`planned=6`和完整状态表保留全部分母；官方分组summary只含实际返回的结果，不能在发生异常时直接把它当作六题成功率。共享服务可能保留前缀缓存，已记录执行顺序，但不宣称各次缓存完全等价。

## 11. 2026-09-30 原始数据恢复与研究复核

本次未运行新GPU实验。通过Kaggle恢复1006个文件（11,674,365 bytes）、panel48/explore18/smoke3共69条完整轨迹；66份原始artifact_manifest中的660个文件大小和SHA-256核对零不一致。归档位于`results/003/recovered-20260930/`，恢复清单与原始完整性报告保存在该目录；两份完整tasks.jsonl未下载，不读取heldout内容。

主结果仍为A0/B各3/24。按既有计划补列：3个smoke为A0 1/3、B 0/3；其余21题为A0 2/21、B 3/21。面板48份原始result与CSV的resolved、exit、tool_calls、total_llm_calls一致。

历史失败分类只作为当时的研究标注保留，不能视作互斥根因。原始证据带来以下修正：

- `rich_3942-B`：88.84秒因ContextWindowExceededError退出；32768上下文中预留16384输出，输入至少16385。因此不是5分钟用尽。
- `rich_3130-A0`：291.48/299.51秒的两次生产源码编辑尝试均为FileEditError；并无“成功修改未被回收”的证据。CSV的first_source_edit_at在此表示尝试，不是确认成功。
- `fastapi_14873-B`：43次FileReadError，其中33次完全相同参数，filepath混入start_line；应同时记录参数/执行循环，不能只归定位。
- 11次localization来自7题；11次localization与14次budget合计25次无最终生产修改，但原因需继续按多标签分开。
- B显式提交7→11次，但生产修改仍11→11次，成功3→3次。平均Agent时间下降约29秒，配对中位差接近0；净节时约六成来自两道失败题提前结束，不是稳健提效。
- 1437个有记录的Agent LLM步骤最大completion6114，无步达到8192；成功轨迹205步最大1494。这支持16384→8192的单变量容量筛选，不证明未来不会截断或一定提速。
- Requests的185个setup错误细分为181个httpbin和4个httpbin_secure递归；A0与参考补丁输出在归一化临时路径、JUnit随机名、耗时后相同。60秒时限不足和环境错误同时存在，不能由此认证A0补丁正确。

完整方法、源码和结果见[描述性重算与原始轨迹审计](../../research/003/review-20260930/README.md)、[运行接口审查](../../research/003/runtime-review-20260930/README.md)。下一轮转入[更新后的P002交接](../plan/002-next-stage-handoff.md)：先做评估与实际参数校准、输出容量对照，再测可提交的定位与编辑辅助。原始CSV不覆盖；本次没有新的修复成功率。
