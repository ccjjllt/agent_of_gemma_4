# 003 进展：运行修正实现与 A0 基线首跑

日期：2026-09-29。状态：R01–R19 代码修正已实现并经一次 A0 smoke 验证；A0/B/C 联合探索载体已提交排队，尚无联合对照结果。

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

## 7. 下一步（待执行）

B/C探索：六题（3 smoke + `fastapi_15280/rich_3905/requests_7315`）×三变体。原单组载体仍保留，新增联合载体实现逐题ABC轮换并已提交，详见第8节。总耗时应包含准备和验证，不能把2.5小时作为已验证上界。

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
