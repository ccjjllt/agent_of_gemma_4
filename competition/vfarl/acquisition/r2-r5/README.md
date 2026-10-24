# R 训练任务首批来源与机制审阅

最终选入20个公开训练任务、20个独立修复家族，R1–R5各4个。它们是待重新采集的任务候选；本目录没有将来源成功标记转换成本地修复成功或六工具训练正例。20个任务均已取得完整题目、造缺陷补丁、F2P/P2P、镜像身份，并重新将各自故障分支解析为40位commit。

- `mechanism-reviews.json`：逐题触发条件、基线错误、生产符号、因果改动、来源实际观察及0起算消息索引。必须连同每题限制阅读。
- `task-acquisition-manifest.json`：最终20题身份、原始任务文件、SHA256、下载方式及完整故障commit。
- `environment-identities.json`：真实公开git引用读取结果；任务ID中的仓库快照短SHA并非执行故障commit。
- `license-evidence.json`：上游项目固定commit许可证文本来源；不代替文件或依赖许可证审计。
- `curation-decisions.json`：未计入最终20题的具体原因，包含退化变异、因果误分类、来源污染和环境约束。
- `dask-matcher-environment-reserve.json`：保留Dask9573真实官方重放；pandas原required通过，但可选cudf跳过被原matcher去参数后匹配，故resolved=false。原报告、required和matcher不改，最终R5换为Jinja PEP451加载器分派。
- `freezegun-environment-reserve.json`：492/493同修复家族的环境保留说明。独立官方重放中Maya required条目跳过，故不能按resolved计入；没有删减required列表。

完整源文件位于被Git忽略的 `artifacts/tasks/` 和 `artifacts/native/`；`artifacts/selected-tasks.jsonl` 供采集器使用。前者来自固定版本的 [SWE-smith-py train](https://huggingface.co/datasets/SWE-bench/SWE-smith-py/tree/77cab9055d42ab4a5c25c89a8f937096db13558e)，后者是重新核对SHA256、instance_id、traj_id后的 [SWE-smith-trajectories tool](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/tree/08e109b4a59eaeebf80e4675cd125d42e7ac99a4) 原始缓存。物理split是tool，train-candidate是当前用途，二者分开记录；没有读取heldout或旧研究eval正文。

SWE-smith-py的patch制造缺陷，[官方文档](https://swesmith.com/guides/harnesses/)对此有明确约定。它只用于离线身份与机制审阅。原始trajectory.patch列没有使用。不能把bug diff反向拼成未经执行的训练回答。

采集器文件和机制审阅含答案侧信息。策略输入必须从允许字段构建，只接收题目与实际故障工作区；不能将bug patch、因果改动描述、gold/privileged验证信息直接加入动作前上下文。来源原始消息还包含旧工具、创建脚本和修改tests的行为，需要真实六工具重新执行，不可改工具名后直接入池。

R2/R5各包含四种不同因果家族。R2是partial异步实例识别、嵌套callable限定名、wrapper默认值元数据、类函数追踪包装。R5是PEP451与ZIP加载器分派、lazy callback导入与构造契约、可选multipart命名空间与warning策略、import期注解导致的初始化循环。Freezegun staticmethod家族不占这四个R2名额。

审阅特别保留了反例：有的issue把正常构造误写成类型检查、有的普通import实际成功但strict warning失败、有的所谓修复前后hash都稳定。来源teacher的完整最终补丁、已改测试后的通过、安装依赖成功、普通import成功，都不能冒充官方任务验证。

本采集工作没有执行来源命令或启动GPU。独立重放工作器的执行证据存放在其运行目录，由上层任务合并；此处静态审阅中的本地结果为null，避免与来源publisher标签混同。

Jinja替换使用额外核对的固定版本tool原始train缓存，来源manifest及hash随selection记录。仅targeted替换Dask一行，保留其余19题以及上层独立复核补充的Starlette2733限制；build_reviews.py为此前批量生成快照，不能未经合并复核注记重新整体覆盖。
