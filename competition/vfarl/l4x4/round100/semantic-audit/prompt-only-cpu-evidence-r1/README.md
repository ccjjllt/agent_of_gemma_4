# 纯题面修订与固定补丁CPU证据

2026-10-28。先形成下面的只读源码和原件审阅；随后按主任务授权实施末尾的最小CPU兼容修复。未新增测试文件、重跑100题或改写旧奖励。

当前固定补丁CPU路径没有使用problem_statement。TaskEnvironment.prevalidate只调用env.verify，未启动agent；_verify直接把固定snapshot和patch交给swegemma 0.2.11 verify_task。实际本地官方模块与已有source pin逐字相同，verification.py SHA为99e2a5fdab2be2b5e0388de29080da9d6e7003cf8b53299cda790464f4ecd2aa。

| 消费位置 | 实际使用的输入 |
|---|---|
| Task.__post_init__ | 规范化FAIL_TO_PASS、PASS_TO_PASS |
| Task.test_files | 从test_patch提取目标文件 |
| ExternalTrainTask.test_files | verification_target_files |
| verify_task准备／过滤／运行 | instance_id、repo、base_commit、test_patch、test_files |
| 成功后的JUnit判定 | FAIL_TO_PASS、PASS_TO_PASS、test_patch |
| 容器setup辅助函数 | snapshot、固定agent patch、repo、config、已存在setup/wheels/工作树资源 |

对verify_task全部Name(task)使用的AST分类没有发现其他whole-task传递：要么是上述属性，要么是字面量getattr。构造数据模型会存储problem_statement，但验证器不读取它。TaskEnvironment的manager.start闭包只使用runtime/profile和workspace，不读取题面。EvalConfig及container_setup使用tasks_path的父目录定位setup.py、wheels等，不加载任务JSON正文。因此实际路径、setup/wheels、runtime、隔离策略、验证器版本和预算等仍是独立条件，不能由“题面字段相等”替代。

已读取固定100训练任务的旧material原件与当前R3对象：61个完整对象相同，39个只改变problem_statement，其余所有字段逐项相等。将旧/新对象分别送入真实Task／ExternalTrainTask定义，验证器实际使用的字段、归一化required nodes和test_files也完全相等。这是实际数据模型执行和源码消费证据，不是新的oracle运行或模型收益。

真实最小例是Markdownify54：历史CPU记录old task SHA20b84861…已取得0/1，当前task SHA12cfb3ac…只改题面。现有cpu_evidence_applicability面对当前真实资产仅因task_input_sha256不符而拒绝，snapshot、patch、chroot、legacy字节码均匹配。旧runtime与新R7不同，不能凭这一例宣称它们等价；R6/R7的四题0/1仍是独立新环境证据。

原始task对象有实际来源：39/39的旧adapted-task路径仍存在，真实文件SHA均匹配原任务身份；固定4b379247…material归档中还保留vfarl-input/assets/<id>/task.json。不能拿revision metadata的“preserved”字符串、原始acquisition source blob或字段白名单代替这些原始字节。逐项路径和SHA见[field-consumption-review.json](field-consumption-review.json)。

初始最小实现建议（后续实现见末节）：

1. 在需要纯题面兼容时，提供真实旧adapted-task字节；其SHA必须等于被引用CPU行的task_input_sha256。当前文件继续绑定当前selection SHA。
2. 对两个完整task对象仅移除problem_statement，然后逐字段完整比较；缺任何旧原件、改变任何其他字段都沿原有严格拒绝。无需推断“哪些其他字段可忽略”。
3. 保留snapshot／固定patch／隔离／字节码检查和现有实际依赖profile判别，不改写旧CPU行，不自动声称运行时等价。兼容返回明确记录“新任务题面版本、相同固定oracle输入”，不要宣称新CPU运行或新版策略轨迹。
4. 载体可在覆盖原material task.json前保存其旧原件，或builder从现有原adapted-task路径冻结这个host-only证据；无须新数据集、额外模型前向或为了题面重跑39个oracle。当前R3仅task字段的等价性已有证据，但最终环境消费适用性仍要另判。

这项证明仅覆盖当前已钉住的固定补丁CPU验证路径。agent采样、SFT prompt、历史行为概率、焦点配对和新题面修复率都会受到题面变化影响，继续绑定完整新任务／prompt身份，不能沿用此CPU兼容逻辑。


## 已实施的最小兼容修复

`cpu_evidence_applicability`保留原来的`matches_current_task`事实及原始mismatch字段，另返回`matches_current_verification`。仅当唯一身份差异是task_input_sha256、真实host-only旧task文件SHA等于原CPU行、两个完整任务对象除problem_statement外逐字段相同，才产生明确的statement-only派生记录。当前任务及prompt身份继续不同；旧行、reward、行为tokens和概率没有改变。snapshot、patch、isolation、bytecode任一不同仍拒绝，实际runtime/profile判别仍由原执行路径负责，没有增加全runtime hash门控。

`kaggle/build.py`在现有question revision provenance流程中保存原始adapted-task字节至`material/provenance/cpu-verification-tasks/<sha>.json`。本次真实39项只需39份、2,891,406字节，现有material manifest覆盖其来源。消费时不相信metadata里的preserved字符串或白名单，而是再次读取原件、重算SHA并比较对象。该目录不进入agent的snapshot workspace或prompt；prepare.py／selection无需改写。

resident、外层runner和lower collect使用verification适用性判断CPU证据，并在重复记录中优先直接匹配当前task的记录，其次才是已证明statement-only兼容的旧记录。lower collect原先直接dict覆盖、随后再次硬比较整个task SHA；这两处也已对齐，防止外层接受后又被旧路径拒绝。当前直接的评分无效记录不会被旧兼容正例覆盖。

[一次性实际消费证据](statement-compatibility-consumption.json)记录：五份冻结原CPU报告逐SHA相同，原row/reward未改；实际100条元数据中61条仍strict match、39条为verification-only兼容。缺原件、原件SHA变化、其他task字段变化、snapshot／patch／isolation／bytecode变化均拒绝；真实CPU5 unknown仍保持不可用。实际resident与lower collector选择语句使用历史CPU、CPU5和CPU7真实行作顺序对照，两端结果一致。该命令没有运行模型、GPU或新oracle，也没有证明当前100个runtime全部可用。

最终实现源见该证据的final_source_sha256：run.py `e72e9233…`、resident.py `847819cf…`、runner.py `b2fc1582…`、build.py `611aacfe…`。此前R6/R7和数值GPU冻结载体保持原件；没有宣称它们已经运行这个新兼容逻辑。
