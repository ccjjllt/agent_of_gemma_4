# 固定100 train：第39项题面修订

2026-10-28。本目录继承已冻结r2，仅修改索引40 PyPika的一个导入示例：`SYSTEM_TIME`从实际public `pypika`导入，不再从不存在该实例的`pypika.terms`导入。原题两端实际ImportError、正确API的目标AttributeError及修复后两条SQL输出均保留在[真实执行材料](../semantic-audit/025-049/pypika-period-sql-teacher-r1/README.md)。

[selection.json](selection.json) SHA`7a6990034f651b932b8d218c8613df7d8f435925e72ef2624d3cc238ecdb893e`；[questions.jsonl](questions.jsonl) SHA`9d223994b7807b1b4ec71064cfeefe25db4b8be18f3bbe74e705e2aa0e02b9a2`。相对r2，99条task对象与question行字节不动；索引40 task除problem_statement外逐字段相同。相对原v4累计39题面修订，库存100ID、顺序、角色、family、snapshot、来源、生产修复与原oracle均保留。revision.json是生成时冻结记录，不将后来的执行结果回写到它或冒充旧示范。

新版40已真实运行原完整40文件／1061用例oracle：baseline两失败，修复后全通过，严格required IDs不变；submitted.patch与旧teacher逐字相同。唯一准入run的90文件内容/mode及初始prompt与本版本精确绑定。第一次wrong-import失败、第二次budget字符串及源archive差异均保留并排除；第三次独立准确输入的运行才进入新teacher。[独立复核](../semantic-audit/025-049/pypika-period-sql-teacher-r1/independent-review.md)另核对API、原输入及8个实际CT动作。

当前完整数据采用[supervision-r6](supervision-r6/README.md)及[trainer-input-r6](supervision-r6/trainer-input-r6/README.md)：805决策／78,924监督tokens，仍为51题本轮真实重放、49题历史来源。最新仅替换0/39共16行，补齐修前可见API故障及Mistune公开渲染整链，98题789行与r5原字节／顺序不变。全部706工具决策与99最终回答保留，最大上下文仍4649；原S600/F600及所有旧版本／失败证据不覆盖，旧r1错误跨度继续禁止训练。

[curriculum-r7.json](curriculum-r7.json)绑定本selection，根据七项具体题内弱oracle证据暂停未来原奖励RL，另4历史runtime unknown共11项暂停；59题待采为32共同＋27专科，35历史领取和奖励不变。新Addict56原已claimed，故该暂停不再减pending；Hamming81由未claimed队列移出。原r6／60题、r5／62题及其他课程均保留。现有正确SFT仍保留全部100题；实际[evidence-course loader独立消费](manual-curriculum-r7-independent-review.json)通过不代表Linux环境全部可用。四题随后在[R7私有CPU运行时](../cpu-runtime-repair/execution-r7-20261028/README.md)取得新有效0／1，但实际采样接入前未自动更改课程。未用本监督数据执行模型训练；辅助recovery-only保持独立协议。

纯题面修订不必重复固定补丁的CPU验收。沿[实际验证器字段消费链](../semantic-audit/prompt-only-cpu-evidence-r1/README.md)证明problem_statement不进入当前固定补丁验证后，维护代码新增明确的verification兼容派生：必须读取SHA恰好等于旧CPU行task hash的真实旧adapted-task对象，且old／current完整对象除problem_statement外逐字段相同；snapshot、patch、isolation、bytecode及实际依赖profile判别继续保留。严格task／prompt身份仍不同，旧CPU行和奖励不改。当前39份旧原件均存在，打包到模型不可见的provenance目录；[实际消费证据](../semantic-audit/prompt-only-cpu-evidence-r1/statement-compatibility-consumption.json)为61项严格一致、39项仅verification兼容，不是100项新Linux验收或环境等价证明。

当前R3完整任务材料现已[本地冻结](material-local-r1/material-build.json)：归档125,224,607字节，SHA `7bb34f0aa992edb8ee09b89d81db1581e2089597e246abafac57ec9bf6de4468`，包含原100题及39份旧task原件（2,891,406字节），当前题目与来源／修复仍绑定原selection。实际归档逐字读回这些原件，并用打包helper和现有runner的报告选择代码消费“历史100行＋CPU R7四行”，得到62项严格匹配、38项仅verification兼容，四条R7记录原样选中，见[消费观察](material-local-r1/consumption-observation.json)。这只补齐任务材料的本地交付，不是100题的新运行时验收；SFT的805条标签仍由上面的trainer-input-r6提供，不包含在该任务材料归档中。dataset metadata名为`jiyujia/vfarl-current100-r3`，尚未上传、没有远端版本；不能把本地名称当作已存在的挂载来源。
