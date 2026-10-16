# E2机制分析：先区分复现判据与生产修复，再解释整体方案差异

日期：2026-10-13。只读本轮三题的12条原始trace、补丁、result、context metrics与裁判输出；没有运行模型、补丁或测试，没有读取其他题。A=冻结exec_closure，B=Budget-Fit V5五分钟参考。官方结果A4/6、B5/6；配对B胜2、A胜1、共同成功3，没有双方共同失败。它仍只有三个已暴露问题，工具、命令预算、输出预留和提示同时不同，不能作单变量因果或跨仓库收益声明。[总结果](results.md)。

**最重要的现象不是B总能更早改代码，而是复现判据是否抓住了题意。** fastapi_5077反复出现“不抛异常/不是字符串”的弱判据，实际结果仍为ForwardRef。成功的两条最终改了原函数globals解析；失败两条分别停留在测试脚本、调试打印。fastapi_14873失败则在题干已明示原因后持续重读和改复现，没有生产修复。技能和图工具在12次里均未被实际调用。

## 全部attempt索引

编号链接直达原trace；下文`sN`为JSON中的`step_id`。首次编辑/submit时刻取工具调用的`extra.elapsed_s`，包括该trace计时起点后的准备时间，不能和独立的`agent_elapsed_seconds`直接相减。表中Agent秒取context metrics；“首次生产编辑”要求工具回包成功，09明确是诊断打印而非修复。已查看run_command动作，没有额外未计入的shell生产编辑；shell写入的是复现文件。

| trace | 任务/重复/臂 | 官方结果 | Agent秒 | 计费工具/LLM调用 | 首次成功生产edit_file：step/秒 | 末次submit：step/秒 | 最终非生产文件 |
|---|---|---|---:|---:|---|---|---|
| [01](evidence-v1/exec_closure/r1/attempts/01-fastapi_14873-r1-exec_closure/traces/trace_fastapi_14873.json) | fastapi_14873 r1 A | 通过 | 300.0 | 27/27 | s15 / 151.3 | 无 | repro.py |
| [02](evidence-v1/budget_fit_5m/r1/attempts/02-fastapi_14873-r1-budget_fit_5m/traces/trace_fastapi_14873.json) | fastapi_14873 r1 B | 通过 | 293.2 | 27/30 | s23 / 190.1 | s31 / 262.1 | 无 |
| [03](evidence-v1/budget_fit_5m/r1/attempts/03-fastapi_5077-r1-budget_fit_5m/traces/trace_fastapi_5077.json) | fastapi_5077 r1 B | 失败 | 300.0 | 24/25 | 无 | 无 | 空补丁 |
| [04](evidence-v1/exec_closure/r1/attempts/04-fastapi_5077-r1-exec_closure/traces/trace_fastapi_5077.json) | fastapi_5077 r1 A | 通过 | 247.8 | 17/19 | s10 / 182.3 | s20 / 237.6 | 无 |
| [05](evidence-v1/exec_closure/r1/attempts/05-rich_3518-r1-exec_closure/traces/trace_rich_3518.json) | rich_3518 r1 A | 通过 | 125.0 | 14/16 | s12 / 87.8 | s17 / 118.7 | 无 |
| [06](evidence-v1/budget_fit_5m/r1/attempts/06-rich_3518-r1-budget_fit_5m/traces/trace_rich_3518.json) | rich_3518 r1 B | 通过 | 86.5 | 16/18 | s13 / 59.3 | s19 / 83.8 | 无 |
| [07](evidence-v1/budget_fit_5m/r2/attempts/07-fastapi_14873-r2-budget_fit_5m/traces/trace_fastapi_14873.json) | fastapi_14873 r2 B | 通过 | 277.5 | 28/30 | s26 / 201.5 | s31 / 271.5 | repro.py |
| [08](evidence-v1/exec_closure/r2/attempts/08-fastapi_14873-r2-exec_closure/traces/trace_fastapi_14873.json) | fastapi_14873 r2 A | 失败 | 300.0 | 40/41 | 无 | 无 | repro.py |
| [09](evidence-v1/exec_closure/r2/attempts/09-fastapi_5077-r2-exec_closure/traces/trace_fastapi_5077.json) | fastapi_5077 r2 A | 失败 | 300.0 | 29/29 | s14 DEBUG / 118.2 | 无 | 无；仅DEBUG |
| [10](evidence-v1/budget_fit_5m/r2/attempts/10-fastapi_5077-r2-budget_fit_5m/traces/trace_fastapi_5077.json) | fastapi_5077 r2 B | 通过 | 297.6 | 28/31 | s19 / 162.7 | s31 / 286.7 | module_a.py、module_b.py、reproduce_issue.py |
| [11](evidence-v1/budget_fit_5m/r2/attempts/11-rich_3518-r2-budget_fit_5m/traces/trace_rich_3518.json) | rich_3518 r2 B | 通过 | 102.5 | 11/13 | s12 / 74.5 | s14 / 98.4 | 无 |
| [12](evidence-v1/exec_closure/r2/attempts/12-rich_3518-r2-exec_closure/traces/trace_rich_3518.json) | rich_3518 r2 A | 通过 | 269.8 | 22/29 | s14 / 122.8 | s30 / 264.6 | 无 |

原件均在每条trace相邻的`result.json`、`context_metrics.json`、`patches/`、`test_outputs/`；第一列也给出精确attempt目录，避免把不同重复混用。

## 三次失败分别卡在哪里

| 失败 | 实际动作与证据 | 结束时状态；不能误称为 |
|---|---|---|
| 03 B / fastapi_5077 r1 | s5、17.9秒已读目标函数；s7缺filepath，s8已纠正。之后生产代码一次未改，全部写/改自建tests。s15、173.1秒自建测试通过；s22、275.8秒打印ForwardRef但`not isinstance(x,str)`仍通过；s23、289.1秒才增加ForwardRef判据，s24首次见正确失败。 | 300秒Agent超时，最终空patch、exit -1，未进入正常补丁验证。不是28调用先耗尽（官方计费24），也不是一个真正修复被裁判否定。 |
| 08 A / fastapi_14873 r2 | s5/6重复畸形`start_line"`，回包仍ok但读文件头；s7纠正。s13起反复改repro。s21错用app.on_startup，s30、223.7秒只输出Inside TestClient、没有startup；s32验证的却是自定义lifespan。s33–44继续读routing/applications。 | 300秒超时、零生产edit，最终618字符全是repro.py；裁判1 failed/9 passed。不是上下文崩溃，也没有已实施的生产修复可供评价。 |
| 09 A / fastapi_5077 r2 | s13、97.1秒打印Successfully created app；s14、118.2秒唯一生产edit只加入两条DEBUG print，没有改变globalns逻辑。s20输出ForwardRef；s25同模块案例globals=True，s28、284.6秒才跨模块证实globals=False。 | 300秒超时，621字符最终patch仅DEBUG打印；裁判1 failed。不能说“零生产修改”，但也不能把这次edit算有效修复。 |

定位证据：[03][03] s5–27、[08][08] s5–44、[09][09] s7–31。两条5077失败都有早期目标文件阅读，继续增加同文件阅读不直接解决问题。03的“测试通过”和09的“成功创建app”没有检验注解是否解析成题意中的真实类型。

## fastapi_14873：第二次分叉与第一次反复修改

第二次B [07][07] s11在44.5秒读到初始化关键区域，也经历错误读参、复现API误用。s25、162.2秒复现只打印Lifespan entered；s26、201.5秒将on_startup/on_shutdown赋值移到super初始化之后，s27、203.7秒随即多出SYNC START。A [08][08] 第二次没有迈出这一步，转去验证自定义lifespan并重新浏览已有区域。两者差异发生在已取得证据之后的行动，不能仅凭工具预算/8192预留解释。

B也并未干净结束：s28改写更大复现，s29再次API错误，s30写新脚本后用满28调用，s31把源码与repro.py一起提交；最终官方10 passed。目标startup回包到submit相隔约67.8秒，中间主要是新的复现而非新的生产修复。

第一次并非A失败/B成功：A [01][01] 在s15、151.3秒已作正确生产修改，后续换lifespan复现并出现API错误，继续读源码至超时，没有显式submit；自动回收仍10 passed，但夹带repro.py。B [02][02] s23、190.1秒修改后，s25、200.7秒观察到同步/异步startup和shutdown完整输出；s26、226.1秒又将赋值移回原位置，s27回包失去START/SHUTDOWN；s28、253.3秒再次恢复正确移动后提交。**B成功中也存在已证修复被主动撤回、再恢复的无效往返**，不能把其整体分数略高描述为全流程更稳。

## fastapi_5077：同一弱复现，各重复却得到相反胜负

- A第一次 [04][04]：s7、65.6秒输出Success同时Signature仍是ForwardRef；s9、147.4秒改为明确比较`param.annotation is ModelA`，输出Failure。s10/11、182.3/190.4秒分别修signature和return annotation，沿`__wrapped__`取原函数globals；s12、192.2秒同一复现返回真实ModelA。随后又grep和回读，到s18清理三个非生产文件、s19 diff check、s20在237.6秒提交，官方1 passed。目标行为回包到submit约45.5秒，部分是必要清理，部分为额外回读。
- B第一次 [03][03]：相近的ForwardRef已经出现，却到289秒才修改判据，未改生产代码，形成A胜的一对。
- A第二次 [09][09]：只插DEBUG后追查compat、重写测试，未实现unwrap，形成B胜的一对。118秒“首次生产编辑”不能支持更早编辑就更成功。
- B第二次 [10][10]：s12同模块案例通过；s16、120.1秒跨模块得到ForwardRef。s19、162.7秒添加get_call_globals并用于signature；s20尝试替换并不存在的return函数旧串而失败。随后s21–24定位/读取真实函数，s25、219.1秒缩为两行exact替换成功，**这是有原始回包支撑的错误后回读并缩小编辑**。之后两次新复现分别因只读`__globals__`赋值、FunctionType重复globals参数而失败；s30再次重写后用满28调用，s31提交，未运行最后那个脚本。官方1 passed，但补丁夹带module_a.py、module_b.py、reproduce_issue.py。

因此本轮B第二次成功不能写成“本地验证已闭环”：成功证据来自最终官方裁判；末尾复现失败是脚本构造错误，不等于生产补丁错误。反之03的自建测试绿也不等于真正解决。下一步应区分目标行为、复现自身错误、环境错误，而不是统一看到红就继续修/看到绿就停。

## rich_3518：四份补丁完全相同，延迟来自编辑协议与判据

四份最终patch均529字符，SHA256均为`558799c2f5077b9e1107c01d77e81e44d259cdcf1bae14893bff1bb3a314ede2`；全为同一处`Column(_index=index, highlight=self.highlight)`，官方都20 passed。A两次Agent125.0/269.8秒，B86.5/102.5秒；这四条能说明本题路径长短，不能外推所有任务。

- A第一次 [05][05]：s12编辑，s13显式列flag正确，s15确认已有显式列False不被覆盖，s16清理，s17提交；邻近条件确认到submit约20.4秒。
- B第一次 [06][06]：s13编辑，s15参数组合断言通过，s17原有定向文件19 passed，s18清理，s19提交。
- B第二次 [11][11]：创建tests/test_table_highlight.py，s11两个断言失败；s12编辑，s13两个断言通过，s14约2.5秒后提交。模型违反了“不创建测试”的提示，但最终补丁中这些测试被过滤，不能把违规动作自动等同于本轮裁判失败。
- A第二次 [12][12]：s10–13四次缺old_string，代码片段掉入参数键且相同重发，s14、122.8秒才正确edit；没有按closure规则一次纠正后换等价入口。其后先错跑不存在的/tmp/repro.py，再用A/B字母比较ANSI输出，正确补丁被误报FAIL；又给add_row传整数触发NotRenderableError，最终改为字符串数字才在s27、246.1秒通过。s28先提交含repro包，s29删除repro，s30再次提交只保留529字符源改动。慢路径主要发生在正确修复之后反复修理见证程序，不能说额外130秒改善了补丁。

## 工具实际采用、静默参数错误与补丁污染

本轮所有记录到的工具调用共299次：read_file104、run_command112、write_file50、edit_file22、submit_patch10、get_status1。官方计费调用合计283；另有5次必需参数拒绝、10次免费submit、1次免费status，不能把二者混成同一分母。**load_skill/run_skill_script等技能工具、三个图工具调用均为0**；run_command中也未见三技能脚本执行入口。不得归因为skill或graph能力收益。

除5次显式必需参数拒绝外，7条attempt共12次read_file含`start_line"`/`end_line"`等错键，典型回包status仍ok却落回1–150行。02另有两次filepath混入start_line字段造成FileReadError，后改sed成功。这说明只响应显式error还不够：有一类不报错的读取偏离。例：[07][07] s5/7/10及其后正确读取、[08][08] s5/6、[02][02] s17/19/20。不是harness忽略了正确的范围参数。

最终非生产污染出现在01、07、08、10四个patch，其中01/07/10仍resolved。03、05、09、10、11曾创建或编辑测试样式路径，部分后来删除/过滤；最终出现与过程违规分开报告。尤其01和12各两次尝试用write_file写/tmp被workspace边界拒绝，随后改到仓库根；“临时文件去/tmp”的提示与选择write_file不相容，只有相应shell入口才是可行机制。这不是在本仓库新增测试：上述为冻结模型轨迹中的行为记录，本次只读。

当前v31还有一条可直接观察的提交语义：[02][02] s29→31从2663/2文件变为1681/1文件，[12][12] s28→30从1821/2文件变为529/1文件，最终patch对应后一次submit。这一实测更新了早期“第一次提交后再删文件无效”的旧栈观察；不是建议靠反复提交替代提交前清理。

## 下一步机制选择

**优先研究有区分力的最小行为见证与证据成立后的直接行动，暂不加角色、图或更多状态技能。** 最有信息量的训练/恢复片段，是同一真实状态里“程序不报错但语义对象仍错”如何被识别，以及错误的复现API/判据如何被纠正；它们应来自新的独立问题，不能把这三个题的具体修法变成通用提示答案。

保留工具协议恢复为辅助目标：05/07等成功说明复杂技能不是必需；12的相同坏edit重复与10的回读缩小编辑提供鲜明正负过程例。单字符串编辑只能减少一类格式失败，不能解决03的弱断言或09的诊断打印。已确认行为后收尾也应保留，但“首次源码改动”“测试绿”“更早结束”均需与最终修复及回退一起看。下一轮不应为了复现这个一题净差继续在三题调提示；E2只是整体方案的局部探索结果。

[01]: evidence-v1/exec_closure/r1/attempts/01-fastapi_14873-r1-exec_closure/traces/trace_fastapi_14873.json
[02]: evidence-v1/budget_fit_5m/r1/attempts/02-fastapi_14873-r1-budget_fit_5m/traces/trace_fastapi_14873.json
[03]: evidence-v1/budget_fit_5m/r1/attempts/03-fastapi_5077-r1-budget_fit_5m/traces/trace_fastapi_5077.json
[04]: evidence-v1/exec_closure/r1/attempts/04-fastapi_5077-r1-exec_closure/traces/trace_fastapi_5077.json
[05]: evidence-v1/exec_closure/r1/attempts/05-rich_3518-r1-exec_closure/traces/trace_rich_3518.json
[06]: evidence-v1/budget_fit_5m/r1/attempts/06-rich_3518-r1-budget_fit_5m/traces/trace_rich_3518.json
[07]: evidence-v1/budget_fit_5m/r2/attempts/07-fastapi_14873-r2-budget_fit_5m/traces/trace_fastapi_14873.json
[08]: evidence-v1/exec_closure/r2/attempts/08-fastapi_14873-r2-exec_closure/traces/trace_fastapi_14873.json
[09]: evidence-v1/exec_closure/r2/attempts/09-fastapi_5077-r2-exec_closure/traces/trace_fastapi_5077.json
[10]: evidence-v1/budget_fit_5m/r2/attempts/10-fastapi_5077-r2-budget_fit_5m/traces/trace_fastapi_5077.json
[11]: evidence-v1/budget_fit_5m/r2/attempts/11-rich_3518-r2-budget_fit_5m/traces/trace_rich_3518.json
[12]: evidence-v1/exec_closure/r2/attempts/12-rich_3518-r2-exec_closure/traces/trace_rich_3518.json
