# P017 C1：五个回退与全量工具行为

2026-10-24。只读已回收C1原题、当前模型响应、工具回包和judge产物；没有调用模型、GPU、网络或新增测试，没有修改Agent或其他报告。**主结果保持Base 14/16、Step68 9/16，W=0、L=5，配对差−31.25个百分点。** RDF1117/1456两题只有F2P，不能把空P2P算通过；其余三个回退有P2P且两臂P2P均通过。

五个回退的Step68都没有成功提交过补丁。主要可见机制分别是重复探索耗尽工具预算、必要修改未完成，以及错误修复后未利用语义反馈；不能用“主训练普通final为0”统一解释这些失败。全32次未发现来源三工具名调用，参数schema异常反而在Base更多，故本批不支持“六工具参数普遍退化导致失分”的概括。

## 来源与计数

源ZIP为13,912,723字节、798成员，SHA256 `4732a49c99dee59ccd515e7b706ab204d7d01d89952d102fd76c858d8d2e8bb4`。已解包根目录为[evidence-c1-v1/weight-ab-c-20261024T070519Z-05d256b1](evidence-c1-v1/weight-ab-c-20261024T070519Z-05d256b1/)。本文`014`等缩写指其中`attempts/014-RDFLib__rdflib-1117-adapter/`，逐题结论对应`agent-task.json / scored.patch / f2p-execution.json / p2p-execution.json / raw-wire.jsonl / raw-adk-events.jsonl`。

[semantic-evidence.json](evidence-c1-v1/semantic-evidence.json)保存32次统计、14条重点轨迹的请求索引/反馈摘录及128个原始成员的字节SHA。请求序号包含摘要请求；时间相对该次运行首个wire请求，以工具返回或模型响应返回时刻标记。仅统计真实当前`wire_response.choices[0].message.tool_calls`，没有重复统计请求历史。工具回包按唯一调用ID关联；已核对每次运行响应ID、ADK事件ID和工具回包ID唯一。

完全重复与D口径一致：同一次运行内，工具名及解析后JSON参数完全相同，首次之外计数；相邻重复仅比较连续两次工具调用。修改后重复验证可以合理，不能把全部重复都判为错误。普通final计数仅指无工具且有可见content的Agent响应，摘要排除；thought-only结束单列。所有pending均无下一动作可见，不能猜它在生成final或submit。

| 指标 | Base，16次 | Step68，16次 |
|---|---:|---:|
| 实际结构化工具调用 | 394 | 302 |
| 来源`bash / str_replace_editor / submit`或其他未注册工具名 | 0 | 0 |
| 参数JSON解析失败 | 0 | 0 |
| schema键/类型/必填字段异常 | 15次，3题 | 2次，1题 |
| 完全重复调用 | 44 | 86 |
| 相邻重复调用 | 12 | 45 |
| 工具`status=error`回包 | 41 | 54 |
| ADK参数调用错误回包，使用`error`字段而非`status` | 15 | 0 |
| submit_patch调用 / 至少提交一次的任务 | 15 / 14 | 7 / 6 |
| 普通content final响应 | 13 | 4 |
| 无content、无结构化tool_calls响应 | 3 | 2 |

工具分布Base/Step68：run_command 189/165、read_file 109/92、edit_file 53/19、write_file 26/18、submit_patch 15/7、get_status 2/1。工具总数包含被拒绝的调用和不计普通工具预算的submit，不能与context计费工具数混用。`status=error`还包含故障复现、grep未命中等正常研究反馈；不是协议错误率。

Base参数异常分别为Pint2089七次edit缺old_string、mashumaro185五次write缺filepath、Soup263三次edit缺old_string并产生错误键。这15次均有实际ADK错误回包。Step68仅Soup119第3、4请求键为`start_line"`，被忽略后返回默认1—150行，下一步改用sed。故JSON合法不代表参数正确，但异常也不只发生于Step68。

无结构化动作的响应中，Base的mashumaro159、SQL19两题共两次，Step68的structlog586、198两题共两次，把目标六工具调用文本留在reasoning；未执行，不能算工具调用。剩余Base RDF1117为提交后的thought-only完成说明，正常退出，不能算普通content final，也不能当作无效工具序列化。

| 问题 | Base完全重复 | Step68完全重复 |
|---|---:|---:|
| networkx5444 | 1 | 0 |
| Pint2089 | 9 | 5 |
| structlog586 | 2 | 1 |
| mashumaro128 | 1 | 1 |
| sqllineage45 | 2 | 2 |
| networkx7024 | 1 | 12 |
| RDF1117 | 1 | 0 |
| Soup119 | 1 | 8 |
| mashumaro185 | 6 | 0 |
| sqllineage38 | 3 | 5 |
| networkx6908 | 2 | 1 |
| RDF1456 | 2 | 0 |
| Soup263 | 4 | 32 |
| structlog198 | 5 | 6 |
| mashumaro159 | 0 | 5 |
| sqllineage19 | 4 | 8 |

## 五个回退的直接证据

### RDF1117：部分修改，关键原题步骤缺失，错误锚点未恢复

题面明确列出csv2rdf的Python3问题：`.next()`、字符串重复解码、stdout包装，并提示detach会破坏help等行为。Base `013`第8请求约130.3秒首次成功生产编辑；随后将`.next()`改为`next(...)`、使用原生`sys.stdout`并避免关闭stdout。第13、14请求约165.6/168.7秒实际验证stdout与文件输出，第20请求提交，正常退出；两项F2P通过。

Step68 `014`前四次读完文件，随后询问stdout名称、configparser版本和Python版本。第8、9请求约145.8/153.3秒成功修改csv_reader和prefixuri，后者属于额外兼容处理；最终补丁仍没有题面明确提到的`.next()`和stdout修复。第10请求摘要之后，第11请求约199.6秒试图替换一个不存在的`__init__(self, base=None, ident=None, label=None, out=None)`块，收到`old_string not found`。第12请求等待100.329秒未返回后超时。没有修后实际转换验证、没有编辑恢复/提交/final；两项F2P失败。这里的直接证据是必要修改未完成与错误锚点未恢复，不能把额外prefixuri修改当作完成了题目列出的事项。

### RDF1456：定位已经接近根因，仍未形成生产修改

题面要求json-ld round trip保留无路径URL，不增加尾斜杠。Base `024`复现后定位`norm_url`，验证空路径被`normpath`改为`.`；第24请求约186.1秒将逻辑改为`normpath(parts[2]) if parts[2] else ""`。第26请求实际验证不同URL情况，第29请求提交；之后删除遗留repro被预算拒绝，但普通final正常结束，原F2P一项通过。

Step68 `023`第1请求建立并运行题面复现脚本，第9请求约46.7秒已读到`util.py`；第10请求摘要后，第11请求约205.3秒用真实Python调用确认`http://virtualparts.org/.`。第12请求等待94.715秒后取消，既没有edit，也没有生产源码diff，最终只有`reproduce_jsonld_bug.py`。原F2P失败。可以确认“已获取关键线索但未落地修改”，不能由无响应片段推断其在思考什么或认定服务端原因。

### Soup119：持续探索到28工具耗尽，第一次编辑已无预算

题面要求`:contains()`接受逗号分隔内容并按列表内任一值匹配。Base `016`虽探索较多，仍在第19、20请求约137.0/145.7秒修改正则与内容解析，第22请求约178.8秒修改匹配逻辑；第24、26请求运行验证脚本，后者显示各项预期数量与实际数量一致。清理脚本被预算拒绝后第31请求submit，普通final退出。原F2P三项、P2P 82项均通过；补丁含遗留辅助脚本，不把它描述成完全干净的提交。

Step68 `015`只有read/grep/sed。两个错误start_line键后改用sed；已经找到`:lang()`的相邻实现，却继续重复读相同区间和PAT定义。第30请求约102.2秒再次读取时普通工具预算已耗尽；第31请求约238.7秒才首次尝试edit，并直接收到`Tool call budget exhausted (28 calls)`。第32请求再等待61.243秒超时。**从未有成功生产编辑，最终patch为空**；F2P三项失败，P2P 82项通过。不能把这种失败描述成“正确补丁没来得及submit”。

### Soup263：无匹配grep循环后，又在预算拒绝上循环

题面要求允许带引号属性值之后紧贴`i/s`标志，并保持无引号值的区分。Step68 `026`第1请求成功复现SelectorSyntaxError，第2—4请求读源码，第5—8请求几种grep均未匹配。第9—28请求连续20次重复`grep -n "attribute" soupsieve/css_parser.py | grep "RE_"`，每次都exit 1、无输出；这与第6请求相同，合计21次该命令。普通工具预算耗尽后，第29—40请求又连续12次调用最初复现命令，均被BudgetExceeded拒绝。全部40个wire请求都有响应，无摘要；约109.48秒耗尽40turns。全程没有edit、submit或final，patch为空，F2P一项失败，P2P五项通过。

这不是提交后循环，也不是最后一轮submit的边界，更不是无返回请求造成的300秒等待。该例直接证明了对“搜索无结果”和“预算耗尽”两类反馈均未切换有效行动。

Base `025`有三次edit参数结构失败，但继续读回源码，第16请求约144.6秒终于成功把属性正则中的标志前空白由`+`改为`*`。第17请求约147.1秒编译题面选择器成功；后续两个错误验证命令也被修正，第20请求检查quoted/unquoted属性模式，第22请求submit、第23请求普通final。F2P一项、P2P五项通过。这里Base胜出依赖的是最终恢复与验证，不是完全没有协议错误。

### sqllineage19：消除断言的错误方向，已有错误结果未被纠正

题面是多行SELECT导致AssertionError。Base `032`第6请求因替换串匹配三处失败，第7请求约55.6秒成功加入临时debug；检查sqlparse后确认Newline是Whitespace的子类型，第14请求删debug，第15请求约92.9秒把三处分支的`ttype == Whitespace`改为`ttype in Whitespace`。第18、20、23请求分别验证SELECT源表、INSERT目标表、WITH情况，均打印预期表名；清理脚本后submit/final。原F2P一项、P2P 22项通过。

Step68 `031`第7请求的token打印已经包含`Token.Text.Whitespace.Newline`；随后四轮写/运行test_sqlparse脚本均失败。第16请求约215.6秒进行唯一生产编辑：保留严格Whitespace相等判断，却把原Identifier断言改为“若不是Identifier，则直接将sub_token.value加入source_tables”。它吞掉newline并提前清除source标志，未解决空白分类。第19请求约223.0秒直接打印LineageParser结果，`Source Tables`只有空白，没有题面要求的`tab1`；模型未据此纠正，又回到token打印。第21请求摘要、第22请求重跑打印，第23请求等待32.802秒取消。F2P失败、P2P 22项通过。

此处不能把无异常/exit 0当成完成修复。已有业务输出已经提供错误证据，模型没有把它与正确源表结果对照。没有新增输入或运行本地测试来得出此判断。

## 两个共同失败的范围

**mashumaro185**要求一个子类匹配多个discriminator值。Base `017`大量读取后五次write缺filepath，最后改用shell写repro，却导入不存在的`MashumaroBase`；只有辅助脚本diff，没有生产编辑。Step68 `018`一直读取/定位，末次read约212.5秒，随后请求无返回直至超时，patch为空。两臂F2P两项都因多标签处理失败，P2P通过。可确定是都未实现功能，不能从Base较多工具错误推出Step68在该需求上更强。

**networkx6908**要求对`nx.join`改为更清楚的树操作名称，并经历弃用周期。Base `021`实现并导出`join_rooted_trees`、给旧join加warning，实际验证新旧接口，提交退出；但冻结F2P要求`nx.join_trees`，三项均因该名称不存在失败。**原题只泛称更明确的新名字，没有直接指定join_trees**，所以要保留“题面命名不充分与来源判据要求不同”的限制，不能单凭失败把Base实现描述成完全不理解弃用。Step68 `022`未编辑，后两个相同grep分别约160.7/299.8秒返回，最终摘要请求尚未返回即超时；F2P也失败。该题P2P为空，回归覆盖未知，主判据未改。

## 研究含义与边界

本批五回退都发生在成功submit之前。它们支持优先研究有限预算内及时编辑、失败后的行动调整、停止无信息增益的重复，以及按业务结果验证修复；仅补普通final不足以直接覆盖这些案例。主训练的来源submit终点与目标运行时结束契约确有差异，但仍只是独立训练数据可检验的假设，不能当作本批退步的已证实因果。

全量参数统计也阻止了另一个过强结论：来源工具调用为0，Base schema异常15次而Step68两次，且Base Soup263能恢复后通过。重点应是异常之后是否恢复、编辑是否落地和反馈是否被正确使用，而非只计工具错误总量。重复调用虽在Step68更集中，仍需区分合理重验证与明确空转；本报告只对有逐步证据的循环下结论。

错误保留原始层级：Base两次timeout，Step68十一次timeout加一次turns错误；`pair.error=null`仅说明题对基础流程有效，不代表Agent没有错误。pending等待不能直接等同于长思考或服务变慢，相关时延与usage另由运行分析解释。C1是实际16个有效配对的确认结果，不与D相加挑选checkpoint或重写主指标。C题、答案、参考补丁及轨迹均不得回流训练。
