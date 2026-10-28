# 固定100题语义深审：原索引50–74

2026-10-28。逐题对照v4题面、实际损坏函数、已审submitted.patch、原F2P正文和相关baseline日志。25题完整记录见[review.json](review.json)。只对具体矛盾做一次性有界CPU源码行为观察，没有运行25题全仓验收，没有新增测试文件、修改题号／原奖励或上传材料。

## 优先澄清的题面

| 原索引 | 问题 | 真实证据与建议 |
| --- | --- | --- |
| 52 APISpec component注册 | 原文要求“用副本”，却与真实缺陷及修复方向不符；示例比较两个局部变量别名，不能观测内容mutation | baseline把eager放lazy buffer、lazy放active；参考修复纠正两者并保留私有helper输入identity。公开schema接口原本已deepcopy输入。应改为普通component即时可见、lazy仅在引用后可见、引用解析正确且公开输入不被意外修改。 |
| 53 APISpec build_reference | 原例参数顺序错误，修复前后都TypeError | 真实签名是`(component_type, openapi_major_version, component_name)`。按正确顺序，baseline的2.x报KeyError、3.x返回definitions前缀及反转名称；reference的2.x/3.x路径正确。建议用公开关键字调用澄清API，保留原命名空间要求。 |
| 54 markdownify pr161 | 题面给的tbody首行全部th，修复前后均已产生正确分隔线 | 真正差异是tbody无thead、首行td：baseline插入空header，reference提升首行为header。建议改成真实td触发例，说明当前是多出空header，而非宣称th例缺分隔线。 |
| 66 PyPika Schema | 原文宣称不同name/parent误判True，方向相反 | 原题三组例子在baseline和reference中均False。真正损坏的是相同name及等价parent也判False；新增同值正例在reference中恢复True，原TableEquality F2P也验证此方向。建议说明同值false negative，并保留异值不等。 |

52/66与较轻的funcy现状矛盾见[behavior-observations.json](behavior-observations.json)；53按原API调用和真实签名的对照见[api-and-lazy-observations.json](api-and-lazy-observations.json)。这些观察分别从对应真实mutant快照运行，再应用原submitted.patch，没有用重写的目标函数代替原源码。

## 另两处范围／现状修正

55 markdownify pr181的参考修复确实把caption后的双换行改为单换行，但题面HTML是caption后直接tr，baseline/reference都没有其Expected中的表头分隔线。原完整oracle使用thead/tbody结构隔离caption换行。建议为例子补公开thead/tbody结构，或只比较caption相邻换行；不要求模型同时修复未包含在当前参考修复中的另一行为。54/55的实际输出和复用bs4依赖的成员哈希见[markdownify-observations.json](markdownify-observations.json)。缺soupsieve仅产生CSS选择器不可用警告，此处走html.parser且不调用CSS选择器。

51 glom YAML题面把当前baseline描述为已能执行恶意YAML，但实际历史baseline完整验收日志是`yaml.load()`缺少必需Loader参数而TypeError；题面CLI例也没有选择yaml target format。`safe_load`修复方向合理，当前证据支持YAML兼容和安全加载要求，不能声称已经复现该环境的代码执行。建议使用原oracle中的`--target-file`及`--target-format yml`接口，给出真实Loader错误和普通YAML输入期望，不再展示未经证实的执行日志。

## 仅修正Actual trace的较轻问题

- 68 without的真实baseline为`[1,1,0,1]`，原文写为`[1,0,1,1]`；正确Expected `[2,3,4]`及修复方向不变。
- 69 lpluck对单字符key `a`真实返回长度1的map列表，原文画了两个map；reference正确返回两个int `[1,10]`。
- 70 flip(dict)真实返回lazy map，求值时因单字符key解包而ValueError，并非原文所说返回原mapping；reference恢复键值反转的dict。最初观察脚本直接JSON序列化map导致的TypeError也原样保留；后续单独记录了真实返回类型及求值错误，没有将脚本编码错误当成库的原始行为。
- 74 interpose真实baseline为`['.','b','.','c','.']`，不仅是原文所列的前置分隔符，还丢失首项、保留尾分隔符；reference正确生成`['a','.','b','.','c']`。

这些任务的正确Expected和原oracle方向仍一致，不能仅因错误Actual转写就重标旧模型奖励或判定原oracle弱。

## 本范围的原oracle边界

本范围每个required F2P正文均有assert或raises类观察，没有发现与另一区间OSA相同的print-only判据。这个源码事实不等于完整语义覆盖，也未运行恶意“常量返回”实现以声称所有oracle可靠。

64 dataset的原test_with明确检查异常退出事务后数据长度不变；其curator题面另要求commit自身失败后的清理，F2P没有直接模拟commit抛错。参考patch恢复了两个rollback分支，但本轮没有构造“完整原oracle通过、commit失败清理却错误”的生产反例，因此只记录覆盖限制，不标成已证实weak_oracle。

50、56–63、65、67、71–73未发现足以改变当前问题核心契约的确证冲突。各题仍保留实际函数、原oracle和未扩展的边界说明，不把“本轮未发现”写成语义正确性保证。全部原100身份、原题面文件、旧哈希、真实验收结果和失败日志保持不变；本文仅提出独立版本化澄清候选，尚未应用新数据revision。
