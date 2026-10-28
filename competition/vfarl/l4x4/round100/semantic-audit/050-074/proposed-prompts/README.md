# 最小问题澄清提案

2026-10-28。仅保存文本候选，未调用prepare、生成新task／selection或打包上传，原v4文件、题号、参考修复、oracle和奖励不改。根代理统一生成后续问题revision。

[mapping.json](mapping.json)绑定每题旧task_input SHA、旧prompt SHA、新statement文件及SHA和行为证据。51/52/53/54/55/66为核心澄清；68/69/70/74只更正已证实的Actual转写或返回类型，不改正确Expected。

51保留原来的数据安全要求，但改为真实Loader错误和明确YAML target格式；52只描述普通／lazy注册可见性、原有公开输入保护与引用解析；53修正公开API参数顺序；54使用真实td触发例；55使用显式thead隔离caption换行；66说明相同Schema值被误判不等。文本不含参考patch、私有验收节点或事后训练标签。

51/52/53/54/55/66的修订例子已对各自真实mutant源码和既有参考patch执行有界CPU观察，代码片段均为有限输入，结果见[revised-mre-observations.json](revised-mre-observations.json)：

- 51普通YAML：baseline为Loader错误，reference得到映射；无副作用的Python tuple tag被reference拒绝。
- 52普通schema即时可见：False→True。
- 53正确公开签名：错误definitions及逆序名称→正确components/schemas路径。
- 54首行td：空header＋数据行→正确首行header。
- 55带thead的caption：双换行→单换行，表头分隔线保持存在。
- 66同值Schema：False→True；异名仍False。

这里只复用已有本地依赖cache，并从原本地runtime archive读出bs4、face、boltons的纯Python文件，成员及哈希保留。未安装或下载新包。bs4缺少soupsieve产生CSS功能警告，但示例使用html.parser且不调用CSS选择器。

这些行为观察不构成新的完整oracle验收、六工具示范、模型rollout或训练收益。后续revision仍须保存原问题版本并按相同原oracle取得该新task hash的真实验收，不得把旧示范改名为新prompt执行记录。
