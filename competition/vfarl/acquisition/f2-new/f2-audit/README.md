# F2 引用与所有权数据复核及补充

2026-10-26。当前准备数据，不启动 GPU 或训练。本目录优先保存真实来源、机制边界和实际六工具结果，旧轮次记录不覆盖。

原有 20 条 `normalize eligible` 都确有双轮官方执行证据，重新读取原完整 JUnit 和原始要求后仍通过。但其中“出现 copy/is 就属于 F2”的归类不成立：

| 修正对象 | 实际主机制 | 原因 |
| --- | --- | --- |
| Jinja 1613 | R2 | async wrapper 的 name/qualname 与模块全局解析支持 pickle，未改变共享可变数据所有权 |
| Jinja 2026 | R3 | Undefined 的 dunder 探测、AttributeError 与 slots 协议 |
| Ariadne 159 | S5 | 新增 AST mixin 指令移除遍历，原路径没有污染存储 AST；deepcopy 保护新引入的遍历 |
| cfn-python-lint 426 | S1 | str_node 与源码 mark 表示转换，旧路径只更新字典 value，copy 保护新键转换逻辑 |
| Qiskit 12620 | F3 | 清空指令的副本需要失效旧参数缓存 |
| aiohttp 8997 | F1 | clone 的 scheme/host 需要传播到绝对 URL；原缓存字典已经独立 |

[20 题修正审阅](reviewed-existing-v3.json)与[修正目录](catalog-existing-v3.json)保留原来源、原审阅和原重放 hash；只有 16 题保留在 F 课程，其中 F2 14、F1 1、F3 1。四条 S/R 题继续保留真实证据，不计 F600 专科。`replay/f2-audit-existing-...` 仅补充元数据，指向同一原运行，未伪造新运行。

新增已通过原 full-required 双轮与 normalize：

- [batch1](catalog-new-batch1.json)：Furl `Path.__truediv__`、`QueryCompositionInterface.args`，不同生产 scope，77 原生 cases 全通过。Path 除法还包含路径顺序问题，作为一个组合任务；不能把整段修改说成已验证的单动作偏好对。args 原题面与固定源码的 live-reference 契约相反，已固定 curator 题面并在此题面下真实重放。
- PyPika `utils.builder` 的新真实重放 1,061 原生 cases 全通过，但与固定共同课 `lm_rewrite__v8d2njaq` 是同一生产修复 scope，已从 [batch3 实际审阅](reviewed-new-batch3.json) 排除。不能因 mutation ID 或 contract 名字不同重复计数。

新增另有 [PyASN.1 构造类型 clone 选项传播](catalog-new-batch7.json)，归 F1：原 26 个 F2P 均在基线失败，修后 1,179 个原生 cases 全通过。它修复 cloneValueFlag 的参数控制，并未发生原对象的别名污染，不能为了配额标成 F2。

当前可交付选择见 [catalog-ready.json](catalog-ready.json)，共 19 个 F 专科来源（F2 16、F1 2、F3 1），最终是否入课由 F600 总课程的实际修复作用域去重和独立复核决定。

未入池的真实结果：GPX 深 clone 的原 143 个 required 均被修复，但官方统一 `python_functions=*` 将原生辅助函数额外收集为测试，产生 6 个 fixture 错误，官方 full-task 拒收；不能用局部通过冒充完成。Cantools 两个 DBC live getter 和 Tester.messages 已做真实尝试，原通用依赖环境为 371 passed/1 failure（CLI 的 warning 断言），首个隔离环境缺原生 parameterized 依赖造成收集失败，均保留；后续补齐 parameterized 与固定源码的真实安装元数据后，仍在 CLI warning 断言上失败。已确认官方 `swegemma/harness/container_setup.py` 生成的 pytest.ini 含 `ignore::UserWarning`，而原生 CLI case 正在录制并断言该 warning；因此三道题全部不计入，无需修改环境过滤器取得表面通过。

Qiskit-Terra 8741 是真实绘图副本隔离缺陷。已固定 0.22.0 CPython3.10/macOS ARM64 官方 wheel，SHA256 `7f661e48a466842a8052d247b9b6485202c413fdd68180c5c409d3744bc518fb`，并用原生编译扩展绑定实际工作区。首轮真实工具仍因无条件导入 tweedledum.classical 缺依赖而停止；该版本无匹配现成 wheel，按本轮“不为单题长时间编译”的执行选择保留 run-error，未伪称 full replay 或计数。

F2 准入依据是生产缺陷改变了本应共享或隔离的对象引用、所有权或身份契约。字面 `is` 断言不是必要条件；人为加一个 `is` 也不足以证明 F2。移除了旧搜索中禁止同仓、禁止同快照、总测试数 100 等人为门槛；同快照不同函数可以独立，但同一修复 scope 的不同 mutation 不重复计数。原题面、原验证节点、原测试字节和原官方 matcher 仍须保留。

`collect.py` / `course.py` 是选定来源的实际数据生产流程，沿用 S600 的正式工具调用器；只执行已审阅的生产修复和 source-bound 行为命令，不生成测试文件，也不生成模型思考、logprob 或关键动作对。`source_task_path`、源页 receipt、完整执行提交、LICENSE、原生资产 hash 均固定。当前证据仍是官方 SubprocessManager 的 macOS CPU 后端，不代表比赛 Docker/Linux 等价。

另完成 root 新18题的[独立二读](root-context-independent-review.json)：逐题实际生产补丁、工具观察、原完整 JUnit 和来源哈希均复核；17F1/1F4支持，附组合修改的归因边界。此复核不把他人来源再计为本分工新增题。
