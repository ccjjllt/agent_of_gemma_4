# F600 独立复核与缓存补充

2026-10-26。本目录保存逐题来源、真实补丁与实际官方工具结果的第二读，不以候选目录、recipe 或类名含 Cache/close/delete 代替机制证据。原始来源、旧审阅、失败运行与 source-snapshot 原字节均保留。

新完成 **26 道外部 train 任务**。它们均通过原始完整官方验证，原 F2P 逐条实际失败，真实 inline 观察为修前 error、修后 ok。进一步交叉复核后，其中 **21 道支持 F 专科：F1 5、F3 13、F4 2、F5 1**；另外 R3 两道、S3 两道、G 一道保留成功证据，退出 F 课程。原始完成清单及初审标签冻结在 [completed-source-batches-v1.json](completed-source-batches-v1.json)。最终边界联合采用 [v2 修订](new26-mechanism-boundary-amendment-v2.json)、[v3 修订](new26-mechanism-boundary-amendment-v3.json)、[v4 修订](new26-mechanism-boundary-amendment-v4.json)及 [独立交叉审阅 v3](independent-f3-resource-review-v3.json)。最后是否进入 F600 的 200 专科由主任务统一选择。

| 新批次 | 最终支持能力 | 原生完整修后 cases | 冻结来源审阅 |
| --- | --- | ---: | --- |
| Funcy 失效、过期、惰性表填充及配置传播 | 4 F3 + 2 F1 | 每题 203，0 skip | [六题审阅](reviewed-funcy-cache-candidates.json) |
| Funcy 写入与过期元数据 | 1 F3 | 203，0 skip | [独立补充](reviewed-funcy-storage-candidate-v1.json) |
| Boltons 链表策略、参数传播与资源生命周期 | 4 F3 + 1 F1 + 2 F4；另 R3/S3/G 各 1 | 每题 423，0 skip | [冻结来源](reviewed-boltons-cache-resource.json)、[完成记录](boltons-cache-resource-completed-v1.json) |
| Jinja LRU 策略、配置传播与状态重建 | 4 F3 + 2 F1 + 1 F5；另 R3/S3 各 1 | 每题 876，0 skip | [完整九题审阅](reviewed-jinja-cache-completed-v2.json) |

分类依据是实际变更：TTL/容量/scoped/default/key_func 原值传递归 F1；真正改变复用、队列淘汰、失效和过期元数据的题目归 F3；状态先恢复再重绑别名归 F5。Jinja.clear 原来已清空条目，实际只修 mapping 形状，归 S3；`__contains__` 仅修 eq/hash 成员协议，归 R3。Boltons.__getitem__ 的 KEY/VALUE 投影归 S3、flush 注入 NUL/返回值归 G、`__enter__` 绑定返回值归 R3。成功事实和原始证据不随标签修订而改写。

[全池实际作用域复核](final-pool-semantic-dedup-v1.json)记录固定版本 210 候选的 361 个真实 changed scopes（含 8 处嵌套函数）与共同 400 题的比对。210/210 均完成源码和真实补丁派生；9 组函数重合均为已二读的 h11/Conan/DVC 不同核心契约，没有相同作用域的同一 before→after AST 对。共享恢复组成项明确保留关联组；这不意味着 210 题全都属于 F，也不表示它们是彼此完全独立的编辑。

Funcy 第七题和 Boltons 十题的发布者题面为空，保留原文本与哈希、原 adapted task，使用独立 curator 版本行为题面真实重放。题面只描述可观察契约；参考补丁、原 required 节点和事后标签不加入策略题面。Jinja 九题与 Funcy 首六题保留原发布者题面。

Boltons 第一轮保留了两类真实环境失败：沙箱禁止本机 TCP 临时端口，以及短路径 alias 中的 `f` 字符触发上游 `lstrip(BOLTONS_PATH)` 的字符集行为。第二轮使用允许 localhost 的 CPU 执行及不含文件名前导 `f/c` 的短路径组 `zz26`，原测试、要求与 matcher 均不改；全部十题第二轮完整通过。不是把首轮失败改写成成功。

复核原材料的交付：

- [F1 原四十题](reviewed-f1-reclassification.json)：三十题支持参数/上下文传播，其余按真实补丁建议重分类。
- [原 F3 preview 四十题](original-f3-preview-audit.json)：未完成或官方未通过的 recipe/目录不能计为准备完成。
- [F3 替代来源四十一题](reviewed-f3-reclassification.json)：通用 cache 流程观察不自动证明实际改动属于缓存失效。保留原成功证据，同时纠正机制标签。
- [F4 原四十题](reviewed-f4-reclassification.json)：二十题支持资源/退出；十题主要是协议或操作状态，另外保留 F1/F3 重分类和重复组成项排除理由。
- [真实作用域碰撞](scope-collision-second-read.json)：过滤注释和 EOF 假重合，区分相同函数中不同契约、共享恢复组成项与完全重复。共同 snapshot/function 仍须分组，不能声称完全独立编辑或做随机评测划分。
- [生产符号纠正](source-symbol-amendments-v1.json)：纠正 Arrow.format、DVC 缓存 helper/restore 和 ThreadPoolExecutor 构造器等错误方法名，旧审阅不改写。
- [root OAuth 十题交叉复核](root-oauth10-second-read.json)：调用者通道、完整来源哈希、原题面与要求及实际成功均核对。

`collect_funcy.py`、`collect_boltons.py`、`collect_jinja.py` 是实际来源准备与官方工具采集入口，不是测试套件。Jinja 早期六题 partial 导出保留，最终使用 `completed-v2` 九题审阅和 `f600-jinja-completed-` 报告前缀。已交付来源审阅冻结，修订应使用新版本名。

这些仍是 Codex 指定并真实执行的 off-policy 冷启动动作，没有 Gemma rollout、生成思考、logprob、同状态关键动作对或 GPU 训练。官方本地 SubprocessManager 的 CPU 结果不证明比赛 Docker/Linux 等价。
