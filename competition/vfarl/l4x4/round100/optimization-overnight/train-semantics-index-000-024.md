# 固定 100 train：数组索引 0–24 的语义审阅

2026-10-28，北京时间。范围严格为 `v4/selection.json` 原 tasks 数组 `[0:25]`，不是题号 1–25 的重新排序。审阅时 selection SHA256 为 `ee270560ab7b27a0322eec8c604a53832d87f491cb6cdd71f680ff38a065d946`。

每题对照了当前 `questions.jsonl` 中的实际 initial user prompt、对应资产 `task.json`、真实 `submitted.patch`、源码 snapshot 中的目标逻辑及 oracle 的相关断言。25 题的资产 task 字节均匹配 selection 中的 task SHA；当前 question 自身指纹正确且包含该 task 的同一问题描述。以下 source 行号来自各题 tar 内的原文件，不是当前项目文件行号。

证据目录统一为 `competition/vfarl/l4x4/kaggle/artifacts/material-v1/material/assets/<完整instance_id>/`；其中 `task.json` 绑定题面与 test_patch，`submitted.patch` 是实际回放交付补丁，`source-snapshot.tgz` 保存未修源码。没有修改这些材料、题面、prepare、runner 或 trainer，没有按模型输赢移除任务，也没有读取其他评测或预留四题正文。验证采用一次性手动命令，没有新增测试文件或 GPU 工作。

## 逐题状态

“契约冲突”指题面直接要求错误 API/返回语义；“复现需修订”指预期目标可成立，但给出的 MRE/actual 不成立；“范围提示”保留可迁移任务，只限制能够声称的行为范围；“本次未发现冲突”不是完整正确性证明。

| 数组索引 | instance_id | 当前结论与强证据 |
| ---: | --- | --- |
| 0 | `Knio__dominate.9082227e.pr_166` | **复现/actual 需修订。** `dominate/dom_tag.py:287` 的真实 `children` 是 list，越界已抛 `IndexError('list index out of range')`，不是题面的 KeyError；实际任务是统一自定义错误消息。oracle `test___get_item___child_index_error` 断言 `Child with index "10" does not exist.`；补丁 catch IndexError 与这一目标一致。 |
| 1 | `PyCQA__flake8.cf1542ce.func_basic__fmtax6ji` | **MRE 需修订，主契约一致。** `FileProcessor.__init__:62` 访问 `options.hang_closing`，题面 `options=None` 先抛 AttributeError，根本到不了 blank counter。oracle `test_visited_new_blank_line` 用真实 `default_options`，断言 0→1；`visited_new_blank_line:156` 的 −1→+1 补丁与该契约一致。 |
| 2 | `PyCQA__flake8.cf1542ce.func_basic__uattrxez` | **已证实 API 与返回契约冲突。** 题面调用 `StyleGuide.is_user_ignored` 并期待 bool；真实类没有此方法，补丁修改 `should_report_error:367`，其返回 `Decision` 枚举。调用方 `handle_error` 用 `is Decision.Selected`。oracle `test_handle_error_does_not_raise_type_errors` 断言 T111 被实际报告，不能用 bool 返回代替。另题面 `OptionManager()` 本身缺少必填构造参数。 |
| 3 | `PyCQA__flake8.cf1542ce.func_basic__w0t2ify7` | **已证实返回层级冲突。** 题面调用 `decision_for` 却预期 `Selected.Explicitly/Ignored.Implicitly`；该 API 真实返回 `Decision.Selected/Ignored`，前两种是 `was_selected:102` 的返回类型。oracle 明确分开 `test_was_selected_*` 与 `test_decision_for`。题面 `OptionManager().get_default_values()` 也不是该 snapshot 的可用 API。 |
| 4 | `bottlepy__bottle.a8dfef30.func_basic__dzcdqzqa` | **主契约一致，附带变更不等于额外需求。** `FileUpload.save:2769` 目录/文件名顺序错误、overwrite=False 时静默 pass；oracle `test_save_dir` 与 `test_save_overwrite_lock` 分别验证目标路径内容和 IOError。参考还把 `_copy_file(..., chunk_size+1)` 恢复为 chunk_size；这部分不应被扩大成题面已自足规定的独立能力。 |
| 5 | `bottlepy__bottle.a8dfef30.func_basic__f94rj5df` | **MRE 需修订，reset 契约一致。** `Route.call:528` 是按需缓存的 plugin apply，`prepare:538` 才强制触发。原题示例从未访问 call/prepare，真实原版与参考版均 Before=MISSING/After=MISSING。`reset:533` 应只删除 call 缓存，保留路由/插件状态；oracle `TestPluginAPI.test_apply` 实际发请求触发 plugin。 |
| 6 | `bottlepy__bottle.a8dfef30.func_basic__r3tamcap` | **MRE/作用域需修订。** `_make_overlay:2545` 初次主动复制值，所以原题首次 `overlay['key']` 在原版与参考版都得到 source。真正损坏的是 `_iter_overlays:2539` 的活 weakref 筛选导致后续更新/删除不传播；oracle `test_overlay` 明确先断言初值，再修改 source→source2 并核对子层更新。 |
| 7 | `facebookresearch__hydra.0f03eb60.pr_1915` | **本次未发现冲突。** `_convert_target_to_string:100` 对 class 误用 `__name__`，使 A.B 变成 module.B；参考统一 callable→`__qualname__`。oracle nested callable target 对实例类型进行判断。仅限可导入的模块级嵌套类，不外推支持 `<locals>` 中任意 callable。 |
| 8 | `getmoto__moto-5200` | **目标一致，oracle 有长度上界缺口。** `secret_arn:66` 从 range(5) 改 range(6)，与题面六字符后缀一致。oracle 新增 `re.match(... + r'\w{6}', ARN)` 没有尾部锚点，不能单凭通过该断言证明不允许七字符以上；不影响实际六字符参考补丁的语义。 |
| 9 | `getmoto__moto-5348` | **本次未发现冲突。** `DynamoHandler.get_item:433` 原先未拒绝 key 属性的空值；oracle 明确传 `Key={'forum_name': {'S': ''}}`，要求 400/ValidationException 和带 key 名的消息。题目的 empty key 应按“属性值空串”解释，不等同于整个 Key 字典为空。 |
| 10 | `getmoto__moto-6159` | **本次未发现冲突。** `APIGatewayBackend.create_resource:1730` 路径正则缺冒号；oracle 将 `{pro:xy+}` 与 `us:er_0-9` 列为有效，保留原斜线等非法情形。参考增加冒号符合该局部字符规则，不声明完整 API Gateway 路径合法性已覆盖。 |
| 11 | `getmoto__moto-6492` | **本次未发现冲突。** Route53 POST 创建逻辑在原 `responses.py:83–84` 返回 201 和旧 headers，参考加 Location；oracle 读取 response Location，并在 mock 模式 GET 该地址确认 hosted zone Name。要求是响应 header 可用，不只是 body 增字段。 |
| 12 | `getmoto__moto-6609` | **支持范围需要明确，不能声称已实现 SM2 密码学。** 参考只改 `Key.signing_algorithms:166` 的枚举与错误列表；oracle 只新增 create-key 算法列表和 invalid-algorithm 消息。真实 `sign:617/verify:642` 明写忽略传入算法并硬编码 RSA-PSS/SHA256。题面泛称“support SM2DSA”比可证明的 mock 元数据支持更宽。 |
| 13 | `getmoto__moto-6755` | **主契约一致，email 属性持久化覆盖较弱。** `_is_verified_address:162` 与 `set_identity_mail_from_domain:569` 改为按邮箱域验证及校验 MAIL-FROM 子域，满足 email Identity 不应报错。oracle 新调用 email Identity 后仍读取 `Identities=['foo.com']`，并沿用之前相同的域配置值；它证明调用可接受，不能证明 email 专属属性读回。 |
| 14 | `getmoto__moto-7061` | **本次未发现冲突。** `describe_tags:1558` 原仅处理资源组/propagate，参考增加 key/value 精确匹配，oracle 分别核对返回整条 tag 字典。题面没有要求新分页能力，不能把相关旧注释当作扩展任务。 |
| 15 | `getmoto__moto-7111` | **PackageType 主契约一致，返回 None 的叙述需降格。** `LambdaFunction` 初始化把未提供 PackageType 的默认值 None 改 Zip；`get_configuration:743` 将该字段返回。oracle 明确检查 list/version/create/get 的 PackageType='Zip'。原题展示的 `list_zipped_lambdas` 函数无论何种输入均返回 list，因此“yield None”不是这段代码真实返回类型；遗漏字段时应为空列表。 |
| 16 | `getmoto__moto.694ce1f4.pr_4924` | **本次未发现冲突。** `ListenerOrBalancerMissingError.__init__:152` 要求了未使用的 arn，参考去掉参数。oracle `test_describe_listeners` 对未给任何 ARN 的调用要求 ClientError、ValidationError 和完整消息；没有隐藏的 ARN 内容构造需求。 |
| 17 | `google__textfsm.c31b6007.func_basic__ajsqn8d7` | **本次未发现冲突。** List.OnClearAllVar:203 将 `[None]` 改 `[]`；oracle `testList/testNestedMatching` 与题面两种输出结构逐项一致，包括嵌套 named groups 和普通 name 值分离。 |
| 18 | `google__textfsm.c31b6007.func_basic__h6yd28ss` | **本次未发现冲突。** Required.OnSaveRecord:129 应对空值 raise SkipRecord，原条件仅在 None 时 return，不能阻止记录。oracle `testParseText` 与 `testList/testContinue` 等保留 required/filldown 行与跳过行为，和题面 required-empty 语义一致。 |
| 19 | `gruns__furl.da386f68.func_basic__1fhv9h7a` | **主契约一致，根因表述可更准确。** `quacks_like_a_path_with_segments:444` 原 OR 会在没有 segments 属性时继续求值 `obj.segments`，真实机制是短路条件错误，不是字符串真的拥有 segments。参考 AND+非字符串可迭代检查，与 Path.load/非零/URL 组合 oracle 一致。 |
| 20 | `gruns__furl.da386f68.func_basic__dc54wz0a` | **题面范围不足，但可由代码/原 oracle恢复。** 题面只说 join 返回 None；`join:1846` 还先 load(url) 丢掉原 base，并把 urljoin 方向写反。oracle `TestFurl.test_join` 同时断言 `f is f.join(...)` 与每步正确 URL、相对路径、query/fragment 和多 URL 顺序。只把 return None 改 self 不能满足已有 join 契约；不因此认定为重复/无效题。 |
| 21 | `gweis__isodate.17cb25eb.func_basic__y3h8khx4` | **actual 需修订，预期锚点契约一致。** `totimedelta:281` 两参数分支 pass 后会落入 `(start-self)+start`；题面所示 datetime 算术实际抛 datetime+datetime 的 TypeError，并非无异常返回错误 timedelta。参考要求两参数 ValueError，并修正 start/end 计算；oracle `test_totimedelta` 直接覆盖跨月份/闰年的正确结果。 |
| 22 | `hukkin__tomli.443a0c1b.func_basic__qlgy79aj` | **已证实 MRE 未触发，实际错误文本也不精确。** `Flags.add_pending:214` 颠倒 `(key,flag)`；原题简单 `[table]\nkey='value'` 在原版/参考版都正常。需要 dotted key 后再进入 table 以 finalize pending。真实有效样例原版 TypeError `'int' object is not subscriptable`，参考版正确解析或对重定义抛 TOMLDecodeError。oracle `TestData.test_valid/test_invalid` 是文档集合判据，而非原题该最小片段。 |
| 23 | `iterative__dvc.1d6ea681.pr_7628` | **导入/适配契约一致，MRE 依赖已有 repo 的前提未说明。** `Tqdm.as_callback:167` 导入不存在的 `_callback.FsspecCallback`，参考调用真实 `callbacks.TqdmCallback(progress_bar=self)`。oracle `test_exists` 等先用 dvc fixture 初始化/添加数据再读 DVCFileSystem；题面直接 Repo() 的片段是否可执行取决于当前目录已有有效 DVC repo，本轮未在完整 DVC 环境运行，保留这一不确定性。 |
| 24 | `jd__tenacity.0d40e76f.func_basic__85q611gl` | **主契约一致，另需保留无底层异常的 exhaustion 规则。** `RetryError.reraise:183` 把 failed 条件倒置并直接 return；参考在 failed 时传播原异常，否则 raise RetryError 自身。oracle `test_reraise_from_retry_error` 验证 KeyError，`test_reraise_no_exception` 验证 callback 总要求重试但函数成功时仍应抛 RetryError；题面主要描述前者，未与后者冲突。 |

## 已执行的强机制证据

一次性命令从对应源码 tar 读取实际函数/类节点执行，或仅在临时目录加载生产包，不调用 pytest，也不创建新的测试文件。

1. dominate 的真实 `__getitem__` 作用于 `children=[]`，索引 10 得到 `IndexError: list index out of range`。
2. Flake8 实际 FileProcessor 初始化方法在题面的 `options=None` 下得到 `AttributeError: 'NoneType' object has no attribute 'hang_closing'`。实际 OptionManager 构造签名要求 version/plugin_versions/parents/formatter_names 四个 keyword-only 参数；原题无参构造实际 TypeError，且类中没有 get_default_values。
3. 编译真实 Flake8 枚举、DecisionEngine 和 StyleGuide 类：`is_user_ignored` 不存在；Decision 的两个枚举值 bool 均为 True，不能代替题面 True/False。原索引 3 的实际 mutant 上，E111/E112 的 decision_for 都为 Decision.Selected；API 返回层级依然是 Decision，而不是 Selected/Ignored 的 Explicitly/Implicitly。这里没有为了匹配题面改写生产代码。
4. 完整加载对应 Bottle 生产模块。reset 原题 MRE 在原版、仅应用参考修改的内存副本中均 Before=MISSING/After=MISSING。overlay 原题初值在两版均为 source；再写 source['key']='source2' 后，原版子层仍 source，参考版为 source2。
5. 用真实 Duration.totimedelta 函数作用于与题面相同的 datetime/十天 timedelta 运算，双锚点落入 TypeError；这是目标方法算术组件验证，不冒充完整 isodate 包评测。
6. 仅将对应 Tomli 生产 `.py` 包载入临时目录，分别保留原版和在内存/临时副本应用实际参考的 tuple 顺序改动：

| TOML 文档 | 原版 | 参考修改后 |
| --- | --- | --- |
| `[table]\nkey="value"` | `{'table': {'key': 'value'}}` | 相同 |
| `fruit.apple.color="red"\n[other]\nx=1` | TypeError，int 不可下标 | 正确 fruit/apple 与 other 字典 |
| `fruit.apple.color="red"\n[fruit.apple]\nx=1` | 同一 TypeError | TOMLDecodeError：不能重复声明 fruit.apple |

这些证据区分“题面错误/复现不足”和“修复目标本身无效”。本轮没有凭这些问题删除题、改变 oracle、改动来源 patch，或宣称任何新增模型修复收益。

## 可采取的最小题面修订

优先修正索引 2/3 的 API 与枚举层级，避免诱导修改正确的返回协议；随后修正 0 的真实异常消息、1 的 options fixture、5 的 lazy apply 前提、6 的后续传播触发和 22 的 dotted-key/finalize 触发。索引 20 可补正确 URL 的例子，21 可将 actual 记为 TypeError。KMS 12 应明确本轮是 mock 元数据/验证枚举范围，不能把参考和现有 oracle 都未实现的 SM2 密码学当作已验证需求。新题面应保持完整任务身份与源码/oracle 不变，并保留源题面及修订来源。
