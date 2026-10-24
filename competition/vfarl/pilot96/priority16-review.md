# 首批96：S2/F2/R2/R5的16题独立复核

16题来源task字节、任务ID、原F2P/P2P计数及authoritative review引用一致；14题Smith为缺陷引入diff，2题V2为解题diff。16题分类均有静态支持，未发现同函数家族重复。当前已读到4题的本地官方成功报告；本复核没有新增执行。

需要保留的限制：Starlette2733只能保留为任务候选，教师删除旧namespace fallback，不收完整兼容正例；pipdeptree同时有遍历状态与错误依赖边，不能拆成单动作因果证据；Tomli错误字符串原本已有行列，真正缺结构化位置属性。PDFminer教师答案硬编码仍隔离。

复核发现PDFminer snapshot前缀曾被误切为`six`。主任务已将生成逻辑及authoritative分组修为`1a8bd2f7`，重新汇总catalog后，本复核确认16/16 review SHA均匹配。原始task字节与完整执行commit始终正确。

| 机制 | 任务 | 独立结论与有效修复契约 | 实际验证边界 |
| --- | --- | --- | --- |
| S2 | `adrienverge__yamllint.8513d9b9.func_pm_op_change_const__l4zuvyrl` | 支持S2；零基/一基行号转换，非格式偏好。 在forbid=True与forbid=non-empty的FlowMappingStartToken分支，将start_mark.line从零基转换为一基；保持列号及空映射规则。 | 原F2P/P2P尚未在目标官方后端重放 |
| S2 | `hukkin__tomli.443a0c1b.pr_238` | 支持S2；结构化源位置契约，错误消息字符串已有坐标。 恢复TOMLDecodeError(msg, doc, pos)构造器、msg/doc/pos/lineno/colno；解析raise位置传入同一原文与pos；同时保留EOF和弃用自由参数兼容。 | 本地官方原2 F2P、14 P2P完整通过；baseline=False、repaired=True；[正式报告](/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/competition/vfarl/replay/tomli-replay-result.json)；human off-policy，非Docker等价 |
| S2 | `pdfminer__pdfminer.six.1a8bd2f7.lm_rewrite__63n4v0qf` | 支持S2；竖排字体坐标推进符号与布局/顺序。 按font/render_char返回advance的既有约定，统一恢复charspace、render_char advance、wordspace的纵向累加；保持数字TJ位移原契约。 | 原F2P/P2P尚未在目标官方后端重放 |
| S2 | `laspy__laspy-327` | 支持S2，兼有F2调用者数据所有权。 ScaleAwarePointRecord的scale/offset异于目标header时，写入及header.grow前转换为目标整数编码，保持物理坐标；正常返回后调用者源record的X/Y/Z、scale/offset不变。PackedPointRecord不自动猜测尺度。 | 原F2P/P2P尚未在目标官方后端重放 |
| F2 | `mahmoud__boltons.3bfcfdd0.pr_375` | 支持F2；sentinel的唯一对象identity须跨复制保持。 Sentinel.__copy__与__deepcopy__返回self；copy/deepcopy外层容器仍产生正常容器拷贝，其中sentinel身份保持。 | run003本地官方原1 F2P、411 P2P完整通过；run001/002路径环境失败保留；[正式报告](/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/competition/vfarl/replay/boltons-replay-result.json)；human off-policy，非Docker等价 |
| F2 | `facebookresearch__hydra.0f03eb60.pr_3009` | 支持F2，混合S5路径规范化；关键是配置树复制边界与parent身份。 区分空/None full_key与整数0；相对root规范化key；OmegaConf.select(root,key)必须是subconfig本体才复制整棵ownership树，否则安全复制已脱离的subconfig；保留父节点插值上下文。 | 原F2P/P2P尚未在目标官方后端重放 |
| F2 | `tox-dev__pipdeptree.c31b6418.lm_rewrite__wjebghtt` | 支持F2的跨root状态隔离，兼有S4；复合缺陷，不是纯alias单故障。 每个root采用本次遍历自己的visited/path语义，按PackageDAG.get_node_as_parent(req.key)或等价真实协议从ReqPackage返回DistPackage节点；不得跟随不存在的requirement.package。输出root定向环及validate展示契约均保留。 | 原F2P/P2P尚未在目标官方后端重放 |
| F2 | `pennylaneai__pennylane-4736` | 支持F2；共享设备identity与隔离可变元数据需要同时满足。 QNode采用定制浅复制；设备等外部对象共享，execute_kwargs、gradient_kwargs与TransformProgram分别复制；dispatcher不deepcopy非pickleable设备。 | 原F2P/P2P尚未在目标官方后端重放 |
| R2 | `encode__starlette.db5063c2.lm_rewrite__12xv5ppk` | 支持R2；partial包装后的真正调用对象与__call__解析。 先解包全部functools.partial，再判断真正函数或可调用实例的异步__call__；不实际调用对象来猜异步性，保留同步negative。 | 原F2P/P2P尚未在目标官方后端重放 |
| R2 | `facebookresearch__hydra.0f03eb60.pr_1915` | 支持R2；运行时callable的可定位限定名，不是显示字符串美化。 类型与其他callable均保留__module__ + __qualname__，使模块中嵌套类路径可由_resolve_target/_locate重新定位；不能用__name__丢掉外层类。 | 原F2P/P2P尚未在目标官方后端重放 |
| R2 | `jd__tenacity.0d40e76f.pr_406` | 支持R2；装饰器保留真实函数默认参数元数据。 BaseRetrying.wraps使用WRAPPER_ASSIGNMENTS并保留__defaults__与__kwdefaults__；不改变retry执行逻辑或以重新生成签名掩盖元数据缺失。 | 原F2P/P2P尚未在目标官方后端重放 |
| R2 | `cool-RR__PySnooper.57472b46.lm_rewrite__urhyaplv` | 支持R2；class装饰器选择真实函数并保留构造/装饰语义。 按inspect.isfunction选择支持的普通函数，包含__init__；显式跳过不支持的coroutine函数，不把任意callable或property当可包装函数；_wrap_function负责注册code。 | 本地官方：baseline=False，repaired=True；human off-policy，非Docker等价 |
| R5 | `pallets__jinja.ada0a9a6.func_pm_ctrl_invert_if__zwhpkn1s` | 支持R5；PackageLoader须按普通/自定义PEP451与ZIP loader的实际ModuleSpec/archive契约分派，保持两种资源加载；R3为次机制。 | 本地官方：baseline1 failed/15 errors/860 passed，修复876 passed/0 skip；原1 F2P+739 P2P全部通过，人工off-policy |
| R5 | `iterative__dvc.1d6ea681.pr_7628` | 保留R5，四题为不同生产符号修复家族 调用as_callback必须lazy导入真实callbacks.TqdmCallback并以progress_bar=self传参；单独fs.exists成功不能替代该触发；完整31F2P/3P2P保留。 | 仅来源局部观察；不等于完整官方成功 |
| R5 | `encode__starlette.db5063c2.pr_2733` | 保留R5，四题为不同生产符号修复家族 两个模块优先导入python_multipart，ImportError后回退旧multipart，无依赖时按原协议置None；新namespace下消除弃用警告且解析可用，同时验证legacy-only/no-dependency兼容；严warning基线失败目前是推断。 | 仅来源局部观察；不等于完整官方成功 |
| R5 | `python-trio__trio.cfbbe2c1.pr_2932` | 保留R5，四题为不同生产符号修复家族 导入定义函数时不得求值尚未初始化的trio.lowlevel.FdStream；延迟注解或等价不成环引用均须使用真实导出；pipe生命周期与跨平台行为需独立验证。 | 仅来源局部观察；不等于完整官方成功 |

R5逐题边界：Dask9573因官方参数matcher把可选cudf skip合入pandas required而转reserve，由Jinja替换；Jinja原生teacher的混合ZIP脚本不足以证明整题，本地另行真实六工具+PhaseB验收通过。DVC必须直接走as_callback，exists本来可能成功；Starlette普通import带warning成功，完整修复须兼顾new/legacy/no-dependency；Trio导入成功不代表pipe全生命周期或跨平台成功。

每题的完整六字段因果链、source/task/native/review/许可blob SHA、修复契约、F2P/P2P计数及实际重放引用见[JSON逐题记录](/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/competition/vfarl/pilot96/priority16-review.json)。三份authoritative来源：[general-36-review.json](/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/competition/vfarl/acquisition/general/general-36-review.json)、[reviewed-sf40.json](/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/competition/vfarl/acquisition/s2-f2/reviewed-sf40.json)、[mechanism-reviews.json](/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/competition/vfarl/acquisition/r2-r5/mechanism-reviews.json)。

该轮只读取已获授权train来源、核对现有字节与记录，没有运行来源命令、增删测试或读取heldout。没有新增on-policy rollout、同状态verified critical pair、模型成功率或GPU训练结果。
