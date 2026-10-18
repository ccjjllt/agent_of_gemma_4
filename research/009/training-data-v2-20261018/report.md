# P009 v2：四个新增问题的真实CPU事务

2026-10-18。只在已分配train的8个repo内，按固定元数据轮转规则选4个未读真实PR；未按结果换题。v1数据、源码、manifest与训练输入均未修改；本目录**未合并进v1、未训练，也不自动授权下一次训练**。没有GPU、付费教师、heldout/选择题读取或测试套件执行。

本次编码：[encoded-v2-12b.jsonl](encoded-v2-12b.jsonl)，282,218 bytes，SHA256 `d3ea694beadc6a58ad11c10159490a6458c8f978ec1ed8e171573e17eb81b186`。**4个新issue、3个repo，21,412输入tokens、3,057监督tokens**；最长10,310。3条≤8,192，mypy长事务完整保留，没有截短来凑8K。

## 1. 实际新增结果

| 问题 | 固定buggy commit | 实际合同与结果 | 输入 / 监督 |
| --- | --- | --- | ---: |
| pydantic PR5568 | 7174a14274c45962198735b2e61438a3afadb6a3 | 原源码缺camel_case/pascal_case公开generator；新增后真实BaseModel验证别名输入、按别名输出，显式field alias优先级保持 | 3297 / 699 |
| mypy PR12943 | bd60b30f880a4b59600bc9002f1b9a404a34aa3c | 独立缩小的继承/property案例在未改源码上触发PartialType INTERNAL ERROR；一行延期处理修复后，合法案例0错误，不兼容property仍正常报override | 10310 / 704 |
| yamllint PR572 | d1e1c32f4a86613cb818ddc15f6aa683239ed1ac | 真实CLI在子目录漏读父配置→正确找到；最近配置、显式配置与全局fallback优先级均保留 | 4370 / 1086 |
| pydantic PR3646 | e5f2e40d31866fcb764d65ad11fb179a1850ca35 | Color同色比较、同alpha、set去重及含Color模型相等性失败→通过；异色、异alpha及非Color不相等保留 | 3435 / 568 |

每条仍为**局部行为成功，task_resolved=null**，没有声称完整仓库benchmark或原Qiskit项目已经通过。所有观察来自本次实际执行，不是上游resolved字段或旧轨迹回包。

用途/精确文件：[data-usage.json](data-usage.json)；完整长度：[encoding-summary.json](encoding-summary.json)；源码branch、下载和许可SHA：[source-manifest.json](source-manifest.json)。

## 2. mypy：坏来源标签与真实验证恢复

该源记录标为resolved=true，但原消息45显示Qiskit文件不存在；随后人为在subtypes.py插入异常分支，消息55可见插入导致的错误缩进；后续反复失败，消息125仍为编辑匹配失败，126却直接宣称已修复并提交。**不能仅按resolved过滤就把这种轨迹当正例。** 该源动作、推理和旧未来没有进入新训练事务。

本次依据真实checker源码注释和崩溃栈，构造两个inline程序，在未改动源码上复现PartialType崩溃，原件见[mypy-inline-reproduction.json](mypy-inline-reproduction.json)。修改仅使尚未完成推断的PartialType基类属性走已有延期路径，避免过早进入subtype比较。

第一轮修后已消除崩溃，但一个B().other循环依赖案例正常报has-type错误；此前“必须零错误”的验证预期过严，故[attempt-01](outputs/mypy-pr12943/attempt-01/transaction.json)被保留为未通过。随后在**同一个已修改源码状态**继续，用无循环的后来C.value案例隔离缺陷，并要求不兼容property产生明确[override]诊断。两个正确判据真实通过后捕获补丁，形成[attempt-02](outputs/mypy-pr12943/attempt-02/transaction.json)。

最终完整历史保留第一次错误验证，没有拼接旧未来。target_assistant_steps=[2,4,5]：真实生产编辑、纠正后的成功验证、提交；失败验证ordinal3不监督。长历史使该样本达到10,310tokens，不能直接装进当前8K配方。32768是本次**计数/编码上限**，不是已证明的训练容量。

## 3. 其余边界

- pydantic PR5568沿用来源的AliasGenerator对象方案，证明实际validation/serialization API行为；未扩展为所有字符串命名边界的结论。
- yamllint仅创建临时配置fixture并向实际CLI送stdin YAML，没有新增测试文件。attempt-02保留原实际优先级合同，同时让祖先遍历包含根目录再退出；根目录分支未在真实文件系统根写配置验证，不额外宣称该边界已实测。attempt-01是同issue旧动作，未重复编码。
- Pydantic两个issue同repo不同完整commit和语义，不把它们说成两个repo。
- 另一路alive-progress PR213属于v1旧candidate升级，不是新issue；其完整API被旧包接口不匹配阻断，见[独立分支](alive-progress/)。它没有进入本4条编码。
- 全部复用现有/tmp CPU依赖和12B CT v2 tokenizer/Jinja；本轮没有安装新依赖或下载模型。原样v31工具函数加P006本地文件/进程transport，仍不是Docker环境等价证明。
- Pydantic/mypy源码为MIT，yamllint为GPL-3.0-or-later，许可证原件及hash保留，不把混合来源统一改称MIT。

## 4. 交付与后续使用

[transactions-v2.jsonl](transactions-v2.jsonl)为4条完整真实事务；[encoded-v2-12b.jsonl](encoded-v2-12b.jsonl)使用同p005六工具协议、官方12B CT v2全对话模板，4/4编码，无截断，tokenizer指纹仍为 `d001718145f4980b65e70b5ffa1dfce93b3eca5a5d8de2446b8c483f2b4b8927`。失败基线与工具回包保留在上下文，只有明确target动作及正文监督。

本轮只推进新数据证据，不用它解释已完成P009的训练收益，也没有把新数据送入运行中的输出预算实验。下一次若使用这些材料，需要另作数据版本与预算安排；mypy长事务应保持完整，或单独研究可因果重建的局部任务，不能直接删掉失败历史。

v1冻结输入复核仍为 `0aca8fc273f6f58e94c641f52b7bf3e4dc039c7275a9b719010ad6300c8a72f5`。本分支到此停止扩量。
