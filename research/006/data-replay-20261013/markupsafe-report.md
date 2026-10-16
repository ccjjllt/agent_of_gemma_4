# MarkupSafe：工具恢复成功不等于语义修复成功

2026-10-13追加CPU工作，仅研究一个新的独立仓库。实际完成了 **2次旧文本匹配失败→刷新源码→官方编辑成功→行为仍失败→新语义改动→同5个案例全部通过**。负过程与新继续分别保存，没有复用源轨迹之后的动作或观察；两者只有1个来源问题、1个工具恢复事件。

## 为什么选择这条

除已处理的click/pydicom外，缓存中另外5条结构化来源分别是DVC、MarkupSafe、Dask、Red-DiscordBot、SQLFluff。P005逐条聚合记录中只有MarkupSafe存在明确编辑匹配失败（2次）；其余4条该计数为0，不代表它们所有动作都无失败。本轮没有为了凑数量执行这些其他任务。

MarkupSafe的失败类型与pydicom不同：pydicom是old_string同时出现两次；MarkupSafe是先前动作已经把函数尾部改为 `s_str = str(s)`，后续仍提交旧版本的 `return Markup(_escape_inner(str(s)))`。两次错误最终通向同一次成功，只计一个恢复事件。

来源实例 `pallets__markupsafe.620c06c9.pr_469` 对应官方镜像真实buggy commit **`000489747bba0b3234687459f883f17b53551b17`**，不是profile commit。仅取回三个生产文件与LICENSE。数据：[固定来源与SHA](sources/markupsafe-instance-manifest.json)、[固定镜像](https://github.com/swesmith/pallets__markupsafe.620c06c9/tree/000489747bba0b3234687459f883f17b53551b17)、[来源轨迹存档](sources/markupsafe-source-trajectory.json)。

## 真实状态和环境

只需将来源消息22的一次编辑通过官方edit_file重放，就能恢复错误前源码；没有执行源bash、pip、安装/构建脚本。恢复后的35–45行与来源当时的可见读取逐行相同。准备过程与编译诊断见[state-preparation.json](outputs/markupsafe-recovery/state-preparation.json)。

为区分“文本匹配”与“实际程序行为”，本轮用macOS Python3.11的公开头文件和 `/usr/bin/clang` 直接编译固定源码中约205行的 `_speedups.c`，没有执行setup.py。所有运行回包明确显示backend为 `markupsafe._speedups`，不是可能掩盖错误的Python fallback。编译产物只在临时工作区，未进入补丁或存档权重。

SimpleLazyObject来自未改写的Django 3.2.20 `functional.py`（14,146字节，只有标准库依赖），作为临时support模块加载；未安装Django。小依赖与许可SHA：[清单](sources/markupsafe-small-dependency-manifest.json)。源码许可均有副本：[MarkupSafe BSD-3-Clause](licenses/markupsafe-instance-LICENSE.txt)、[Django BSD-3-Clause](licenses/django-3.2.20-LICENSE)。

## 负过程：成功回包掩盖了未修复和回退

依次真实调用官方工具：先观察当前SimpleLazyObject错误；重放两次stale old_string尝试，两次均 `FileEditError` 且文件未变；read_file读取当前35–45行；使用当前 `s_str` 文本重新编辑，exact匹配一处成功；最后重复原行为命令，仍得到原生 `_escape_inner` 的SystemError。

新观察给出明确机制：SimpleLazyObject代理会让 `isinstance(value, str)` 为true，但 `type(value) is str` 为false。来源中“先 `if not isinstance(s, str)` 再转换”的修改让代理绕过转换，仍把非真实Unicode对象交给C函数。

此外，该修改把 `__html__()` 结果继续送去escape，违背同一函数docstring中“返回值已经安全”的约定。一次性inline合同案例实际给出：

| 当前实现上的案例 | 语义修正前 | 新语义修正后 |
|---|---|---|
| 普通字符串 `<value>` | 正确转义 | 正确转义 |
| SimpleLazyObject返回 `<value>` | SystemError | 正确转义 |
| Markup已安全字符串 | 被重复转义 | 保持安全HTML |
| 自定义 `__html__` 返回安全HTML | 被重复转义 | 保持安全HTML |
| 数字123 | 正确转字符串 | 正确转字符串 |

因此修正前为2/5，修正后5/5。案例来自issue、当前源码docstring和实际新观察，不来自参考patch或隐藏测试。没有创建测试文件或运行测试套件。

负例：[14消息事务](outputs/markupsafe-recovery/transaction.json)、[真实回包](outputs/markupsafe-recovery/events.json)、[负例状态](outputs/markupsafe-recovery/state-after.py.txt)。它只有局部工具匹配成功，**不得作为成功修复SFT正例**。

## 新继续：依据真实失败修改语义

本轮在负例实际执行后的仓库状态上继续，用新回包和当前源码确定最小修改：只有 `type(s) is str` 才直接进入原生字符串路径；有 `__html__` 的对象直接返回其安全HTML；其余对象先转成真正字符串再escape。通过官方edit_file actual exact匹配一次，再执行完全相同的5案例命令，全部通过。官方submit_patch返回1个生产文件、811字节补丁。

正过程不是旧来源后续动作拼接：它完整包含前14消息的真实负前缀，加上新合同观察、源码读取、语义改动、同合同复验和提交，总24消息。负前缀与正过程前14消息已手动逐项核对完全一致。正例监督仅选择新继续中的读取/语义编辑/复验/提交（assistant ordinal 7/8/9/10）；先前那个语义失败的exact编辑ordinal4仅作上下文，不作为正目标。

正继续：[24消息事务](outputs/markupsafe-semantic-repair/transaction.json)、[真实回包](outputs/markupsafe-semantic-repair/events.json)、[最终生产补丁](outputs/markupsafe-semantic-repair/production.patch)、[最终源码](outputs/markupsafe-semantic-repair/state-after.py.txt)。两阶段可用于研究错误恢复的质量，**目前不是可直接用作DPO的同前缀二选一偏好对**，本轮没有构造这种训练目标。

## 编码与用途

| 记录 | 12B CT v2 输入/监督tokens | 31B CT v2 输入/监督tokens | 用途 |
|---|---:|---:|---|
| 工具已恢复、语义仍失败的负过程 | 4,153 / 295 | 4,153 / 295 | candidate负诊断，禁止参数更新，禁止成功修复SFT |
| 保留负过程后新增语义修正 | 6,968 / 688 | 6,968 / 688 | candidate未分配train/select，禁止参数更新 |

两种模型分别使用真实CT v2模板和官方六工具schema，无截断。计数：[markupsafe-summary.json](markupsafe-summary.json)；编码：[12B候选集合](outputs/encoded-12b-markupsafe-candidates.jsonl)、[31B候选集合](outputs/encoded-31b-markupsafe-candidates.jsonl)。编码中仍保留用于研究的监督位置，但 `meta.data_usage.allow_parameter_update=false`；**不能将这些候选直接交给训练器**。

既有用途已冻结且未改变：click=train容量，pydicom=select诊断；各自已分文件，原两条合并编码只是收集归档。[用途清单](data-usage.json)记录所有sample ID、允许用途与文件指纹。MarkupSafe两个记录共享一个issue和状态链，不能当作两个独立问题或两个独立恢复事件。

边界：5/5是本轮局部CPU合同案例结果，未运行Windows、完整Django、官方Docker或整个仓库套件；所有记录 `task_resolved=null`。本轮证实的是局部程序行为改善及“工具成功≠修复成功”的具体反例，不是比赛分数提升、官方整题resolved或SFT增益。
