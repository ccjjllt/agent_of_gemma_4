# 前 25 题的最小题面澄清提案

2026-10-28。仅处理原 `v4/selection.json` tasks 数组索引 0、1、2、3、5、6、21、22，保持对应身份和原行为范围。原 selection、task、源码 archive、参考 patch、oracle 与已有奖励均未改。这里保存的是待统一版本化的题面提案，不是新动作示范或新训练结果。

`proposal-manifest.json` 逐题绑定原 task SHA、原 statement/prompt SHA、新 statement SHA、实际源码 archive/已有 submitted.patch SHA，以及本轮一次性真实源码行为观察。策略文本只在 `proposed-prompts/`；来源映射和这些审阅证据不应序列化进 policy prompt。

| 原数组索引 | 提案文件 | 修订范围 |
| ---: | --- | --- |
| 0 | [000.txt](proposed-prompts/000.txt) | 原来已经是 IndexError；纠正为缺失自定义错误消息，删除虚假的 KeyError actual。 |
| 1 | [001.txt](proposed-prompts/001.txt) | 提供该版本 FileProcessor 所需的 options，使示例实际到达 blank-line counter，而不是先抛 None.hang_closing。 |
| 2 | [002.txt](proposed-prompts/002.txt) | 改用真实 should_report_error API，保留 Decision 枚举返回和选中/忽略语义。 |
| 3 | [003.txt](proposed-prompts/003.txt) | 分清 was_selected 与 decision_for 的返回层级，用显式 Namespace 替代不存在的 OptionManager 无参/默认值 API。 |
| 5 | [005.txt](proposed-prompts/005.txt) | 先 prepare 实际触发 lazy plugin apply，再观察 reset 后附加状态是否保留。 |
| 6 | [006.txt](proposed-prompts/006.txt) | 原初始读取本来成功；增加已创建 overlays 后的 source 更新，实际触发传播故障。 |
| 21 | [021.txt](proposed-prompts/021.txt) | 最小纠正双锚点示例的 actual：TypeError，而不是无异常返回；期望的 ValueError 契约不变。 |
| 22 | [022.txt](proposed-prompts/022.txt) | 采用 dotted key 后进入另一个 table 的真实触发边界，并保留 invalid TOML 应抛 TOMLDecodeError 的原行为范围。 |

每份提案里的 Python 代码块都在一次性临时生产源码工作区执行。原版触发已确认的错误，应用各题已有真实 `submitted.patch` 后全部代码块完成；没有复制 patch 实现到新题面，也没有把原 oracle 节点或事后机制标签加入策略文本。index 3 另外直接观察隐式默认选择的枚举值，index 22 的两个代码块分别执行，没有因首个 baseline 错误跳过第二个验证。

Dominate、Bottle 和 Tomli 使用真实生产包/模块；Duration 使用完整真实 duration.py 模块及其类。Flake8 使用从对应 snapshot 直接编译的真实类/函数，略去无关依赖导入，示例显式传入 options，未用替代逻辑模拟选择算法或计数。这是方法/组件级机制证据，不冒充完整官方 oracle 或模型修复结果。

临时工作区只写入相应生产源码，参考改动由实际 `git apply submitted.patch` 应用；未提取/新建测试文件。未启动 GPU、远程或模型采样。其余前 25 题没有为了凑数量生成新的题面，已有范围提示仍留在上级逐题审阅中。
