# P006：三个外部仓库的真实 CPU 工具重放

2026-10-13，北京时间。当前已覆盖 **3 个不同仓库、3 个来源问题、2 个独立工具恢复事件**。首批click/pydicom见下；同日追加MarkupSafe保留“工具恢复成功但语义仍错”的负过程，以及基于新观察产生的语义修正过程，二者共用一个issue/event，不能重复计数。[MarkupSafe追加结果](markupsafe-report.md)：5个inline合同案例从2/5到5/5，完整新继续6,968输入/688监督tokens。没有 GPU、教师调用、模型权重下载、现有测试套件运行或新增测试文件；未读取比赛 heldout 或预留评测题。所有文件写在本研究目录，临时源码工作区在 `/tmp`，运行后已清理。

**用途已冻结（同日追加授权）**：click只用于train容量/导出；pydicom只用于select诊断，禁止更新参数。原 `encoded-*-ct-v2.jsonl` 是两条收集归档，不能整个传给训练器。正式分开的文件为[12B train容量](outputs/encoded-12b-train-capacity.jsonl)、[12B select诊断](outputs/encoded-12b-select-diagnostic.jsonl)，31B有同名对应文件；各文件meta与[用途清单](data-usage.json)一致。本轮新增第三仓库的恢复样本先标candidate，不自动划入训练或选择集。

## 首批实际结果

| 来源与范围 | 本轮实际观察 | 12B CT v2 输入 / 监督 tokens | 31B CT v2 输入 / 监督 tokens |
|---|---|---:|---:|
| click `pr_2555`：从固定 buggy 源码开始的新完整任务事务 | 缺失 stdout、stderr 两种案例修改前均 AttributeError，修改后无异常；显式 StringIO 输出前后正确；官方 submit_patch 捕获一个生产文件修改 | 3,368 / 519 | 3,368 / 519 |
| pydicom `pr_2079`：明确给定中间状态的局部恢复子任务 | 官方 edit_file 拒绝两处重复匹配且文件不变 → 实际 read_file → 增加已观察到的 docstring 上下文 → exact 一处编辑成功 → 实际回读 | 2,055 / 226 | 2,055 / 226 |

两条合计745个监督tokens，监督密度分别15.41%和11.00%。完整保留**新事务**的system、issue、当前状态及每次实际工具观察，没有截断、没有复制旧未来。两种模板指纹不同，但这两条纯文本样本的input_ids与labels实际完全一致；不能外推所有输入都相同。数据及证据：[transactions.jsonl](outputs/transactions.jsonl)、[12B编码](outputs/encoded-12b-ct-v2.jsonl)、[31B编码](outputs/encoded-31b-ct-v2.jsonl)、[计数与手动核对](length-and-replay-summary.json)。

**两条的 `task_resolved` 均为 null。** click只验证issue所述的Python流对象行为，未在Windows/pythonw执行，也未运行官方整题测试；pydicom仅证明匹配错误可恢复，未运行mypy或确认其最终类型方案正确。上游resolved=true仍只作为来源标签。不能把这两条称为两次官方benchmark修复成功或训练收益。

## 必须修正的来源理解：profile commit 不是任务初始状态

原P005通过官方profile把instance前缀映射到原始仓库commit，这一步适合身份核对，却不足以恢复PR回归任务。实际取回源码后发现：

- click的profile commit `fde47b4b4f978f179b9dff34583cb2b99021f482` 已含缺失控制台的None保护。
- pydicom的profile commit `7d361b3d764dbbb1f8ad7af015e80ce96f6bf286` 已含后续类型改动，和轨迹最初读到的代码不同。

官方SWE-smith `RepoProfile.get_container()` 在环境内执行 **`git checkout instance_id`**，镜像仓库由 `swesmith/<owner>__<repo>.<profile_commit_prefix>` 命名。本轮因此通过官方公开镜像解析并固定实际任务分支：

| 实例 | 真实 buggy commit |
|---|---|
| `pallets__click.fde47b4b.pr_2555` | `8154a324c30a33f618b7226cc5be18f54bacac99` |
| `pydicom__pydicom.7d361b3d.pr_2079` | `0f0e22c5fcc977fc211bd08854308e3b2aee459c` |

click的目标源码SHA256为 `45de8f7dee99c7e86844d45c3191bda72d4920505664108edef0733c3d950897`；pydicom为 `eae6f36ff304aeb7a5616318707fb4045a5bf9a910c21fc5d46257396d5da4ed`。本轮只提取click生产包及许可证、pydicom目标生产文件及许可证，不提取或执行仓库测试。提交基线是这些生产源码子集，**不是声称已构建完整官方容器**。

来源：[SWE-smith官方profile实现](https://github.com/SWE-bench/SWE-smith/blob/main/swesmith/profiles/base.py)、[click固定镜像](https://github.com/swesmith/pallets__click.fde47b4b/tree/8154a324c30a33f618b7226cc5be18f54bacac99)、[pydicom固定镜像](https://github.com/swesmith/pydicom__pydicom.7d361b3d/tree/0f0e22c5fcc977fc211bd08854308e3b2aee459c)。快照：[loader源码](sources/swesmith-base.py)、[固定来源清单](sources/pinned-instance-manifest.json)、[先前分支取样SHA](sources/instance-source-manifest.json)。后续扩量应以buggy instance commit分组，profile commit作为仓库环境来源字段保留。

## click：不是把旧轨迹裁短，而是重新执行短事务

旧来源虽然只有8个assistant步骤、上游resolved=true，其自建 `program.py` 在修改前后都只是打印Hello world，并未触发Windows缺失控制台的故障。本轮不沿用“旧程序已验证修复”的表述，也不保存那个非生产脚本。

新事务只用来源公开issue中的需求与定位线索，依次实际执行：

1. 官方read_file读取 `src/click/utils.py:222–315`。
2. 一次性inline Python将sys.stdout/sys.stderr临时置None，捕获并还原标准流；实际得到两次AttributeError，同时确认显式StringIO输出正常。命令故意以失败退出，真实回包是CommandError，保留为上下文。
3. 使用来源中的局部修复意图与原old/new文本，调用**官方edit_file实现**。实际exact匹配一处、写回生产文件并返回diff。
4. 执行完全相同的inline命令，三案例通过；无测试文件或持久复现脚本。
5. 官方submit_patch从真实git工作树抓取并过滤补丁；一个文件、增加None保护四行。

本轮没有把 `str_replace_editor` 改名后复制其回包，而是在不同工具语义下真正执行动作。源末尾submit无回包的问题，通过新事务实际调用submit_patch得到真实回包解决。新assistant文字是简短程序化示范说明，不伪装成源模型thinking，也未调用教师模型。标签只监督成功编辑、重新验证和提交三步；旧失败复现仅作上下文。

证据：[实际事件](outputs/click-echo/events.json)、[生产补丁](outputs/click-echo/production.patch)、[完整事务](outputs/click-echo/transaction.json)。`origin_kind=cpu_replayed_expert_demonstration` 明确标注这是来源驱动的新演示，不是源Claude完整episode的无损格式转换。编码器的synthetic=false表示真实源码/执行数据，与toy占位样本区分，并不宣称自然采样策略数据。

## pydicom：恢复状态可重建，局部含义必须保留

在固定buggy `codedict.py` 上，本轮只重放到首次编辑匹配错误之前的6次源码替换（来源消息14、16、18、26、30、32）；每次均调用官方edit_file并成功。没有执行来源里的pip安装、mypy或任意bash。中间state及所有真实准备回包存档，原轨迹当时可见的70–75行与恢复状态逐行一致。

局部子任务输入明确包含：原issue、这是已有CodeLike改动的中间状态、当前生产diff、当前要求修改 `_CID_Dict.__getattr__` 的注解。它不是假装从原始issue自主定位到这个文件；文件和操作在来源失败动作前就已确定，目标修复未从未来gold patch推导。随后四次新工具调用组成完整局部因果链：

- 短old_string真实匹配两处，官方工具拒绝，文件SHA不变。
- 读取当前70–82行，得到方法docstring和函数体。
- old_string加入刚观察到的docstring，官方exact成功一处。
- 回读同一行段，确认目标注解文本改变；离线ast.parse通过，只说明语法可解析。

失败动作ordinal0明确不监督；监督读取、纠正、回读ordinal1/2/3。**不得把2,055 tokens写成原始10,734-token恢复前缀被“无损压到2K”**。缩短来自明确改定义为当前状态局部子任务，省去的历史变成已物化状态和当前diff；不能据此评价原模型整题导航能力。

证据：[中间状态准备](outputs/pydicom-recovery/state-preparation.json)、[恢复前源码](outputs/pydicom-recovery/state-before.py.txt)、[恢复后源码](outputs/pydicom-recovery/state-after.py.txt)、[实际错误/纠正回包](outputs/pydicom-recovery/events.json)、[完整局部事务](outputs/pydicom-recovery/transaction.json)。其production.patch包含之前6次准备改动；本次恢复新增部分以events中的成功编辑diff为准。

## 工具实现、部署模板和许可

数据生产器：[replay.py](replay.py)。它加载v31原样的 `tools/base.py`、`tools/workspace.py`、`tools/execution.py`、`swegemma/edit.py` 和官方 `adk_eval_core.editing`。为避免安装整个ADK/evaluator，仅对submit_patch依赖的两个保护文件函数与路径解析函数加载原AST，未改函数体。代码指纹：[implementation-manifest.json](implementation-manifest.json)。

执行边界是macOS Python3.11、临时目录和受限文件映射；不是Linux Docker仿真。read/edit/write/response与patch过滤调用原实现，文件传输由本地适配器负责；命令新起 `/bin/bash -c`、清理环境、固定超时。没有执行存档中的shell字符串，也没有未知安装脚本。平台差异仍可能影响完整任务，故未报告官方验证成功。

两次编码均复用现有 `training/encoding/gemma_tool_chat.py`，官方六工具schema SHA `4645523d1a8d46e334d04cb8fcce2053616a2dd9fcf330136da78b1723a73100`，thinking on、thought-loss mask、完整上下文；没有真实reasoning_content被补造。模型模板分别为：

- 31B CT v2：16,934 bytes，SHA `94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413`，资产在本目录[tokenizer](tokenizer/manifest.json)。
- 12B CT v2：17,466 bytes，SHA `36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0`，复用另一CPU工作取得的 `training/artifacts/tokenizers/gemma4-12b-ct-v2`。
- 两份tokenizer.json实际SHA同为 `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`。编码meta保存完整tokenizer/template/tools指纹；`training_eligible=true`是编码管线字段，不代表整题成功或已证明训练价值。

来源轨迹固定dataset revision `08e109b4a59eaeebf80e4675cd125d42e7ac99a4`，原messages SHA与P005清单重新手动对上。数据卡MIT归档在[来源卡](licenses/SWE-smith-trajectories-README.pinned.md)。click当前实例LICENSE为BSD-3-Clause；pydicom目标文件首行指向BSD根LICENSE，GDCM字典例外存在于根许可但本轮未触及其字典文件。许可证副本：[click](licenses/click-instance-LICENSE.txt)、[pydicom](licenses/pydicom-instance-LICENSE)。官方wheel的源码许可见各wheel内METADATA，不将外部源码改许可。

## 失败信息与研究含义

真实遇到的资料获取失败：HF datasets-server按两个instance过滤时，click返回HTTP500，pydicom请求超时；没有把空返回当成没有数据，随后按官方loader找到镜像分支并固定commit。首轮本地工具接线运行在“预期AttributeError”的断言处终止，第一轮原始回包未归档，因此不对原因作更强推断；显式把运行解释器目录加入PATH后，新运行产生上面完整可核查回包。

首批的实质进展是：修正SWE-smith任务初始状态认定，取得一个真实故障前后变化和一个真实拒绝后恢复，且得到可直接喂入现有编码路径的外部样本。追加MarkupSafe后仍只有3个源问题，规模远不足以宣称跨仓库SFT收益。click已冻结为一次真实训练容量/导出样本，pydicom留作select工具恢复诊断，MarkupSafe正负过程尚未分配train/select且禁止参数更新；不要用同一条样本同时训练和声称独立收益。
