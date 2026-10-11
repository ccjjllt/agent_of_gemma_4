# 外部 SWE 训练材料：先做可追溯的小批量 CPU 筛选

核实日期：2026-10-11。比赛允许公开、平等可访问的外部数据，以本轮已完成的官方 MCP 规则核实为前提；最终部署仍使用规定的 CT v2 模型。这里评估数据材料，不引入其他部署模型。P002 的 9 个 eligible issue 是本地旧环境筛选范围，不是外部训练数据的合法范围上限。

## 三个来源与实际接入顺序

| 来源 | 已核实的许可、规模与格式 | 用途和接入限制 |
|---|---|---|
| **1. SWE-bench/SWE-smith-trajectories，优先 `tool` split** | 数据卡 MIT；`tool` 24,100 行，字段 `instance_id/resolved/model/traj_id/messages/patch`，`messages` 是 JSON 字符串。来源为 SWE-agent 实际执行轨迹，有成功也有失败。最小 tool 分片 `tool-00004-of-00008.parquet` 为 **77,754,558 bytes**。 | 来源 ID 和成功标记齐全，最适合先筛。只选 `resolved=true`，不把所有行或旧卡片“5017条训练轨迹”叙述当作全量成功数。实际样本混有合成变异、`pr_...` 和数字命名任务，不能把整库统一标为天然 issue 或全合成。 |
| **2. SWE-Gym / OpenHands-SFT-Trajectories** | SFT 卡明确 MIT，`train.success.oss` **491 行**，下载 **10,379,409 bytes**，解压数据约31.8 MB；只有 `messages[{role,content}]`。项目任务集为真实 Python issue，任务表带 `instance_id/repo/base_commit/problem_statement/patch/test_patch`。 | 小而真实，适合补充合成任务分布；但精简 SFT 缺任务 ID、成功报告和结构化 tools，不能直接做可靠去重。先恢复与原任务/采样轨迹的精确映射，无法恢复者暂不进入训练。 |
| **3. R2E-Gym/R2EGym-SFT-Trajectories，备选** | 官方代码仓库确认是 editing-agent 的预采集 SFT 轨迹；HF显示 **3,231 行、约56 MB**，仅 `messages`。代码仓库 Apache-2.0；本次未在该数据卡核实独立数据许可，README raw访问失败。 | 不把代码许可自动当作数据许可，也不把全部消息视作已逐条核实成功的修复。须先获得明确数据许可和任务/成功映射；本轮不下载，不优先投入转换成本。 |

一手来源：[SWE-smith 数据卡和字段](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/blob/main/README.md)、[最小分片与 SHA](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/blob/main/data/tool-00004-of-00008.parquet)、[SWE-smith 官方生成流程](https://github.com/SWE-bench/SWE-smith)；[SWE-Gym SFT 元数据](https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories/blob/main/README.md)、[任务元数据](https://huggingface.co/datasets/SWE-Gym/SWE-Gym)、[官方项目](https://github.com/SWE-Gym/SWE-Gym)；[R2E 官方轨迹说明](https://github.com/R2E-Gym/R2E-Gym#-training-open-weight-swe-agents)、[轨迹文件](https://huggingface.co/datasets/R2E-Gym/R2EGym-SFT-Trajectories/tree/main)。

两个容易误用的细节：SWE-smith 当前目录总计约4.22 GB，卡片配置下载量约3.17 GB，且 `tool/xml/ticks` 是不同表示，不能相加当独立问题数；本轮不全量下载。[目录](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/tree/main/data)。SWE-Gym 的原始 [OpenHands-Sampled-Trajectories](https://huggingface.co/datasets/SWE-Gym/OpenHands-Sampled-Trajectories/blob/main/README.md) 有6,055条、约301 MB，具备 `instance_id/resolved/messages/tool_calls/tools/test_result`，更适合恢复来源和行为，但其卡片未声明许可；不能把491条 SFT 的 MIT 自动外推到整个原始采样集。

## CPU 筛选与协议转换边界

先只读来源列，记录 dataset revision、分片 SHA、原始行号、`instance_id/traj_id/model/resolved`。按任务表或官方 repo profile 补齐原始 GitHub repo、完整 base commit、来源许可；SWE-smith 的镜像仓库名和短 commit 不能直接当作规范 repo/完整快照指纹。先分 repo、快照、issue 去重，再算轨迹数；多种表示和同题多次采样都不增加独立 issue 数。

仅对通过来源筛选的行读取 messages。记录角色和调用格式、工具名/参数、调用与回包 ID 是否齐全、是否有失败调用→同文件/操作的纠正→成功回包。`resolved=true` 是来源的任务成功标记，不是本项目重新执行验证；某一步 exit 0 不代表任务解决。失败动作留作上下文，只监督可核实的成功恢复/编辑动作。judge 的 `test_result`、参考 `patch/test_patch` 留在离线元数据，不伪装成 Agent 当时看到的观察。

工具转换必须保持参数和观察语义：可检查 shell 命令是否能等价表示为 `run_command`；查看/替换/插入/撤销工具不能简单改名成官方 `read_file/edit_file`。无法保持语义的样本记录为待适配，不编造官方工具回包。本轮只做格式抽样，不写转换器。

长度分两层记录：先按 CT v2 tokenizer 统计原始完整对话/完整因果前缀的文本 token；只有未来完成语义转换并用正式 Gemma tool chat template + 官方工具 schema 渲染后，才报告真正的训练长度。不得把字符数、其他模型 tokenizer 或去掉历史的局部窗口当作≤8K证据；分别统计4K/8K/16K完整交易数、独立问题数、监督 token 密度和恢复事件数。当前尚不能承诺这些来源能解决≤8K覆盖问题。

## 排除重合与下一轮材料

初批直接排除比赛涉及的全部原始 repo：`fastapi/fastapi`、`psf/requests`、`Textualize/rich`、`encode/httpx`（以 `task_splits.json` 元数据集合为准），包括可识别 fork/镜像别名；不读取这些 repo 的外部 messages/patch。这比逐题比较更稳妥，且不需要打开 heldout 或预留八题内容。其余数据仍按规范 repo、完整 commit、issue/PR ID、来源内容 SHA 去重；映射不明确者不作为独立来源。训练和后续外部 select 按 repo/快照分组，不能随机拆同题轨迹。

实际优先级：先用 SWE-smith 一个小分片验证“有多少独立 repo 的成功完整前缀可用”，再决定是否扩大；并行核实 SWE-Gym 491条的来源映射。后续交付应是带来源指纹的候选清单和真实长度分布，而不是先假定有足够8K数据再启动SFT。来源许可和对应代码片段原许可均保留归档，数据集 MIT 标签不代替逐仓库许可记录。

## 已完成的单分片 CPU 抽样

按追加授权下载一个分片并核对公开 SHA，没有下载模型或运行 GPU。固定 revision 为 `08e109b4a59eaeebf80e4675cd125d42e7ac99a4`，文件 **77,754,558 bytes**，SHA256 **`e41ffbd3f9d1571917ac5b9f4d0108ded4799d9776f597a054203467135a5715`**；[固定下载 URL](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/resolve/08e109b4a59eaeebf80e4675cd125d42e7ac99a4/data/tool-00004-of-00008.parquet)。同 revision 的 MIT 数据卡已本地归档。现有环境无 Parquet reader，仅向 `/tmp/p005-parquet-reader` 装了 DuckDB 1.4.3（13.1 MiB），未改变项目训练环境。

先投影读取 `instance_id/resolved/model/traj_id` 和行号：**3,012 行，110 个解析来源 repo，1,255 行 resolved=true，覆盖105个成功来源 repo**。该分片没有比赛四个 repo 或同名仓库命中；未读取任何比赛任务内容，也未读取分片 `patch` 列。以固定种子20261011的哈希顺序选20个不同 repo、每 repo 一条成功轨迹，选择发生在读取 messages 和计算长度之前。随后仅反序列化这20条 messages，并原样保存；原始完整前缀均保留。repo 来自 instance_id，完整 commit/逐仓库许可仍待官方 profile 补齐。

实测格式不是“tool split 全部结构化”：**7条**具有 OpenAI 风格 `tool_calls`，工具为 `bash/str_replace_editor/submit`，观察用 `tool_call_ids` 关联；**13条**是 `system/user/assistant` 文本对话，动作包含 `<function>/<parameter>` 标记，未转换。7条结构化调用的参数都是 JSON 字符串，直接套 Gemma 模板会报参数需 mapping；只做 `json.loads(arguments)` 后 **7/7 渲染成功**，未改工具名、观察或原始存档。所检查的结构化观察 ID 均能连到先前调用。

| CPU 长度口径 | ≤4K | ≤8K | ≤16K |
|---|---:|---:|---:|
| 20条完整源对话，经 CT v2 tokenizer / 原生模板渲染（7条仅反序列化参数；未供 tools schema） | 0 | **1** | **6** |
| 20条对话中，从起点到每个 assistant 步的完整原始 JSON 前缀（含重复元数据） | 42 | 74 | 205 |

完整源对话 token 最小 **7,341**（`pallets/click`）、中位 **22,760.5**、最大 **56,600**。第一行不是官方六工具协议下的正式训练长度；第二行是序列化代理口径，既非独立任务数，也不代表这些步骤都是成功编辑。不能把74个≤8K前缀当作74条可直接训练样本。当前样本的来源已比本地9题广，但短完整任务仍少。

在 `pallets/markupsafe` 和 `pydicom/pydicom` 的结构化观察中找到 **5次**明确的编辑匹配失败（`No replacement was performed`，含多处匹配）；它们是恢复行为候选，尚未核实逐事件“失败→同操作纠正→成功”，不报告已得到5个独立恢复样本。20条来源 `resolved=true` 仍只是上游标签，本轮未重放验证。

下一步应先处理这7条结构化来源的真实动作语义与身份/许可补全，优先确认 `pallets/click` 的完整短轨迹，以及两个编辑失败仓库的完整恢复前缀；不因模板可渲染就宣布已经适配训练。原始样本和分片保存在 `training/artifacts/external/swe-smith/`；固定来源、选样行号、每条长度和存档指纹见 [swe-smith-inventory.json](swe-smith-inventory.json)。本轮没有创建转换器、训练标签、测试文件或 GPU 实验。

## 7条结构化轨迹的语义接入复核（同日，只读）

已核对 [canonicalize.py](../../../training/data/canonicalize.py) 和 [本地官方工具 schema](../../../training/configs/harness_tools.json)，并读取现有 `swegemma-0.2.11` wheel 的 `tools/{execution,workspace}.py`、`sandbox/subprocess.py`、`edit.py`（wheel SHA `63f054a3d42eb6c582e2b42195f96b22f4b9ee69d6e42ae62cedeb4dd6cf5480`）。现有 importer 不设别名，三个源工具均不兼容；7条原始输入首先因 list content 被拒，不能把“模板已渲染”写成“importer可直接接入”。

**可无损处理的是表示层**：参数 JSON 字符串→对象；样本中166个 content list 全为单个 text block，可保留原文展平成字符串；159个观察的 `tool_call_ids` 均为单元素且能连到先前调用，可显式对应单个 `tool_call_id`。原始额外 `thought/action` 元数据须归档，不能把已在 content 中出现的 thought 再监督一遍。这些处理不改变工具语义，也不产生官方回包。

| 源工具 | 与官方工具的边界 |
|---|---|
| `bash(command)` | 参数形状类似 `run_command(command)`，但 source 路径是 `/testbed`。官方 subprocess 每次新起 `/bin/bash -c`、`cwd=workspace` 和净化 env，因此前次 `cd/export/alias` 不成为后次 shell 状态。样本62次 bash 中30次显式带 `cd`，未见 export/source/alias/后台进程；这不足以断言源 bash 本身无持久 shell。另有15次 pip install，依赖安装和文件修改确实影响后续调用；官方无网络、超时/截断及 JSON stdout/stderr/exit_code 反馈也不同。不能只改工具名或全局替换 `/testbed` 字符串。 |
| `str_replace_editor(command,path,...)` | `str_replace` 的 `old_str/new_str` 看似可映射到 `edit_file.old_string/new_string`，但官方实现实际委托多级 exact/flexible/regex 等匹配，并默认 `allow_multiple=false`；源匹配失败不保证官方也失败。`view` 的带行号文本与官方范围/截断 JSON 不同；`create` 是否拒绝已存在文件，不能从样本推定与可覆写的 `write_file` 等价。本次只有 view/create/str_replace，不泛化到 insert/undo。需要逐动作重放或明确的表示对照。 |
| `submit({})` | 只在参数形状上像 `submit_patch({})`。官方捕获相对 baseline 的 diff 并返回 JSON；7条源轨迹都以 assistant 的 submit 调用结束，没有对应 tool 观察，而 importer 要求每个调用有回包。必须保留该事实，不能伪造提交成功回包填平序列。 |

下面长度均从原始对话起点完整保留历史，只反序列化 arguments 后用同一 CT v2 模板计数，**没有官方 tools schema，仍不是正式训练长度**。消息索引从0开始；“后续成功”仅指同路径 str_replace 的源观察明确写出文件已编辑，不是官方重放或新增 resolved 判分。

| 来源分片行 / 错误观察索引 | 完整前缀到错误观察 | 首个后续同路径成功编辑观察 / 完整前缀 |
|---|---:|---|
| markupsafe 行289 / 31 | 15,537 | 37 / 16,775 |
| markupsafe 行289 / 33 | 15,915 | 37 / 16,775（与上一行共享一次后续成功） |
| pydicom 行574 / 35 | 10,158 | 39 / 10,734 |
| pydicom 行574 / 45 | 11,376 | 47 / 11,796 |
| pydicom 行574 / 73 | 15,644 | 75 / 16,052 |

五次错误全部属于已抽中的7条，来自两个 issue；**没有≤8K的完整错误/恢复前缀**。三个 pydicom 后续成功前缀在源表示下≤16K，最后一个余量很小；markupsafe 两次错误最终汇到同一次成功，不能重复算独立恢复。`verified_recovery_events` 继续为 null；下一步优先判断这两个真实失败链能否在官方动作语义下重现，而不是截短历史凑8K。本次未新增样本、转换文件、报告、测试或GPU工作。
