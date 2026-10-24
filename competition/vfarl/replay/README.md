# 官方六工具 CPU 重放

`run_official.py` 正常导入原版 `swegemma 0.2.11`、`adk-eval-core 0.1.0` 和官方缓存的 `adk-submission 0.2.13`；没有重写同名工具或伪造工具响应。实际运行环境与 wheel 指纹见 `runtime-manifest.json`。

重放使用官方 `SwegemmaContext`、`EvaluationBudget`、`HarnessLimits`、`SubprocessManager` 和六个官方工具包装函数。任务源码先复制到新目录，移除下载缓存中的 `.git` 历史，重新建立 `_swegemma_baseline`。输入树指纹、真实工具参数与返回、实际预算计数、提交补丁分别保存。

动作由审阅后的 recipe 给出，类型必须是 `human_reviewed_off_policy_demonstration`。每条 `run_command` 需要记录精确命令 SHA-256，防止将未审阅的来源命令直接执行。写入和编辑不得触及官方保护的测试、配置或启动文件。模型没有参与这些示范，不能据此声称 on-policy 数据、旧策略 logprobs、关键动作因果对或恢复后成功。

`--verify` 会真正调用原版 `verify_task`：在另一干净 workspace 解包基线快照，重新应用经过官方保护规则过滤的提交补丁，再运行官方验证和 JUnit 核对。其结果字段明确写作 `resolved_by_local_official_verify_task`。当前 Docker daemon 未运行，因此此处是官方支持的本地 subprocess 后端，**不等价于比赛 Docker/Python/依赖环境**。官方 subprocess 路径会跳过依赖安装函数；必须另外记录并准备上游所需依赖。

`--verify-baseline` 另跑一份无修复的干净验证。停止 sandbox 前仅复制实际 JUnit 报告，再调用原版 stop 清理。`--source-layout src` 在每份 sandbox 自己的 venv 中以 `sys.path.insert(0, workspace/src)` 优先绑定当前源码，并保存三阶段实际 pth 内容。Jinja 首次运行暴露普通 pth 追加路径可能被已安装同名包覆盖，原观察与修正记录均保留。若原训练任务以 F2P/P2P 节点要求未改动的测试文件，`ExternalTrainTask` 显式提供这些节点的文件并集，原工具和 `verify_task` 均不修改。这个数据适配可能暴露来源与比赛契约的冲突，不能用删节点或放松保护规则来掩盖。

已执行结果（均为人工审阅动作示范，无模型推理）：

| 任务 | 原版 Phase B 基线 | 提交后 Phase B | 数据用途 |
| --- | --- | --- | --- |
| h11 协议切换，F5 | 1 failed / 77 passed | 78 passed，原 1 F2P + 75 P2P 核对通过 | 本地官方 subprocess 已验证的冷启动示范 |
| apispec RFC 表示顺序，S1 | 2 failed / 567 passed | 569 passed，原 2 F2P + 567 P2P 核对通过 | 本地官方 subprocess 已验证的冷启动示范 |
| parse 数字基数，S1 | 2 failed / 94 passed / 1 skipped | 同基线；实际提交为 0 字节 | 来源契约隔离，不能计成功 |
| Freezegun 静态方法绑定，R2 | 1 failed / 132 passed / 4 skipped | 133 passed / 4 skipped；必需 maya 节点 skipped | R2 因果机制与 F2P 局部验证，整题未准入 |
| Tenacity 随机等待下界，G | 1 failed / 122 passed / 1 optional skipped | 123 passed / 1 optional skipped，原 1 F2P + 122 P2P 核对通过 | 本地官方 subprocess 已验证的冷启动示范 |
| PySnooper 方法识别与包装，R2 | 8 failed / 70 passed | 78 passed，原 8 F2P + 70 P2P 核对通过 | 本地官方 subprocess 已验证的冷启动示范 |
| Boltons sentinel 复制身份，F2 | 1 failed / 422 passed | 423 passed，原 1 F2P + 411 P2P 核对通过 | 短路径 CPU profile 下的已验证冷启动示范 |
| Tomli 错误位置，S2 | 2 failed / 14 passed | 16 passed，原 2 F2P + 14 P2P 核对通过 | 本地官方 subprocess 已验证的冷启动示范 |
| Dask backend 导入，R5 | 2 skipped，未记录为 failed | pandas 参数真实 passed / cudf skipped；原 matcher 仍拒绝 | R5 局部机制与准确 F2P 节点阳性，整题隔离 |
| Jinja importer 分支，R5 | 1 failed / 15 errors / 860 passed | 876 passed，原 1 F2P + 739 P2P 核对通过，无 skipped | 本地官方 subprocess 已验证的冷启动示范，含真实终止回答 |

分别见 `h11-replay-result.json`、`apispec-replay-result.json`、`parse-replay-result.json`。parse 原 P2P 包含生产文件 `parse.py` 的 doctest；完整目标映射使原版提交保护过滤掉该生产文件，且 README doctest 与官方目标选择不兼容。真实工具观察虽显示局部修复生效，整题仍未通过官方流程。h11 首次采集的相对路径问题及原始错误也保留，未算作任务或模型能力失败。

Freezegun 详见 `freezegun-replay-result.json`。其源码把 maya 依赖限定在 Python < 3.12，而原 P2P 又要求 `test_maya_datetimes` 通过；当前 Python 3.12 本地执行会跳过该节点。原版验证器在 pytest exit 0 后仍正确判定 `resolved=False`。没有删除这一要求或把 skipped 改成 passed。

Tenacity 详见 `tenacity-replay-result.json`。它的唯一跳过项为原始 F2P/P2P 没有要求的可选 Trio 场景；全部 123 个必需节点都真实通过。修复仅把 `random.uniform(0, high)` 的第一个参数改为 `self.min`，没有采纳来源 teacher 附加的负值截断行为。

PySnooper 详见 `pysnooper-replay-result.json`。真实类包装复现原来因协程被错误包装而抛 `NotImplementedError`；修复后类执行、初始化方法注册、协程身份及实际 trace 都有效，全部原始 78 个必需节点通过。

Boltons 详见 `boltons-replay-result.json`。原测试误用 `str.lstrip(BOLTONS_PATH)`，宿主长路径中的字符会把 `fileutils.py` 的前缀一并剥掉。run001/002 的这一无关失败原样保留；run003 采用显式短路径 profile，以每个干净 workspace 的符号链接导入实际 `boltons/__init__.py`，未构造替代模块、跳过真实初始化、修改测试或删 required。三阶段实际包路径、同一源码目标、空文件 SHA、pth 内容及树指纹均保留。该 profile 仍不等价于比赛 Docker。

Tomli 详见 `tomli-replay-result.json`。局部真实观察从文本已有 line/column、结构化属性却全部缺失，变为 pos=13、line=2、column=8；EOF 位置和旧参数弃用警告也实跑验证。恢复结构化构造器后，让现有集中错误辅助函数传递原始文档和位置，无需回退所有 parser raise 点。原 815 份测试/数据资产均按固定上游 SHA 核对。

这两题实际使用 Python 3.12.13、pytest 7.2.0、pytest-cov 4.0.0。Boltons 原 conftest 的旧 path hook 与 pytest 9 不兼容，故按其声明对齐 pytest；此前成功 run 的旧包版本及结果保留，未重跑。每次完整 freeze 见相应 `run-summary.json`，不声称复制了原镜像全部依赖。

Dask 详见 `dask-replay-result.json`。修复后真实 pandas DataFrame 同步计算、entrypoint 扫描和带 backend 上下文的错误都生效；原 required `[pandas]` testcase 也真实 passed。然而原版 JUnit matcher 会将去参数后的同名 `[cudf]` skipped 一并归到 `[pandas]`，因此实际 `resolved=False`。原测试、参数、required 和 matcher 均未修改，也未安装 CUDA/cudf；该题不能计整题 SFT 正例。真实 XML、原 matcher 的跨参数匹配结果与函数指纹均保留。

此 Dask 新 run 的初始 user 内容由原版 `build_agent_prompt` 生成，包含真实预算、环境、提交和终止规则。reviewed recipe 的人工 text-only completion 只在实际 `submit_patch` 返回后记录，单独标记 `human_authored_reviewed_off_policy_recipe`；没有伪称模型生成。旧成功 run 的 raw messages 不补写、不重跑。

Jinja 详见 `jinja-replay-result.json`。修复使非 ZIP PEP 451 loader 使用文件系统根目录，ZIP loader 使用 archive 元数据；真实 custom loader 与 zipimporter 从错误变为分别渲染 BAR/FOO。run001 因已安装 Jinja2 覆盖追加的 src 路径而在第 3 个事件停住，原观察保留且不算模型失败。run002 三阶段明确优先绑定同一源码，实际命令核对 `__file__` 并打印修改前后 SHA，完整 876 个 testcase 通过。只按固定上游恢复缺失的原 `test_loader.py`，30 个其余测试/资源文件与故障分支字节一致；没有删除 740 个原 required。此次按上游声明使用 pytest 8.3.3、trio 0.27.0，完整环境 freeze 已保存。该成功 run 真实记录官方初始 prompt 和提交后的人工 text-only completion。

历史源码绑定复核见 `historical-source-binding-review.json`。旧六个成功中仅 apispec、tomli 用 src 布局，各自当时 freeze 均没有同名安装包；tomli、h11、PySnooper 的实际故障栈直接指向当前 workspace，Boltons 三阶段记录实际导入路径。apispec 和 Tenacity 没有单独打印 `__file__`，其原依赖清单、仅改源码后真实行为及完整 Phase B 的对应变化支持执行当前源码；这项证据边界明确保留，没有补写旧观察或重跑旧成功题。

总览见 `replay-summary.json`：当前 10 个独立任务、83 个主要 run 的真实工具事件，其中 7 条整题本地官方验证成功，G/S/F/R 分别为 1/2/2/2；另保留 h11 首次采集故障的 8 个事件、Boltons 两次环境 profile 和 Jinja 首次路径绑定尝试的 19 个事件。没有生成 on-policy rollout 或同状态偏好对。`supervision-interpretation.json` 明确记录：首次 `run_command` 返回 error 是有意复现仓库缺陷的正确诊断动作，不能因此充作 rejected action 或策略关键错误。

六种真实 Google ADK FunctionTool 声明保存在 `adk-six-tool-declarations.json`，参数名/required 与先前 AST 核对见 `adk-schema-comparison.json`。当前成功示范只需五种工具；没有为覆盖 `write_file` 而增加无效写入。原始 `messages.json` 独立保留，Gemma导出已实际完成，见 [gemma-sft-manifest.json](gemma-sft-manifest.json)。工具arguments的JSON字符串先解析成对象，再交给固定官方模板。

`export_prompts.py`在本运行环境调用原版`build_agent_prompt`，只取来源可见题面/提示和原预算，输出版本化的官方任务prompt。早期raw仅含题面，canonical导出明确记录这项增补，不修改raw观察，不声称模型在增补后prompt下采过样。新版`run_official.py`直接记录官方builder生成的初始prompt。可选workspace tree留空，没有从未来状态反推目录。

`../format_replays.py`用官方31B tokenizer/template与实际声明导出整段审计渲染及逐动作训练输入`sft-actions.jsonl`。逐动作输入使用真实生成入口`add_generation_prompt=True`，覆盖关闭thinking时模板补的空thought控制块；该块只作prompt，loss仅覆盖已审阅的真实assistant动作。当前版本`thinking_budget=0`、无新增system prompt；其他预算/提示需另行版本化。没有生成思考或logprob；新Jinja含实际人工text-only结束回答，旧6条没有这项监督，保持原样。当前导出7题、59个决策（58工具动作+1实际结束回答）、6,466个loss tokens，最长逐动作上下文7,659 tokens；不能把补丁验收成功等同于所有会话终止行为已覆盖。

只有来源明确属于训练池、基线状态核对过、任务机制经人工审阅的任务可调用：

```bash
competition/vfarl/replay/.venv/bin/python competition/vfarl/replay/run_official.py \
  --task /absolute/path/task.json \
  --workspace /absolute/path/source-at-task-base \
  --recipe /absolute/path/reviewed-actions.json \
  --output /absolute/path/new-replay-directory \
  --verify
```

任务原始 `FAIL_TO_PASS` / `PASS_TO_PASS` 必须保留。若节点格式与官方 JUnit 匹配规则不同，应写明映射依据，不能删掉要求取得成功。来源 bug-introducing patch、gold solution、示范动作和验证规格属于不同字段；特别是 SWE-smith-py 的 `patch` 不能当作参考修复。

本程序是生产数据采集入口，不是新增的测试/门控套件。它不自动采样模型、不调用付费 teacher、不启动 GPU。
