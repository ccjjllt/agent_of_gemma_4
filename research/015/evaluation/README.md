# P015 天然 PR 开发面板与 CPU 执行证据

2026-10-22。根负责人运行的 CPU v2 `zhuating/p015-natural-pr-panel-cpu/2` 已 COMPLETE，实际 Linux / Python **3.12.13** 记录覆盖全部 18 题：12 道最终 dev 和三道 train 通过前后区分及既有回归，voluptuous 三道候选在 buggy/reference 的 F2P/P2P 四态均 collection 失败（exit 4）。根已在模型结果出现前按预冻规则替换整个 voluptuous 仓，并冻结最终 12dev＋3train。随后首轮 base 模型修复已完成，下节仅核对其结果元数据与计数，不把 dev 解题内容转作训练。dev 不产生训练数据，合并运行的三道 train 保留原训练用途。

## 来源与用途

`natural-pr-freeze.json` 在读取内容前按身份元数据冻结四仓各三题及 faker 替补顺序。来源是 SWE-Lego Real Data 的 `resolved` split，固定 revision `879c96b91dfa321b4d1c621d149a9200f8a1d118`。这是来源本身的成功轨迹筛选集，不能当作无偏抽样全部真实 issue；本次不使用轨迹里的解题动作。取行用 HF `/rows` 的精确 `offset=<frozen row_index>&length=1`，每次核对 instance_id / base_commit；数据集 revision 在获取前后均一致。

dev 候选按仓库级排除了历史 71 个训练仓、旧开发/诊断仓及官方四仓；只核对划分身份元数据，没有读取 heldout 题内容。data 负责人已将全部 dev 候选仓及家族从 P015 训练排除；单独合并的三道历史 train 身份保持训练用途。旧 `candidate-identity-freeze.json` 保留为合成 SWE-smith 候选来源记录，本轮新面板在任何模型运行前改用天然 PR。已缓存的 SWE-smith 00/01/02 分片大小及 PAR1 footer 一致；未重复下载。

## 首轮 31B base 实际基线

真实运行 `run-015-31b-base-and-train-sampling-v1-20261022T050436Z-7bae0a25` 的 `summary.json`、`analysis-rows.json`、`experiment-manifest.json` 已归档于 `../inference/evidence-v1/`；完整输出位于 `../inference/output-v1/run-015-31b-base-and-train-sampling-v1-20261022T050436Z-7bae0a25/`。本次只读取逐题结果、`context_metrics.json` 与服务元数据来核对计数和时限；未读取 dev 补丁、对话或推理内容用于训练。

15/15 次均取得有效 host 判定，worker 全部 exit 0，没有将环境错误从分母剔除。**dev 为 base 5/12，train 为 base 1/3**，两者分别统计。dev 已修复身份是 arrow-986、arrow-1035、arrow-906、faker-2190、faker-1869；三个 sqlglot、三个 pyupgrade 及 faker-1741 的 F2P 仍未通过。全部 12 道 dev 的 source P2P 判定均通过；沿用 CPU 中 source skip/xfail 的既有范围，不能因此声称覆盖全部潜在回归。

| 冻结 dev 顺序 | F2P＋P2P 结果 | Agent 实测秒数 | 工具 / LLM 次数 | 结束证据 |
|---|---|---:|---:|---|
| sqlglot-1001 | 未修复；P2P 通过 | 300.03 | 21 / 21 | 5 分钟会话超时 |
| arrow-986 | 通过 | 196.61 | 28 / 30 | 正常结束 |
| pyupgrade-530 | 未修复；P2P 通过 | 300.02 | 6 / 6 | 5 分钟会话超时 |
| faker-2190 | 通过 | 92.66 | 13 / 15 | 正常结束 |
| sqlglot-2047 | 未修复；P2P 通过 | 291.89 | 28 / 32 | 已提交，未报会话超时；F2P 失败 |
| arrow-1035 | 通过 | 174.29 | 28 / 33 | 正常结束 |
| pyupgrade-133 | 未修复；P2P 通过 | 300.01 | 16 / 16 | 5 分钟会话超时 |
| faker-1741 | 未修复；P2P 通过 | 300.01 | 7 / 7 | 5 分钟会话超时 |
| sqlglot-3401 | 未修复；P2P 通过 | 300.02 | 6 / 6 | 5 分钟会话超时 |
| arrow-906 | 通过 | 83.80 | 12 / 14 | 正常结束 |
| pyupgrade-656 | 未修复；P2P 通过 | 300.01 | 9 / 9 | 5 分钟会话超时 |
| faker-1869 | 通过 | 216.85 | 22 / 24 | 正常结束 |

七道 dev 失败中，六道有明确 `Agent exceeded session timeout (5.0 min)`，工具数仅 6–21，不能将这些失败统称为 28 次工具预算耗尽。另一道 sqlglot-2047 使用 28 次工具、32 次 LLM，约 291.89 秒结束，`agent_error=null` 且 `patch_submitted=true`，最终 F2P 失败；这里能证明的是预算计数和未修复，不能仅据此确定工具上限是否是唯一失败原因。dev 成功题 Agent 用时 83.80–216.85 秒，失败题 291.89–300.03 秒。上述是元数据层面的失败分类，不是源码遗漏或推理错误的语义诊断。

train 中 boltons-298 在约 206.76 秒、21 次工具 / 23 次 LLM 后通过 F2P/P2P。xmltodict-81 与 boltons-338 均约 300 秒会话超时；前者 19/20 次工具/LLM、已提交候选，但 F2P/P2P 都失败；后者 19/19 次、未提交，F2P 失败、P2P 通过。**1/3 只是训练仓采样成功率**，不加入 dev 成绩，且修复通过不能代替 raw 轨迹完整性与可训练性审计。

本轮复用冻结 runtime-v2 base Agent、官方 CT v2 31B，单题 5 分钟、28 次工具、40 turns、60 秒命令上限，研究四并发。coordinator 实测墙钟 1495.95 秒，其中服务启动 413.79 秒；这不是完整 Kaggle 配额账单，也不能据四并发墙钟承诺正式串行时限。

服务实际以 `enable_lora=true` 挂载容量诊断 adapter，但逐题修复请求仍走 **base**，未使用 adapter。独立诊断仅为 `p015_capacity_diag` 的一次 8-token hello 请求：实际 CT HTTP 200，返回 model alias 与请求相同，服务记录只有一次成功启动，没有发生 fallback；其输出不计入上述修复成绩。它证实导出 adapter 的服务兼容性，不证实任务后训练增益。后续权重 A/B 必须匹配这次 `enable_lora`、挂载 adapter、rank、并发及服务设置。

本轮没有任务 LoRA 修复对照，也没有新增正式提交分数。**既有正式成绩仍是 0.15**（`57471497`，原始回执在 `../../014/submission/runtime-v2/result-20261022.json`）；`5/12` 是本次训练外开发面板的 base 结果，`1/3` 是 train 采样结果，三者不能相互替代，不能宣称 LoRA 或正式榜单提分。

## Linux 真实结果与最终选择

证据为 `cpu-prevalidation/actual-results-from-live.json` 的 18 个实际 records，以及已下载的原生 `cpu-prevalidation/output-v2/p015-panel-prevalidation/prevalidation-results.json`；两份 records 已逐行核对一致。live 解析文件 SHA256 为 `40864f428a63681f5cdcbe362147659ece05df1588c34d270a5ebf7f45857e3f`，与 `final-selection.json` 的证据指纹一致。每题记录包括真实 base Git HEAD、当前 checkout 的模块 origin、四态命令、退出码与输出 tail。

- **12dev**：sqlglot、arrow、pyupgrade、faker 各三题。每题 buggy F2P 均有真实失败，reference F2P 通过，buggy/reference P2P 均 exit 0。对应数量与下方本机表中这四仓相同；arrow-1035 仍为 47 pass＋1 skip，pyupgrade-133 仍为 327 pass＋3 xfail，不能把它们算成全部节点 pass。pyupgrade-133 的 F2P 依然是一个原失败、一个原已通过，参考两者通过。
- **三道 train**：xmltodict-81 的两个既有 unittest F2P 在 buggy 出错、参考通过，16 个 P2P 两态通过；boltons-338 的一个 F2P 失败后参考通过，25 个 P2P 两态通过；boltons-298 的五个 F2P 失败后参考通过，119 个 P2P 两态通过。这三题保持 train，仅用于训练仓采样，不计入开发题修复分母。
- **voluptuous-155/231/380**：三题所有四态退出码都是 4，tail 指向对应 `tests.py` 的 collection ERROR，没有形成 Linux 上有效的代码缺陷前后区分。最初 live tail 被警告信息占据，缺少首个异常。随后取回的 `cpu-prevalidation/output-v2/p015-panel-prevalidation/logs/dev/alecthomas__voluptuous-155/buggy-f2p.log` 直接证明 **155** 的导入链为 `tests.py → nose.tools → nose.importer → from imp import ...`，最终 `ModuleNotFoundError: No module named 'imp'`。这条精确根因只归于已取回完整日志的 155；231/380 仍仅确认四态 collection 失败，不推断异常完全相同。

这与本机 Python 3.9.6 的主要差异是：本机 voluptuous 三题都能完成真实 F2P 区分及 P2P；Linux 3.12.13 环境中四态都无法完成 collection。按读取模型结果之前写下的整仓替补规则，根将三个 faker 整仓替换 voluptuous，不按模型输赢挑选。`final-selection.json` 冻结于 `2026-10-22T05:01:17.794944+00:00`，`model_results_seen=false`，恰好包含全部 15 个 Linux `valid_prepost=true` 身份，未混入其余三题；其中 train 三题前置，dev 按四仓轮转，各三题。CPU reference 通过是环境及判据证据，不是自主解题成功。

## 本机 Python 3.9.6 对照记录

所有行均使用来源已有 `test_patch` 与 F2P/P2P；没有编写新的测试文件、专用测试套件或自造断言。source criterion 只临时应用于 host workspace 中的上游既有测试文件。下表的 P2P 在 buggy/reference 两态均没有失败。原始日志及模块来源在 `evidence/<iid>/`。

| iid 后缀（完整身份见冻结文件） | buggy F2P | reference F2P | 既有 P2P 每态 |
|---|---:|---:|---|
| sqlglot-1001 | 1 fail | 1 pass | 16 pass |
| sqlglot-2047 | 1 fail | 1 pass | 37 pass |
| sqlglot-3401 | 1 fail | 1 pass | 41 pass |
| arrow-986 | 2 fail | 2 pass | 208 pass |
| arrow-1035 | 1 fail | 1 pass | 47 pass，1 source skip |
| arrow-906 | 1 fail | 1 pass | 42 pass |
| voluptuous-155 | 2 fail | 2 pass | 16 pass |
| voluptuous-231 | 1 fail | 1 pass | 60 pass |
| voluptuous-380 | 2 fail | 2 pass | 106 pass |
| pyupgrade-530 | 8 fail | 8 pass | 47 pass |
| pyupgrade-133 | 1 fail，1 pass | 2 pass | 327 pass，3 source xfail |
| pyupgrade-656 | 1 fail | 1 pass | 28 pass |
| faker-2190（替补） | 1 fail | 1 pass | 134 pass |
| faker-1741（替补） | 1 fail | 1 pass | 61 pass |
| faker-1869（替补） | 1 fail | 1 pass | 331 pass |

pyupgrade 来源日志解析把含空格的参数 node ID 截断。例如 `test_fix_py3x_only_code[from` 不能直接执行。读取任何模型结果前，以原节点作为前缀，在当前 upstream patched test collection 中确定性展开完整节点；保留 `source_f2p_nodes` / `source_p2p_nodes` 和 `source-node-expansion.json`。保留前缀可能匹配到的所有节点，P2P 去掉与 F2P 重叠节点。133 的展开 F2P 中一个原本已通过，因此只能报告 1 个真实失败见证。原先节点找不到的执行不计为代码缺陷证据。

faker 的上游 pytest fixture 需要显式加载 `faker.contrib.pytest.plugin`；关闭 autoload 而不加载时出现递归 fixture 错误，该环境错误已修复后重跑。预序第三题 faker-318 的 source P2P `test_documentor` 在 buggy 已失败，另用已移除的 `inspect.getargspec`，故在任何模型运行前按身份 hash 序下一题 1869 替换；318 的原始失败日志保留。voluptuous 的老版本涉及 `collections.Mapping` / nose 的 `imp`，本机 3.9 可以执行。实际 Linux 结果已按上述规则导致整仓替换；155 的完整 Linux 日志现已确认 nose 导入已移除的 imp 导致 collection 失败；231/380 的精确首个异常尚未从现有 tail 定位，不把三题归为同一个已证实异常。

## 材料契约

- `material/agent-tasks.json`：原始 primary 候选 12 行列表，每行只有 `iid, repo, base_commit, issue`。它保留最初的 voluptuous 候选，不能直接当最终选择；模型执行使用 `final-selection.json` 对 18 题合并输入清单作出的明确选择。
- `material/snapshots/<iid>.tar.gz`：顶层 `repo/`，真实 buggy base_commit 的独立浅 git checkout。实际核对 HEAD、无 alternates、git fsck。快照在应用 criterion/reference 之前生成，不包含新增判据、参考补丁或轨迹。
- `material/host-criteria.json`：`{schema_version:1,tasks:[...]}`，包含 source/展开后的 `f2p_nodes,p2p_nodes,module,timeout,reference_patch_path,criterion_patch_path,environment`。参考及 criterion 不进入 Agent。
- `material/substitute-agent-tasks.json` 与 `substitute-host-criteria.json`：三道已冻结 faker 替补，CPU 材料作业也实际执行它们。
- `material/environments/<iid>/{requirements.txt,wheelhouse/}`：每题独立目录，固定 pip freeze；CPython 3.12 / manylinux x86_64 wheels 已下载，纯 Python wheel 同样保留。不安装全局环境，不混装不同项目。所有任务 `install_editable=false`；从 workspace 直接导入。

上游判据命令模式（已在本机执行，shell 临时变量以真实路径替换）：

```sh
PYTHONPATH="$task_repo" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$task_python" -m pytest -q -p no:cacheprovider -o addopts= <source node ids>
```

arrow 增加 `-p pytest_mock`，faker 增加 `-p faker.contrib.pytest.plugin`。导入来源用 `importlib.util.find_spec(module).origin` 或模块 `__file__` 核对当前 checkout。

## 已运行材料与后续模型对照

`cpu-prevalidation/prepare.py` 是本次冻结面板的 CPU 材料实验入口，非新测试套件；它只执行数据源已有判据，不调用模型。`kernel-metadata.json` 禁 GPU、禁互联网、无模型来源。根负责人已上传私有数据集 `zhuating/p015-natural-pr-material` 并完成 v2 CPU 的 18 道实际 pre/post：15 道 dev 候选（含三个预序替补）及 data 负责人交付的三道 train。平台最初递归展开归档，v1 因内层 snapshot 路径缺失而 ERROR，尚未执行题目验证；v2 将原 tar 的同字节内容改以 opaque bin 承载并显式解包，恢复逐文件 hash 校验后实际完成。这两次准备结果不混作任务代码失败。统一清单的 host 每题保留 `split` 与 `panel_role`，各用途不转换。train 的 xmltodict 复用上游 unittest 命令，其余复用 pytest 节点。本子代理未推送 Kaggle。

该 18 题最终上传包的大小、SHA256 和逐文件指纹在 `cpu-prevalidation/input-manifest.json`；`material/material-manifest.json` 保留早先 15 道开发题材料的指纹，不是最终上传包。所有解包 material、checkout 和 venv 位于 `/tmp/p015-panel-prevalidation`，每题执行后删除临时 checkout/venv。`/kaggle/working/p015-panel-prevalidation` 仅保存 `logs/{train,dev}/<iid>/`、结果与来源两个 JSON，以及唯一的 `prepared-panel-material.tar.gz`（本次输出 provenance 证实与原上传同为 81,303,711 bytes，SHA256 `eeebb65c1a03d71d42ed29d6d081b27af59a78a9c1c32ab7593d69b31f6ff8cc`），不会输出依赖目录树。最终 12 道 dev 已由根按实际 CPU 结果与事先规则冻结；后续同 Agent 的模型修复以这 12 道 dev 单独报告。train 的预验证及随后采样始终只用于 train。CPU 结果只证明判据和参考区分；随后实际 base 自主修复结果为上节的 dev 5/12，与 adapter 收益或正式分数分开记录。

补充实际 Linux 异常：`cpu-prevalidation/output-v2/p015-panel-prevalidation/logs/dev/alecthomas__voluptuous-155/buggy-f2p.log` 完整日志显示 `tests.py -> nose.tools -> nose.importer -> from imp import ...`，最终为 `ModuleNotFoundError: No module named imp`。这是155的直接根因证据；另外两题已知四态均收集失败，未据此虚构它们逐条相同的 traceback。
