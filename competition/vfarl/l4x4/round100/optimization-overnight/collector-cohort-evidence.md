# 固定 cohort 的结果依赖丢样修复

2026-10-27 夜间至 10-28 凌晨，北京时间。只进行了本地代码审阅和一次性临时 Python 机制实验，没有启动 GPU、远端作业或正式训练，没有读取隐藏评测或预留四题正文，没有新增测试文件。

## 已确认缺陷

`resident.py` 的 `live-original-pair` 固定每批两个任务、一个 mixed-group 更新目标、最多两个状态。然而原 collector 在 original 完成后再次通过 `TaskQueue.state_available()` 和 `reserve_state()` 判定是否执行恢复分支。两处判定都受已经完成的 mixed-group 数量影响。

因此，即使第二个任务已经被 claim、已经在真实失败反馈后预选 checkpoint 并采到 original action，第一个先结束的任务一旦产生 mixed 结果，第二个任务的恢复分支就可能被取消。调换两个任务的完成先后或第一个任务的奖励，可能改变第二个样本能否进入训练。此前“固定 cohort 全部完成再更新”的注释并不能保证实际采样行为。

一次性实验只构造临时 `TaskQueue`，使用虚构标识 `task-a/task-b`，未加载任何研究任务。修复前实际输出：

```text
claimed=['task-a', 'task-b']
second_state_available_before_first_result=True
second_state_available_after_first_result=False
second_reserved=False
```

## 修复行为

resident 的固定 live cohort 显式绑定 `complete_fixed_cohort=true` 到执行契约，并经 resident worker 传到 collector；历史协议和没有该声明的独立 live collector 保留原有停止方式。

固定 cohort 的 mixed-group 数量只决定完整批次结束后的更新资格，不再裁掉批次内其他任务。截止时间继续阻止新 task claim；已经 claim 的任务可以完成自己的 original/restore/branch 配对。最大状态数仍然约束全部分支，不增加任务库存或每题预算。运行者需要把该截止时间理解为“停止开始新任务”的时间，不能当作正在执行任务的硬停止上限。

collector 的 capture、环境 control 与 fork 判定均携带当前任务标识，只有真实已经 claim 的 cohort 任务取得继续资格；不会因为一个任意任务 ID 就绕过截止时间。

## 实际手动验证

临时目录中的纯队列实验得到：

| 情形 | 第二任务可 claim | 第二状态可 reserve |
| --- | --- | --- |
| 原 historical，先有 mixed 结果 | 已提前 claim | 否，保留历史行为 |
| 未声明 fixed 的独立 live collector | 已提前 claim | 否，保留原行为 |
| fixed live，两个任务提前 claim | 是 | 是 |
| fixed live，第一任务 mixed 后第二任务才 claim | 是 | 是 |
| fixed live，两个任务 claim 后越过新任务截止时间 | 是 | 是 |
| fixed live，截止时间前只 claim 第一任务 | 第二任务不能新 claim | 第一任务可继续，未 claim 第二任务不能继续 |

两次状态 reservation 后，`maximum_states=2` 仍拒绝后续状态。三个修改文件均经一次性 AST 解析，`git diff --check` 没有报错。这些结果只证明队列机制已修复，尚无真实 GPU 配对、更新或修复收益证据。

## 已修复的生产修改漏计

当前 `execution-recovery` 只将成功 `write_file` 与带非空 diff 的成功 `edit_file` 算作 mutation。Agent 若用 `run_command` 实际改写生产文件，后续失败的 pytest 会因为缺少 mutation 计数而被分类为普通 diagnostic failure，错过恢复 checkpoint；失败命令也可能在失败前留下真实修改。命令名称不能证明是否改过代码。

现在 live collector 在 episode 起点记录生产工作树内容指纹，随后每次真实 `run_command/edit_file/write_file` 返回或异常后重新观测。候选路径来自冻结 baseline 清单、当前 Git tracked 清单与 nonignored untracked 清单的并集，按官方保护规则和额外 test_files 过滤；逐路径读取文件字节、符号链接目标与执行位，明确删除/新增。不运行 `git add -N`，不改变真实 index，不扫描 venv 或建设新的快照框架。已知生产与新增未分类路径采用独立指纹，详见下文修订后的边界。

初始已有 dirty 内容只建立初值。stage/unstage 没改变实际内容时不增加 mutation 计数；真实失败命令若在失败前留下已知生产修改，仍记为实际变化。指纹改变只证明已知生产范围内状态变化，不代表修复正确。读取失败或只有新增未分类文件变化时显式保留 `changed=null`；对应测试失败归为 `mutation_unknown_test_failure`，**该 unknown 本身不能触发 execution-recovery**，也不伪装成“没有修改”或赋予任何成功奖励。

临时生产仓库的一次性手动实验实际核对了初始 dirty、stage/unstage、未跟踪新增/删除、tracked 文件从 index 移除后的删除、执行位变化、符号链接目标变化、ignored 文件、官方保护配置以及额外 protected 路径。真实 `python` 子进程先改生产文件再退出 1，观测得到 `changed=true`。另对分类函数使用明确的合成错误输入核对得到 `repair_attempt_failed`；`changed=null` 得到上述 unknown 类别。这些是机制验证，不是实际任务执行或 pytest 成绩。

成本测量在本机已有 Python 3.11.15 中解包两份属于固定 100 train 且不在预留四题名单中的真实源码 snapshot；只在过滤后读取生产内容。官方保护判据函数从仓库中冻结的官方源码提取执行，因为本机 venv 未安装完整 swegemma。每份连续观测三次，指纹一致：

| Train snapshot | 读取路径数 | 内容字节 | 三次墙钟秒数 |
| --- | ---: | ---: | --- |
| iniconfig `func_basic__vb9u5hga` | 11 | 12,826 | 0.027761 / 0.011684 / 0.010168 |
| moto `pr_4924` | 1,940 | 72,086,093 | 0.199505 / 0.157866 / 0.161838 |

约 0.16 秒的 moto 本机热读成本支持先采用完整内容指纹；Notebook 文件系统上的实际成本仍待短验证观察，不能把本机耗时视为 GPU 环境测量。最初系统 Python 不支持 `tarfile.extractall(filter=...)`，因此成本测量改用已有 venv，并未把失败尝试算入成功测量。

## 独立审查后的运行产物与 unknown 边界修复

初版指纹把所有未受保护的新增文件都看作生产文件，会把首次诊断命令生成的 `report.xml/.coverage` 误计为修复修改；初版分类还允许历史 unknown 计数永久授予后续测试失败焦点资格。这两项均已修正，不能沿用初版的资格解释。

最终边界为 `frozen-baseline-production-and-new-source-v2`：

- 冻结 baseline 中已有的非保护文件保留所有类型。因此既有 YAML/JSON/XML 配置、无后缀生产文件、模板和静态资源的真实修改仍可见；不靠后缀排除这些既有文件。
- 新增文件只对明确源码/模板类型给出已知生产资格，包括 Python/JS/TS/C/C++/Java/Go/Rust 等代码、shell/SQL/GraphQL、HTML/CSS/Jinja/Mako/Mustache/SVG 等模板与前端源码，以及 Dockerfile/Makefile/Rakefile。常见 cache、htmlcov、coverage、build、dist 等生成位置中的新增文件不自动取得资格。
- 新增 XML/JSON/YAML、无后缀以及其他未分类文件保留独立内容指纹、路径数与路径样例。仅这些文件变化时得到 unknown，不能当作生产 mutation；`git add report.xml` 也不会将其升级为 baseline 生产路径。新配置修复留作显式待审证据，没有静默抹掉，也没有假装已经识别为生产代码。
- unknown 计数只供历史说明，不参与恢复触发。当前 unknown 测试失败不 eligible；后续已知无变化的诊断失败也不因历史 unknown 变为 eligible。只有已知生产修改观测可以将测试失败归为 `repair_attempt_failed`。

一次性临时生产仓库实验实际得到：依次新增 `report.xml`、`.coverage`、`htmlcov/index.html`、`new-settings.yaml` 时均为 `changed=null / focal_eligible=false`；同样保留 `production_mutation_unknown_observations=100` 的 ledger，下一次无变化时均回到 `diagnostic_test_failure / focal_eligible=false`。对 report 单独 stage/unstage 不变更已知生产指纹。既有 `source.py/settings.yaml/page.jinja2` 的修改，以及新增 `new-module.py/new-template.jinja2` 均得到已知变化；合成失败测试反馈据此 eligible。没有实际执行 pytest 或建立测试文件。

这个边界仍是保守的采样证据，不是文件语义分类器：源码扩展的生成文件若落在未识别的输出位置，仍可能进入已知范围；新配置/数据文件则需要显式来源审阅后才能扩展边界。它不把任何这类调度信号当作正确性奖励。
