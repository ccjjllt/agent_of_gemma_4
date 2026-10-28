# 问题版本修订的来源绑定与消费路径

2026-10-28。审阅 `prepare.revise_problem`、本地 `round100/question-revision-r1/`、Kaggle material builder、runner 与 resident 的实际消费路径。仅使用既定 100 train；无隐藏/预留题正文、远端提交、GPU 或模型训练。没有把未执行的新示范记为失败。

## r1 实际保留了什么

原 selection 的第 86 个数组元素（question_row=87）是 split_by。实际比较确认：100 个 ID/顺序、role、submechanism、repair family、snapshot group、source task/review、recipe 与原 oracle 字段不变；只有这一题的 task.problem_statement 改变。其余 99 个 selection task 对象和 question 行不变。新 task SHA 为 `7c3aa4c6a25d8347401ce5b4deac41fc63ad852c9087e0e5e6df6afcebde123c`，新 prompt SHA 为 `4b50474c4a88a3ca35602a8e5396b91e62e8d7023936fbe0076dbec27f321995`。

用冻结 `swegemma-0.2.11` wheel 中原 `build_agent_prompt` 函数体，在原 report 的 budget/harness/tools 设置下执行，原 parent 与 revised prompt 都逐字重建成功，且各自 SHA 与 selection 一致。使用函数体是为了避免安装不相关运行时依赖，不是另写一个近似 prompt builder。父版本读取的本地 semantic venv 路径已不存在，因此没有声称从该不存在环境完成了导入运行。

r1 已有的本机 CPU 新旧完整 oracle 比较保持其原记录：原/新版 baseline reward=0，参考 reward=1；这是 macOS CPU 结果，不能代替当前 Kaggle chroot 验收，也不是新 prompt 的动作示范。

## 已确认并修复的来源缺口

原 r1 selection 继承了 `substitutions='substitutions.json'` 及 SHA，但 r1 目录没有该文件。builder 原来只检查文件是否存在，缺失时静默不打包，形成悬空来源。

现已从已绑定的 v4 parent 验证后逐字补入 r1，SHA 为 `f6fb185ddb4c1343bbc5695c4fb84fb789cd3e6dea80884f5fa6ac9c5ff27d6d`。r1 selection/task/prompt/revision/奖励均未改；selection 仍为 `4a3d1f7ede038183d5bc455f4f71c73392f10cc608d6b288fbe6784569dd3977`。未来 `revise_problem` 继承 selection 声明的替代元数据文件及原 SHA，不重新选择任务。

原 builder 复制新 task 却同时复制旧 run-summary，且 selection 所引用的 revision.json 没有实际进入包。现在保留旧 run-summary 原字节，另增每题 `evidence-provenance.json` 和 runtime 中的当前 task/prompt SHA。绑定同时列出历史 summary 自身的 task/prompt SHA、适用范围与原文件 SHA，不会用当前字段覆盖历史证据。

包内 `provenance/question-revisions/<SHA>.json` 保存明确引用的 revision 原件，`material-manifest.json.question_revision_provenance` 把原 selection 的 host 来源引用映射到真实包内位置，并绑定 SHA。原 selection 字节不重写；原 host 路径不冒充包内相对路径。`problem_revisions` 累积显式版本引用，保留同一题多次修订的历史；只复制这些小 metadata 与替代元数据，不递归复制来源库。一次性临时 provenance 组件实验连续生成两次仅换行的临时版本，确认三条 revision 来源均可冻结、替代元数据逐字继承；临时版本全部删除，未作为新训练任务或示范保留。

来源适用性采用三态，避免将旧 schema 缺字段当作没执行：

| 情况 | 包内解释 |
| --- | --- |
| 历史 task/prompt 身份均存在且一致 | 历史报告对应同一版本；不声称新模型 rollout |
| 已知修订题的历史 task/prompt 不同 | 旧报告只对应历史版本；新 prompt 动作示范未重放 |
| 历史 summary 缺 task/prompt 字段 | applicability=unknown；不据此说未执行或失败 |

实际 r1 包为 98 题历史身份相同、split_by 一题已知不同、Tenacity PR425 一题因旧 summary 缺 prompt SHA 而 unknown。Tenacity 的 task SHA 仍匹配；没有把旧字段缺失升级为任务失败。

## CPU usable 的提前版本核对

实际旧 Linux split_by CPU 行是 usable=true，但 task SHA 为旧 `b3286d75…c2673d`。原 runner/resident 仅按 ID 接纳，直到 collector model_ready 后才发现 task hash 不匹配。原路径最终不会制造 reward，却会误报可用数并浪费模型加载。

现在 runner 和 resident 父进程在模型/worker 启动前，按实际当前资产 task SHA、源码 snapshot SHA 和 execution_isolation 核对 CPU 行。旧行保留旧 usable/reward，不改成 failure；新版本缺少适用证据时只排除该版本的模型采样资格。部分 material 解包没有载入源码 snapshot 的任务单独说明 `current_asset_not_materialized`，不能因旧 ID 存在而开工。

合并多份 CPU 报告时优先保留与当前版本匹配的记录，不能让后面的旧 full100 覆盖前面的新版本记录；同一匹配版本仍沿用最后记录。一次性执行真实 runner merge 循环，使用显式合成且 unusable 的“当前版本”行核对两种输入顺序，结果都保留 unusable，不制造验收通过。实际 resident 函数使用旧版单题行，在读取模型配置或创建 worker 之前拒绝，未生成 bootstrap。

对完整 r1 material 的真实旧 Linux 100 行核对，99 题适用，只有 split_by 需要该 task 版本的实际 Linux CPU 验收。当前 r1 的 macOS CPU 比较不能被改名为 chroot usable。历史 CPU 文件字节保持不变。

## 真实本地打包结果

最终审阅包位于：

`/private/tmp/vfarl-r1-material-reviewed-2h9rcqx7/dataset/vfarl-train100.bin`

- 100 个任务，124,731,426 bytes。
- archive SHA256：`2c643683608eb1c09b525badf40473b0b58a10b4d1c0a28d7ed769bd411a1a29`。
- manifest SHA256：`6b716279061a973f4acfc98d685ec820ee53010e2643aa546646a6955378df4d`。
- 实际读取 archive 与 material，确认 revision refs 经 manifest 映射都可解析、SHA 一致；substitutions 可解析且原字节一致；全部资产文件匹配 manifest；旧 summary 与原来源文件逐字相同。
- 该次打包时 source_files 与当时工作区没有差异。后续源代码或题面修订需要重新构建，不能把此审阅包当作未来代码的身份凭据。

此前两个临时审计包仅用于发现/修正 metadata 适用性问题，不作为最终审阅结果。没有上传 dataset、启动 Kaggle 作业或 GPU，也没有新增动作示范/训练完成/修复收益。必要的一次性 AST 与 `git diff --check` 通过。
