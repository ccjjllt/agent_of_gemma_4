# 固定 100 train：38 条题面批量修订

历史版本保留：后续真实执行又发现PyPika40的SYSTEM_TIME导入错误，当前入口为[question-revision-r3](../question-revision-r3/README.md)，累计39题面修订及监督r3。本文的38／62、47新teacher及监督r2均描述本目录冻结版本，不自动适用于新40题面；原selection/questions/revision与实际结果不回写。

2026-10-28。依据已冻结的 `question-revision-r2-plan.json`，从原 `v4/selection.json` 一次生成本目录；生成前目录不存在，没有覆盖 r1、v4 或任何已有结果。

- Plan SHA256：`533876ae89fdf18bc181cace7907c5788ae816d20dc1a916de71c2bfd6eb5aa1`。
- [selection.json](selection.json) SHA256：`e665c576d29541fd65f1610157986ab4e21fa959cddbec60a1f79d8c75b80764`。
- [questions.jsonl](questions.jsonl) SHA256：`345078173a32f5203944d24bb29d8c96398fcefb4bbb266f2bb775fc46dd54bb`。

本目录仅有一份 100 题 selection/questions。38 道题各自在 `tasks/<ID>/` 保存 problem-statement.txt、task-v2.json 和现有 v1 format 的小 revision.json；[batch-summary.json](batch-summary.json) 绑定原 selection、plan、所有新旧 task/prompt SHA 与未执行边界。不会为每条修改生成一套 100 题文件，也不重新选择任务或修改机制标签。

## 实际手动核对

62 个未改 task 对象的原始 JSON 字节、62 条未改 question 行的原始字节均与 v4 完全相同。100 个 ID/顺序、question_row、共同/专科角色、family、snapshot、来源与 recipe 指针保持不变；每份新 task 与原 task 逐字段比较，只有 problem_statement 不同，原 patch/test_patch/F2P/P2P/完整 target files 未变。新 question 的内容 SHA 与自身记录及 selection 一致。

38 条 plan 的 index、ID、原 task SHA、新 statement SHA 均在写入前核对。原有 substitutions 元数据逐字继承；38 份 revision 原件及其 SHA 位于 top problem_revisions 和逐题记录中，沿用现有 builder 可消费的格式。

官方 `build_agent_prompt` 使用冻结 `swegemma-0.2.11` wheel 的未改函数体执行，wheel SHA256 为 `9ad876581f3acde8ec3c369858f0834e2b9f41a3cffce3ca42f09cb090aea9d3`；并非手写近似 builder。每道题先用历史 budget/harness/tool 来源逐字重建旧 prompt、核对原 SHA，再生成新 prompt。

Tenacity PR425 的旧 canonical report 缺 initial_prompt_declared_tools，但同一已绑定 report 明确 six_tools_only=true，并保存六个实际 tool_imports 及实现文件 SHA。该题仅从这些真实名称恢复 builder 输入，在 revision.prompt_tools_source 单独说明；未回写缺失的旧字段，也未把 tools_called 的调用子集当作声明全集。其原 prompt 仍通过逐字和 SHA 核对。

## 批量入口

`prepare.revise_problems(previous_selection, plan_path, output)` 与 `--revise-selection ... --revision-plan ... --out ...` 接受显式 entries：index、instance_id、old_task_input_sha256、statement_path、statement_sha256。statement_path 可以是项目根相对路径或绝对路径；plan 的 previous_selection/previous_selection_sha256 也会与实际 parent 绑定。原 revise_problem API、单题 CLI 和单题输出布局保留。

一次性两题临时演练已实际核对：只写一套 selection/questions，98 个原 task 对象与 question 行字节保持不动，旧 revision refs/substitutions 继承，单题布局兼容；错误 statement SHA 在创建输出前拒绝，已有输出不覆盖。临时演练全部删除，没有当作新增研究题或动作示范。没有新增测试文件或长期检查脚本；源文件完成 AST 与必要的 `git diff --check` 核对。

生成时38道新问题版本均未取得新动作示范，历史SFT与CPU usable没有重新命名。此后的独立实际进展如下，核心selection/questions/revision原件不回写。

- [material-consumption-evidence.json](material-consumption-evidence.json)：本地打包100个当前task、原snapshot与38个revision；旧Linux CPU只匹配62个未改task，38个新版本均不适用。新增参考补丁哈希绑定还防止只按task/snapshot复用另一份control patch的旧CPU结果。归档`material-local-r1/dataset/vfarl-train100.bin` SHA为`3e2d90ebfeec251a2b9c70bd3326b823a72dd28ba458567b44ee449e7b178589`，未上传。
- [curriculum.json](curriculum.json)：保留100题库存与旧历史，明确38条历史观察属于旧task版本；暂停OSA弱oracle后，待采64题为32共同＋32专科。原4个runtime unknown仍暂停。排序函数的独立核对不是100题CPU通过。
- [split_by新版真实示范](../question-revision-r1/teacher-sft-r3/README.md)：按本selection索引86相同task/prompt哈希重新执行7次工具调用，原203个节点baseline失败、修后通过，导出8条off-policy决策。其他新题面不会因此自动获得示范或CPU usable。

后续每个新task版本仍需适用的真实CPU证据，未执行不记为失败。上述都没有模型rollout、训练更新或GPU作业；本轮私有Kaggle代码payload提交尚待用户回应自动审批要求。

## 后续完整消费版本

当前38条新题面已全部按同一原oracle真实重放并严格准入。与8个moto行为验证增强、OSA正确距离示范及其余53题历史teacher组成[完整100题当前监督r2](supervision-r2/README.md)：805决策、77,187监督tokens，47题新执行；8题66行另修正CRLF渲染错误，739行与r1原字节相同，52题416行仍为原S600字节。最大context4649，14条动作超4K，未截断。旧r1已撤回，原消息与完整模板／工具／终止动作均已复核；没有将teacher当作Gemma行为数据或伪造logprob。

最新RL调度使用[curriculum-r4.json](curriculum-r4.json)：原100ID及正确SFT仍在，基于实际原oracle反例暂停84/85/95，待采62题为32共同＋30专科。此前64／63题课程原件保留，实际新版Linux CPU适用性仍独立核对。新material本地归档尚未上传；短GPU诊断也未提交，正式训练保持关闭。
