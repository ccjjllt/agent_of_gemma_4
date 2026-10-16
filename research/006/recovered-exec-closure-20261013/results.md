# 已完成的P005执行收尾实验：2026-10-13恢复

这是一轮**2026-10-11已实际完成**、但此前未进入当前main研究记录的实验。本轮在启动新GPU前通过账号Notebook清单发现并取回；不能记作10月13日新跑出的收益。来源：[私有运行](https://www.kaggle.com/code/ccjjllt/p005-exec-closure-pair)，run ID `run-005-exec-closure-pair-20261011T082035Z-7fcb29d6`，远端为COMPLETE。

## 核对结果

原件ZIP为3,166,279字节、222项，SHA256 `b277e9e188c99e2eeea9505eafda617946e9b08ced8f6a8cda4c4633fb8864f0`。下载到`output/results/005/`，只展开12个已暴露attempt的结果、轨迹、补丁供分析；没有读取heldout。逐文件核对：A single_state的15个文件与正式0.13 submission完全一致；B exec_closure只改变prompts/system.md。B是旧版执行提示，与10月13日新写closure_prompt措辞不同，二者不能混称同一版本。

| 题目 | A single_state | B exec_closure | 重复结果 |
|---|---|---|---|
| fastapi_14873 | 1/2 | 1/2 | 两臂均第一次成功、第二次超时失败 |
| fastapi_5077 | 0/2 | 1/2 | B仅第二次成功，第一次仍只有复现脚本 |
| rich_3518 | 2/2 | 2/2 | 均成功；B分别132.58/217.08秒，A306.16/129.31秒，节时方向不一致 |
| 合计 | 3/6 | 4/6 | 一个配对新增成功、零成功回退；不是6个独立问题 |

12次均完成并有原始轨迹。B四次成功不能证明增益稳定；新增fastapi_5077也只出现一次。所有12条轨迹均未调用load_skill/run_skill_script等新增技能，仍不能归功于状态技能。

fastapi_5077第二次B确实改了`fastapi/dependencies/utils.py`：沿`__wrapped__`获取原函数的globals，分别修改typed signature与return annotation；约167.95和176.51秒执行两次生产edit_file，236.30秒submit_patch，官方结果resolved=true。第一次B仍只留下`reproduce_issue.py`。这是执行成功与失败的具体分叉，不应把第二次补丁当所有输入的已证明通用方案。

原实验为v30、31B CTv2、一次服务加载、两路隔离Evaluator；5分钟/50调用/80turn/60秒命令与验证，16384输出/4096thinking。启动约408.50秒，整轮wall 2387.65秒（39.79分钟），按2倍估算约1.32647额度小时；这是历史wall估算，不是本次扣额或已核实独立账单。原始参数/结果见`output/.../run_manifest.json`与[恢复分析](recovered-analysis.json)。

## 对本轮执行的影响

补充手动轨迹核对：[动作与最终文件清单](execution-mechanism-observations.json)。5次失败中有4次最终只有非生产复现文件，没有生产代码补丁；另一失败是single_state的fastapi_5077 r1，约259秒首次edit失败、298秒才成功改typed signature，最终仍未通过。不能把“最后没submit”单独认作失败原因，官方会抓取现存补丁。

exec_closure的fastapi_5077 r1在约94秒已经明确提出沿`__wrapped__`取得globals，随后却继续构造未触发问题的复现、追逐旧traceback行号，直到超时仍未编辑源码。相同策略r2在168秒开始生产编辑并成功。因此执行延迟有直接轨迹证据，但仅凭这些公开任务不能把它当0.13隐藏失败的总原因。另有5次越出工作区的write拒绝、1次old_string找不到和2次缺old_string参数；局部错误计数与整题修复分开解释。

下一次对照应看实际生产补丁、保留行为和裁判结果，而不是只看工具报错减少、是否写“已理解”或是否调用技能。本轮不根据单个失败继续改十版提示。

先前006计划把同类三题重复比较当作尚未执行，是因为仅依赖本地main记录；恢复远端原件后已纠正。无需为了接近措辞的新prompt再消耗一轮GPU重复这项机制筛选。新写closure_prompt保留为未运行草稿，不赋予本实验4/6成绩。

将用户本轮授权的≤1.5额度小时用于新增信息：旧exec_closure冻结版本与可核实Budget-Fit V5来源的五分钟匹配版本对照。原版0.18是八分钟条件，新五分钟参考不继承该成绩；比较整体策略，不作单提示/单参数归因。CPU外部数据重放与12B适配继续并行。
