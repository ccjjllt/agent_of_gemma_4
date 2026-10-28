# 固定100题：split_by问题版本修正

2026-10-28。只修改第87行、`Suor__funcy.207a7810.func_basic__ew5ofxg0`的问题说明。100个题号、顺序、共同／专科身份、S4机制、repair family、snapshot、原参考修复、原F2P/P2P及五个完整验收文件保持不变；其他99行questions和99份selection任务对象均已逐字节核对相同。v4原件、旧哈希和旧奖励未改。此版本已在本地审计打包，未上传或用于模型训练；包内来源绑定及CPU版本适用性见[消费链审阅](../optimization-overnight/question-revision-consumption-evidence.md)。

## 真实矛盾

发布者原题面将`split_by`误写为全序列按谓词分组，声称`[1,2,3]`的正确结果是`([1,3],[2])`，现状为`([2],[1,3])`。冻结源码的docstring定义的是“最长连续满足谓词的前缀”和剩余序列；一次性真实源码运行得到baseline `([2,3],[1])`，应用已有参考修复后得到`([1],[2,3])`。原`tests/test_seqs.py::test_split_by`也明确要求`([1],[2,3])`。

实际命令输出、源码snapshot哈希和原oracle原文见[manual-reproduction.txt](manual-reproduction.txt)。这里只修正错误行为要求，没有把参考补丁的takewhile/dropwhile实现或私有验收节点写进新策略题面。

新[problem-statement.txt](problem-statement.txt)明确：第一部分是最长连续真前缀，第二部分从首次不满足谓词的元素开始；后来再次满足谓词的元素仍留在第二部分，保持顺序和原惰性行为。

## 版本绑定

| 绑定 | 原版本 | 本地修订 |
| --- | --- | --- |
| task_input SHA256 | `b3286d7593040bf3ccd2c231bcb81841fcd4a5729e15c3053ef2526f73c2673d` | `7c3aa4c6a25d8347401ce5b4deac41fc63ad852c9087e0e5e6df6afcebde123c` |
| initial prompt SHA256 | `55979595a68936c1547c1c97b22e0977dd44605e2dfb4f6a718a5345cd36c666` | `4b50474c4a88a3ca35602a8e5396b91e62e8d7023936fbe0076dbec27f321995` |

生成器为`prepare.py`中的显式单题revision入口。它只读取既定100题，不调用600题重选过程；先用实际官方`build_agent_prompt`逐字重建原prompt，再在相同历史预算与工具声明下构建新prompt。只有problem_statement改变，其余task字段逐项相同。完整来源与新旧哈希见[revision.json](revision.json)，可供下一轮使用的本地文件为[selection.json](selection.json)和[questions.jsonl](questions.jsonl)。

已有canonical report、recipe、动作与短示范继续表示旧问题版本。**没有按新prompt执行的动作示范或模型rollout**；selection和revision均明确标注待真实重放，不将旧token数据重新命名。旧CPU usable不能直接用于新task hash。

## 本地CPU的同一原oracle比较

复用已有official/pytest依赖cache；首次四条完整调用均因缺少whatever而collection失败、reward=null，真实记录保留在`cpu-local/`。随后仅从已存在的`jiyujia-base-runtime/vfarl-cpu-stage.bin`恢复9个whatever 0.7依赖文件，未安装或下载新包；依赖成员及哈希保存在真实结果中。

修正依赖后的四次官方`verify_task`使用同一个源码快照、同一已有参考patch和原完整验收要求，结果如下：

| 问题版本 | 原始源码 | 已有参考修复 |
| --- | --- | --- |
| 原题面 | 有效reward 0；1 failed、202 passed | 有效reward 1；203 passed |
| 新题面 | 有效reward 0；1 failed、202 passed | 有效reward 1；203 passed |

全部`verification_complete=true`。原始结果、官方判定和JUnit XML保存在`cpu-local-complete/original/`及`revised/`，汇总见[cpu-local-complete/cpu-results.json](cpu-local-complete/cpu-results.json)。这是macOS CPU、Python 3.12.13及已复用pytest 8.3.3环境，不证明Kaggle chroot或比赛Docker等价；没有通过它追认模型修复收益。

## 五道既有curator题面复核

固定v4中的TinyDB缓存失效、dataset事务退出、WordStart、tomli Flags.set、pyparsing.line五个curator题面，均未发现参考patch代码、私有oracle节点或事后机制标签泄露。零宽断言、存储类型和局部／递归flag约束属于行为说明。原题面四份为空，tomli原文则错误声称Flags.add及重复显式表应合法；版本化澄清修正这些来源问题，保留原完整oracle。

五题实际`agent-prompt.txt`均逐字等于v4 questions，且与真实messages首轮用户内容、selection/canonical report/run-summary的prompt哈希一致；当前合并CPU证据绑定相同task-v2哈希、usable=true，原F2P/P2P逐项未改。curator身份本身不构成数据污染。

95个原题面同snapshot的相似变异中，cut_prefix/cut_suffix可由函数及输入输出区分，聚合函数DISTINCT与查询DISTINCT也属于不同公开契约；不能因文字或补丁相似误删。上述split_by是本轮找到并实际复现的题面／原oracle冲突。
