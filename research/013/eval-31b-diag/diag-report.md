# P013 step-1 31B诊断报告：public022_v10 vs p013_candidate（4旧题×2臂=8 attempts）

作业 `ccjjllt/p013-31b-diag-pair-v2` COMPLETE (run-013-31b-diag-20261020T054234Z-c452d9ba)，8/8 completed。
配额：推前9.14h → 现8.18h（支出约0.96h，另有v1错形状0.03h）。原件 /tmp/p013-gpu-v2（zip+解包）。

## 逐题结果（同预算 8min/28calls/80turns）

| 题 | A public022（调用/编辑/提交） | B p013（调用/编辑/提交） | 机制对照 |
|---|---|---|---|
| rich_4006 | 6 calls, 0 edits, session timeout, 无提交 | 21 calls, edit@call9, timeout10验证通过, handoff@step18, review git核验, 超时未来得及提交 | B过程严格占优（历史首次编辑464s；A零编辑）。124正确记阻塞。结局同timeout |
| rich_3180 | 13 calls, 0 edits, timeout?（结果timeout） | 22 calls, 0 edits（纯读+2 repro脚本）, 无handoff, session timeout, 残留453字 | B负证据：call-12必编辑规则被模型忽略，旧失败模式复现 |
| requests_7205 | 25 calls, 1 edit, 脏提交1389/6→清理→503/1, 裁判FAIL(exit1) | 31 calls, 1 edit（同类netrc修复）, handoff@25, review提交2395/6脏包→发现脏但预算耗尽 | 两臂同找到修复；B review脏包检出有效但交接太晚；A干净包仍FAIL |
| requests_6757 | 38 calls, 3 edits, 11提交（多0/0,一次415/1）, FAIL | 32 calls, 3 edits（首edit@9，含tox.ini）, handoff@25, review连交2次0/0（第二次违反禁重提规则）, FAIL | B违反P1-4禁重提；A旧空包循环复现 |

resolved: A 0/4，B 0/4。按§4.3这不是提交门（门是12新题净增≥2），是机制诊断。

## 四项机制核验结论
1. 短超时截断（P0-2）：B-4006上 `timeout 10` + exit124记阻塞成立，未误撤回。PASS（单样本）。
2. 真实源码核验（P0-1）：B全程 `PYTHONPATH=/workspace` 前缀；`module.__file__` 打印到sandbox workspace即判有效（对sandbox现实的正确适配）。PASS。
3. 首次编辑提前（P0-3）：4006 edit@call9（vs历史464s/A零编辑）PASS；3180零编辑 FAIL——软上限无约束力。
4. 两阶段交接（§2.2回退判据）：handoff 3/4（4006/7205/6757），但review在3次中均无墙钟完成提交（4006超时前、7205预算尽、6757空交）。3180未交接。**阶段预算风险成立**：repair在step25（~2/3 steps）才交接，review无墙钟。且工具调用中位耗时~2-4分钟（sandbox+推理），8分钟墙钟只够约4-6个慢调用，call计数预算非绑定约束。

## 决定：针对性修正一次后复测（§4.3未达标路径），不直接冻结
- repair.md：加墙钟交接规则（~120s读完/~200s改完/~300s前交接；time<180s立即交接）+ call12未改则下一调用必须edit + handoff附time_remaining。
- review.md：收紧提交终局（脏包先清后交一次；0/0后永不再交；终身最多2次且第二次须有真实新edit+git diff证据）。
- 复测仍4旧题×2臂（~1h配额），判据：handoff≤step20且review有提交动作的比例、零编辑复现是否消失。
