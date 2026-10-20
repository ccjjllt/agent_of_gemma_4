# P013 step-1 v3复测报告：修正有效（部分），冻结SequentialAgent进12新题配对

作业 `ccjjllt/p013-31b-diag-pair-v3` COMPLETE (run-013-31b-diag-20261020T064452Z-d520e540)，8/8 completed。
配额：推前8.18h → 现7.22h（支出约0.96h）。原件 /tmp/p013-gpu-v3。

## 逐题（同预算8min/28calls）

| 题 | A public022 | B p013 v3修正版 | 相对v2变化 |
|---|---|---|---|
| rich_4006 | 8 calls, 0 edits, timeout, 无补丁 | 20 calls, edit@call8, handoff@18, 超时未提交但树内留1237字补丁（cells.py+1 repro残留） | 持平偏好（补丁出现） |
| rich_3180 | 13 calls, 0 edits, timeout | 14 calls, 0 edits, 无handoff, timeout | 负证据持续两轮 |
| requests_7205 | 28 calls, 脏提交1695（含.netrc/debug/repro），FAIL | 25 calls, edit@7, handoff@17（v2为25）, 1次干净提交503（仅utils.py）, FAIL | 三判据全改善；同生产修复，B干净/A脏 |
| requests_6757 | 23 calls, 0补丁, timeout | 28 calls, edit@7（v2为9）, handoff@30（v2为25，反而延后）, 1次提交377（仅setup.py, 无0/0连交） | 提交纪律修复；交接延后 |

resolved: A 0/4，B 0/4（两轮一致，地板效应：4006/3180两臂两轮全超时）。

## 门判据（复测前设定）回检
1. 交接提前：7205达标（25→17）；6757未达标（25→30）；4006持平18。
2. review提交动作：7205达标（干净单提交）；6757达标（违规连交消失）；4006超时无提交（墙钟）。
3. 3180零编辑消失：未达标（两轮0 edits；A同样0 edits，任务侧难度）。

## 冻结决定：冻结p013_candidate（SequentialAgent + single回退），进§4.2十二新题配对
- 依据：两轮8+8 attempts中B过程证据（edit时机/124规则/交接3/4/脏包检出/提交纪律v3修复）占优，无resolved回退（地板0=0）；补丁卫生B 3/4非零且7205干净对A脏；单Agent回退件已备（agent-single.yaml）。
- 残留风险：探索型任务（3180类）修好的概率低，模型忽略硬性edit规则；墙钟仍是绑定约束，review平均只剩~1次提交机会。12新题配对以净增≥2为唯一验收。
