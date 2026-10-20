# P013 旧题机制诊断小结（CPU 只读复核，未跑 31B）

日期：2026-10-20。范围：只读 `research/008/review-20261018/agent-review.md` 引用的已有 trace/patch 原件；
未启动任何 31B 模型运行、未读 heldout、不计入修复成功数。31B 实测待 GPU 授权。

基址（下文 RUN）：`research/007/public12-pair-20261017/gpu-output-v2/results/007/run-007-public12-pair-20261017T063527Z-5d0f2c69/behavior_v1/r1/attempts/`。

## 1. rich_4006：死循环复现卡 240s，首次编辑 464.65s（P0-2 + P0-3）

- 原件：`02-rich_4006-r1-behavior_v1/traces/trace_rich_4006.json`（11 steps）；agent-review §"rich_4006" 确认双方无短超时的复现各卡 240s，B 首次编辑 464.65s 且只修 `\ufe0f` 漏相邻 `\ufe0e`，最终裁判超时。
- 机制映射：P013 repair prompt 强制 `timeout 10 python3 -c` 单行复现、124=阻塞未完成严禁撤回；Call 12 前完成主体+关联分支（对称条件一次读完集中写）。
- 未跑原因：需 31B + 8min 预算的 GPU 运行；CPU 无法产生新的机制证据。

## 2. rich_3180：415.41s 才首次编辑，补丁混入 repro（P0-3 + P1-4）

- 原件：`06-rich_3180-r1-behavior_v1/traces/trace_rich_3180.json`（11 steps，step 10 在 415.41s 首改 `rich/cells.py`，step 11 又写占位 `repro_2.py`）；agent 超时后靠恢复树补丁通过 114 tests，但 patch 含 `repro.py`/`repro_2.py`。
- 机制映射：repair 软上限 call 7-12 完成全部生产修改、定位后禁写第二版 repro；review 收尾 `git status --short && git diff --stat` + 清理 `repro*`/`*.tmp`。
- 状态：B 的生产改动为真（独胜），但"验证闭环完成"表述已由 agent-review 收窄；P013 不把该题记为已解决。

## 3. requests_7205：导入系统旧包，PYTHONPATH 纠正为真机制证据（P0-1）

- 原件：`18-requests_7205-r1-behavior_v1/traces/trace_requests_7205.json`（19 steps）：step 14 自查 `requests.__file__` 命中 `/usr/local/lib/python3.12/dist-packages/requests/__init__.py`；step 15 改用 `PYTHONPATH=/workspace/src` 后观察到正确行为；step 17 非空提交。
- CPU 一次性核验（手动命令，已执行）：上述 trace 文件存在且 step 15 确含 `PYTHONPATH` 字样——见本机 `grep` 输出（12 traces 中四题文件齐）。未重跑模型。
- 机制映射：P013 所有 python 检查强制 `PYTHONPATH=/workspace/src:/workspace` 前缀 + 打印 `module.__file__` 且要求 `/workspace` 前缀；区分目标代码错/复现命令错/环境缺失，复现写错只修命令。

## 4. requests_6757：连交 10-11 次空补丁（P1-4）

- 原件：`19-requests_6757-r1-behavior_v1/traces/trace_requests_6757.json`（41 steps，A/B 分别 11/10 次 `submit_patch` 均 `patch_size=0, files_changed=0`）；agent 在"git diff 明明有改动"与 submit 0 之间空转耗尽调用。
- CPU 一次性核验（手动命令，已执行）：trace 中 step 15 起反复出现 `patch_size: 0, files_changed: 0` 自述与重复 `submit_patch` 调用（见本机 step 扫描输出）。
- 机制映射：repair 无 submit_patch（半成品无法提前终止）；review 规则 `patch_size==0 → 严禁直接重调 submit_patch`，必须回源码做真实编辑并 `git diff` 确认。注：该题疑似配置类不可交付任务，P013 规则至少节省空转调用，不承诺解出。

## 5. 脏补丁（rich_3180 / fcd12 cc-2010）

- 原件：agent-review §补丁污染（B 的 rich_3180、fastapi_14463、requests_7427/7502 携带 scratch）；P012 fcd12 结论 `f` 的 2010 patch 含 `generate.py.tmp`、`reproduction/`、`test_cookiecutter.json` 残留但判据仍过。
- 机制映射：review 收尾白名单（仅生产源码可改）+ 删除 `*.tmp`/`repro*`/自建目录；提交前 `git diff --stat` 确认。

## 待 GPU 确认（非本轮 CPU 产物）

1. 短超时是否截断死循环且无误撤回；2. 路径核验是否每次命中 /workspace；
3. 首次生产修改是否进入 Call 12；4. SequentialAgent 第一阶段是否按时交接（否则切单 Agent）；
5. 终补丁干净且无空提交循环。
