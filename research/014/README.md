# P014 研究状态与交接依据

**2026-10-21：正式提交57438326因整体运行超时失败，无分数。** 完整API同时返回COMPLETE与error_description；旧CLI默认列表没有错误字段，不能据COMPLETE认定成功。已冻结runtime v2：每题5分钟、单命令60秒、40轮、28工具，模型采样保留；补充edit_file必填参数与拒绝循环切换。修复版已重新提交 **57471497 / PENDING**，单臂原八题 `zhuating/p014-31b-runtime-v2/1` 已完成8/8通过，5分钟截止真实生效；四卡4并发；不重跑基线、不新增测试。见[实际失败及修复](submission/failure-20261021/README.md)。下文10月20日PENDING均为历史状态。

**20:47 执行更新：已真实提交 `57438326`，当前 PENDING，分数尚未返回。** 31B 新候选与正式基线各 **8/8 F2P 通过**，W=0、L=0，没有证明净提分；冻结候选按探索性提交上传，ZIP 未在实跑后改动。12B 行为 SFT 为 base **4/4**、adapter **3/4**，暂停原34条配方继续训练。两个 GPU 作业均 COMPLETE，共24次真实 attempt，没有新增测试。见[执行记录](execution-20261020/README.md)、[31B 结果](eval-31b/result-review.md)、[12B 结果](eval-12b/result-review.md)、[提交回执](submission/README.md)。下文为19:42交接快照，“新 GPU 作业 0/新正式提交 0”和“尚待评测”均仅指该历史时点。

日期：2026-10-20，北京时间。代码起点 `59f562d`。执行方案见 [P014 提分与正式提交交接](../../docs/plan/014-score-first-submission-handoff.md)。

本次完成只读 API 查询、现有作业日志/权重取回与交接整理；新 GPU 作业 **0**，新正式提交 **0**。没有读取 heldout 内容，没有新增测试。

## 19:42 左右的官方查询

每个进程独立绑定账号凭据，经 API 核对实际 username；没有覆盖全局凭据。原始 stdout/stderr、查询命令与 UTC 时间保存在 `status-20261020/`，时间展示转换为北京时间。

| 项目 | 结果 | 原件 |
|---|---|---|
| zhuating GPU | used 0.00h / remaining **30.00h** | [quota](status-20261020/quota-zhuating.json) |
| ccjjllt GPU | used 24.16h / remaining 5.84h | [quota](status-20261020/quota-ccjjllt.json) |
| qingyu1232 GPU | used 21.88h / remaining 8.12h | [quota](status-20261020/quota-qingyu1232.json) |
| GPU 刷新 | UTC 10-24 00:00，即北京时间 **10-24 08:00** | 同上 |
| 正式成绩 | `57340172` COMPLETE，**0.15**；旧 `57091482` 为 0.13，无第三次提交 | [submissions](status-20261020/submissions.json) |
| 可提交数量 | `numTotal=2, numAllowedNow=1`；上场前重读 | [limits](status-20261020/submission-limits.json) |
| 主赛 | 3305 队，截止北京时间 **12-03 07:59** | [metadata](status-20261020/competition-metadata.json) |
| 公榜前 100 | 0.24：3 队；0.22：32 队；0.20：65 队 | [leaderboard](status-20261020/leaderboard-top100.json) |

配额为共享账号的四舍五入读数，不据差额推算本项目实际消耗。榜单返回仅前 100；不能从它推导铜牌/银牌边界或私榜成绩。CLI 的榜单 stdout 在 JSON 数组前含下一页 token，解析时应跳过这一提示行。

## 已取回：12B 行为 SFT 真实完成

作业 [`ccjjllt/p013-12b-behavior-sft-34/1`](https://www.kaggle.com/code/ccjjllt/p013-12b-behavior-sft-34) 的 API 状态为 COMPLETE；进一步用训练日志核实：

- 34 独立问题、20 仓，8 epochs，连续 **72 optimizer steps**；4×NVIDIA L4。
- 输入 token `1,221,688 = 152,711 × 8`；监督 token `556,696 = 69,587 × 8`。
- supervisor exit 0、没有超时，墙钟 **1817.58 秒（30.29 分钟）**；独立导出在北京时间 **17:56** 完成。
- 最终 adapter **115,020,728 bytes**，已经下载并在本地独立核对 SHA256：

```text
54de0ca38cdcb284509f7678cd59077217b169e3ee84649e78791faa8b5ae2a3
```

权重路径（按项目规则不入 Git）：

```text
research/014/recovered-20261020/sft-v1/results/013/
  run-013-behavior-sft-20261020T092606Z-521525b9/training/adapter/
```

下一步直接加载此 adapter，完成同 Agent 的 **12B base vs adapter** 修复对照。导出清单仍为 `runtime_compatibility_verified=false`；`diagnostic_runs=0`、`repair_gain=null`。训练拟合改善不是修复率或公榜收益；不重复训练，也不继续旧 P012 假完成标签的 Stage B。

便于 Git 交接的短原件：[训练摘要](evidence/sft-stage-summary.json)、[总结果](evidence/sft-run-summary.json)、[supervisor](evidence/sft-supervisor-report.json)、[硬件](evidence/sft-hardware.json)、[下载文件指纹](evidence/recovered-files-manifest.json)。完整 export manifest、metrics、日志与权重在 `recovered-20261020/`；没有下载四份完整 optimizer checkpoint。

## 已取回：CPU 预验证在读取输入时失败

实际可查询作业为 [`qingyu1232/p013-preverify-cpu-clone/1`](https://www.kaggle.com/code/qingyu1232/p013-preverify-cpu-clone)，ERROR。启动约 1.8 秒：

```text
AssertionError: missing criteria index under /kaggle/input/p013-task-material//kaggle/input/p013-task-material
```

原始错误见 [CPU 日志](evidence/preverify-cpu-error.txt)。已取回[线上源码](recovered-20261020/preverify-source/p013-preverify-cpu-clone.py)与 metadata：源码依赖同目录输入文件，但上传的 code_file 没有嵌入这些内容；metadata 也没有 dataset/kernel sources。启动时找不到同目录 tasks/index，退回旧 dataset 路径后再次失败。它没有执行到环境准备与修前/修后验证，因此目前不能统计有效新题。

另外，源码仍有新 venv 依赖不全、任意 ImportError 判环境错误、重复同文件当独立回归等问题，修输入后还需按交接方案处理。`dev12-freeze.json` 的占位内容不代表题集已经通过。

取回时带 `/1` 的 `kernels pull` 返回 403；不带版本的 pull 成功。`p013-preverify-cpu` 这个暂存 metadata 中的 slug 查询失败，实际在线 slug 如上。错误响应如实保留，避免交接时沿用错误命令。

## 保留的研究判断

P013 两轮旧题诊断仍是基线 0/4、候选 0/4，不能改判。reviewer 缺 `{problem_description}` 已核实，源码与 v3 实跑嵌入包一致；新包需要同步修正。旧 Requests 裁判还存在系统安装路径和 fixture/证书错误，需要区分环境污染与真实补丁失败，不能只核对 Agent 自己的 Python 命令。

本轮交接以新的逐题胜负、可复现 submission 和官方回执为交付。当前未证明新方案提分，未取得奖牌。
