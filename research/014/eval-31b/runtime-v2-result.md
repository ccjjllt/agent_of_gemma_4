# P014 runtime-v2：缩短预算后原八题仍8/8，已重新正式提交

2026-10-21。作业 `zhuating/p014-31b-runtime-v2/1` COMPLETE，运行 `run-014-31b-runtime-v2-20261021T070628Z-6d621ac4`，8/8 attempts返回。正式提交 `57471497` 已由根代理上传；此报告没有查询其分数。用户将等待分数后通知，当前不继续启动研究或监控。

**5分钟/60秒命令/40turns 的实跑配置已经生效，原八个冻结F2P见证全部通过，8/8 fresh judge 均确认从新工作树导入源码。** 这次是旧开发面板上的单臂运行修复复测，没有重跑基线，也不是独立提分证明；八题局限、空P2P和共享推理/自定义subprocess环境仍与v1相同。

| 问题 | F2P | Agent秒数 | tools / LLM calls | 运行异常 |
|---|---|---:|---:|---|
| Click2604 | 通过 | 274.76 | 24 / 26 | 无 |
| MarkupSafe402 | 通过 | 219.69 | 20 / 22 | 无终止异常；仍有一次编辑匹配失败 |
| MarkupSafe418 | 通过 | 270.99 | 11 / 13 | 无 |
| MarkupSafe379 | 通过 | 205.43 | 15 / 17 | 无 |
| MarkupSafe400 | 通过 | **300.006** | 25 / 26 | **5分钟Agent超时，回收补丁仍通过** |
| pydicom1938 | 通过 | 280.90 | 21 / 25 | 无 |
| pydicom1867 | 通过 | 244.94 | 25 / 27 | 无 |
| pydicom2163 | 通过 | 202.13 | 19 / 21 | 无 |

MarkupSafe400的原始错误为 `Agent exceeded session timeout (5.0 min)`；worker总时长313.61秒包含运行初始化与独立裁判。不能把它写成运行全程没有超时，也不能因Agent超时而误记无补丁或修复失败。这一结果直接证明5分钟限额被研究runner执行，但不保证正式全部任务能够在12小时内完成。

## 参数拒绝循环的观察

v1的MarkupSafe402曾连续12次 `edit_file` 缺 `old_string` 而被拒绝。v2此题的4次编辑请求全部包含 `filepath/old_string/new_string`，**没有再次出现该缺参循环**；全八题合计16次edit请求也均齐备这三个参数。

但不能宣称编辑问题已经彻底修复：402先有一次 `old_string not found`；随后修改误插入字面量换行转义，再读取并纠正，之后还调整了空行。这题最终通过F2P，仍有探索与格式修正浪费。本批一次运行不能证明prompt变更因果上消除了未来所有无效调用。

## 时长与GPU采样

coordinator总墙钟 **1600.05秒（26.67分钟）**。Kaggle日志记录安装完成89.81秒、模型启动124.76秒、模型就绪1133.96秒、FINAL1689.87秒、最后平台日志1696.15秒。manifest的 `model_startup_seconds=1044.14` 从coordinator起点计，包含启动前准备；日志模型启动至就绪约 **1009.20秒**，不能把1044.14或整个会话时长都称纯权重加载。

| 已采样区间（coordinator相对秒） | 帧数×卡数 | device utilization均值 |
|---|---:|---:|
| 全部35.00–1590.05秒 | 308×4 | 34.58% |
| 就绪前35.00–1040.77秒 | 200×4 | 1.28% |
| 就绪后1045.82–1590.05秒 | 108×4 | **96.25%** |

加载等待占比高，推理期间维持四卡高利用率。采样不覆盖最初和最终平台收尾；就绪后也包含工具/裁判等待，不等于纯decode效率或整会话平均。最后平台日志只给可观察执行时长，不能当完整配额账单。

## 原始证据

[汇总](evidence-runtime-v2/summary.json)、[实际条件与逐文件指纹](evidence-runtime-v2/experiment-manifest.json)、[逐题裁判](evidence-runtime-v2/analysis-rows.json)、[预算和编辑调用观察](evidence-runtime-v2/attempt-observations.json)、[GPU采样口径](evidence-runtime-v2/gpu-sampling-summary.json)。原始ZIP位于 `output-runtime-v2/p014-31b-runtime-v2-output.zip`，270742 bytes，SHA256 `8239bf4a9d632e67e2712068c2c2a002819aaf85085e4ac99a2896ec66a20bc6`。

没有更改冻结候选，没有新增测试或再次启动GPU。下一步等待用户带回正式分数或错误后再决定。
