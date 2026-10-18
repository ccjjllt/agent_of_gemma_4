# P010 实际结果与下一轮训练决策

复核日期2026-10-18。只读P009/P010已下载原件，未启动新GPU、提交或改动原件。P010 run为 `run-010-budget-pair-20261018T084132Z-9eeaa68e`；本目录下 `gpu-output-v1/results/010/<run>/` 为原件根目录。对照P009 run `run-009-12b-sft-pair-20261018T075746Z-5ad7782d`。

## 结论

**P010真正完成了六次Agent任务，不只是平台COMPLETE；base 2/3，adapter 1/3。思考/总输出预算确实收紧，但没有观察到adapter净收益。下一步应停止围绕这三题重复小探针，转入批量真实事务数据和较大批次SFT。** P009已经是真实训练（7仓库8条事务、2epochs、4次DDP Adam），但规模仍是链路探针，不能称为大规模训练。

P010没有重新训练，adapter权重与P009固定相同。两轮差别是两臂联合修改 thinking4096→1024、总输出8192→4096；不能拆成单变量效果，也没有排除随机生成、共享服务缓存和跨轮顺序影响。这三题已重复使用，是诊断材料，不能作为独立泛化提升的统计证据。

## 六次任务逐项结果

| 题目/臂 | P009 resolved / 秒 | P010 resolved / 秒 | P010 实际产物与机制 |
|---|---:|---:|---|
| Rich base | false / 491.15 | false / 246.70 | 零patch；24次run_command、4次read_file，依旧探索循环。失败从8分钟超时变为更快耗尽28次工具调用。 |
| Rich adapter | true / 418.07 | true / 391.19 | `rich/highlighter.py` URL字符集增加`~`，82 passed。最终655字符patch与P009逐字节相同，SHA `a9f9dcce3c6737e474127daef2158cf59f6af8f77e15f6fd8a57e282bd0faeb5`。首次edit由158.88秒提前到68.40秒，但本轮7次edit包含4次anchor匹配失败，仍有重复工作。 |
| FastAPI base | true / 240.77 | true / 238.37 | 导入deepcopy并将`additional_response.copy()`改成`deepcopy(additional_response)`；首次edit由175.27秒提前到62.98秒。完整结果成功。 |
| FastAPI adapter | false / 500.50 | false / 490.87 | 从无patch变为只添加`import copy`（356字符diff），关键浅复制代码未改；官方目标仍1 failed。认识到deepcopy方案，但反复搜索/自我重复，只完成前置导入，没有实施主体修复。 |
| Requests base | false / 487.75 | true / 484.17 | 本轮真正编辑`src/requests/utils.py`：空auth `('', '')`不返回；206 passed、13 skipped。patch另含遗留`test.netrc`文件，不能描述为干净单文件补丁。虽然没有显式submit_patch调用，官方harness捕获的工作区patch已通过。 |
| Requests adapter | false / 487.77 | false / 472.36 | 无patch；错误使用netrc API、缺失默认netrc文件后没有形成正确最小复现，反复读取同一段代码，最终28calls耗尽；随后又生成4096 tokens重复正文。 |

事实原件：`analysis_rows.json`、`joint_summary.json`；每个 `base|new_adapter/r1/attempts/<attempt>/` 下的 `result.json`、`patches/`、`test_outputs/`、`traces/`。本轮base新增Requests成功；adapter跨两轮都仅Rich成功。P010内base胜2题、adapter胜1题，而非adapter增益。

## 预算缩短的是单次响应，不是整个失败链

用固定SHA的官方tokenizer，重新编码每个trace的`event_type=thinking`文本（不含parser剥离的`thought\n`及边界）：P009长思考最多4093–4094tokens，P010各题最多1022–1023tokens。P010可见响应均不超过4096 completion tokens，限制得到执行。

| 题目/臂 | 最大completion：P009→P010 | 可见completion合计：P009→P010 |
|---|---:|---:|
| Rich base | 8192→4096 | 10264→8199 |
| Rich adapter | 4107→1297 | 12604→13006 |
| FastAPI base | 936→4096 | 6477→7744 |
| FastAPI adapter | 4423→1085 | 13877→5420 |
| Requests base | 8192→4096 | 20754→19806 |
| Requests adapter | 4137→4096 | 8156→20092 |

这些合计只含有metrics的已归档响应，不含未完成请求或全部内部摘要调用，不能当服务总token计费。Requests adapter虽然思考段变短，仍可用更多回合重读，合计反而增加；Rich base则更快失败。预算控制本身不等于正确行动或修复。

还有一个已确认的配置覆盖范围，须在下次主实验说明中保留：官方ADK1.36.1 `google/adk/apps/compaction.py:236` 使用 `agent.canonical_model` 建立摘要器；`llm_event_summarizer.py:102–105` 创建的 `LlmRequest` 只有model和contents，没有Agent的generate_content_config。因此采样YAML的max_output_tokens不自动覆盖该内部摘要请求；模型additional_args中的thinking预算可以继承，总输出不能直接认定仍为4096。

P010 FastAPI adapter最后可见响应prompt为16546tokens，已超过配置的14336 compaction阈值；161.4秒编辑后到490.9秒结束没有新trace事件。这与内部摘要消耗时间相符，但没有带request ID的完整HTTP请求体，**不能确证这段空档全由compaction造成**。这一源码边界不要求再开独立GPU探针；可在下一次批量训练后的整体运行中按实际路径处理并记录。

## 时间、利用率与配额

P010监督器报告 `status=completed`、6/6 completed、`timed_out=false`，总 **1117.63秒（18.63分钟）**，服务只启动一次、启动234.29秒。P009总1721.87秒中含371.01秒训练阶段，而P010没有训练；不能把总墙钟减少全部归因于预算改动。

减去服务启动后，batch窗口约P009942.11秒→P010766.04秒（包括调度/收尾，不是纯模型计算）。较早结束的Rich base失败释放了配对worker槽位，有助于后面的Requests两臂更早启动；这不代表每题都同等加速。

P010 evaluation有75个GPU采样，四卡平均利用率 **94.60%**，对照P00995.30%；最多4个请求同时生成，waiting采样峰1。采样覆盖746.32秒、服务计数生成96422 tokens，属于共享服务整体（含思考等），不当作单卡或单题速度。硬件没有因这轮低吞吐闲置，但大量忙碌仍可消耗在无效循环。

根代理同期账号配额读数27061.109→29307.285秒，差2246.176秒，约 **0.62394配额小时**；这是账号共享累计差值，保留“同期差值”的归因边界。原件为`supervisor-report.json`、`run_manifest.json`、`gpu-telemetry.jsonl`；账号配额原件由根代理保存。

## 从小探针转入批量训练

现在已有足够的运行证据开始扩量：真实QAT/NF4基座加载、6K级样本反向、四rank同步Adam、独立导出、官方12B CT adapter路由、完整任务闭环都已运行。无需再用一轮六题证明相同链路。当前证据尚未证明训练改善泛化，但这正是扩大合格数据后需要研究的目标，而不是继续只调三题参数的理由。

建议下一批直接构造 **64–128条有真实前后结果的独立训练事务**，按仓库/issue去重并提前固定train角色，排除保留评测和这三道诊断题。优先补本轮缺口：明确证据后的完整主体修改、工具API错误后的正确恢复、anchor失败后的重读与一次有效替换、实际生产行为验证、完成后清理临时文件和提交。保留真实上下文与观测，监督实际动作；不把正确口头方案、导入语句、上游resolved标签或重复长思考当作成功修复训练目标。

这不是要求64–128条立即证明金牌，而是摆脱8条/4步的容量探针。固定同一训练配方做一次连续批次，不每几个样本重新启动模型。按P009同长度、global batch4、2epochs的实测约25秒/step粗估，64–128条约32–64steps、纯循环13–27分钟，另加加载与保存；仅为预算估算，长度分布和数据质量改变后不能保证相同成本。之后使用预先固定、与训练分离的开发问题做一次整体效果比较，不再在这三题上选训练配方。

本轮没有修改实现，没有写新的测试/门控文件，没有另开GPU。
