# P005 single_state 完整运行载体

**历史单臂入口。2026-10-11起，本轮研究使用[四题双臂载体](005-t0-single-state-pair.README.md)，共用一次加载并发比较T0/single_state。下面的原六题panel包含现已预留的`rich_3130`和`requests_6644`，不用于当前调参。**

入口是 `005-single-state.ipynb`，其单个代码cell与 `005-single-state-single-cell.py` 完全一致。当前已补齐真实模型服务、逐题Evaluator、结果/补丁/轨迹归档及清理流程；没有在本轮启动GPU、提交Kaggle或取得新修复率。

与旧DRAFT不同，本载体嵌入候选agent的全部技能正文及逐文件SHA，无需在Kaggle克隆本仓库。运行时生成独立agent目录和submission ZIP；ZIP只作为输出，不自动提交比赛。

## 使用

1. 使用 **Linux x86_64、CPython 3.12、CUDA** 的全新Kaggle kernel。v30有cp312专用二进制wheel；如果当前kernel是Python3.13，载体会在任何包安装、模型加载之前明确停止。请选择Python3.12镜像/kernel后重跑，不强装不兼容wheel；脚本不自动更换解释器。
2. 挂载比赛数据、`metric/gemma-4-developer-agent-wheelhouse` **v30**，以及 `google/gemma-4/other/gemma-4-31b-it-qat-w4a16-ct/2`。载体不下载这些大输入。
3. 历史默认 `RUN_MODE = 'smoke'` 只执行已暴露的 `rich_3518`；原 `'panel'` 会执行六题，现包含预留任务，不应按本页旧配置启动。当前四题双臂研究使用上方新入口。
4. 执行代码cell即启动真实GPU模型。所有所选题共享一次服务启动，每题新建Evaluator/工作区；发生任务异常记录原始异常并继续下一题，不自动重试。

六题为 `rich_3942`、`rich_3130`、`fastapi_14873`、`fastapi_5077`、`requests_6644`、`rich_3518`。其中两个历史裁判条件有歧义，已在manifest说明。这个单臂载体研究的是整个single_state组合，不能把其分数差全部归给状态卡或某个单独技能；smoke更不是效果证据。

采样从候选目录读取，当前研究起点为温度0.2、top_p 0.95、输出上限16384、thinking_budget4096并保留思考。每题5分钟/50工具调用/80轮、单命令及验证60秒沿用研究预算。这些参数写入归档，不声称已知最优。

## 指纹与实际执行

- 四个官方包的版本+SHA从已有v30载体继承：swegemma0.2.11、adk_submission0.2.13、adk_eval_core0.1.0、google_adk1.36.1。
- 全部v30 wheel名称/字节数来自官方MCP文件清单；安装前检查Python/platform标签，再归档所有wheel的实测SHA。不会遇到不兼容wheel仍强制安装。
- 比赛tasks.jsonl固定SHA `e4b3fd60f69dbc2b9213e54eeb9636db78aefe92c1d06269d73d9f5f8f3c8ad6`、1984455字节。
- 模型版本固定为官方CT v2；tokenizer.json经2026-10-10 Kaggle MCP单文件下载核对，32169626字节，SHA `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`。不把这个tokenizer指纹宣称成所有权重分片的哈希。
- 使用官方 `AdkSandboxCodeExecutor`。它通过 `SubprocessManager` 执行技能，实际工作区由candidate脚本解析；载体不伪造 `/workspace` 宿主目录、不修改官方host回调。

输入版本变化应形成新运行条件。当前CPython3.13问题是二进制ABI不兼容，不能靠删除检查解决。

## 输出

每个run写入 `/kaggle/working/results/005/<run_id>/`：

```text
run_manifest.json                 # 版本、GPU、配置、预算、任务、启动次数
frozen_bundle/                    # 本次全部agent内容、SHA、wheel/model指纹
attempt_status.json
analysis_rows.json
joint_summary.json
vllm_server.archived.log           # 在官方stop删除日志之前复制
single_state/r1/
  submission-005-single_state.zip
  submission_manifest.json
  attempts/<attempt_id>/
    attempt_manifest.json
    result.json
    context_metrics.json
    harness.log
    patches/<task>.patch
    traces/trace_<task>.json
    test_outputs/<task>.log
    artifact_manifest.json
    attempt_status.json
```

任务异常的`result.json`保留`resolved:null`及错误详情；若ctx已有提交补丁，则另存`submitted_patch_before_exception.patch`。轨迹缺失被记录为`completed_evidence_gap`，不会伪造轨迹。服务启动失败也归档异常并清理，尚未执行的题记为`not_run`。

## 从实时candidate重新生成

```sh
python3 notebooks/builders/build_005_single_state.py
```

生成器只用标准库，读取当前 `research/005/candidates/single_state/agent/`；已有P002冻结载体只作为服务/评测实现来源，不修改它。候选改动后重新生成，不能直接编辑生成文件里的大段BUNDLE。

生成后手动核对候选内容、运行预算、任务列表和来源指纹；使用一次性命令观察关键路径，不再维护独立测试脚本。真实速度、工具行为和修复收益由开发题对照实验判断。

旧 `005-single-state-draft.py` 保留为历史草稿，不作为执行入口。`one_handoff`是另一个候选agent包，本载体暂不把它混入单Agent实验。
