# P017 官方公开题：单批32题身份冻结

[selection.json](selection.json) 固定本次唯一批次：**FastAPI 16、Rich 12、Requests 4，共32个独立repo＋base_commit，计划64次Base／Step-68尝试**。运行载体采用两worker同时运行一对A/B；这里不启动作业，不设“小样本后再确认”阶段，也不依据模型输赢换题。

来源为比赛 `gemma-4-developer-agent` 官方公开 `tasks.jsonl`，既有来源pin为 **1,984,455 bytes**，SHA256 `e4b3fd60f69dbc2b9213e54eeb9636db78aefe92c1d06269d73d9f5f8f3c8ad6`。本地身份母集是 [research/task_splits.json](../../task_splits.json)，其完整字节SHA及大小保存在selection。此次未读取任务正文；源tasks文件的pin继承既有身份记录，由Kaggle载体对实际挂载官方文件复核。

只从既有 `dev_panel` 24题和 `dev_future` 76题抽取，排除 `heldout` 28题及 `explore_singleton` 1题。另读取 `training/data/sample_behaviors.py` 的既有 `PAIRING_RESERVED_ISSUES` 常量，排除其8个身份及对应repo＋commit整个组；selection只保留这些排除集合的count和规范JSON哈希，不列出排除身份。过滤后剩92个唯一组：FastAPI53、Rich34、Requests5，足以满足固定16／12／4配额。

确定性规则为：种子 `p017-official-single32-20261024`；各仓按 `SHA256(UTF-8(seed + ':' + instance_id))` 升序、再按instance_id排序；同repo＋commit只取首个，继续固定顺序补足配额。最终按FastAPI→Rich→Requests轮转，已满额仓跳过。抽中30题来自dev_future、2题来自dev_panel。selection记录每题的排序哈希、仓内rank、原split及执行顺序，所有旧冻结清单保持原样。

先前使用情况仅通过 `research/001..016` 下已有trace文件名核对，未读trace正文或旧输赢；7个选定身份有既往公开评估产物，已逐题标记。其余仅表示本次路径盘点未发现证据，**不声称题目未见或全局盲评**。既往D/C结果已经被观察，但不参与本次hash排序。

载体应直接挂官方competition source，从 `tasks.jsonl` 先提取身份、只解码本次32题白名单，再交给官方 `swegemma` Evaluator及官方snapshot／test_patch判据。不得加载heldout正文到模型，也不得把参考patch混入Agent输入。当前官方runtime与Step-68私有权重dataset可复用；外部C16和D12材料不属于这个批次。

冻结文件：29,414 bytes；SHA256 `5b9e22f78b6473fddc862def16f8c7b4e132cb872aaef670020ff3de561420b6`。本次未联网、未用GPU、未新增测试。官方评估题、解答与产生的轨迹均不得回流训练。
