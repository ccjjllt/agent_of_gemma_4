# 两道已冻结 RDFLib 候补的来源材料

按上级 `replacement-identity-freeze-v1.json` 获取，原24题冻结与acquisition均未修改。仅取得公开任务/原判据/参考patch等host字段，未取得teacher messages/meta/hints。沿用固定SWE-Lego revision和已有严格Parquet列范围读取，只有两道冻结row被解码导出；精确Git base浅取后确认HEAD与工作树干净，再归档为`repo/`快照。

| 原题 → 候补 | 来源判据 | 环境准备建议 | 当前证据 |
|---|---|---|---|
| RDFLib-1022 → 2474 | 1 F2P、40 P2P；`test_translate_algebra.py` | 来源Python3.9；先用固定Py3.12工具链和来源库pins，直接从checkout导入。原pyproject的editable需要`poetry-core>=1.4.0`，来源pip freeze没有该backend；环境计划显式提出先不editable，必须核对模块origin和真实四态结果 | 只完成来源、快照、SHA；未运行CPU判据 |
| RDFLib-2084 → 1117 | 2 F2P、无P2P；原test_patch增加`test_csv2rdf.py`及CSV fixture | 来源Python3.7；保留editable，因为原判据通过子进程执行`csv2rdf`。venv/bin必须优先，防止执行系统版CLI；原选定判据未导入nose，不能仅因源requirements含nose就预判无法运行 | 只完成来源、快照、SHA；未运行CPU判据，回归未覆盖 |

`agent-tasks.json`只含Agent可见题面/身份；`host-source/`包含原始参考答案和判据，不能放进Agent workspace。`source-acquisition.json`、`snapshot-acquisition.json`记录真实bytes/SHA与来源；`environment-plans.json`是拟用于CPU的明确配置，不是通过记录；`candidate-material-summary.json`绑定本次材料和计划。两份snapshot合计13,250,013 bytes。

2474检查的判据是本地SPARQL代数转换及本地fixture；1117是本地CLI/CSV。此次静态检查没有发现原1022/2084那种DBpedia远程服务前置条件，但不替代实际离线CPU运行。无P2P按既定规则保留null覆盖，不补造回归判据。仅在真实来源前后区分得到确认后，根才能更新最终C选择；这些材料不可回流训练。

初次受限网络读取未成功，随后使用已授权的联网读取完成；没有启动GPU、执行仓库测试、上传或push。源checkout保持不变，后续CPU必须从快照另建独立工作目录。
