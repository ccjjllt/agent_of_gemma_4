# 索引75–99：六条新版题面的真实示范

2026-10-28。针对当前question-revision-r2索引77、78、83、89、91、99，逐题从原source inventory重建文件内容与mode，保留原recipe、原预算及harness数值类型，并按新题面真正重放。六条实际官方prompt SHA和原source tree均与各自pin逐字一致；没有复制旧动作换题面。

原完整oracle保留，六题均有真实baseline0／reference1：TextFSM96、Graphene450、qrcode81、xmltodict60、pyparsing1852、funcy203个原节点。严格normalizer全部准入，无缺项。节点数量只是执行范围，不是模型研究收益。

两组增量均复用同一已验CT tokenizer/template，实际CPU编码版本见manifest：

- [independent-sft-r1/gemma-sft/sft-actions.jsonl](independent-sft-r1/gemma-sft/sft-actions.jsonl)：77/78/89/99四题，32决策、2674监督tokens，最大context3335；SHA `d401a787840306f736ba81a3dffb94ee5bded3c4f058bed0e0a72269a3f8955d`。
- [independent-sft-r2/gemma-sft/sft-actions.jsonl](independent-sft-r2/gemma-sft/sft-actions.jsonl)：83/91两题，16决策、1286监督tokens，最大context3286；SHA `16953754aae81c71306cc52f5d0ae75d0f917afb573bfae2671bae18399adde5`。

合计6题、48条真实决策、3960监督tokens，当前task/prompt→canonical→SFT的逐题连接见[current-version-teacher-sft-index.json](current-version-teacher-sft-index.json)。thinking0，无生成思考、行为logprob、模型rollout或训练更新，旧S600/F600保持原样。

首轮qrcode/pyparsing的完整验收失败保存在83/91的actual-run，不当作有效baseline因果0或参考1。qrcode缺自身真实distribution metadata、PNG/Pillow资源；pyparsing缺railroad。按既有Linux缓存中的纯Python资源与uv内实际Mac CP312 Pillow10.4.0构建小task profile后，在actual-run-r2从头重放并取得完整原0/1。原oracle/源码/stdlib未改，profile来源与逐文件SHA见small-task-profile-r2.json；没有下载依赖或模型，也没有把首次失败静默变成成功。

这是macOS CPU真实材料，不能证明Kaggle/Linux等价、31B梯度正确性或独立修复收益。
