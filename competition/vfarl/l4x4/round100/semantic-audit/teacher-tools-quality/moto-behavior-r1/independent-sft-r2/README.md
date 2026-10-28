# 两题Moto行为示范：独立Gemma SFT增量

2026-10-28。仅包含固定100内两个train ID：`getmoto__moto-5200`、`getmoto__moto-6609`，两个G common任务。输入为 [gemma-sft/sft-actions.jsonl](gemma-sft/sft-actions.jsonl)：2题、16条助手决策（14次真实工具调用+2条实际终止回答）、1349个助手监督tokens，最大动作上下文3652tokens，均在4096内。动作文件SHA256为 `20de65f2593e50c9458bc3ce3aeed09914875ec42d49dc63597882829d32bd22`。

这是Codex off-policy新示范；thinking_budget=0，空thought控制为prompt且不进loss。无模型rollout、行为logprob、生成reasoning或训练更新。旧S600/F600、其manifest以及原有同ID语法检查示范没有重写或自动替换。新增行为观察并不代表训练收益。

每题新recipe/messages/tool-events/验收均绑定邻近对应 `actual-run-r2/`。完整原官方文件范围、原F2P/P2P和题面不改；5200实际baseline1失败84通过→reference85通过；6609实际baseline3失败113通过→reference116通过。真实source API断言在改前失败，改后成功，source_tree及initial_prompt字节SHA匹配旧真实pin；详细来源/失败环境见上一级README。首次官方收集失败的actual-run不进入本导出。

原source task和机制review使用原真实pin。缺失原blob已按review明确缓存单行精确SHA恢复，无近似原source。`source-catalog.json`仅复制原两条来源记录；新raw reports只写本次实际指针和review信息。未改动normalize_results、export_prompts或format_replays，严格准入两题eligible/reasons为空，没有例外。官方builder prompt逐字等于新实际prompt（保留CRLF），没有collector侧prompt增补。

复用已验证31B CT的research/006 tokenizer与模板：tokenizer SHA `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`；template SHA `94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413`。编码使用既有CPU transformers5.13.1/tokenizers0.22.2/jinja2 3.1.6。一次性手动核对16条均只含这两个train ID，token/mask长度、监督数、thinking0、无logprob及真实14条工具来源一致。没有新增测试文件、网络或GPU。
