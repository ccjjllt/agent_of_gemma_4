# Fixed100 SFT trainer input r2

这是当前805条 action-view 的显式版本化训练器适配。输入100题、805条、2047562个上下文token，保留全部原input_ids及assistant_loss_mask位置；attention_mask全1，labels在原mask为1处等于原token，其余为-100。没有截断、重写问题/模板、丢行或新增模型样本。最大上下文4649；8192只是数据读取上限，未测模型显存或反向容量。

数据：`encoded.jsonl`，SHA256 `ffcd3e5d366ab9100849ba8fd74d805b664743024b31646702f6651a7eb14cbf`。原action-view SHA256 `a416cda43d8733d5b90f19e6ff744b9e3e007de932922c8c178d5b510c00869d`。工具706条、完整终止99条。因无参工具schema元数据错误，`../trainer-input-r1/`仅作保留草稿，不可用于训练。

所有轨迹均为Codex编写并审阅、真实六工具执行的离策略示范，没有Gemma采样、行为logprob或生成思维。`meta.synthetic=false`沿用现有编码器“真实执行轨迹、非SyntheticTokenizer/冒烟fixture”的语义；任务来源独立记录为81道SWE-smith合成变异、11道PR复现及8道SWE-Gym PR修复。旧canonical中human命名的准入字段与旧evidence_kind仅保留证据来源含义，不声称人为创作或同策略样本。

每行绑定当前train selection、publisher原blob SHA、adapted task SHA、当前官方prompt SHA、严格canonical、实际messages和原action源文件/行/SHA；manifest的rows_source_index记录一一对应。`source-supervision-manifest.json`冻结生产时的父manifest，SHA256 `7a4553cba668b478a2d5ce13bf9cc3ddcb0a323b58fac44456fbae8564a016a4`；`source_supervision_manifest_path`仅指可更新的原推荐入口，不要求它在添加新推荐指针后仍等于冻结SHA。原100ID、角色、family、snapshot及原verifier未改。弱oracle正确示范可用于SFT，RL暂停状态另由curriculum管理。

训练器必须合入以下实际配置，并将`--data`指向本目录`encoded.jsonl`。`trainer-overrides.json`可作为这些字段的来源；模型与训练超参需另行明确配置。这份导出使用独立tools/schema/protocol，不能使用默认harness_tools或旧protocol。

```json
{
  "protocol_id": "vfarl-fixed100-codex-action-view-v1",
  "tokenizer_name_or_path": "/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/research/006/data-replay-20261013/tokenizer",
  "chat_template_path": "/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/research/006/data-replay-20261013/tokenizer/chat_template.jinja",
  "tool_schema_path": "/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/competition/vfarl/l4x4/round100/question-revision-r2/supervision-r2/trainer-input-r2/deployment-tools.json",
  "max_sequence_length": 8192
}
```

协议`vfarl-fixed100-codex-action-view-v1`的protocol_hash为`7efde6aba2f9a9131111a3ab40a49be81bcbcaa4c0428f9145055ffc7835ccfc`。真实部署tools hash为`43118e0d2729405fe1613b5ee22930cde39d701c19fb754a6f618fbc0d3f6199`，逐字采用原formatter对get_status/submit_patch的空object/properties/required默认值。实际AutoTokenizer类GemmaTokenizer的语义fingerprint为`d001718145f4980b65e70b5ffa1dfce93b3eca5a5d8de2446b8c483f2b4b8927`；原tokenizer.json原始SHA单独保留，不能替代这一语义fingerprint。

已用真实`load_encoded_jsonl(max_length=8192)`读取805条，用实际SelectedLabelCollator在CPU对全部行做混合长度右padding并核对causal shift：77187个监督目标，prompt/pad均不新增损失。`validate_tokenizer_protocol`接受明确配置。805条既有字符串由实际AutoTokenizer复现原IDs；新deployment-tools与原formatter工具列表完全相等，100个实际messages重建全部805个generation prefix与原prompt相等。证据在`manual-consumption.json`及`actual-tools-prefix-reconstruction.json`。首行原action-view被现有训练器真实拒绝的原始错误也保留在前者。

生产入口为`competition.vfarl.prepare_trainer_input`，沿用`training/encoding/gemma_tool_chat.py`的fingerprint/normalize_tools/protocol hash逻辑；生产时必须能从真实messages和tools重建原prefix，输出仍直接映射既有IDs/mask。入口拒绝覆盖目标目录。此次没有模型/优化器/GPU/新oracle重放，不表示学习或修复收益。

独立复核（2026-10-28，`/root/data_status`）未发现本范围的实质问题：逐行核对805条原IDs/mask到labels/attention及源行SHA、100个train身份和81道变异题来源；另以真实AutoTokenizer与训练器协议校验确认语义fingerprint和原formatter无参object默认声明，并用3条含最长行的CPU collator样本核对padding忽略。冻结父manifest无r2循环引用。详见`independent-consumption-review.json`；本次没有重复805条oracle或CT前缀验证，没有改manifest/encoded，辅助12条recovery仍仅保留独立action-view。
