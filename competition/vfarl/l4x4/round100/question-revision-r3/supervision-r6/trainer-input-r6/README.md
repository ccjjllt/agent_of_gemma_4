# R6现有SFT trainer输入

直接消费[encoded.jsonl](encoded.jsonl)，不要把父目录的action-view JSONL交给trainer。encoded SHA`6329db6393dbf703709d7427e923f0fb392256114ff2c3472305bf359e012c0f`；[child manifest](manifest.json) SHA`ceeca71f249965a56a573edf60428637cdc8e6fe69e63f5cc50de159b52be616`。805行覆盖固定100 train，78,924个shift监督标签、2,062,458个输入位置；706工具动作与99最终回答全部保留。

必须采用[trainer-overrides.json](trainer-overrides.json)中的协议与实际[deployment-tools.json](deployment-tools.json)，并加载匹配CT tokenizer/template。协议为`vfarl-fixed100-codex-action-view-v1`；tokenizer采用trainer语义fingerprint `d001718145f4980b65e70b5ffa1dfce93b3eca5a5d8de2446b8c483f2b4b8927`，不是拿tokenizer.json文件SHA替代。工具SHA`43118e0d2729405fe1613b5ee22930cde39d701c19fb754a6f618fbc0d3f6199`，模板SHA`94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413`。无参工具保留实际formatter的object默认schema。

所需配置为`protocol_id`、`tokenizer_name_or_path`、`chat_template_path`、`tool_schema_path`及`max_sequence_length`；具体绝对路径在overrides中。示例数据路径为`competition/vfarl/l4x4/round100/question-revision-r3/supervision-r6/trainer-input-r6/encoded.jsonl`，输出目录须另选新路径。当前overrides使用8192上限，实际至少需4,649；19行7题超过4K，没有截断或丢行，也未证明31B加载或反向容量。

固定producer `3455007f…`实际重建805个公开任务／真实messages前缀，逐行复制原IDs及mask，并映射为labels/全1 attention。现有`load_encoded_jsonl`消费全部805行，实际GemmaTokenizer通过部署协议核对；一次18×4649的CPU collator覆盖16新行及全局长短行，prompt及39,627个padding位置保持ignore。具体证据见[manual-consumption.json](manual-consumption.json)。第一次一次性观察误读不存在的`meta.instance_id`，在collator前因KeyError停止；改用真实协议的`meta.group.issue`后消费通过，导出产物未改。

来源为Codex指定、实际执行工具的off-policy示范，thinking0，无Gemma采样、旧logprob或RL pair；legacy human schema名称不表示用户手写或on-policy。task-level来源仍单独保留81项synthetic mutation、11项SWE-smith PR及8项external PR，不把真实工具轨迹冒称天然human题。当前selection与课程r7原样绑定，7weak＋4unknown的未来RL暂停不删除正确SFT或改旧奖励。

导出输入父manifest已逐字冻结为[source-supervision-manifest.json](source-supervision-manifest.json)，SHA`132ec073f7c3669fd5a2a0913c81922abbcd959c475c2015db33ad90da79b4d2`。父目录随后添加本child推荐指针，冻结副本不再回写。相对R5只换0/39共16行，其余98题789行源bytes保持；R5 parent/child/report/data未改。没有加载模型、执行optimizer或GPU；draft模型配置的独立更新不改变本输入协议证据。
