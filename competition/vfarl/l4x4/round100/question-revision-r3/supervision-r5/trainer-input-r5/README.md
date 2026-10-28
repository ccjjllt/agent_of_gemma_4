# Fixed100 SFT trainer input r5

这是当前805行／100题的现有训练器格式导出：78,423个causal-shift监督目标，encoded SHA `a5dea1aecd85bf8478d29cd1ebf2588df498a3baa8c0ee4b7d20eaa43f20d704`。源action-view SHA `ed99e56fc669134945766e563b604f678df91d4aef3c9037bed86acc26578bc1`。只把原IDs／mask映射为attention1与labels(token／-100)，不重编码、不丢行、不截断。

使用本目录 [encoded.jsonl](encoded.jsonl) 作为 `--data`，将 [trainer-overrides.json](trainer-overrides.json) 合入另行明确的模型／超参配置：`protocol_id=vfarl-fixed100-codex-action-view-v1`；tokenizer与chat_template使用研究006 CT缓存；`tool_schema_path` 使用本目录 [deployment-tools.json](deployment-tools.json)。不能直接把source action-view当作reader输入，也不能沿用不匹配的默认tools／protocol。

真实GemmaTokenizer语义指纹 `d001718145f4980b65e70b5ffa1dfce93b3eca5a5d8de2446b8c483f2b4b8927`；tools hash `43118e0d2729405fe1613b5ee22930cde39d701c19fb754a6f618fbc0d3f6199`；protocol hash `7efde6aba2f9a9131111a3ab40a49be81bcbcaa4c0428f9145055ffc7835ccfc`。无参工具使用与原formatter相同的object／properties／required默认值。producer源码 `3455007ffdf35820255b91f5e857bfd420ef30cb1309f0266d7aa33ed12145e2`、formatter `5a9ce116d2c7835c854e17b3109812bec5050037dc865cff94bcd8b78bca04eb` 均未改。

仅Hamming81的8行替换为实际truncate／integer-aware teacher，99题797行源字节保留；全部706工具＋99terminal、51题真实新重放与49题历史来源保持。所有trace为Codex scripted/off-policy，无Gemma生成思维或行为概率。题目级81 mutation／11 PR复现／8 Gym来源与实际trace标记分别绑定。

[source-supervision-manifest.json](source-supervision-manifest.json) 是生产导出时冻结的输入父manifest，SHA `9e6999d2493a454b2e4360a67e75da9e4d96162010c1edb398a1f9ff3d0dafe9`。父文件在该冻结之后才增加child推荐及实际消费记录；该副本保持原字节，避免循环hash。另有R4最终parent原件冻结副本，R4全部旧文件保持。

生产导出器实际重建805个部署模板前缀；`load_encoded_jsonl(8192)`读取805行。实际 `validate_tokenizer_protocol` 与一次CPU `SelectedLabelCollator` 的10行混长batch `[10,4649]` 通过，包含8个新Hamming行和全局最短／最长；1048个batch监督目标，所有prompt及pad都忽略loss。完整805行IDs、attention和mask→labels逐项一致，详见 [manual-consumption.json](manual-consumption.json)。最大上下文4649，19行／7题超过4096；8192仅为reader上限，不表示硬件反向容量。

数据绑定selection SHA `7a6990034f651b932b8d218c8613df7d8f435925e72ef2624d3cc238ecdb893e` 及课程r7。未来RL暂停7weak＋4unknown的决策与正确SFT保留均不改旧奖励。没有加载模型、optimizer或GPU；当前31B NF4模型的加载／训练入口兼容仍是单独工作。draft模型训练配置未在这里更新。
