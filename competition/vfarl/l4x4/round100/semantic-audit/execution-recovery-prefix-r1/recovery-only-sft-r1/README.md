# Recovery-only SFT辅助视图

2026-10-28。只含固定100索引8 Moto5200和索引26 Tenacity wait_fixed的两条已记录恢复轨迹；未重跑模型、工具或原官方oracle。原始logs、recipe、baseline0/final1与当前100主SFT都未改。前缀由Codex人为错改，不是Gemma采样，也不是RL pair/新奖励或模型收益。

**eligible_for_full_episode_SFT=false；eligible_for_recovery_only_SFT=true；included_in_current100_main_SFT=false。** 推荐的独立辅助文件为[recovery-actions.jsonl](recovery-actions.jsonl)：12条决策（10真实恢复工具动作+2实际终止回答）、1060个监督tokens，SHA `e6e40c6758f2b493f33253bef3417660d5f09ed68758bda3f09b3105863da294`。此文件仍是action-view，不可直接作为现有train_sft/train_fsdp的--data；当前固定100主分支仍要求100题严格canonical，不接收本辅助研究视图；独立recovery-only生产分支现已导出[trainer-input-r1/encoded.jsonl](trainer-input-r1/encoded.jsonl)，协议与消费说明见[独立README](trainer-input-r1/README.md)。恢复边界partial provenance保持独立，不放宽full-episode准入。实际最大上下文5250，9条超过4096；完整上下文保留，不能直接当4K材料截断使用。

边界固定为失败event4响应之后，即rawmessage10之后。输入完整保留原user、events0–4的assistant动作与所有工具观察；这些prefix tokens全部为0 loss。输出只含原action indices5/6/7/8/9/10，对应raw assistant indices11/13/15/17/19/21；10是实际终止回答，不是tool event。每条样本仅当前恢复completion有loss，连历史已恢复动作也只作当前样本的输入。核心动作6改正源代码、7真实验证，另含5定位、8状态、9提交和实际终止文本。

屏蔽按原消息/call身份，不按文本去重。event4失败验证和event7成功验证可以是同一命令；这里只监督event7的真实成功状态下动作。正确edit6的old_string必须包含人为错误源码7/2倍，这不等于把错误edit3的new_string作为正向标签。两条闭环已人工逐条核对：Moto5→7→6，Tenacity0.25→5.0→2.5，actual edit6的old_string等于干预edit3的new_string；最终真实API通过且正确full oracle沿用原证据。未评估的干预状态official result保持null，不补0。

严格复用已修复[format_replays.py](/Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/competition/vfarl/format_replays.py)的schema_for_template/action_examples，固定SHA `5a9ce116d2c7835c854e17b3109812bec5050037dc865cff94bcd8b78bca04eb`。没有调用render_episode或CLI全监督入口，也未创建完整episode的input_ids/loss mask。这里只先渲染无标签审计上下文，再提供6个明确选中的span给action_examples。context-records.jsonl和*.context.gemma.txt只作审计，不能作为全监督训练输入。

上下文按UTF8 bytes写入，修复后的action_examples按原bytes+SHA读取；Moto52处CRLF保真。把它们转换为LF会错位这6个所选completion，已有静态对照记录，但没有产出错误标签。tokenizer/template继续使用已验证31B CT研究006缓存；CPU实际transformers5.13.1/tokenizers0.22.2/jinja2 3.1.6。无thinking、behavior logprob、onpolicy或生成reasoning；模板强制空thought属于prompt且mask0。

manifest逐条绑定boundary/rawmessages/recipe/events/run-summary/原官方报告、初始prompt、task、source tree、tokenizer/template/formatter及每个prompt/completion/token IDs/mask SHA。原source、rawmessage首prompt和当前r2 task SHA均核对，12个completion均精确来自实际选中的原message；末尾仅增加原模板turn结束符。原boundary JSON是前轮冻结的“尚未导出”快照，保持原hash；本轮是否编码以本manifest为准。

第一轮builder缺required initial_prompt_augmentation元数据，在任何action row写出前停止，只有无标签context文本；补齐实际prompt元数据后完成。没有绕过canonical准入或把研究trace伪装成完整canonical，也没有把错误prefix先导出为全监督再过滤。
