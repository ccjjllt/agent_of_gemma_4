# 固定100监督r6：补足0/39的真实修前API观察

本版本以[冻结监督r5](parent-supervision-r5-manifest.json)为父，只替换Dominate0和Mistune39各8行；其余98题789行、顺序及原始行字节全部保留。当前仍为100题805决策，含706工具动作与99最终回答；51题413行来自本轮真实teacher，49题392行沿历史来源。监督标签为78,924，累计输入位置2,062,458，最大上下文4,649；19行、7题超过4K。

Dominate0原teacher的修前命令只成功import。新[真实重放](../../semantic-audit/000-024/dominate-visible-reproduction-teacher-r1/README.md)前后执行同一`div()[10]`及消息断言：修前实际输出`IndexError list index out of range`并exit1，修后返回正确异常文案并exit0。移除了纯来源打印，保留精确工作区模块路径断言。原24文件bytes/modes、当前公开prompt、int60预算及510-byte正确patch不变；原完整65-case oracle真实0→1。[独立复核](../../semantic-audit/000-024/dominate-visible-reproduction-teacher-r1/independent-consumption-review.json)另核源身份、原scope及8个实际CT动作。

Mistune39由旧私有helper观察改为[公开MarkdownRenderer实际重放r2](../../semantic-audit/025-049/mistune-public-renderer-teacher-r2/README.md)，前后渲染同一`"- list\n- \n"`并断言完整返回值。命令仅catch预期`IndexError`，从真实exception对象及`__traceback__`输出类型、消息和最后工作区帧`src/mistune/renderers/_list.py:43 in _render_list_item`；其他异常不catch。修前共同断言仍exit1，修后真实返回原Markdown并exit0。原122文件bytes/modes、公开prompt、int60预算、460-byte patch及完整945-case oracle均保持，真实0→1。[独立复核](../../semantic-audit/025-049/mistune-public-renderer-teacher-r2/independent-consumption-review.json)核对了这些绑定。

Mistune完整framework traceback的[新r1](../../semantic-audit/025-049/mistune-public-renderer-teacher-r1/README.md)仍原样保存。r1→r2是重新执行的命令，不是对旧stdout、prompt或mask做过滤：8行累计输入位置34,368→18,450，最大上下文5,623→2,998，而监督标签551→885。相对当前父r5，R6总输入位置反而增加4,626，标签增加501，因为补上了两题缺失的真实修前API观察；没有模型、GPU吞吐或修复率收益结论。

直接训练消费使用[trainer-input-r6](trainer-input-r6/README.md)，源[action-view](sft-actions.jsonl)不能直接传给trainer。固定producer `3455007f…`实际重建805个CT prefix，沿用formatter `5a9ce116…`；现有reader核对805行IDs、attention及mask→labels，18行CPU混长collator覆盖16新行及全局长短两端。[组装记录](assembly-observation.json)和[独立轻核](independent-assembly-review.json)保留具体来源、SHA与范围。

selection仍为question revision r3 SHA`7a6990034f651b932b8d218c8613df7d8f435925e72ef2624d3cc238ecdb893e`，课程仍为r7 SHA`f51cdba0d2a93d8d910536ba2997c46622a0b90a67efd8bd7afd65be14a916b7`：59待采为32共同＋27专科，7项weak oracle与4项runtime unknown仍暂停未来RL。正确SFT保留全部100题；原奖励、oracle、R5及更早数据不改。所有99终止行保留，辅助recovery视图未混入。训练配置草稿由独立配置工作管理；本次未加载模型、启动GPU、optimizer或正式训练。

[manifest.json](manifest.json) SHA`37c44e710133c057310cad31f72b53ca3b8a6ef00d2a68682d60806b4fc2bf19`；actions SHA`9f0cfee5ce3dbb62b4363131f8f78bf9f342bbca9e6f79a7eedf36639fc56907`。child冻结的是导出前父manifest SHA`132ec073f7c3669fd5a2a0913c81922abbcd959c475c2015db33ad90da79b4d2`；其后父manifest只添加消费结果与向前推荐，不形成循环hash。
