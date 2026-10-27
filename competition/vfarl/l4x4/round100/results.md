# 原100题16K RL实际执行

北京时间2026-10-27 16:22:57，原材料上传完成；16:23:40回读确认私有数据集v1 ready，文件124,593,514字节。

16:23:56，GPU Notebook v1已显示RUNNING，回读私有设置、输入和源码SHA一致。

作业：https://www.kaggle.com/code/jiyujia/vfarl-train100-16k-resident-r1

固定原67共同＋33专科题；16K上下文，Unsloth双常驻副本，每份使用两张L4。每批最多4题，累计2组真实异奖后更新并同步权重。

初始LoRA沿用已完成短轮的导出，Adam在本轮首次创建后持续复用。采集新行为，不复训此前已消费的奖励批次。

提交前实际剩余配额17.43349小时；新任务领取预算7小时，平台会话时限30,000秒。实际覆盖和更新数以输出记录为准。

本记录仅确认提交和平台RUNNING。模型ready、真实采样及optimizer更新尚须实际日志证明；不能将该状态当作训练完成或收益证据。
16:28:02 实测GPU 0–3均为NVIDIA L4，单卡23034 MiB，驱动查询无错误。NF4本地复制18,295,563,073字节，用时144.98秒。随后两个worker均发出resident_model_loading，context=16384，初始adapter来自已完成短轮。

对上轮短更新的只读数学复核：matched no-grad奖励方向−0.000109963；固定IS的PPO＋KL目标上升0.000383885，其中PPO项0.000375365、KL项0.000008520。公式、mask及next-token索引未见明确反转；IS浮点舍入不足解释该变化。本轮记录gradient_dot_parameter_delta及同目标的实际前后变化，以区分优化器方向、有限步长和前向/backward数值问题；完成step本身不代表收益。

16:33，两个resident模型均ready，完整策略ID一致，16K上下文、100题池。首批开始领取任务。16:35，Dominate首个真实动作完整返回127 tokens，finish_reason=tool_calls，耗时90.19秒（首次prefill39.61秒）。权重加载进度中的1188指张量条目。

16:35:31实读剩余GPU配额17.042468小时，启动消耗0.391025小时。当前只有真实采样证据，尚未观察到本轮optimizer step；常驻采样—更新循环继续运行。精确事件见startup-evidence.json及live-events.jsonl。
