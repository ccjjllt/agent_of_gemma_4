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

北京时间17:10:05的实时日志观察：首批仍在采集，尚无本轮optimizer事件。剔除原动作replay后63动作、8699输出token；sample累计3863.87秒（两worker相加），原episode扣prefill热decode约2.525 token/s。KV55/63命中（87.3%），命中prefill中位0.723秒；动作到下次请求中位间隔约0–1秒。Dominate唯一已完成focal组奖励[1,1,1]，无异奖。worker1已结束后因4题批屏障空等约209秒；当时GPU 0–3利用率21/6/0/0%。原始事件及计算见utilization-live-events.jsonl、utilization-evidence-summary.json。

源码核对：Unsloth loader的fast_inference=False，沿用单序列Python逐token sampler，每token有host同步；两卡balanced层分片非TP，focal reference也逐token评分。async episode内部sample与工具均同步阻塞，分支顺序执行。当前低吞吐主要证据指向decode实现及整批屏障，慢工具与跨题prefill不是本观察段主因。

工具边界：六基础工具取自官方create_tools(ctx)，使用官方SubprocessManager、SwegemmaContext与verify_task；研究循环为自写VFARL，executor外包chroot/UID/seccomp，未完整复刻官方默认Docker的4GiB/2vCPU限制，不能声称完整官方Docker环境等价。

后续优化优先：真正的批量推理backend与独立session/KV；冻结采样窗口内动态补任务以减少短任务等待；工具移出事件循环与独立sandbox进程池；分支公共前缀缓存。更新前仍排空本策略在途动作，保留真实token/logprob/mask及权重身份，不能事后改写已采样命令或把旧验证结果当新奖励。

17:25:20，新作业 jiyujia/vfarl-train100-16k-batched-r2/1 平台RUNNING，私有输入与冻结源码SHA 0cbe7709007822a2ad68f04a124b697f2587e00cbbf9370f53ca8fd4b1a28572 回读一致。

R2：每双卡副本4个独立任务session，生成action microbatch最大2，共用单owner推理线程；采用metadata-only共同/专科交错及repo轮转，动态补未领取任务直到2组真实异奖或时间预算。原100任务保持，排除旧轮已领取4题，续未领取96题；旧轮3完整＋1中途取消分别记录，不能算100题全部完成。

奖励/PPO+KL目标不变，actor执行profile更新为v3；保留实际生成token、概率、forced mask、batch/padding和配置来源。工具、验证、快照移出event loop并按独立task root锁；不同任务workspace/context互不共享。更新前排空在途请求，Adam在本轮各步持续复用。reference改为256-token backbone块与16-token head投影，独立zeroLoRA评分。

实现取舍：当前并非continuous slot refill；早结束行等待同batch，action batch之间不保留KV。一次性tiny Gemma4 CPU结构核对误差小于5e-7不代表真实NF4 GPU速度或16K显存证据，待实际batch日志。GPU实读启动配额15.728333小时；新claim预算18000秒、平台session时限25200秒。

17:38:23 GPU实际证据：两模型ready身份一致、16K/96题池，每副本4独立任务真实进入采集；两个副本均已完成真实B2。worker0首B2共453 tokens，prefill47.53秒、decode127.62秒（3.5495 token/s）；worker1首B2共512 tokens，prefill7.917秒、decode113.957秒（4.4929 token/s），含prefill整体4.2003 token/s。按唯一(worker,batch_index)计时，未将两条action的batch wall重复加总。任务与长度不同且仍含冷形状开销，只能视为初步吞吐信号，不是受控倍速或RL收益。最新显存9250/12692/9276/12728 MiB，无worker错误；尚未本轮optimizer更新。

旧取消轮已正规API完整恢复：43研究文件、68动作，actual_optimizer_steps=0；唯一完整focal组无异奖。exact-token数据与队列SHA匹配。详细凭据见cancelled-r1/recovery-summary.json。

18:25:27，R3 jiyujia/vfarl-train100-32k-session-r3/1 平台RUNNING，私有设置/输入/源码回读一致，source SHA 073af7857f11cb7f11fe886e67c5002c31d6c22e3f511e8dd8ec662779317afb。

R3容量设置32K，未强制把任务填满；每副本6独立任务，max action batch4，padded active token预算32768、inactive session KV LRU预算32768；长请求自动拆批。单次响应1536、思考256、episode输出8192。原100题排除前两轮已领取16题，续84未领取，partial与completed分别保留。新claim预算16200秒、session23400秒，启动前实际剩余配额14.1809869小时。

v4机制：各session按policy ID与严格prefix复用B1 KV，增量prefill新增工具后缀后合并B2/B4；行停止当步提取full/sliding缓存，避免dummy尾部挤掉有效sliding窗口；独立reference缓存，episode结束release、更新前aclose排空清缓存，保留真实old概率及mask。实际tiny Gemma4 CPU手工核对跨滑窗/不同prefix/B4尾部、下一turn命中与LRU/epoch，最大KV差约1.27e-5；这不是GPU吞吐或32K反传证据。

本地官方31B配置的理论BF16 KV：50层1024滑窗+10层full，单序列16K约2.03GiB、32K约3.28GiB，未含权重/激活/allocator/打包克隆峰值；R3会直接记录实际cache layer类型。官方vLLM/PP2+NF4为后续独立引擎候选，当前并未接入。

19:06:51 R3实际GPU证据：B4与v4 session缓存已跑通，两副本策略身份一致，32K上限/84题池/每副本6session。缓存类型实际50 Sliding+10 Full。33个唯一batch，B4 13/B3 15/B2 3/B1 2，排首形状后整体4.377 token/s；最近十分钟整体4.190、decode4.750 token/s（两副本unique batch时间相加口径，非整机墙钟tps），prefill占11.8%，cache87/105命中、复用306382 tokens。R2长上下文窗口2.565 token/s仅作粗参照，任务不同不是受控实验或RL收益。

四卡最近十分钟均值26.2/64.9/16.0/62.9%，整体42.5%，显存13878/21288/13998/21088 MiB；无worker异常/OOM。长上下文自动缩B3、LRU出现淘汰重prefill；重卡接近容量，不能无依据扩大KV预算。每题请求还存在73–83秒队列等待及241–343秒总延迟，吞吐收益不代表单题反馈延迟下降。

当前13题领取，仅1 episode结束且reward=null/verified=false，不能计失败或成功；focal0/update0。最长实际前缀约9.3K，不能声称实际32K前向或反传已被验证。

R2取消结果后续已完整正常Range恢复：确认12领取/4完成/8中断、focal0/update0；此前publication_pending已解除。原始malformed为真实多余引号，旧严格parser未误拒规范参数。768截断两类分别是长工具脚本与重复^退化输出，增加单次预算只可能帮助前者。
