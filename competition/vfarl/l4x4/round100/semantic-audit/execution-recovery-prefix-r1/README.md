# 受控前缀后的真实执行恢复研究

**eligible_for_full_episode_SFT=false。禁止直接交给全episode formatter或拼入当前100题SFT。** 两条都只是Codex人为干预后的恢复监督候选；未调用normalizer/formatter，未生成canonical/SFT、logprob或新奖励，没有模型采样/训练/GPU/网络提交，未改当前100数据或算法。

选定真实轨迹只有两条：`008-moto5200/actual-run-fresh-B`的实际API值为5→7→6，`026-tenacity-fixed/actual-run`为0.25→5.0→2.5。前者人为把ARN随机后缀从5错改7，再恢复6；后者人为将0.1倍错改2倍，再恢复配置的2.5秒值。Tenacity只直接计算wait，不重试或sleep。两条各10个实际工具事件与真实终止回答，recipe/call/event/observation逐项配对，source tree与当前r2 task/prompt SHA匹配；最终patch字节等于原正确参考。原完整官方baseline/final独立保留：Moto1fail+84pass→85pass；Tenacity13fail+111pass→124pass。节点数只说明验收范围，不是模型收益。

统一边界是event4失败工具响应之后（rawmessages索引10之后）。events0–4全部只作输入前缀，错误edit是event3；它是Codex故意注入，不是Gemma行为，也不能做正向监督。event4本身是正确诊断命令，因属于固定条件前缀而mask，不表示“失败命令是坏动作”。候选恢复工具为events5/6/7/8/9；最小核心为6的正确edit与7的真实验证，5的定位及8/9控制动作可选。实际终止回答另列未来decision10/rawmessage21，它不是tool event。候选assistant rawmessage indices为11/13/15/17/19/21；所有工具观察只作输入。

后续partial formatter应保留完整真实历史，所有prefix assistant/user/tool-observation tokens为0 loss，只在明确选中的当前恢复assistant completion上给loss；实际终止文本可选。不要补写新消息、thinking或概率，也不要先用当前全量formatter把错误edit生成标签后再声称它是正常示范。本轮没有编码mask，这些只是与真实call IDs绑定的候选indices，详见recovery-boundaries.json。两条错误前缀状态都未跑original official oracle，其官方结果是null/未评估，不能把API assert失败补成official0；不是同状态偏好对、onpolicy pair或模型恢复成功率。

首个Moto `008-moto5200/actual-run`完整8事件失败原样保留：5→7后，正确6源码已写入且打印SHA0cecd13e…，API仍返回7，轨迹在event7停止。原workspace被官方stop清理，源mtime和pyc头未被日志捕获，明确不可回填。正确6源码可从原snapshot和记录edit逐字重建，SHA匹配event7；重建字节/大小及缺失字段已放入recovery-boundaries.json，临时副本在/private/tmp，未冒充恢复的live文件。此不完整运行不进入两条恢复候选，也没有最终oracle1。

另做了有界真实Moto源码缓存机制复现（moto-timestamp-cache-reproduction.json）：在临时源码中控制7/6文件相同mtime与大小3211，不同Python PID仍从timestamp+size pyc加载7；记录了实际源码SHA/mtime_ns/size、pyc16字节头/flags0、compiled integer constants及真实API。另一个新PID单加-B也读到旧7；仅删临时目标pyc后-B才加载6。这里确认的是匹配的新进程缓存机制，未声称找回首轮原pyc头。选定fresh-B轨迹从无source pycache开始，三个新PID实际显示bytecode_writes_disabled=True、workspace_bytecode_exists=False、loaded constants5→7→6；-B不禁止读取已有pyc。

只读审了官方SubprocessManager环境构造：host设置PYTHONDONTWRITEBYTECODE会被SAFE_ENV allowlist丢弃；底层builder支持显式extra_env，但Manager.exec当前没有caller env参数。每次exec确实调用新Popen并在结束清理进程组，当前heredoc不是同进程已import未reload。未来若root选择新runtime policy，collector需显式记录并向所有repo subprocess注入flag，且在workspace materialize后、任何source preload/import前只清生成的source字节码缓存；不可仅设host env或宣称-B全面禁读，也不处理同进程reload语义。只读来源与提案在prospective-runtime-policy.json；没有改主harness或旧runtime。

边界event编号是原始tool-event顺序，不等于charged tool budget计数；JSON保留event4后的真实budget_after。root新发现旧formatter的read_text会CRLF→LF而沿用旧offset，本目录未编码，不受该错位污染。未来恢复-only视图必须等换行保真修复，从真实字符串重新计算span；这里的rawmessage/call indices不能直接充作token offsets。旧bad SFT不覆盖。

2026-10-28后续：已另生成recovery-only-sft-r1辅助视图，只用选定恢复spans监督；原boundary JSON保留前轮未编码快照hash，原真实logs不变。完整episode依旧禁用，辅助视图不混入当前100主SFT。视图最大5250tokens，9/12动作超4K，未截观察。
