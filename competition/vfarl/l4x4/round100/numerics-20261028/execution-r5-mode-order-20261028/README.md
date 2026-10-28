# 零反向模式对照：真实partial结果

私有版本`jiyujia/vfarl-focal-mode-order-zero-update-r5-bounded/1`，session `363028286`，载体SHA `e0e0dadc4b10e57735ccc67c8fc953d5cf9cd44fc9e6a5addf3382103b51707f`。03:25:13 UTC单次提交，03:29:38首次读取RUNNING。按既定预算在03:44:23请求取消，API于03:44:24接受；03:45:13的提交到停止确认目标未能获证。03:46:16实际GPU reserved已为0且日志END_OF_LOG，04:00:20首次读到CANCEL_ACKNOWLEDGED。这里分别保留取消请求、资源预留、日志结束与平台终态；不将请求接受当作已及时停止。

完成的是A-first进程的13次观察（动作A 8次、动作B 5次），耗363.6045秒，参数哈希前后相同。第二个B-first进程完成模型构造、原pending reference暖机及training模式切换，最后记录到B的N0开始；没有任何已保存的forward_complete或完整worker报告。因此总共13／26个计划观察，不能称为完整的交换顺序实验。原始[阶段记录](exact-output/vfarl-train/mode-b-first/phases.jsonl)与[收尾摘要](closeout-summary.json)保留这一边界。

已完成A-first进程中，模式差异持续存在：

| 同一动作的对照 | selected logprob最大绝对差 | selected hidden最大绝对差 |
|---|---:|---:|
| A：G0－N0 | 0.0600976944 | 3.0 |
| A：G1－G0 | 0 | 0 |
| A：N1－N0 | 0 | 0 |
| A：N1－G1 | 0.0600976944 | 3.0 |
| A：N2－N1（中间插入R1） | 0 | 0 |
| A：G2－G1 | 0 | 0 |
| A：R1－R0 | 0 | 0 |
| B：G0－N0 | 0.0551054478 | 2.53125 |
| B：G1－G0 | 0 | 0 |
| B：N1－N0 | 0 | 0 |
| B：N1－G1 | 0.0551054478 | 2.53125 |

表中零差对照的hidden hash也逐字相同。结果来自[实际A-first报告](exact-output/vfarl-train/mode-a-first/mode-order-study.json)明确列出的13个forward文件，完整selected LP向量、hidden hash/差统计、410层LoRA状态和执行flags均保留，未按结果挑选位置或过滤观察。这个worker没有backward，所有观察的parameter grad tensor count均为0；optimizer、rollout、参数扰动和adapter导出也均为0。其before/after adapter SHA同为`25639062b38cc88c07ead3eee8423f626bf811c60dd6f54dd81ed5b515da1656`。

[独立逐向量复算](independent-a-first-analysis.json)再次确认上述结果：A、B的hidden相对L2差分别0.0177654、0.0187842。所有5865个模块保持training，88个checkpoint属性均为`unsloth`；410层actor scaling2／reference scaling0，记录的dtype／shape／ptr／version／cache属性未变。`_caches`仅记录dict类型，内部内容没有观测，因此不能把这些字段稳定扩大成“所有隐藏缓存都未变化”。

在这次零反向路径中，重复前向与warm reference插入没有消除N/G差异，差异已存在于输出层之前的selected hidden，不能仅归因于输出层。原r4中第二动作前发生过第一动作backward，这次没有，因此原r4第二动作约1e−5与当前约0.055的差别仍需检查backward／checkpoint等持久状态变化；不据此直接改训练anchor或选择学习率，也不将唯一根因先定为BF16。

两个fresh进程实际共用一次18.295GB缓存复制，本次复制及核对186.510秒。A-first构造224.744秒，其中backend import50.52秒、model load29.81秒、adapter attach7.72秒、saved adapter核对15.25秒、loaded-base逐tensor核对110.47秒；B-first构造197.499秒、base核对110.07秒。原reference暖机完整保留，没有用已补分数据跳过。状态读取、同步与CPU复制增加观测开销，不能宣称完整时序／显存生命周期与r4相同。

取消打断了正常清理，发布的[guard receipt](exact-output/vfarl-budget-receipt.json)只有初始running状态，没有最终cleanup确认。平台最终取消ACK、GPU预留0分别由API核实；不从partial receipt推断清理完成或B参数最终恢复。

精确单文件接口在发布未完成时对nested研究路径返回404，终态后成功。最终只按白名单回收26个研究文件、9,041,205字节，见[精确下载清单](exact-output-download-manifest.json)；没有读取ZIP body或下载权重／运行时包。此前通用输出遍历在分页时中断，额外94个小型准备文件共281,923字节原样保留在`actual-output`，不计入研究观察证据。临时访问URL没有保留在精确下载清单中。

最终[账户GPU额度](final-status-and-quota.json)为used25.7523533522／30小时、remaining4.2476466478小时、reserved0；相对本次开始共享账户差0.4963740036小时。这是GPU配额口径，不能当作美元费用或CPU作业消耗。没有自动补跑第二进程，也没有下一次GPU提交。
