# 真实31B零更新数值诊断：r4 bounded

2026-10-28，用户在新额度窗口明确要求继续优化后，仅一次私有提交`jiyujia/vfarl-focal-numerics-zero-update-r4-bounded/1`，实际session `363019705`。02:38:47 UTC提交，02:57:15读取到COMPLETE，早于辅助取消时间；没有取消请求、重试提交或正式训练。源`67968495…`精确包裹原`ca576bc7…`数值payload及37项冻结源，不包含工作区后来的修改。来源及实际观察集中在[收尾摘要](closeout-summary.json)。

实际分配4张NVIDIA L4，模型使用其中2张。完整输入分别2415／2413 tokens；8K只是上限，不能作为16K容量证据。已有18,295,563,073字节NF4缓存的本地复制和逐文件SHA核对耗424.998秒，没有新模型下载。远端wrapper总运行982.149秒，child returncode 0，cleanup confirmed，无错误及存活后代；[预算原件](actual-output/vfarl-budget-receipt.json)没有自行推断研究完成。研究完成来自[实际numerics-study](actual-output/vfarl-train/numerics/numerics-study.json)，SHA `568a059ec28c90beb1df84deecb32c6a78ecbe4b8c1c92b862444b58a2c7f87c`，状态`zero-optimizer-numerics-study-completed`。

本次有真实完整焦点反向，梯度范数0.4231025346，optimizer steps=0、new rollouts=0、adapter exported=false。暂态方向探针结束后实际adapter tensor SHA恢复为`25639062b38cc88c07ead3eee8423f626bf811c60dd6f54dd81ed5b515da1656`，与开始值相同。两卡峰值allocated分别9,952,005,120及13,213,278,208字节。数值研究入口耗446.984秒；这些是机制数据，没有独立任务收益。

同一实际梯度前向的8个源结构位置用于输出层隔离，hidden维5376、词表262144、softcap30。融合实现对分块FP32参照的selected logprob最大差9.06e−6，hidden梯度余弦0.99999648、相对L2差0.00275863；普通BF16投影对同参照的logprob最大差0.0292586、hidden梯度相对L2差0.0964236。该局部8行比较支持融合head的这组数值行为，不能单独证明整个NF4 backbone反向正确。

方向探针保持相同grad-enabled前向、固定old/reference/IS，使用测得梯度的负sign，anchor目标0.03164151310920715：

| 暂态epsilon | 一阶预测目标变化 | 实际负sign方向目标变化 | 中心差分导数 |
|---|---:|---:|---:|
| 1e−6 | −0.00242084723 | +0.00042729080 | +312.73067 |
| 5e−6 | −0.01210423617 | −0.00091673434 | −153.72038 |

第一个动作grad/no-grad selected logprob最大差0.07084899，第二个仅9.5889e−6，存在明显不对称。暂态探针anchor与原backward anchor相等，但现有一次观测尚不能区分first-call热身、模式／reference切换、算子行为与BF16有限精度等来源，也不能选学习率或证明旧short-r1负更新的唯一根因。没有接受、选择或训练任何探测参数。

只对原先缺失的原始动作补了1次真实reference前向；[新两行数据](actual-output/vfarl-train/numerics/focal-reference-completed.jsonl)SHA `a9e2e282ebfab39e2f625b5ddc77c251c8ba9c8a0bd86d4cfa39b25582fee5bb`。一次性逐字段比对确认：原始行仅5个reference相关字段改变；原branch1行完全相同；两行行为概率、prompt／completion tokens、mask、奖励及采样身份均原样保留。原pending文件SHA `aa988529…`未变。[reference来源](actual-output/vfarl-train/numerics/focal-reference-completed.reference.json)记录新的评分执行路径，不冒充当时采样概率；这批行为尚未消耗训练机会。

root另使用当前`train_vfarl.py`真实`prepare_rows()`及fixed-scale分母函数读取该新数据，得到2行／1异奖组／分母2，优势−0.5／+0.5，selected tokens99／97。见[当前消费记录](current-consumer-reference-admission.json)。此为CPU数据可消费性，没有模型加载或optimizer；不能将它当作后续工作区源码已完成GPU兼容验证。

最终[真实账户额度](final-status-and-quota.json)为used25.2559793486／30小时、remaining4.7440206514小时、reserved0；相对本次开始账户差0.5488261111小时。此为共享账户GPU配额口径，不是美元费用，也不归因于并行CPU作业。小型原件与哈希见[下载清单](actual-output/download-manifest.json)，未下载权重或运行时大包。后续没有自动提交新GPU作业。
