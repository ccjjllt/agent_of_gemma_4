# 异奖筛选后的k3不保证拉近完整reference

2026-10-28。一次CPU float64精确枚举，实际调用生产`reduce_focal_terms`；没有模型、采样轨迹、optimizer、GPU或配置修改。[完整输入及结果](mixed-group-k3-selection-20261028.json)包含全部9个有序动作对。

取三个单token动作，actor与冻结learner均为`p=(.77,.22,.01)`，reference为`q=(.85,.12,.03)`，确定性奖励为`(0,0,1)`。每组独立采两次，完整动作分母为2，固定token尺度128，TIS=1；这里只隔离beta=1的k3分量，PG不参与。采样概率在微分时冻结，不能对它一并求导。

当前mixed-only regularizer只在异奖组计算。给定动作a，其对端异奖概率为`w(a)=1−P(R(a))`，所以本例`w=(.01,.01,.99)`，整个mixed概率为.0198。其总体固定数据目标为：

`J(theta) = sum_a p(a) * w(a) * k3(q(a), pi_theta(a)) / 128`

在`theta=log(p)`，令`z=w*(p−q)`，则logit梯度为`[z−p*sum(z)]/128`。实际梯度的下降方向为`(-.00011165625,-.0000415,.00015315625)`。

沿此方向作单位logit步，自己的冻结mixed-k3目标下降约`3.76390e-8`，完整`D(q||pi)`却增加`1.72010e-6`，完整`D(pi||q)`也增加`5.61023e-6`。这不是backward与自身目标冲突：它确实在降低一个受到奖励类别筛选的目标，完整reference距离是另外的量。独立纯数学重算确认了w、梯度及有限步数值。

在同一枚举中，让所有完整组都计算k3（w全1），该点下降方向令两种完整KL都下降。这里仍须区分：固定数据k3在anchor取`D(p||q)`的值，但因采样权重不微分，其梯度对应`D(q||p)`，不能宣称两个可微目标整体等价或全程下降。

目前代码与文档明确声明mixed-only sampled regularizer，因此本项增加的是定量机制边界，并未找到“实现不符合声明”的新bug。不能用mixed-k3日志下降证明完整KL变好。若以后比较全cohort正则，还须统一处理全同奖cohort当前不更新、Adam时钟、样本分母及额外评分成本；只在异奖update里多算几行不等于此处无条件all-complete对照。

本例的非零adapter/reference差异也不同于旧short-r1的首个零adapter anchor，不能用来解释该轮负方向。单位logit步不是学习率或Adam方案；尚未据本反例改变实际loss、启用新算法或声称31B收益。
