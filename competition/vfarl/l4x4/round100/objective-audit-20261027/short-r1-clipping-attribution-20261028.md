# short-r1：旧概率向量中的 clipping 归因

一次性 CPU 数值研究；只读取四个真实训练动作的概率/数字及已 pin 的冻结源码，没有模型、optimizer、GPU、问题正文、heldout、概率改写或源码/epsilon 修改。原 mode reconstruction 报告和所有旧产物保持原 SHA。完整来源、精度、逐行/逐 token 分解见 [short-r1-clipping-attribution-20261028.json](short-r1-clipping-attribution-20261028.json)。

固定原 learner_old、reference、实际 FP32-rounded rho 和四动作等权均值。545 是 selected-token 总数，未改成全 token pooled mean。unclipped 仅为同向量上的诊断：将 PG 中 min/clamp 换成 `-r*A`，KL 完全相同，不是替代训练目标或新的模型结果。

| FP64 复算，保留实际 FP32 rho | pre-no-grad | post-no-grad | 增量 |
|---|---:|---:|---:|
| 原 clipped PG | 0.235803361982107 | 0.236178787609493 | +0.000375425627386 |
| 诊断 unclipped PG | 0.235803361982107 | 0.236129321425521 | +0.000325959443414 |
| 同一 KL penalty | 0.000000000000000 | 0.000008520398140 | +0.000008520398140 |
| 原 clipped 总目标 | 0.235803361982107 | 0.236187308007633 | +0.000383946025526 |
| 诊断 unclipped 总目标 | 0.235803361982107 | 0.236137841823661 | +0.000334479841554 |

clipping 的保守项为 **+0.000049466183972**，占原匹配 no-grad 目标上升的 **12.88363%**。其余 unclipped PG 增量 +0.000325959443414 和同 KL 增量 +0.000008520398140 仍使总目标上升。混合 pre-grad→post-no-grad 口径的相应比例为 12.88566%，结论相同。

真正 active min-clipping 只有第四行动作（A=+0.5、165 tokens）的 selected-vector 索引97、117（从0开始），ratio 分别1.312368368516109、1.221597065663753；两项对全四动作目标的贡献分别 +0.000041491556562061、+0.000007974627409792。该行动作还有一个 ratio=0.6755342014546163，虽越过下界但 `A>0` 时 min 仍取原 `r*A`，clipping excess 为0。所以它的 outside fraction 为3/165=1.81818%，active fraction 为2/165=1.21212%；前三行动作无 outside/active token。全部545 tokens 的 active 是2个。pre-grad、pre-no-grad 两个锚点均无 active clipping。

原 weighted `mean(rho*A*action_mean(delta_logp))` 仍为 -0.000272085411372，完全不依赖 min/clamp。其负值已使 unclipped PG 的线性项为 +0.000272085411372；有限 exp 余项再增加 +0.000053874032042。post 诊断 clipping 不能解释这个原始 logp 方向，也不能作为已记录更新梯度的成因。

主口径用保存的 FP32 数值、实际 FP32 rho，在 FP64 中做 exp/reduction；不是旧 GPU scalar 的直接重跑。另存 CPU FP32 旧公式复算，matched 增量为 +0.000383991748095，clipping 直接项为 +0.000049466161727，单独 reduction 的末位不加和误差在 JSON 中保留。

仍没有 post-grad 向量，无法恢复同一 grad-enabled 路径的有限步目标或根因。这里不能推导放宽 epsilon、重训旧 batch、backward 正误、独立任务修复收益或 unclipped 训练轨迹。
