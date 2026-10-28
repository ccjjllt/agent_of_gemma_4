# short-r1 真实数值：grad / no-grad 锚点重建

2026-10-28，一次性 CPU 数值复算。只读旧训练数值产物、训练 focal 和 carrier 中明确选取的冻结源码；没有读取 heldout/预留验证题内容，没有模型 forward、optimizer、GPU，也没有改写旧记录、概率、源码或配置。

旧诊断确实混用了 **pre-grad current 与 post-no-grad**。不过这里独立保存了真正的 **pre-no-grad learner_old**，所以可以消除这个已记录的模式混杂。结果仍然变差，而且幅度略增。该混杂不能解释这次真实结果的符号或主要量级；缺少 post-grad 向量，仍无法辨识同一 grad-enabled 路径的有限步目标变化和根因。

## 源与字段身份

完整数字、逐行结果、来源 SHA 和误差在 [short-r1-mode-reconstruction.json](short-r1-mode-reconstruction.json)。权威 training manifest SHA 为 `ab87c8cf6ab5b60f3c806d2ee7953bef10f53cc944f904c14c4902219abfe528`，与最初下载副本逐字节一致。真实 focal SHA 为 `c1e7dd56322899e95773a972a1326be9a453b63ed7fac3244ccae9d05cec7c07`，与训练声明一致。

冻结 `training/train_vfarl_unsloth.py` SHA 为 `b092ea87903e1a245afb2e6b667e329f6a0234fae33d12f607a9bc8b407d4763`；冻结 `training/train_vfarl.py` SHA 为 `6295d1f30bb68c344b34aa8638d60aec11158d10d3b820ab15d158bf38019b31`。均从该次 carrier 的 `SOURCE_BUNDLE` 对应源码键独立解码，与实际执行 carrier 和发布 manifest 的 pin 相符，没有用当前修改版源码代替。

- 冻结 Unsloth learner 第 102–105 行：`no_grad + BF16 autocast` 计算 learner_old，并在零 LoRA scaling 下独立计算 reference。
- 第 106–107、133 行：grad-enabled current 保存成 `post_update_comparisons.before_update_logprobs`。
- 第 139–142 行：分别保存 actor 行为/参考、真正的 `frozen_learner_old_logprobs` 和 `actual_reference_logprobs`。
- 第 204–210 行：post 向量在 `no_grad + BF16 autocast` 下计算。
- 第 132 行：每行 loss 除以 `len(rows)=4` 再 backward。不是 545-token 全局均值，也不是后来增加的 fixed-sum 分母。

4 个动作共 545 个被选择 token；原 focal 的 mask 与 actor 数值逐行完全相符。没有用 actor behavior 或 reference 冒充 before-no-grad。此批数据中 actual learner reference 恰好与独立保存的 learner_old 逐元素相同，但身份仍分别保留。源码在 export 前后没有显式切换训练模式；导出函数深拷贝 PEFT config 后才设置其 `inference_mode`，没有修改 live model config。

## 只用旧公式

每行固定 `rho = exp(min(sum(float64(learner_old)-float64(actor_old)), log(2)))`，再按实际 current dtype 转成 FP32。四个实际 loss 权重为 `1.3352116346359253, 0.8895184397697449, 1.928138256072998, 0.48740455508232117`，与原记录一致，均未触及上限。

`ratio = exp(current - learner_old)`；PG 为 `-min(ratio*A, clamp(ratio,.8,1.2)*A)`；KL 为 `exp(reference-current) - (reference-current) - 1`。先逐 token 相加 `PG + .02*KL`，每动作取 token mean、乘固定 rho，再对四动作取 mean。after 始终使用相同 frozen learner_old、reference 和 rho，不重新定 old 或 IS。

## 重建结果

| 数值口径 | pre-grad | pre-no-grad | post-no-grad |
|---|---:|---:|---:|
| 原记录的 pre-grad 行 loss 算术均值 | 0.23580341786146164 | — | — |
| CPU FP32 逐 token 旧公式复算 | 0.23580341413617134 | 0.23580336198210716 | 0.23618735373020172 |
| FP64 + 原 raw rho（此前报告口径） | 0.2358034190460078 | 0.2358033584346821 | 0.23618730445781616 |
| FP64 + 实际 FP32-rounded rho | 0.23580342259343087 | 0.23580336198210716 | 0.236187308007633 |

此前 `.235803419 → .236187304` 是向量上的 FP64 分析重算，并非两次 GPU 直接记录的 scalar loss。现在完全复现该报告，两端误差均为 0。FP32 CPU 重建与直接记录的 pre-grad 单行动作 loss 最大误差为 `1.4901161193847656e-8`；CPU 与旧 CUDA 的 exp/reduction 末位舍入仍可能不同，因此没有虚报精确重跑旧 GPU scalar。

按此前 FP64 口径，混模式 loss 增加 `0.00038388541180836566`；同 no-grad 锚点增加 `0.0003839460231340741`。模式混杂使原报告的上升量少报 `6.06113257084484e-8`，只占原上升量约 **0.01579%**。真实 FP32-rounded rho 口径结论相同。匹配锚点下，weighted PG 从 `0.2358033584346821` 到 `0.23617878405954845`，KL penalty 从 0 到 `0.00000852039826765528`；不是单由 KL 项造成上升。

| 奖励方向指标（四动作等权） | pre-grad → post-no-grad | pre-no-grad → post-no-grad |
|---|---:|---:|
| mean(A × action-mean Δlogp) | -0.00010989460001216945 | -0.0001099628952435072 |
| mean(raw-rho × A × action-mean Δlogp) | -0.0002720247955638146 | -0.00027208540835567407 |
| mean(FP32-rho × A × action-mean Δlogp) | -0.00027202479858173687 | -0.00027208541137159963 |

两种锚点的奖励方向都为负。逐行 pre-grad/no-grad 最大差为 `0, 5.054473876953125e-5, 6.490154191851616e-9, 4.3213367462158203e-7`，其真实影响已经按原向量量化，而不是由 toy 反例推定。

没有 post-grad 向量、旧参数上的新 forward 或重复 anchor，不能据此判断真实 grad-enabled finite-step loss 是否上升、backward 是否错误，或把 BF16 舍入认定为旧负更新根因。这里只确认：这四个旧训练动作的已记录 no-grad 前后 surrogate 与方向指标仍变差；没有独立任务修复收益结论。
