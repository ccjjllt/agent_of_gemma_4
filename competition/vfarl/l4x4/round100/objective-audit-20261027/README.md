# 焦点动作目标与长度归一化审计

2026-10-27。本轮只做源码审阅与一次性CPU计算，没有运行GPU、模型训练或optimizer step，没有读取独立预留题内容。原short-r1行为token与old/reference记录均保持原样。

## 已确认的目标差异

原实现把每个完整焦点动作内的PPO及k3项按本动作的有效token数取均值，再对动作取均值。完整动作的终局策略梯度包含该动作的token score之和；除以动作自身长度改变了目标，不能解释为全批共同的学习率缩放。[Dr. GRPO原始论文](https://arxiv.org/html/2503.20783v1)分析了这类长度归一化偏置。本项目下面的有限动作反例为独立推导与CPU穷举结果。

令三种完整动作概率为 `(0.05, 0.05, 0.9)`，终局奖励为 `(0, 1, 0)`，有效token数为 `(1, 100, 100)`。每组独立采两次，优势为组内均值中心化终局奖励；行为与learner完全相同、IS为1、KL系数为0、在更新前PPO anchor计算梯度。完整动作的参数依赖可集中在首个选择token，后续为固定概率续写。

- 真实成功概率为0.05，其softmax-logit梯度为 `(-0.0025, 0.0475, -0.045)`。
- 原action_mean的期望上升方向为 `(-0.001188125, 0.000299375, 0.00088875)`，与真实成功梯度内积为 `-2.2803125e-5`。沿该方向移动一单位，成功概率降至约0.04997720。
- 固定尺度128的token和方向为 `(-9.765625e-6, 0.000185546875, -0.00017578125)`，内积为 `+1.6748046875e-5`，成功概率约0.05001675。
- 只保留异奖组相当于除以该例的异奖概率0.095，不改变反例方向。两样本的组均值基线只产生共同的1/2因子。

详细输入、穷举结果、实际reduction函数的梯度及scalar/tensor目标比较见[length-normalization.json](length-normalization.json)。这证明旧目标可能与终局收益方向不一致，**不能据此认定它就是旧short-r1实测loss上升的根因**：固定批上的反向与自身目标仍须独立核对。

## 显式新模式

旧配置默认保持 `objective_normalization: action_mean`，实际仍调用旧的`.mean()`路径。新增[固定尺度配置](../../../../../training/configs/vfarl_l4x4_live_pair_sum.yaml)声明 `action_sum_fixed_scale` 与 `action_token_scale: 128.0`：

`L = sum_mixed_actions[rho_action × sum_selected_tokens(PPO_token + beta × k3_token) / 128] / complete_group_action_count`

PG与KL共用相同固定尺度；每个完整动作的权重不再额外乘自己的1/长度。128是固定数值尺度，接近旧短轮动作长度量级，用于限制梯度大小；它是待实测选择，没有证明优于其他固定常数。梯度裁剪、Adam和有限步长仍会影响实际变化。

KL在这里表示完整焦点动作沿实际选中token的k3之和/128，再乘同一action IS权重；不是全词表精确KL或整条Agent轨迹KL。两后端均保存算法版本、归一化、每行分母和分别归一化的policy/KL项；原mean-token方向指标保留，另记与实际新目标同尺度的sum/128方向指标。训练器没有重写行为概率，采集协议、奖励、mask和完整prefix均不变。

在精确on-policy、无clip/KL的理想条件下，新目标在选定状态估计固定continuation策略下的局部终局收益梯度（两样本组均值带1/2）。实际seq-TIS截断、焦点选样、只更新一个动作及旧continuation仍使它不是完整task-level无偏梯度。16K仅为配置上限，未新增16K反向证据。

## 数值侧边界

[numerical-boundaries.json](numerical-boundaries.json)使用与GPU记录相同SHA的pinned融合输出层源码做CPU手动计算。无softcap及softcap=5时，融合与普通softmax梯度余弦约1，最大元素差小于1e-6。这未覆盖CUDA Triton、NF4骨干或实际Unsloth checkpoint backward，不能排除这些路径的问题。

另一个三类概率CPU计算表明：当采样分布固定，直接微分sampled k3的期望梯度并不等于微分其期望值所对应的reverse KL。anchor处它等于`KL(reference || policy)`梯度，而k3值的期望等于`KL(policy || reference)`。目前保留原sampled-k3 regularizer，只明确其估计边界，不在同轮静默更换KL算法。它在首个零adapter anchor理论上为零，因此也不能直接解释short-r1首步负方向。

后续短GPU只做零optimizer数值诊断：比对实际反向与同路径无梯度评分、梯度方向有限差分、Adam候选方向及真实精度下的局部响应。先定位目标内下降是否可靠，再决定后续训练实验。

## 完整cohort的动作分母

第二轮深审发现，先丢掉同奖组，再除以剩余训练行数，仍会让批分母依赖奖励。这与token长度偏置不同，固定两题、等长动作、精确on-policy、IS=1、KL=0时也可反转平均终局收益梯度。

[cohort-normalization.json](cohort-normalization.json)用真实`prepare_rows()`及新分母函数精确枚举16种结果：两题共享一个参数，A的成功率0.1、success logit斜率+3，B成功率0.5、斜率-1；真实两题平均成功率梯度为+0.01。每题独立采两次、组均值优势时，旧“仅异奖行平均”的期望梯度为-0.0125；保持完整4-action质量后为+0.005，正好保留两样本基线的1/2因子。所有数值再除固定128不改方向。

新fixed-scale模式从**完整原始focal文件**重建`eligible_complete_group_action_count`，保留已验证同奖组以及同动作异奖组的零PG质量。后二者没有额外backward；相同焦点动作异奖不当成动作偏好。缺失partner、评分无效或unknown不补0，也不补动作数量。旧action_mean配置保持原训练行分母。runner和resident的原始focal拼接目前保留所有完整组，没有只保存`prepare_rows()`输出。

KL仍仅在异奖、不同动作组上实际评估，因此此处是带零PG质量的mixed-group近似regularizer，不能称已估计所有完整组的平均KL。这个修复没有消除自适应选题、等待异奖才更新、有限样本和Adam的其他影响，也不承诺每步真任务收益单调提高。

进一步的[三动作k3精确枚举](mixed-group-k3-selection-20261028.md)量化了这一边界：在actor等于current、TIS1、固定分母的条件下，奖励筛选后的k3下降方向仍可同时增大两种完整reference KL，而其自身冻结目标确实下降。完整组对照在同一点方向相反。当前声明与实现一致，尚未据此改loss；同奖cohort跳过和Adam时钟也不能靠多算几行正则消除。

继续审阅resident后同步修复了跨cohort的隐式稀释：live-original-pair每批update-input只来自本次预先固定的至多两题；全同奖批保留原件和记录，但不拼回未来更新。旧historical-three-branch保留原pending积累语义。否则前面随机遇到多少无信号cohort，仍会成为后续梯度的随机分母。[cohort-input-scope.json](cohort-input-scope.json)直接执行该段实际源码AST进行一次性手动核对，确认live模式每轮替换输入、空输入清空，历史模式仍积累。没有启动worker、模型或optimizer。

## Off-policy与续写的准确边界

[offpolicy-boundaries.json](offpolicy-boundaries.json)另保存两项有限MDP精确枚举，用于解释现目标，不在同轮更换算法。

第一例的focal actor与learner完全相同，首动作IS恒为1，仅其后续执行分布不同。focal-only修正的梯度与真实learner终局收益梯度内积为-0.0025；采用共同续写随机数或独立续写随机数，两种枚举结果相同。因此同prefix、逻辑预算和共同续写seed能控制部分实验差异，却不能证明已校正continuation分布。现实现的正确表述仍是固定actor continuation下的`Q_actor`局部目标。

第二例在确定性终局奖励下，未截断IS与真实learner收益梯度内积为+0.00204375，cap2后为-0.000740625，说明截断与组基线共同构成有偏surrogate。这里没有据反例移除cap；真实行为差异与方差仍须实测。若将来研究后缀IS，还必须重审“相同focal动作异奖丢弃”的规则：引入依后缀而变的权重后，两个同焦点动作贡献不再必然相消，不能只给现有损失乘一个后缀ratio。

## Adam续接的实际绑定

原`optimizer.load_state_dict()`仅按参数槽位续接，没有参数名、adapter或基座来源绑定。同形状参数乱序时可能静默载入错误moment；这是一项实际续接缺陷，与旧short-r1首个新Adam步无直接因果证据。

运行模块`training/vfarl_optimizer.py`新增带来源的state格式，绑定按optimizer组顺序的参数名/shape/dtype、实际adapter tensor hash、NF4/tokenizer/template/tool核心身份及实际LoRA scaling。inference/training执行模式与运行时hash不属于权重身份；切换模式不会误拒绝正确续接。保存与恢复不重置moment或step，恢复后显式采用本轮声明学习率。旧裸optimizer state因缺少槽位来源明确拒绝；可保留adapter并显式从新Adam开始，不能伪造其历史绑定。

[optimizer-binding.json](optimizer-binding.json)记录一次性微型CPU Adam手动操作：正确state的moment和step逐tensor完全一致、学习率5e-6→1e-6、仅runtime模式改变仍可恢复；同形状乱序、adapter tensor变化及旧裸state分别被拒绝。没有加载比赛模型或执行GPU研究训练。

## Unsloth导出adapter加载语义

Unsloth两处加载先按当前cfg创建LoRA层，再仅复制A/B。旧`inspect_adapter()`已校验rank和部分普通LoRA结构，却只要求alpha为正，未要求保存的alpha/dropout与当前层匹配。同tensor hash并不能确定相同LoRA函数；[adapter-scaling-identity.json](adapter-scaling-identity.json)用小CPU张量给出直接例子。

现在构造器和resident原位同步都复用原`inspect_adapter()`，随后在读取/复制tensor前匹配saved r/alpha/dropout并拒绝rsLoRA及非普通变体；匹配语义写入加载说明。没有更改历史行为概率、reference记录或policyID定义。native加载器走`PeftModel.from_pretrained()`，会采用保存的PEFT配置，而不是Unsloth这条“当前cfg建层后只复制tensor”的路径，本次未扩大修改通用native加载器。

[adapter-load-contract.json](adapter-load-contract.json)记录实际tiny safetensors/JSON读取：完全相同的tensor文件配alpha32，旧结构检查通过、新加载契约明确拒绝；真实历史r32/alpha64/零dropout/plain配置通过。dropout、rsLoRA及非普通变体错配也在tensor加载前拒绝。没有证据表明旧实验实际发生过此类错配，也没有运行GPU。

## 本轮补充的真实小架构与失败归因证据

[真实Gemma4＋PEFT梯度对照](real-gemma4-peft-fused-gradient-20261028.json)使用2层真实Gemma4、r2七投影与未改的pinned融合函数原生CPU fallback；production focal scorer与同模型full LM selected logprob最大差2.38e-7，全部28个LoRA张量的梯度相对L2差4.03e-7。真实nonreentrant checkpoint发生重计算并得到相同值／梯度，参数字节未变、没有optimizer。这个FP32 CPU结果未发现该路径错误，不覆盖NF4、Unsloth骨干、Triton或31B。

[实际生产cache边界](real-gemma4-cache-boundary-20261028.json)在真实小Gemma4上核对变长3/7前缀、window4、停行抽取及严格延伸；未发现该范围内语义差异。sample_batch在CPU的CUDA显存统计处真实失败，保留该失败与已执行的实际采样值，不以替换kernel／缓存掩盖。它没有新增GPU性能或容量结论。

[exit1与required覆盖研究](verifier-exit-one-completeness-20261028.json)确认上游exit1不自动校验完整JUnit，但不能据此把缺失/error/skip统一改成unknown：坏生产patch也会造成这些失败，删掉它们会引入动作依赖筛选。默认官方命令未启用continue/maxfail，R3四条真实exit1均覆盖所声明required，原reward0保持。研究输入用于说明函数分支，不冒充新真实任务结果；早先过强的unknown建议已撤回，未新增验收门控或修改奖励。

[当前方法对照](current-methods-comparison-20261028.md)进一步区分SPROUT指导式分叉、trajectory-return组中心化及future-KL stationary-target路线。独立两步表格枚举不能证明当前focal-k3/TIS具有论文保证，也没有据此盲改loss。另以真实pinned LoRA cast-cache函数及PEFT小层核对两次CPU小张量更新和原位同步，缓存均正确失效，见[窄关闭证据](pinned-lora-cast-cache-update-20261028.json)；不排除真实Unsloth CUDA/NF4路径。
