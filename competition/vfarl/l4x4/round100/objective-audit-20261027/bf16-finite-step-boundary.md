# BF16有限步与autograd下降预测的CPU边界

2026-10-28。固定32维输入／hidden、r4、64词表的小LoRA，FP32参数、B精确为零、两动作各2个token、优势±0.5、固定old/reference、TIS=1、无梯度裁剪。loss严格按每token先合并PPO与0.02*k3，再动作token求和与两动作平均。所有复评分与±探针都启用grad forward，随后读取标量；不借no_grad数学路径比较。AdamW是每档重新从零B出发的CPU首次更新，betas=(0.9,0.999)、eps=1e-8、weight_decay=0、foreach/fused=False；moments保持FP32。

预先限定256 seed或60秒，发现首个上升例后停止。实际只执行0–8共9个seed、0.49秒；保留全部尝试和所选输入，不能把筛出的例子当作发生频率。seed8结果如下（每档均独立从锚点出发）：

|BF16导出实际参数步|autograd·实际delta|同BF16 loss变化|同实际delta的FP32 loss变化|
|---|---:|---:|---:|
|sign 1e-6|−1.52260e-5|0|−1.51396e-5|
|sign 5e-6|−7.61300e-5|+8.34465e-7|−7.61151e-5|
|fresh AdamW 1e-6|−1.52260e-5|0|−1.51396e-5|
|fresh AdamW 5e-6|−7.61300e-5|+8.34465e-7|−7.61151e-5|

这不依赖两种precision选了不同方向：FP32交叉控制使用同一BF16导出的实际FP32参数delta。seed8在1e-6时仅1/128个hidden元素变化，词表logits无元素变化；5e-6时只5/128个hidden与3/256个logits变化。LoRA分支虽有更新，大部分变化在BF16 residual加法及输出计算中被舍入。PPO项已经上升，5e-6的未加权k3约3.91e-11，KL不是该上升的主要来源。

同方向中央差分也会如此：BF16在1e-6的导数为0、5e-6约+0.09537，而autograd预测约−15.226；FP32分别约−15.110／−15.187。PyTorch autograd穿过BF16 cast与舍入传播的是surrogate；实际有限精度forward呈局部常数与跳变，因此同grad路径的loss不降本身不自动证明backward实现有错。FP32 head隔离也不能验证LoRA／残差／backbone中的舍入。

精确输入张量、hash、源函数字符串及其SHA、PyTorch git/build版本、固定锚点、Adam状态dtype、实际delta张量、前后loss、全部尝试及选定seed的分层变化见[bf16-finite-step-boundary.json](bf16-finite-step-boundary.json)。每个选定步后都真实恢复FP32参数并按两种precision再评分：参数bytes与锚点scores完全一致，梯度bytes未因step／复评分改变。初次诊断将PG/KL分开reduce引入约1e-12以下的FP32顺序差异，最终证据已按原目标严格顺序重跑。

这是实际PyTorch CPU小张量机制证据，未加载31B、未用GPU、未改主代码或配置、未建立额外载体、未开始正式训练。它既不能归因此前真实负更新，也不能从toy所选步长推断31B学习率或计算精度方案。后续真实零optimizer probe应分别报告autograd点积、同路径有限变化及FP32 head的边界，不把其中任一项单独当作算法／backward正确性的判据。
