# 首次前向与reference切换：冻结的零反向对照

针对真实r4诊断中第一个动作grad/no-grad最大差0.07084899、第二个9.5889e−6的现象，当前载体只观察前向，不构造loss或optimizer，不做backward、rollout、参数扰动或adapter导出。独立审阅通过后，03:25:13 UTC已按实际配额及并发读回作单次私有提交；实际版本与输出见[执行目录](../execution-r5-mode-order-20261028/)，冻结manifest的not-submitted描述仅对应生成时点。

`run.py` SHA `e0e0dadc4b10e57735ccc67c8fc953d5cf9cd44fc9e6a5addf3382103b51707f`，内嵌payload SHA `819dec5703f0b567977155dfa45115b6df1765d4865814e9a2e09920d0868bb7`。原r4的37项源中35项字节不变，另外两项仅增加CLI研究分派及可选构造阶段日志；增加独立研究模块，原loss和旧numerics函数AST完全相同。来源差异见[source-delta.diff](source-delta.diff)，具体入口见[冻结研究模块](source-overrides/training/vfarl_mode_study.py)。

同一个父进程只准备一次既有NF4缓存，然后按A-first、B-first顺序启动两个全新的Python进程，均使用GPU0、1，前一进程退出后才启动后一进程。两个worker都加载相同原adapter和原pending两行数据，先完整保留原`complete_reference`推理暖机及其模式恢复，再执行原显式`switch_execution_mode('training')`。因为缺失reference的始终是A，所以A/B-first仅指训练前向次序，不表示两种输入都从未被模型见过。

N表示actor no-grad前向，R表示零LoRA scaling的reference no-grad前向，G表示grad-enabled actor前向。每个worker首动作8次，次动作5次，共26个完整前向：

| 位置 | 固定顺序 | autocast边界 |
|---|---|---|
| 首动作 | N0 → R0 → G0 → G1 → N1 → R1 → N2 → G2 | N0/R0共同；G0、G1各自；N1/R1/N2共同；G2独立 |
| 次动作 | N0 → R0 → G0 → G1 → N1 | N0/R0共同；G0、G1、N1各自 |

原N0/R0/G0之间不插额外actor调用，保留原N0/R0共同autocast上下文。G0/G1观察首次G差异，N1/R1/N2观察warm reference插入后的恢复，G1/G2观察之后G是否变化。cold reference插入与首次grad调用仍然混杂；若只有cold异常而warm控制稳定，仍需要另一个独立fresh无R对照，不能直接修改训练anchor或认定唯一根因。此轮也省略了旧研究G0后的backward，因此第二动作不能冒称旧反向后状态复现。

每次真实前向后立即detach并丢弃图，保存全部selected logprob、target IDs、selected hidden的实际dtype/shape/hash、对同动作前面各次观察的max／relative L2差，以及module training、checkpointing、LoRA scaling、A/B存储元数据和真实耗时。只保留小型CPU detached selected hidden用于比较，不导出完整activation。状态观察不额外调用`get_lora_parameters`：其实际源虽已绑定，泛型TP／fake-quant／FP8分支可能有副作用；记录被动属性和分支存在性，避免干扰冷路径。每次forward增量保存，预算耗尽时已有观察仍可回收，缺少最终参数哈希时不会推断完整成功。

两个worker共享一次缓存复制，但各自仍做真实模型加载、已加载base identity核对及原reference暖机。前次缓存复制425秒、完整旧worker447秒；没有足够细分实测可保证两个新worker都在20分钟内完成。本次增加构造/加载/adapter/base验证/暖机/模式切换/每次前向的monotonic日志，不延长预算来追求完成数。

父guard仍1180秒执行＋20秒cleanup；除期望payload SHA、落地文件名和说明外，所有进程所有权／预算／回收函数AST与已经真实CPU/GPU验证的control相同，见[wrapper-delta.diff](wrapper-delta.diff)。所有原配置、NF4/runtime/adapter依赖和原pending SHA保持不变。实际提交必须使用显式jiyujia连接、私有单次提交和最少25分钟可用GPU配额，绝不通过默认CLI账号；本冻结不含新的提交记录。

记录所需的同步、CPU复制与状态读取会改变观测开销和张量驻留；本对照保留原cold前向调用顺序及autocast跨度，不宣称完整时序或显存生命周期完全相同。原r4首动作之后曾发生backward，本方案没有，因此第二动作是零反向顺序对照。
