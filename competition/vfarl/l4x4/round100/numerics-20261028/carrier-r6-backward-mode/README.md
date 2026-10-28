# 一次backward前后的模式对照（零optimizer）

冻结载体SHA `9de2b42aa2a24002b1d9a39d12352824698580531556bd5483f8181b3b1536d5`，child SHA `e46de57695e84807126dccd14b8a81a9992a607ca505681f922c3d46d01e7556`；独立入口见[source-overrides/training/vfarl_backward_mode_study.py](source-overrides/training/vfarl_backward_mode_study.py)。原模型、adapter、pending两行数据、配置和reference暖机均保留。相对r5只增加该研究模块及CLI分派，37项旧bundle内容不变，主训练loss／optimizer函数未改。

单个fresh进程，13次完整前向及一次顶层scalar backward：A原N0→R0共用no-grad/autocast，G0独立grad/autocast；保留真实G0图，按原fixed-sum/128、序列TIS、组分母2构造原A focal scalar并直接backward一次。第一次backward之前不加入额外N/G重复。省略原r4仅用于head系数诊断的`autograd.grad(scaled, current)`，它本来不进入head/backbone；原scalar梯度路径不因此换公式。

之后先B的N1→R1→G1→G2→N2，再A的同五步。N1/R1共用原no-grad/autocast，其他各自；post G图立即脱离，不再backward。A已有.grad保留至全部post观察结束，范数归约在所有post之后才执行，避免在backward与紧接B之间加入额外CUDA归约／显存分配。没有optimizer构造／step、梯度裁剪、参数扰动、rollout或adapter导出。

每次前向保留完整selected LP、hidden hash与差统计、training/scaling及被动CUDA math flags。读取GC模块实际源SHA及`FIRST_PASS`、`BACKWARD_PASS`、index、offload、double-buffer等已有标量；记录各block `__dict__`中的checked/mode两项和实际block-compile模块源SHA，不调用descriptor或额外forward。

G0与post G1图只遍历`next_functions`并读取匹配Unsloth checkpoint节点的自定义`ctx.__dict__`已知元数据。读取dtype、shape、slot index与stream存在性，绝不访问saved tensors或触发unpack，不调用/修改第三方GC函数及globals。没有节点或缺元数据只记未暴露，不能据此排除FP32压缩。单次手动CPU autograd API核对确认该遍历没有触发saved-tensor unpack；它不验证真实Unsloth offload，实际节点计数须由GPU原件给出。

一次backward同时改变多项运行状态；本轮只形成前后观察，不能从关联直接认定唯一根因。没有设置exact-offload开关、替换主训练anchor或选择学习率。完整保存每个完成的观察与backward前后记录；预算中断保留partial，不自动补跑。

仍为1180秒执行＋20秒cleanup的同一进程预算机制，私有jiyujia单次提交并读取实际配额／并发。单worker避免重复约200秒构造；先前13-forward worker实测364秒，新增一次backward尚无精确耗时，预计连缓存约10–15分钟，不保证完整完成。实际执行必须受20分钟上限约束。
