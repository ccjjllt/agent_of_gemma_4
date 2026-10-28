# 一次backward前后：真实零optimizer结果

私有`jiyujia/vfarl-focal-backward-mode-zero-update-r6-bounded/1`，session `363043672`，载体SHA `9de2b42aa2a24002b1d9a39d12352824698580531556bd5483f8181b3b1536d5`。04:38:27 UTC单次提交，04:38:34已RUNNING，04:52:24真实API确认COMPLETE，早于取消时间。13次完整前向、1次顶层原focal scalar backward全部完成；optimizer=0、rollout=0、参数扰动=0、无adapter导出。实际结果见[完整报告](exact-output/vfarl-train/backward-mode/backward-mode-study.json)，SHA `406df2d0326ab406afe875f699e2570232975f7c0ccb88abbfeb85c5342a6816`，以及[收尾摘要](closeout-summary.json)。

原N0/R0/G0调用与autocast跨度保留，A原固定128缩放、序列TIS及完整组分母2的scalar执行一次backward。随后B的五次控制，再A的五次控制；期间保留A的.grad，最后才测量范数。worker耗415.920秒，唯一backward45.493秒，保留梯度范数0.558820033、820个grad tensors。实际adapter before/after同为`25639062b38cc88c07ead3eee8423f626bf811c60dd6f54dd81ed5b515da1656`。

| 同一次运行的对照 | selected LP最大绝对差 | hidden hash相同 |
|---|---:|---|
| A pre：G0－N0 | 0.0775868893 | 否 |
| A post：G1－N1 | 0.0526647568 | 否 |
| A：post G1－pre G0 | 0 | 是 |
| A：post N1－pre N0 | 0.1302516460 | 否 |
| A：G2－G1、N2－N1、R1－R0 | 各0 | 是 |
| B post：G1－N1 | 0.1689448357 | 否 |
| B post：G2－G1、N2－N1 | 各0 | 是 |

因此没有观察到一次backward消除N/G差异；A改变的是no-grad actor输出，grad-enabled输出和零LoRA reference保持相同。A的post测量之前还执行了B的五次控制，B没有本次运行内的pre对照，不能将跨run B差别单独因果化。实际scalar使用的保留baseline/reference向量分别与立即保存的N0/R0完全相同，未见简单输出别名覆盖。[独立复算](independent-backward-mode-analysis.json)核对了原scalar公式与这些完整向量。

实际checkpoint状态在backward前后发生变化：`FIRST_PASS true→false`、`BACKWARD_PASS false→true`、`CURRENT_GC_INDEX 62→0`、`LAST_GC_INDEX=62`；CPU slots归还60。`USE_DOUBLE_BUFFER`在立即after-backward时仍false，在之后首个B N1完成时才变true。CUDA math flags始终为highest／TF32false／BF16 reduced reduction true。实际GC模块源SHA为`2a914bad…`，block-compile模块源SHA为`2e233c63…`，均匹配已审候选；请求的block属性没有在`named_modules().__dict__`中暴露，空记录不能证明编译内部状态不变。

G0图实际遍历68节点，暴露60个Unsloth checkpoint节点且60个offloaded；post B G1为60／59，post A G1为60／60。遍历未截断、所有offloaded节点两项dtype均可见，全部`torch.bfloat16→torch.bfloat16`，实际FP32→BF16 offload路径计数0。因此该特定FP32存储cast候选在这些已观测图中没有触发；这不是“没看到节点所以算0”，也不能扩大为所有训练形状都无此路径。未访问saved tensors、未触发unpack、未修改第三方GC或CUDA math开关。

多个执行状态与backward一起变化，当前结果仍不能给出唯一根因或训练收益；没有因此更换主训练anchor、选择学习率或启动训练。原参考暖机、模型/adapter/数据来源保留，所有原始观察可在[精确下载清单](exact-output-download-manifest.json)中核对。

[实际guard](exact-output/vfarl-budget-receipt.json)记录child returncode0、683.583秒、cleanup confirmed、errors／survivors为空，未调用取消。共回收27个明确研究文件23,624,282字节，没有下载权重／运行时大包或遍历环境目录。最终[GPU额度](final-status-and-quota.json)used26.1354727967／30小时，remaining3.8645272033小时、reserved0；相对本次开始共享账户差0.3831194444小时，不是美元费用。没有自动追加GPU作业。
