R7 已正常 COMPLETE：[私有会话](https://www.kaggle.com/code/jiyujia/vfarl-focal-ng-checkpoint-zero-update-r7-bounded)，version 1，session 363051991。提交 2026-10-28 05:13:10 UTC，首次终态 05:25:38；未取消、未补跑。源码 run SHA `611c78fa284601e78555c5074894b1b721382095227d796b57af9c994032b67d`，所有实际文件哈希见 [回收清单](exact-output-download-manifest.json)。

本次完成 21 次 full forward、1 次原 A focal scalar backward、0 optimizer 更新、0 参数扰动/新 rollout。保留原 pending-reference 暖机及训练模式切换。adapter 前后 SHA 均 `25639062b38cc88c07ead3eee8423f626bf811c60dd6f54dd81ed5b515da1656`；worker 462.796s，backward 46.244s。

A、B 各做 legacy → candidate → restored，每组 N/R/G。仅 candidate N/R 临时替换 60 个文本层 checkpoint callable，G 前恢复原对象。结果如下：

| 动作 | legacy N/G 最大 LP 差 | candidate N/G 最大 LP 差 | restored N/G 最大 LP 差 |
|---|---:|---:|---:|
| A | 0.0600976944 | 0.0600976944 | 0.0600976944 |
| B | 0.0551054478 | 0.0551054478 | 0.0551054478 |

每个动作的 candidate/restored N、R、G 分别与 legacy 对应模式的完整 selected LP 逐值相同，hidden SHA 相同。A 在这次原 backward 前后对应 N/R/G 也完全相同。候选没有改善模式数值差异，不能据此认定根因或学习收益。

候选作用域的完整被动 GC 状态前后相同：A CURRENT_GC_INDEX 62→62，B 246→246；legacy/restored 的每次 N/R 仍 +1，所有 G 仍 +60。每个候选 scope 记录 120 eager Python direct 调用、0 compile trace、0 grad delegation，每层 2 次，退出精确恢复全部原 callable。计数不是 compiled graph 执行次数；恢复 Python callable 也不证明编译缓存完全没有变化。本次只证明狭窄候选可消除这些额外 NG checkpoint 状态变化且保留本输入数值，不修改训练默认、主 loss、精度开关。

[预算原件](exact-output/vfarl-budget-receipt.json) child return 0，715.958s，cleanup true、errors/survivors 均空。最终 GPU used 26.5365866856/30 h，remaining 3.4634133144 h，reserved 0；账户差额 0.4011138889 h。只回收 37 个研究文件，共 35,308,457 bytes，没有环境目录遍历、权重/runtime/ZIP 下载。两个可选路径 actual-config.json、gpu-topology.json 返回 404；实际配置仍绑定 frozen source，真实硬件记录在 carrier.json，未把缺失文件当作证据。

具体向量和状态在 [原始研究摘要](exact-output/vfarl-train/no-grad-checkpoint/no-grad-checkpoint-study.json) 与逐 forward 文件；紧凑结果在 [closeout-summary.json](closeout-summary.json)。本次结束后未启动任何新 GPU。
