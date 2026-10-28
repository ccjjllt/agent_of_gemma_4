R7 临时 no-grad checkpoint 干预，独立 opt-in。原 A N0/R0/G0 和 scalar backward 保留，随后 A/B 各 legacy/candidate/restored N/R/G，总 21 forward、1 backward、0 optimizer step。

Candidate 仅 N/R 在 60 个已绑定文本层实例上临时直接调用 function(*args)，G 始终先恢复原 callable。finally 保存 GC globals、Python trace/eager 调用计数及精确对象恢复。计数不代表 compiled graph 层执行次数。既有主训练 loss、精度标志与默认路径不变。原 remote guard 1180+20 秒，无自动延长或补跑。
