# Native optimizer 与 adapter 提交的异常边界

2026-10-28。发现并修复了一处实际记录缺陷：native learner 的 Adam 已成功返回并改变参数，但随后 adapter 导出或提交 I/O 失败时，没有留下 optimizer-step 记录。旧 carrier 缺省读到0步；另一方向上，已有 Unsloth partial 记录的1步也不能代表 adapter 已可复用。

[一次性CPU证据](gradient-update-boundary-review-20261028.json)执行真实 native 源码的 post-step 小段、真实 FP32 Adam，以及明确注入的 export I/O 异常。旧段实际1步、参数 delta 非零，但原 reader 公式报告0；新段保留 rank0 partial 和每 rank 的实际返回记录。非零rank仅写自己的记录，不声称全局完成；已有完整提交字节保持不变。原 reader 两行是修改前实际阅读所得，未捕获完整旧runner SHA，不伪造其来源。

`train_vfarl.py` 将 optimizer.step 到全局提交段包入异常处理。step 返回立即记录本 rank 的事实；异常处理不发起新 collective。partial 明示全局完成未确认、adapter 未提交且不是可恢复 checkpoint，禁止自动重复训练。完整 native commit 显式声明 `adapter_export_completed=True`。loss、采样、概率、反向累积与裁剪未改；去掉新增 guard/metadata 后，整份源码 AST 与修改前相同。[最小改动差异](gradient-update-boundary-native-change-20261028.patch)及修改前源码快照均已保存。

root 独立修复 carrier consumer，以 immutable optimizer-step 的来源、adapter 文件 SHA/bytes 和 export 状态决定是否可复用，training_manifest 仅供后续诊断；[rank-scope证据](optimizer-commit-consumer-rank-20261028.json)保留“只有rank2已返回而全局未知”的情况。独立只读复核未发现这一范围内另一条实际误接路径。resident worker 已在成功返回前拒绝未导出的 partial，未因此修改 resident。

本机真实 Gloo world1 初始化因 loopback 地址解析失败，发生在任何 optimizer step 前。成功的后续证据明确省略单rank SUM identity collective，因此不称分布式/FSDP 验证；rank/world 元信息只核写入归属。CPU微张量不验证NF4、CUDA、31B、硬终止或证据存储本身失效，也不解释旧负更新或证明任务收益。没有新增测试/脚本、GPU或正式训练。
