# Session prefix rewind：显式resident入口

本轮只接通`training/vfarl_batched.py`新接口的CLI/runner参数，不改训练算法、run.py、数据、奖励或历史实验。训练代理负责sampler v5实现和数值机制；这里没有加载模型、采样、优化器、真实子进程、GPU或上传，也没有生成/回写已冻结numerics carrier-r4。

Kaggle builder新增`--resident-session-prefix-rewind`，必须同时显式使用`--resident-rl`和正数`--resident-session-cache-tokens`。只True时RESIDENT_SETTINGS新增`session_prefix_rewind=True`，runner原样转交resident.run；非resident或无cache会在环境/输出创建前拒绝。

resident.run新增默认False的session_prefix_rewind参数；只True时写bootstrap和summary键，并在worker命令追加`--session-prefix-rewind`。直接resident CLI也提供`--session-cache-tokens`与`--session-prefix-rewind`，使依赖能显式满足。worker检查CLI与bootstrap显式值一致，并检查正数cache，随后对configure_batched_policy和BatchedPolicyService均传bool(args.session_prefix_rewind)。不从cache大小推断或自动开启。

False仍选原sampler v3（无session cache）或v4（已有cache），settings/bootstrap/runtime中不写新False键；只有True才选REWIND_SESSION_BATCHED_EXECUTION_PROFILE v5，真实SessionCacheStore.session_prefix_rewind=True。版本指sampler协议，不是historical carrier轮次。

该开关只允许在相同pre-action prefix下回退cache后prefill不同response tail。当前collector仍rawappend，未集成官方结构化history renderer；单开此flag不会使现有history变成官方history。full cache crop仍可能保留原backing storage，不据此声称显存释放、吞吐或修复收益。

一次性手动CPU证据在manual-observations.json：实际builder parser/settings AST、runner kwargs/guard AST；actual resident.run生成bootstrap/worker argv并在Popen前拦截；actual worker.main接收参数并完成collect/close dispatch，实际configure/service/store构造与aclose。模型、隔离、collect用惰性替身；metadata-only临时库存不作为真实训练题/CPU验收，均已删除。三入口实际--help正常。观察到False cache0→v3、False cache64→v4且runtime无新键；True cache64→v5且真实store开关为True。无cache、非resident及bootstrap不一致均在加载策略前拒绝。未执行真正的cache数值/模型路径。

四源码前后SHA和构造时sampler runtime SHA均保存。build_runner末尾原numerics command的`--timeout 1200`与time_limit元数据已逐字保留；未将本轮源码功能回填冻结payload。此前各文件已有的改动以修改前SHA为基线保留。
