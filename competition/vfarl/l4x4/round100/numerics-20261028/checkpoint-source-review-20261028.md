# 首次 backward 前后状态：固定依赖源码审阅

2026-10-28，只读源码和已有真实 GPU 输出；本审阅没有执行模型、backward、optimizer 或新提交。以下状态转移有源码依据，但尚未证明它们造成实际 G/NG 概率差异。R4 的第二动作在一次真实 backward 之后，R5 的第二动作没有任何 backward；二者不能直接视为同一执行历史。

## 来源和实际绑定边界

R5 `execution-r5-mode-order-20261028/exact-output/vfarl-train/mode-a-first/loaded-identity.json` 记录 torch 2.10.0+cu128、transformers 5.13.1、Unsloth / Unsloth Zoo 2026.10.9、两张 L4、BF16 计算、FP32 LoRA、`gradient_checkpointing=unsloth`。其真实 utils / loader / bitsandbytes forward **文件** SHA 分别是 `ffe3907694c38536906988d34a79b126b58510f780d6e808d67ef777c7b59c07`、`29db53e1dca4f84b29dbdc944d83004285030cff588e6b60ce6f58b459a1dd9b`、`647bcadab8933532eb8c47d5100db4d1b3f3849b2949aa9471540618e8eb3cde`，与本地缓存源一致。

实际 stage 的 `actual-output/vfarl-cpu-stage/preparation.json` 固定公开 wheel：Unsloth SHA `feb40b99bc4f8c543a36cbd0fe6e1410d6e4ac16bd4d0c731f607adce3d63bb0`，Zoo SHA `f68e0576b0e9bd2ef85f19bf665132c576b52ee4df923fbb16c81538a3cde676`。本次读取以下提取源码，并手动核对每份原始 bytes 与对应 2026.10.9 `.dist-info/RECORD` 一致：

| 本地源 | 原始 bytes SHA256 |
| --- | --- |
| `/private/tmp/vfarl-pinned-unsloth_zoo/unsloth_zoo/gradient_checkpointing.py` | `2a914bad043607b2a3b3ba46241e0a1fba0ee76c732c6e257b1478ed97bfcfbd` |
| `/private/tmp/vfarl-pinned-unsloth/unsloth/models/_block_compile.py` | `2e233c6385e3f76892a6530ece93228263220a82b699e3634486fab2a31cb72d` |
| `/private/tmp/vfarl-pinned-unsloth_zoo/unsloth_zoo/temporary_patches/gemma4.py` | `23e9b5570ab46036769e60ece313114c4baa556f0bba47f6aa93fcb906070962` |

R5 没有记录实际已加载 GC / block-compile 模块文件 SHA 或每层实际 checkpoint callable；因此这些文件是匹配固定 wheel 的源码候选，不能冒称已有运行时逐函数绑定。下一次已授权短诊断应记录实际 module file SHA 和 checkpoint callable 名称。Zoo 源有 CRLF，hash 必须用原始 bytes，不能对 `read_text()` 规范化文本取 hash。

## GC 的真实状态转移

`gradient_checkpointing.py` 1234–1333 初始化 `FIRST_PASS=True`、`LAST_GC_INDEX=CURRENT_GC_INDEX=0`、`BACKWARD_PASS=True`、`USE_DOUBLE_BUFFER=False`。`FP32_OFFLOAD_EXACT` 由 `UNSLOTH_OFFLOAD_FP32` 决定，默认 `bf16`，即 False。

`UnslothCheckpointFunction.forward` 1389–1400：仅当第一个参数是 requires-grad tensor 时，若 FIRST_PASS 则 LAST 加一；CURRENT 每次加一。1408 的 offload 条件依赖大小、CURRENT/LAST 和 FIRST_PASS。1425–1445 在 BACKWARD_PASS 后首次 offload 开始新 generation，清 prefetch 状态；FIRST_PASS 已为 False 且显存余量足够时可永久启用双缓冲。1510 每次 offload 翻转该设备的 buffer slot。

1543 的 `run_function` 总在 no-grad 下运行，而且使用原始参数。offload 保存的输入主要用于 backward 重算，因此不能仅凭保存 dtype 或双缓冲改变，推出本次前向输出必然改变。

`backward` 1627–1632 设置 BACKWARD_PASS=True、FIRST_PASS=False、CURRENT=0，LAST 不归零。1668 在加载存储输入后恢复保存的 dtype；1672 在 enable-grad 和保存的 autocast 下重算。535–560 的 recompute 线程局部标记通过 finally 恢复；未发现该标记按正常路径泄漏。

`_CpuSlotLease.release/__del__` 1219–1232 只回收 CPU 槽和记录 stream fence，不恢复 FIRST/LAST/CURRENT。因此 G→detach→丢弃图而不 backward 会推进这些全局计数；释放图不等于恢复算法执行状态。N/REF 是否也推进计数取决于实际 checkpoint 输入的 requires-grad（例如 embedding hook），不能只由最外层 grad mode 推断，更不能先假设每个全前向必定增 60。

已有真实证据：R4b `actual-output/vfarl-train/numerics.log` 第 10–11 行记录 smart offload 和双缓冲启用；日志没有给出它在动作间的精确时间。相同日志未检出 NF4 kernel failure 或 block-compile failure 警告。

## FP32 offload 的候选边界

1403：若输入 dtype 是 FP32 且 FP32_OFFLOAD_EXACT=False，保存 dtype 改为 BF16；1511–1514 将原 dtype、保存 dtype 和 offload metadata 写进 ctx。1668 的 BF16→FP32 恢复不能恢复已丢的尾数。这是可能影响重算及梯度的明确数值机制，但当前没有实际 31B checkpoint 输入为 FP32 的记录。

可被动遍历 G0 `grad_fn.next_functions`，保留 node 引用并去重，仅检查类型名为 `UnslothCheckpointFunctionBackward` 的显式 Python ctx 属性：`_requires_gradient`、`_saved_dtype`、`_store_dtype`、`_offload_generation`、`_saved_metadata`。metadata 中只保存 new_size、shape、CPU slot、device index、buffer slot；stream 仅记录是否存在。dtype 属性只在 offload 路径存在，缺失不得解释为某种 dtype。零 GC 节点应报告图未暴露，而不能直接断言零 FP32 节点。

不得读取 `saved_tensors`、`_raw_saved_tensors` 或自动生成的其它 `_saved_*`，以免触发 unpack / offload 行为。若实际确认所有已暴露 offload 节点均非 FP32，则本次关闭 FP32→BF16 候选，不改环境开关。

## block compile 与 Gemma4 边界

R4b 日志实际记录启用 `Gemma4TextDecoderLayer` 整块编译。`_block_compile.py` 950–960：首次 no-grad block 调用若该层尚未完成 grad-mode 编译试运行，会 fork RNG、enable-grad，内部递归调用该层 forward，再执行原 no-grad 调用。这是层内预编译，绕过外层 checkpoint wrapper，并非额外一次完整模型 G 前向。

该 wrapper 逐层持久保存 `_unsloth_block_checked`（bit1=no-grad、bit2=grad）和 `_unsloth_block_mode`；首次编译成功或编译异常降级会修改它们。可直接读 layer.__dict__ 两字段；不调用 forward、不开新 hook、不变更编译开关。CUDA graphs 默认 False；首次 backward hook 克隆梯度的分支只在显式 CUDA graphs 路径生效。

Gemma4 patch 435–495 的强制非 reentrant 路径以真实 KV sharing 为条件，缓存 `_unsloth_gemma4_has_kv_sharing`，不能仅凭 Gemma4 类名认定触发。当前31B配置的本地对应材料是无 KV sharing，但仍应读取实际模型字段/层计数确认。`_AutoCheckpointSkip` 仅显式 auto 模式安装，且多 GPU 层布局直接拒绝；与当前两 GPU 默认设置不构成已知触发路径。

## 旧 directional probe 的执行历史限制

R4 冻结 `training/train_vfarl_unsloth.py` SHA `2663f4af9c19b331c2a285cecfc8f6b412827bff53a1d9d9f7d63020333bd996`。其 `focal_directional_numerics` AST 与本次审阅时维护源相同：每次 evaluate 都 enable-grad 调 `focal_logprobs` 后立即 detach，不 backward；baseline 只执行一次，然后依次测两个 epsilon 的正负方向。finally 只恢复并核对 adapter bytes。

因此旧 probe 已保证参数数值恢复，但没有证明 GC/编译/缓存历史对目标值无影响；各 evaluate 可能推进上述状态。`place(0)` 的 copy_ 也会改变参数 version；虽然 utils 的 `_fast_lora_weights` 缓存会读取 version，其所在 fast-linear 路径在 q_len!=1 时提前返回，当前完整序列是否实际消费该缓存并未证明。未来若继续使用方向差分，应先有固定参数重复 objective 和恢复 anchor 的实际对照，再解释差分；不因此直接宣布旧数值错误。

当前最小观测是一个 fresh worker：保持原 N0→R0→G0，立即只做一次真实 focal backward，再测同权重同输入 A/B 的 N/G 重复。记录标量 GC globals（FIRST_PASS、BACKWARD_PASS、CURRENT/LAST、CPU_INDEX、OFFLOAD_GENERATION、USE_DOUBLE_BUFFER、FP32_OFFLOAD_EXACT、GC_PREFETCH、MINIMUM_SIZE、NEXT_BUFFER_SLOT），上述安全 ctx metadata 和两项 block attrs。第一 backward 前不得插入额外 G；不重置内部 globals，不改第三方源码，不引入 optimizer。

## R6 实际结果补记：no-grad 评分进入了 offload

原件目录：`execution-r6-backward-mode-20261028/exact-output/vfarl-train/backward-mode/`。本次只读取该目录，没有新 GPU 提交。`active-checkpoint-sources.json` 已把实际加载的 GC 和 block-compile 模块绑定到本文的 `2a914bad…` / `2e233c63…` 文件 SHA；先前的这两项来源缺口已补齐。实际 text layers 的 checkpoint callable 是 60 个 plain `torch.utils.checkpoint.checkpoint`，没有 partial 参数；其 wrapper 文件来自 torch `_compile.py`，故该文件 SHA 不能替代 checkpoint 函数实现 SHA。实际快照的 `compiled_blocks` 为空，不能仅凭 module loaded 把旧 R4 block 编译状态迁移到 R6。

| 实际调用 | 外层 grad | CURRENT 前→后 | LAST 前→后 | 其它变化 |
| --- | --- | --- | --- | --- |
| A N0 | False | 0→1 | 0→1 | generation 0→1，CPU slot top 0→1，free slots 0→1 |
| A R0 | False | 1→2 | 1→2 | 复用一个 CPU slot，结束后 free=1 |
| A G0 | True | 2→62 | 2→62 | 60 个 GC 节点，60 个 offload，free=0 |
| A backward | True | 62→0 | 62→62 | FIRST True→False，BACKWARD False→True，free=60 |
| B N1 | False | 0→1 | 62→62 | generation 1→2，双缓冲 False→True |
| B R1 | False | 1→2 | 62→62 | 无 backward，继续推进计数 |
| B G1 | True | 2→62 | 62→62 | 60 个 GC 节点，59 个 offload |

G0 的 60 个实际 offload 节点全部 BF16→BF16，post-backward B G1 的 59 个和 A G1 的 60 个也全部相同 dtype；本次没有支持 FP32 压缩的实际节点，关闭该候选，不改 `UNSLOTH_OFFLOAD_FP32`。

N0/R0 的 slot allocation/release 与计数变化已证明外层 no-grad 时发生了多余的激活保存，而且它们被累计进“首轮层数”LAST。G0 只有 60 个节点，LAST 却为 62。后续重复 G→detach 继续使 CURRENT 增至 122、185、245，而 LAST 保持 62。这是实际状态污染，不代表其必然造成数值偏差。根任务的同参数比较显示 A 的 G0/post-G 完全相同，而 N0/post-N 改变；该数值现象仍需干预证据。

入口链的固定源码依据：

- Transformers 5.13.1 `modeling_layers.py` SHA `2f0028eefdfecdf3423f44cccf549ec51e01536104ff124cd6e3dfcb63ae82fb`，60 行入口只判断 gradient_checkpointing 和 training，没有检查 grad mode；92 行传 `partial(super().__call__, **kwargs)` 与位置参数。
- Zoo `temporary_patches/misc.py` SHA `c631496dab0deccd1a5c3da06b02eae40af6de0605a3e42cc259c00c75aefd14`，2573–2595 的 keyword-input patch 同样没有 grad-mode 排除。有 requires-grad keyword tensor 时，它把 kwargs 转成 `_KeywordArgumentCall` 和尾部位置参数；否则回到原入口。两条路径都只给 checkpoint callable 传位置参数。
- Zoo `peft_utils.py` SHA `90f6fc67df08780b6eef070bd3dccd1c581a0a647766419cd1e36d2f69405fa1`，694–699 的 eager post-hook 会将 embedding 输出设为 requires-grad，即使外层处于 no-grad；仅编译中的 no-grad 分支跳过。实际 hook 身份本次未记录，不能冒称已逐函数绑定，但这条源码机制与每个 NG 只有第一层计数 +1 相符：第一层的显式 requires-grad 输入触发保存，而 no-grad 的 Function.apply 输出不建立后续梯度链。

这三个文件均手动核对了本地对应固定版本 RECORD；R6 未直接记录实际 modeling_layers / misc / peft_utils 文件 SHA，保留这一绑定边界。

最小后续诊断建议，仅限新实验，不改训练默认：在每个 N/REF 评分的紧凑 scope 内，临时替换实际 60 个 text layer 实例的 `_gradient_checkpointing_func`。replacement 在 no-grad 时直接调用传入 function 和位置参数；若遇到 enable-grad，原样交回该层捕获的原 callable。保持 gradient_checkpointing 标志、training、embedding hooks、autocast、参数和原 loss/anchor 不变。这样仍走 outer layer `__call__`，保留 use_cache/past 清理和模块 hook，也保留 partial / `_KeywordArgumentCall` 对模型 kwargs 的重建。

当前固定入口不向 replacement 传 kwargs，所以最窄实现可将意外非空 kwargs 作为明确的实验失败，不应静默删除或把 `preserve_rng_state/use_reentrant/context_fn/debug/determinism_check/early_stop` 等 checkpoint 选项传给模型。无需建设通用 wrapper 协议。factory 必须逐层捕获原 callable，避免循环闭包 late binding；保存原实例属性是否存在及原 object，以 try/finally 逆序恢复（原先无实例属性则 delattr）。scope 在 G0 前完整退出，确认恢复的是相同 object，G0 与 backward 消费原路径。不要调用 model.eval()/for_training()/gradient_checkpointing_disable()，不要清理或改写 GC globals。

干预的可解释结果应包括：N/R scope 内 GC globals 是否保持原值、G0 是否从 0 增至 60、G0 的相同参数输出和实际梯度、N/G 数值差是否变化。若只有多余 offload 和 LAST 污染消失而概率差仍在，则这项优化只解决多余保存与计数语义，不能宣称解决数值根因。

## LoRA 分支的静态条件与尚缺的实际调用身份

本次追加仅阅读固定依赖源码与 R5/R6/R7 已有记录，没有新模型、GPU、网络或 CPU 模拟。原始 bytes 与同目录 Unsloth 2026.10.9 `dist-info/RECORD` 的 SHA 和长度再次匹配：`unsloth/kernels/fast_lora.py` 为 `589df65aae64fdab7f32b8266e76ca7948e944927f7bd15b8632a3106aa231ae`（39,457 bytes），`unsloth/kernels/utils.py` 为 `ffe3907694c38536906988d34a79b126b58510f780d6e808d67ef777c7b59c07`（67,853 bytes）。后者还与实际 loaded identity 的 accessor 文件 SHA 一致；实际运行未单独保存 `fast_lora.py` 的加载文件身份，因此其边界仍是固定 wheel 源码。

`fast_lora.py` 220–274 的 `_lora_dropout_and_bias` 返回融合路径的 dropout／bias 参数，或返回 `None` 让调用方回到各 projection 的普通调用。246–260 附近的 `grad_enabled`、`X.requires_grad`、`_active_lora_requires_grad` 判定只位于 adapter 缺失、disabled 或 merged 的支路：存在梯度需求时回退，以免融合 backward 缺少所需 LoRA 权重。它不是所有 G／NG 都会经过的分流。此前 `_is_plain_lora` 全满足还会直接返回。实际记录中的 410 个 LoRA 层均为单个 default、disabled=False、merged=False；reference 仅临时将 scaling 置零，没有 disable adapter。因此不能拿这个源码条件直接解释本次 actor／reference 或 G／NG 差异，也不能从这些记录推断未记录的 projection 路由。

`utils.py` 1825–1852 的 `addmm_` 仅在 custom op 可用、正在编译、grad 关闭且 alpha 是 Python 数值时，调用 `unsloth::addmm_` mutation op；两条源码分支最终都执行 `out.addmm_(A, B, alpha, beta)`。其设计差别是编译时避免 functionalization 为输入先复制，不能仅据此声称换了数学公式。1855–1902 的 `matmul_lora` 顺序为 base dequant／matmul，再 `XA`，再以 scaling 为 alpha 的原位 addmm；若实际回退到分开的 PEFT LoRA线性、缩放和相加，有限精度算术顺序可能不同，但本次没有证据绑定这种切换。`fast_lora.py` 604–650、974–1022、1115–1130 的 MLP／QKV／O helper 确有融合与 projection 回退分支；是否真的调用这些 helper 尚未记录。

还需保留嵌套执行模式的限制：固定 `fast_lora.py` 34–43 在 torch<2.11 时用 Dynamo disable 包装 `_apply`；本次 torch 是 2.10.0。若走该融合自定义 autograd Function，其 `forward` 内部又在 no-grad 语义下执行。因此外层标为 G 或 N，不能直接推出 `addmm_` 内部的 compiling／grad 条件不同。这两处源码都尚未构成本例已激活的根因链。

现有 `resolved_nf4_dispatch`（`training/vfarl_unsloth.py` 222–241）只记录无 `base_layer` 包装的裸 bitsandbytes `Linear4bit` 的 **`type(module).forward`**；明确跳过 PEFT 包装，亦不等于逐实例实际绑定的 forward。R5/R6/R7 的状态记录包含 LoRA 参数 metadata、scaling 和 checkpoint callable，但没有实际 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj.forward`、attention 的 `apply_qkv/apply_o` 或实际 MLP融合入口身份。下一次解释应先补实际绑定函数、来源及路由证据，不能把文件里存在某分支或模块已加载当作它已执行，也不据这一静态候选追加 GPU、修改 loss 或 precision。

## R7实际干预及跨运行边界

[R7原始结果](execution-r7-no-grad-checkpoint-20261028/README.md)采用更窄的post-backward干预：保留原始N0/R0/G0和一次A backward，再按两动作分别作legacy／candidate／restored N/R/G。每个candidate scope有120次真实direct调用，完整GC状态前后相同，全部60个原callable退出时精确恢复。它没有把最初的N0/R0或原始backward改成候选路径，也没有重置此前的GC历史。

两动作对应LP向量及hidden SHA在三种路径中全部相同；N/G最大差仍为A0.060098、B0.055105。由此关闭“在这些post评分中绕过checkpoint即可消除数值差异”的候选，不把它推广为训练修复，也不声称所有GC机制都已排除。A在R7原backward前后N/R/G也全部相同，未复现R6的A N位移；R6前后隔着B的五次调用，不能认定backward是唯一因果。

[跨运行比较](cross-run-mode-comparison-20261028.json)绑定30份实际来源，R5与R7的六组A/B N/R/G均LP逐值及hidden SHA相同，R6六组均不同。三次记录的base、adapter、math、package、NF4 dispatch及loaded runtime identity一致。R5/R6/R7的loaded-identity、checkpoint source和现有日志均没有实际Gemma4 generated/compiled forward文件路径及SHA；R5已保存的首200项发布列表也没有可定位条目。本次没有翻页、猜路径或请求大包。固定依赖源码身份不能替代实际执行图身份，编译来源仍是尚未检验的线索。
