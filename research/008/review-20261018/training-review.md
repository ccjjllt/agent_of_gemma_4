# 12B真实训练与CT探针复核

> **P009执行更正（2026-10-18）**：本报告下文“override是自定义路径、默认Unified失败所以正式兼容未通过”的推断遗漏了已归档官方证据。Ryan在[discussion748287](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748287) comment3536939（2026-10-11 14:39 UTC）明确scorer已经加入`Gemma4ForCausalLM/model_type=gemma4`覆盖；原件位于`research/006/official-data-20261013/topic-748287.json`，P006报告第38行亦有记录。V6使用这一架构覆盖符合官方说明。ADK默认`extra_args=[]`不能推翻scorer的per-model映射。保留下文当时诊断作为可追溯记录，但不再据此阻止新数据训练；实际未证明部分仍是32768等完整运行条件、真实Agent修复及正式得分。

复核日期：2026-10-18。起点HEAD `be9b27b578127cc4939933ba2e8d4eb6d9db49ea`，比较`5a979ef..be9b27b`；只读训练代码、NF4生产脚本、V1–V6原件与冻结click训练/pydicom诊断材料。随后按主代理明确授权，仅修改`training/train_sft.py`的缓存恢复身份和来源字段。未运行GPU、模型加载、训练或新测试脚本，未读取heldout或预留题。

## 可以确认的进展

**真实12B训练已经开始并完成过首步；原生submission兼容与修复收益仍未证明。** V1–V6都从同一NF4缓存、同一click样本各自完成一个AdamW步，六次不是累积六步。V6完成了独立导出及自定义text-only CT服务中的数值作用对照。其动作探针没有成功的生产编辑，也没有整题运行。

| 层次 | 本次核实结论 | 直接证据 |
|---|---|---|
| 基座来源 | 固定HF QAT commit；CPU下载时核23.9GB原权重SHA，之后生成NF4训练缓存，未把NF4当比赛基座 | [CPU脚本](../../007/12b-qat-cache-and-train-20261017/cpu-nf4-cache/cache_12b.py):232–319；[缓存manifest](../../007/12b-qat-cache-and-train-20261017/cpu-cache-output/cache-manifest.json) |
| 真实训练 | 一条click，3368输入/519监督tokens，`synthetic=false`、未截断；pydicom未进入反向 | [冻结用途](../../006/data-replay-20261013/data-usage.json)；[V6训练manifest](../../007/12b-qat-cache-and-train-20261017/gpu-output-v6/p007-12b-probe/train-run/training_manifest.json):3705–3736 |
| 反向/更新 | 生产训练loop执行backward、裁剪与AdamW；240个LoRA B均非零，28,737,536个参数 | [训练loop](../../../training/train_sft.py):518、577–584；[phase1记录](../../007/12b-qat-cache-and-train-20261017/gpu-output-v6/p007-12b-probe/phase1-training-report.json):13–21 |
| 独立导出 | 单独CLI导出；训练adapter与独立副本的safetensors本地SHA一致，未合并基座 | [构建器](../../007/12b-qat-cache-and-train-20261017/build_12b_gpu_probe.py):113–149；[导出manifest](../../007/12b-qat-cache-and-train-20261017/gpu-output-v6/p007-12b-probe/export_manifest.json) |
| CT数值作用 | 完整519/226个监督位置均有logprob，base=zero，真实adapter产生变化 | [teacher-forced结果](../../007/12b-qat-cache-and-train-20261017/gpu-output-v6/p007-12b-probe/phase2-teacher-forced.json) |
| 原生部署 | **未通过**：V6改用Gemma4ForCausalLM，并改上下文/批次限制；默认Unified路径有实际启动失败 | [V6服务命令](../../007/12b-qat-cache-and-train-20261017/gpu-output-v6/p007-12b-probe/phase2-server-cmd.json)；[V2日志](../../007/12b-qat-cache-and-train-20261017/gpu-output-v2/p007-12b-probe/phase2.log):17、220–270 |
| 实际修复 | **未证明**：两局部状态都未得到成功edit，更没有官方整题裁判 | [V6动作记录](../../007/12b-qat-cache-and-train-20261017/gpu-output-v6/p007-12b-probe/phase2-ct-probe-report.json):137–2138 |

## 首要部署边界：V6绕过了默认12B架构

官方CT配置的`architectures`是`Gemma4UnifiedForConditionalGeneration`，[原件](../../006/official-data-20261013/models/12b-ct-config.json):2–4。V2日志实际解析成`TransformersMultiModalForCausalLM`并回退到Transformers模型；profile阶段经`embed_multimodal → get_image_features → embed_vision.patch_dense → LoRA shrink`，触发`token_lora_mapping.size(0) == M`断言。该失败发生在服务启动，不是训练反向OOM，也不证明adapter权重没被加载。

V6显式传入`--hf-overrides '{"architectures":["Gemma4ForCausalLM"],"model_type":"gemma4"}'`，并设置max_model_len=8192、max_num_batched_tokens=512、max_num_seqs=1。[构建器](../../007/12b-qat-cache-and-train-20261017/build_12b_gpu_probe.py):356–389；[V6原始服务日志](../../007/12b-qat-cache-and-train-20261017/gpu-output-v6/p007-12b-probe/vllm_12b_server.log):7–13确认使用该路径。原生text-only实现有conditional权重/LoRA键到文本层的映射，且跳过视觉/音频权重；对应代码来自本地归档的[vLLM wheel源码](../../007/12b-qat-cache-and-train-20261017/cpu-prepare-output-v4/p007-12b-probe/vllm-gemma4-sources.json)，`gemma4.py`源内1439–1457、1572–1674行。

ADK的[VllmConfig](../../006/official-data-20261013/wheels-source/adk_submission/server.py):160默认`extra_args=[]`，build_cmd只把所给extra_args附加到命令。探针agent.yaml只声明model/adapter/tools，未承载上述server override。因此使用官方CT权重、官方VllmServer类，并不等于已经走过普通声明式submission的原生12B路径。**准确表述应为“12B CT权重在自定义text-only服务中，真实LoRA产生可测分布作用”**。没有隐藏scorer的当前启动命令，不能进一步断言所有正式12B提交必定失败。

## V1–V6不是六步训练曲线

| 版本 | Phase 1 | Phase 2实际结束点 |
|---|---|---|
| V1 | click真实1步＋导出完成 | 导入不存在的`adk_eval_core.editing.edit_file_content`失败 |
| V2 | 同样从头1步＋导出完成 | 默认Unified/Transformers multimodal路径的LoRA profiling断言 |
| V3 | 同样从头1步＋导出完成 | text-only服务后，长prompt `/completions` HTTP500；未保存服务器根因日志，不补猜原因 |
| V4 | 同样从头1步＋导出完成 | 三个model短prompt smoke通过；长prompt log_softmax申请2GiB而剩余1.27GiB，推理OOM |
| V5 | 同样从头1步＋导出完成 | 8192/512后继续到结果处理，误把字符span当token_span，`KeyError: token_start` |
| V6 | 同样从头1步＋导出完成 | teacher-forced与局部动作生成完成，summary success=true |

各版本`phase1-training-report.json`与`train-run/training_manifest.json`均记录1条、1步，同一数据SHA `efd43a741f5a80dfe86bf25a0147cc81105fde2cc446f43ff6e3c760709f00c6`。原始`phase2.log`保存上述失败；V4根因在[vllm日志](../../007/12b-qat-cache-and-train-20261017/gpu-output-v4/p007-12b-probe/vllm_12b_server.log):196–198。运行版本号不能作为epoch或有效样本数。

## 数值、显存与导出证据的范围

V6训练CLI总95.491秒；报告中的`optimizer_step_wall_seconds=14.827`实际取自整个`run_training_loop`计时，包含checkpoint与最终adapter保存，不能简称纯Adam或纯前反向耗时。519/14.827=35.004监督tokens/s是这段单步loop的观察，不能直接年化成稳态大语料吞吐。[构建器](../../007/12b-qat-cache-and-train-20261017/build_12b_gpu_probe.py):105–111、212–220；[训练计时位置](../../../training/train_sft.py):550–603。

训练发生在`cuda:0`，单L4进程峰值allocated为9.7362GiB；整个Notebook仍申请4×L4。峰值在加载后reset，反映训练阶段PyTorch allocated，不包括加载瞬态和进程外显存；`peak_cuda_reserved_bytes_after_eval`来自随后另一个no_grad模型评估，不是训练reserved峰值。不能从3368直接宣称16K、DDP或大batch容量。V6外层完成493.674秒，CT启动195.243秒；KV token容量与68.78x theoretical concurrency不是实测Agent并发吞吐，实际server max_num_seqs=1。

本地一次性读取safetensors header并重算SHA：训练adapter及独立导出都是480个F32张量、28,737,536参数、115,020,728 bytes，SHA均`f292a8e1ac092cb6cd5744e70eae5f32c4ffcc0faa1321dbe1bac28b2de5994e`。zero副本SHA为`aea9e0ec212f0b1141c664047c5fb7cda0d22d680f238600eaabbea26afb7eea`；源码复制A并把B置零，配置相同。独立导出和后续重载成立，不只是内存中训练对象自评。[exporter](../../../training/export_peft.py):102–132仍保持独立PEFT文件与`runtime_compatibility_verified=false`，这个保守字段应保留。

| 位置 | click训练样本 | pydicom冻结诊断 |
|---|---:|---:|
| NF4 zero→adapter selected NLL | 0.530146→0.515842 | 1.654296→1.653286 |
| 自定义CT base→adapter NLL | 0.507977→0.503882 | 1.686483→1.684741 |
| CT base与zero最大logprob差 | 0 | 0 |
| CT共同覆盖监督tokens | 519/519 | 226/226 |

这些是同一teacher-forced序列各位置的条件概率，不是模型自由执行整条轨迹。NF4和CT差异还混合HF Unified/原生text-only实现，不能全部归因为量化。当前数据中没有缺失logprob位置，因而完整NLL比较成立；未来若缺位置，构建器当前会取三臂交集，不能仍称全监督NLL。

训练与诊断编码SHA、token长度、label与input_id对应均已手动复核；V6 source-manifest记录的四个训练文件在本次修复前与HEAD逐SHA相等。未重新执行模型或重新编码。训练模板/工具schema核对成立，不代表live chat入口和训练的每一个中间前缀已逐token证明一致。

## 动作探针存在的实质限制，尚未修改

[构建器](../../007/12b-qat-cache-and-train-20261017/build_12b_gpu_probe.py):546–590的`execute_edit_and_verify`不是完整官方工具执行：只取第一个tool call，忽略filepath正确性，把old/new字符串套到预选的源码字符串；成功后仅compile，再用字符串包含关系设置click/pydicom行为标志。没有落盘、没有真实API行为执行、没有完整仓库/裁判。README“executes generated tool calls against real buggy source files”应收窄为“对实际源码文本试用官方替换函数”；本次没有改该探针实现或README。

V6实际也没有正例：click base/zero重发run_command，adapter改为`ls -R`；pydicom base/zero的短old_string有两处匹配而失败，adapter改为read_file第100行。六个`edit_ok`均false。adapter生成不同动作只能证明策略变化，不能证明恢复成功；click的`ls -R`甚至提示下一步可能偏离任务，但单状态单次结果不足以估计泛化退化率。

另一个运行续训边界：[构建器](../../007/12b-qat-cache-and-train-20261017/build_12b_gpu_probe.py):204–208主动删除所有checkpoint与optimizer状态。V6保存的adapter适合当研究产物，**不能从它恢复原Adam状态继续第2步**；若使用其权重再训，应明确是新优化器warm-start。六版本重跑Phase1也不构成真实GPU resume验证。

## 已发现并获授权修复的两处问题

只修改[training/train_sft.py](../../../training/train_sft.py)，没有重写memory/export或生成冻结V6 bundle。

**1. 原恢复签名未绑定有效NF4基座。** 旧`_signature`绑定原始QAT身份，但忽略`model_info.nf4_cache.manifest_sha256`及实际缓存config。本次先用V6真实run_config重算旧签名匹配；只更换NF4缓存manifest指纹后，旧签名仍相同。因此重新量化/更新缓存后，旧优化器可能恢复到不同有效冻结权重上。

修复位于265–278、437–484行：读取实际`config.json` SHA并对照创建manifest；`effective_nf4_base`绑定cache manifest SHA与实际config SHA。缓存路径、tokenizer/模板/schema路径用内容身份代替，换挂载位置不变。无缓存的历史31B与pinned签名保持原算法；旧cache签名不再静默兼容，缺少config身份的旧metadata直接拒绝。权重分片仍沿用不可变挂载的manifest＋大小校验，没有增加GPU阶段整模型重哈希。

**2. 缓存分支误报`local_weights_hash_checked=true`。** 原值仅由`Path(source).is_dir()`决定；缓存分支没有重新核全部权重SHA，却写true。修复位于308–336行：缓存分支为false，并显式记录`local_nf4_cache_manifest_checked=true`；实际逐文件核验的未量化本地路径仍为true。它描述当前调用做了什么，不抹掉CPU创建阶段已经记录的完整来源SHA。

## 本次一次性CPU验证与局限

未新建测试文件。Shell heredoc在TemporaryDirectory构造**仅供元数据判断的小文件，绝非模型权重**，调用生产`validate_pinned_model_source`、`validate_nf4_cache`、`_signature`和`load_resume_state`。临时目录退出即删除。实际结果：

- 同cache复制到另一挂载路径，并移动source manifest：签名相同，恢复身份比较接受。
- 更换cache manifest：新签名不同，旧恢复身份被拒绝。
- 等字节数修改实际config：在缓存验证阶段拒绝；同时更新manifest中的config SHA后，仍因有效基座身份变化拒绝旧恢复。
- 旧cache签名与缺少实际config身份的旧metadata均拒绝。
- 对照`be9b27b`的旧函数，未使用缓存的pinned与31B签名逐值保持相同。
- 缓存分支的权重hash标志为false，manifest标志为true；小型未量化文件分支相反。

现存`/private/tmp/p006-training-cpu/bin/python`已没有torch；首次尝试实际torch状态读取时立即`ModuleNotFoundError`，未加载模型。后续明确使用**JSON state-loader替身**给`load_resume_state`提供已保存元数据，只验证真实函数的身份接受/拒绝分支，不把结果写成torch checkpoint反序列化、优化器恢复或12B训练恢复成功。另完成AST语法解析和`git diff --check`。没有安装依赖、GPU或训练。

## 下一步训练规模的依据

1. **先解决原生12B提交路径的真实兼容，再扩大训练支出。** 允许CPU准备新独立issue材料，但不能用V6 text-only成功直接推出正式adapter可提交。需要与实际普通提交入口一致的加载/数值/修复证据。
2. **容量起点可以从“真实3368 tokens、519监督、r8、microbatch1首步已成功”开始。** 不再重复合成2048诊断。用新增、已明确分配训练用途的数据测少量连续更新；下一个真实长度应由可用样本分布决定，不先扫描16K/32K网格。
3. **click容量样本不能变成收益语料，pydicom诊断不能进梯度。** 训练规模要按独立issue计数，当前1条训练＋1条诊断没有泛化收益。长序列、padding、不同监督密度和多步Adam稳定性仍未实测。
4. **保存可恢复状态、将加载与稳态吞吐分开。** 现有探针配置`max_steps=1`且每步checkpoint、随后清除状态，适合容量/导出探针，不是正式多步训练运行模板。扩大前使用本次修复后的生产代码重新打包；冻结V6 bundle保留为历史原件，不能冒充含本次修复。

本报告所引原始`.log`及adapter二进制存在于当前本地下载目录；它们受gitignore影响，不能仅凭`be9b27b` Git对象重建全部运行产物。复现交接应保留原Notebook版本及下载产物，勿把JSON汇总当完整逐token原始响应。
