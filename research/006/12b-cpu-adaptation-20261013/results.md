# 12B Unified CPU适配实际结果

日期：2026-10-13，北京时间。本轮按P006实施授权只做CPU代码适配、小资产获取和一次性手动数值核对。没有下载预训练权重、没有GPU、没有真实12B训练，也没有新增测试文件或检查脚本。

## 已改动的路径

- `training/train_sft.py` 支持 `gemma4` / `gemma4_unified` 两个完整conditional-generation类。新增12B固定来源manifest；配置、tokenizer、CT模板先核对，再加载权重。远端HF必须是固定repo/commit，本地QAT镜像需匹配公开权重大小及SHA。12B拒绝复用31B缓存；随机微型诊断显式标为fallback，不能当生产QAT。
- `training/memory.py` 对两种类保留shift、softcap、监督token均值及显式分母语义；输出类匹配各自架构。新增Unified RMSNorm识别，大矩阵保持原dtype，仍用non-reentrant checkpoint。未启用默认关闭的grouped attention。
- `training/export_peft.py` 限制实际adapter键位于language/text decoder层，继续排除vision/audio/embedder/norm/head；保留实际PEFT键，不改名或合并基座。导出manifest携带训练时固定来源，运行时兼容仍标为未验证。
- 新增 `training/configs/sft_12b_qat_nf4.yaml`、`gemma4_12b_sources.json` 和CT v2 Jinja。16K是编码接收上限，不是显存实测；真实首步仍待额度安排及明确train样本。

## 真实来源与本轮下载

QAT来源：`google/gemma-4-12B-it-qat-q4_0-unquantized`，revision `b6ed86275a6a5735884e208bfed95b445a684ca2`。4,310-byte config SHA `a323d02f68420f6fa3a3548130a0d36356075a4047a622e57148558f8eee7077`已取回并核对。权重仅记录官方元数据：23,919,549,408 bytes，SHA `26f2cee4292298a3f9f92209643c37c80e34e011381e22434088870d9439a0a0`，本轮未下载。

部署文件直接取自 Kaggle `google/gemma-4/Other/gemma-4-12b-it-qat-w4a16-ct/2` 的单文件接口：[下载manifest](source/tokenizer-manifest.json)。CT Jinja为17,466 bytes，SHA `36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0`；tokenizer.json为32,169,626 bytes，SHA `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`；tokenizer_config为2,774 bytes，SHA `ee66825f1a587ca13bcb90d7e60f59bc99e84f3b8bc0cefdc3ea595cd5abf773`。

tokenizer主体与数据分支的31B CT资产逐SHA相同，但这里独立保留12B来源和完整CT模板/config。下载启动后才收到避免重复取主体的消息；本次确实产生了一份重复32MB tokenizer，未再次下载。数据分支已收到统一缓存路径 `training/artifacts/tokenizers/gemma4-12b-ct-v2/`。HF QAT当前的模板及tokenizer_config与该CT版本不同，不可静默替换。

## CPU实际观察

环境为 `/private/tmp/p006-training-cpu/bin/python`，Python3.11.15、torch2.14.1、Transformers5.13.1、PEFT0.21.2、safetensors0.8.0。原项目venv无ML栈，未改全局环境。最初尝试复用uv中torch2.14.0，但依赖解析实际安装了2.14.1并下载121.4MiB wheel；其余依赖安装在独立临时环境，使用独立临时cache。CPU版本不等于Kaggle CUDA栈。

| 实际操作 | 观察 | 证据边界 |
|---|---|---|
| 随机两层Unified和原Gemma4，各有sliding/full层、K=V、padding/稀疏labels，原HF与selected loss比较 | FP32 loss差均0；LoRA梯度最大绝对差分别8.94e-8、1.27e-7；训练态checkpoint调用非零；未标注forward仍返回原logits | 微型模型；不是12B pretrained或NF4 |
| 随机Unified BF16、小norm FP32，同样的稀疏标签比较 | loss与LoRA梯度最大差均0，20个LoRA梯度张量均有限且非零 | 此随机输入观察，不证明CUDA位级一致 |
| 经真实loader加载保存的随机Unified，执行训练loop | 两步Adam，7个监督next-token标签、15个输入tokens；基座未变、LoRA更新 | 合成token用于实现核对，无修复意义 |
| 从第一步checkpoint恢复到第二步 | 与连续训练的全部LoRA参数最大差0 | 同一CPU环境/随机小模型 |
| 独立PEFT导出 | 10个文本投影、20 tensors、832个LoRA参数；无vision/audio权重 | 导出不代表官方CT加载成功 |
| 完整官方12B config在meta device构造 | 48层、240个q/o+MLP目标；r8共28,737,536参数（BF16约57.5MB，FP32约115MB） | 只分配meta形状；不是显存/训练峰值 |

原始手动记录：[FP32比较](manual-forward-results.json)、[BF16比较](manual-bf16-results.json)、[优化/恢复/导出](manual-training-results.json)、[完整配置meta形状](full-config-meta-results.json)。随机微型adapter只在临时目录，不放入正式提交。

meta构造的Parameter元素数是11,959,730,176；加48个persistent `layer_scalar` buffer，恰等于HF safetensors元数据11,959,730,224。最初直接把Parameter计数等同存储元素数时相差48，实际源码定位到该buffer后消除疑问，没有据此伪造权重加载。

## 与真实数据分支的接点

已读取数据分支 `research/006/data-replay-20261013/outputs/encoded-12b-ct-v2.jsonl` 的编码记录，**只核对模板/tool/tokenizer指纹与长度，没有训练**。两条分别3,368/519和2,055/226输入/监督tokens；来源与实际重放结论由数据分支报告负责。它们不是预先冻结的train/select，也不代表完整benchmark已resolved。[协议核对原件](source-protocol-results.json)。

同次手动调用实际拒绝了31B cache标志、无manifest的12B、普通IT冒充QAT与错误HF模型来源。新增的生产Unified分支另要求12B配方，避免只换`--model`后仍保留31B标签。后续远程训练必须自行取得固定来源权重，不能把本轮微型CPU模型当作12B训练完成。

## 一次性命令摘要

所有计算使用Shell heredoc，未保存可运行的test/check/verify/gate文件。依赖安装用 `uv pip install --python /private/tmp/p006-training-cpu/bin/python --cache-dir /private/tmp/p006-uv-cache ...`；Kaggle下载显式设 `ApiDownloadModelInstanceVersionRequest.path` 为单个tokenizer文件，未调用整模型下载。

数值命令分别调用现有 `resolve_target_modules`、`prepare_model_for_memory_efficient_kbit_training`、`install_selected_token_loss`；对同随机权重的原HF/selected两臂实际 `loss.backward()`。优化命令调用生产 `load_model_and_tokenizer`、`run_training_loop`、`export_adapter`，然后从checkpoint-000001重载。第一次临时手写输入遗漏`attention_mask`，collator明确失败；补齐该必需输入后完成上述两步，没有改生产collator来容忍错误。

固定来源命令通过HF小config实际执行 `validate_pinned_model_source`，并用真实CT tokenizer对数据分支的两条编码执行 `validate_tokenizer_protocol`。配置dry-run与必要diff检查均为手动命令；检查数量不作为研究收益。

## 尚未运行

12B预训练权重加载、CUDA/NF4、真实长序列首步、12B CT adapter推理、31B/12B真实修复配对均未运行。当前预算优先主代理E2，12B GPU研究等后续周额度及明确运行安排。静态PEFT键与HF结构兼容不能代替v31官方推理服务实际加载与数值作用证据。

## 同日终审修复：checkpoint必须绑定实际基座身份

只读终审发现旧`_signature`只绑定配置路径与编码数据，未绑定已验证的来源manifest内容。本地镜像与manifest同路径更新后，可能将旧LoRA/优化器接到不同冻结基座。已为有manifest的配方加入manifest SHA、QAT repo/revision、model_type、原config SHA以及权重SHA/大小；模型、tokenizer、模板、工具schema和manifest的本地路径在签名中用固定身份/指纹替代，避免换挂载位置就失去恢复能力。

单卡与DDP入口都在完成来源核对后计算签名，DDP使用修改rank-local device前的共享配置；签名不符时先拒绝，随后才允许覆盖run_config或恢复优化器。未启用新manifest的31B路径保持历史签名格式，并排除新引入的`model_source_manifest=None`字段；旧12B无基座身份签名不静默接受。

一次性CPU命令实际使用保存的随机微型Unified模型及它自己的config/权重SHA构建来源manifest：先完成两步，再将模型、tokenizer、模板、工具schema和manifest移到不同路径，从第一步checkpoint恢复；最终全部LoRA参数与连续训练差0。仅修改同路径manifest时恢复被拒绝；修改微型模型q_proj的真实权重并相应更新manifest后也被拒绝。旧31B签名按历史算法手工重算一致。没有构造专用测试文件，也没有声称实际12B权重或CUDA恢复已验证。[本次结果](resume-identity-results.json)。
