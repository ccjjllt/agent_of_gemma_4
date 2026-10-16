# P005：从真实轨迹到独立LoRA

本目录包含实际数据转换、Gemma工具对话编码、BF16 LoRA/同源NF4 QLoRA训练、checkpoint恢复和独立PEFT导出。修复后已在Kaggle四张L4上完成真实31B NF4 LoRA的四个诊断更新（256、256、1024、2048长度），四rank参数一致，云端adapter已保存。该轮使用合成代码诊断数据，尚未做真实修复语料SFT或取得比赛增益。详见[四卡实测结果](../research/005/kaggle-training-probe-20261010/ddp-result-summary.json)；[首次FP32转换OOM](../research/005/kaggle-training-probe-20261010/result-summary.json)保留为历史。

NF4默认只将小norm参数提升FP32，保持大embedding/head为原dtype；`loss_backend: selected_chunked`保留完整上下文、只投影监督位置，以checkpoint分块CE降低中间显存。[四卡DDP入口](train_ddp.py)复制基座到每张卡，仅同步LoRA梯度；两进程CPU实测已与单进程全局批次对齐，并通过尾部不满批次和断点恢复。31B实测2048步骤15.89秒（含约0.70秒诊断），每卡峰allocated20.398 GiB；更长上下文和真实数据容量仍需另测，不能直接外推4K/16K。

## 1. 依赖和输入

本地集成使用Python3.11；Kaggle Python3.13.15已通过训练依赖与官方tokenizer导入。requirements.txt记录本地CPU集成使用的版本；训练环境与比赛vLLM环境分开。正式NF4训练需CUDA和适配该GPU的bitsandbytes。单设备使用train_sft.py，多卡使用train_ddp.py；DDP每卡各放一份基座，不能把总显存直接相加，也不使用device_map=auto。

    python -m pip install -r training/requirements.txt

此命令是给实际运行环境的安装说明，训练脚本不自行安装依赖。默认local_files_only=true；缺少依赖、模型、数据或权重返回非零，不再输出假成功。

**Kaggle例外：保留镜像自带CUDA torch，不直接安装上面的本地CPU版本表。** 使用[requirements-kaggle.txt](requirements-kaggle.txt)在CPU整理wheel；已完成的私有`ccjjllt/p005-training-cpu-stage`输出可直接挂载给离线GPU Notebook。实际准备代码与资源探查见[CPU准备入口](../research/005/kaggle-training-probe-20261010/cpu-stage/stage.py)，训练无需比赛41个推理wheel。四L4利用方案及显存优化交接见[第8节](../docs/plan/005-implementation-handoff.md#8-每周30小时配额下的训练资源交接2026-10-10)。

准备官方未压缩QAT基座和相同部署tokenizer的本地目录。指定CT/W4A16文件只用于最终推理，不能直接拿来做普通QLoRA。普通IT备用实验需要明确标注，不能仅改adapter配置名称。

当前第一版训练协议为 **p005-harness-tools-v1：官方六工具**，schema在 [harness_tools.json](configs/harness_tools.json)。先在同一六工具底座上比较base/adapter；P005额外skill协议的训练数据需实际采样或重放，不能把旧edit_file改名成span apply后冒充同协议。

## 2. 真实数据入口

### 已归档本项目轨迹

    python training/data/prepare_local.py --output-dir training/artifacts/local-diagnostic

只读已有ATIF、result和元数据划分；在解析轨迹前排除heldout，不读任务答案，不运行模型。默认选最终resolved且任务ID在后来0B eligible名单的轨迹；这项后来资格过滤不证明历史环境完全等价。

本轮实际取得15条去重轨迹、5个独立任务、89个目标片段。它们属于已经暴露的开发任务，只适合诊断输入管线，不是跨仓库训练规模或独立泛化证据。原始错误工具调用保留在前文，但不作为正向动作监督；工具回包和源码不凭空重建。

部分ATIF步骤明确带event_type=thinking及原文，导入器保留为reasoning_content。不能因为旧索引没保存thinking就宣称全部原轨迹都没有；没有记录的仍不补写。

### 外部或新采样轨迹

convert支持两种明确输入：标准messages/raw_turns对象，或trajectory字段内的ATIF v1.x。每行必须有trace_id、group(issue/repo/commit)、source和来源类型。外部特定scaffold不符合这些结构时需要来源适配器；当前不宣称任意OpenHands/SWE-agent文件都能直接读取。

    python training/data/canonicalize.py convert \
      --input /path/raw.jsonl --output training/artifacts/canonical.jsonl --compatible-only
    python training/data/transactions.py split \
      --input training/artifacts/canonical.jsonl \
      --output training/artifacts/transactions.jsonl \
      --labels-out training/artifacts/offline-labels.jsonl

不做工具名语义替换。未知工具明确需要重放；官方工具参数错误保留为历史，并屏蔽对应坏动作。标准tool_call_id必须能与回包匹配，多调用ATIF没有明确回包对应时拒绝。

每个片段保留原system/user及其之前所有真实观察，只选择2–6个assistant动作计算loss；不会删除失败动作后拼接旧未来。超长前文交给编码器明确拒绝，不能只保留最近一条回包。

## 3. 分组与实体输出

    python training/data/split_registry.py register \
      --input training/artifacts/local-diagnostic/transactions.jsonl \
      --registry training/artifacts/local-diagnostic/splits.json
    python training/data/split_registry.py assign \
      --registry training/artifacts/local-diagnostic/splits.json --whole-repo
    python training/data/split_registry.py materialize \
      --input training/artifacts/local-diagnostic/transactions.jsonl \
      --registry training/artifacts/local-diagnostic/splits.json \
      --train-out training/artifacts/local-diagnostic/train.jsonl \
      --select-out training/artifacts/local-diagnostic/select.jsonl

每issue默认最多3条；按完整内容去重，不把“工具调用顺序相同”误当重复任务。相同issue或相同repo+commit相连的样本始终同组；whole-repo进一步整仓留出。支持--heldout-ids列表，排除传播到同快照组。这个本地诊断集分出6条train、9条select，不改变原项目任务划分。

## 4. 真Gemma编码

    python training/encoding/gemma_tool_chat.py encode \
      --input /path/train-transactions.jsonl \
      --output training/artifacts/encoded-train.jsonl \
      --tokenizer /models/gemma4-tokenizer \
      --tools training/configs/harness_tools.json \
      --thinking on --thought-loss mask --max-seq-len 4096 --on-oversize skip

使用完整会话的官方chat template与工具schema，核对整段token ID，按精确offset只给选中assistant内容/工具动作设labels；角色前缀、用户和工具输出为-100。reasoning_content保留原文，默认不计loss；如需监督真实thinking，显式用--thought-loss include。单独thinking事件不能被官方模板正确表达时明确报错，不改写成最终答案。

默认不下载tokenizer；HF ID需要预缓存，显式--allow-download才允许取tokenizer资产，不涉及模型权重。--chat-template可指定部署覆盖模板，其指纹进入数据；实际训练会核对模板一致性。--synthetic仅供演示，输出标记禁止真实训练。

**历史数据的实际长度限制：**本轮登记后6条train完整上下文均超过4096和8192；使用16384后4条能完整编码、2条被跳过，共55,536输入tokens、1,559监督tokens。没有截断工具JSON，也没有伪装成4K可训数据。若使用这批诊断数据，需另存16384长度的训练配置；默认4K配方保留给真正较短的外部/新采样片段。没有宣称31B的16K训练能适配某张显卡。

输出包含tokenizer/template/tools/protocol指纹、来源、分组、监督位置，以及OUTPUT.report.json。零样本会返回失败，不生成貌似可训的空文件。

## 5. 训练、恢复和导出

### Kaggle四卡路径

先在CPU准备数据及同源NF4缓存，再挂载给GPU会话。[CPU缓存入口](../research/005/kaggle-training-probe-20261010/cpu-nf4-cache/cache.py)先用微型Gemma4确认真实量化、保存和重载，内存足够后才处理官方31B；缓存带来源、版本、文件大小和SHA清单。`allow_cached_nf4: true`只接受带官方QAT v2来源manifest的本地NF4缓存，不接受任意预量化模型。

私有CPU缓存已实际完成：601个量化模块，18.30GB，来源config/index指纹与首次官方v2探针一致。GPU四rank并行加载与准备约181秒，整轮四步264.48秒、账号净扣0.15h。全程GPU采样均值仅约8.4%，因为短探针的大部分时间用于加载；正式实验应一次加载后集中完成有用步骤。不要反复重建同一缓存，也不要把诊断adapter当修复SFT产物提交。

    torchrun --standalone --nproc-per-node=4 -m training.train_ddp \
      --config training/configs/sft_qat_nf4_ddp.yaml \
      --model /kaggle/input/notebooks/ccjjllt/p005-cpu-nf4-cache/qat-nf4 \
      --data /path/real-encoded-train.jsonl --out /kaggle/working/repair-r8

配置从1024长度开始，只是容量研究起点；现有历史诊断数据完整前文超过此长度，不能截断后冒充可训。DDP按所有rank、全部累积microbatch的监督token总数归一化；最后一批空rank通过零权重占位参与同步，不把重复样本计入目标。checkpoint记录各rank RNG，恢复必须保持world size、数据和配置；命令追加`--resume /kaggle/working/repair-r8/checkpoint-000100`。指标记录全局输入/监督tokens、每步耗时及每卡峰显存，PEFT adapter由rank0保存。

DDP关闭冻结基座的启动广播，并验证各rank初始adapter指纹；其前提是所有rank加载同一不可变基座。当前不是FSDP，显存仍不足时需另实现兼容的分片训练，不能把uint8量化存储普通转换成BF16来冒充FSDP兼容。

### 单设备路径

先做不加载权重的计划检查：

    python training/train_sft.py --config training/configs/sft_qat_r8.yaml --dry-run

真正训练（必须提供匹配长度的真实编码数据）：

    python training/train_sft.py \
      --config training/configs/sft_qat_r8.yaml \
      --model /models/gemma4-qat-unquantized \
      --tokenizer /models/gemma4-tokenizer \
      --data training/artifacts/encoded-train.jsonl \
      --out training/artifacts/repair-r8

配置起点为rank8/alpha16/LR5e-5/4096序列，q/o+MLP文本层，冻结基座。训练真实执行梯度累计、按监督token归一化的loss、优化器、调度器和checkpoint；不训练视觉塔/embedding/lm_head。NF4在配置中设quant_backend:nf4，并配合CUDA依赖。

    python training/train_sft.py \
      --config training/configs/sft_qat_r8.yaml \
      --model /models/gemma4-qat-unquantized \
      --tokenizer /models/gemma4-tokenizer \
      --data training/artifacts/encoded-train.jsonl \
      --out training/artifacts/repair-r8 \
      --resume training/artifacts/repair-r8/checkpoint-000100

恢复要求同一数据、参数、协议和训练计划；checkpoint包含优化器、调度器、随机数和进度。不要通过改max_steps让已完成计划伪装为中断恢复。输出目录不覆盖已有不同运行。

    python training/export_peft.py \
      --adapter-dir training/artifacts/repair-r8/adapter \
      --destination training/artifacts/submission/adapters/repair_lora \
      --training-manifest training/artifacts/repair-r8/training_manifest.json \
      --out training/artifacts/repair-r8/export_manifest.json

导出真实adapter_config.json及adapter_model.safetensors，校验A/B矩阵、rank、目标层和文件指纹；不合并或替换官方CT基座。缺权重即失败。导出成功不表示已经通过官方vLLM/CT加载或提高修复率。

### 12B Unified 的独立准备路径（2026-10-13）

[12B 配置](configs/sft_12b_qat_nf4.yaml)固定官方 HF QAT revision `b6ed86275a6a5735884e208bfed95b445a684ca2`，来源/权重/CT tokenizer 指纹见 [source manifest](configs/gemma4_12b_sources.json)。入口按实际 model_type 选择 `Gemma4ForConditionalGeneration` 或 `Gemma4UnifiedForConditionalGeneration`；生产 unified 路径必须使用固定12B配方。普通IT、31B NF4缓存及未核实的12B缓存均不能重命名当作该QAT来源。若传本地QAT镜像，加载前核对原config以及24GB权重文件SHA；HF路径要求固定repo/commit。

训练编码显式使用[部署CT v2模板](configs/chat_templates/gemma4_12b_ct_v2.jinja)，17,466 bytes，不能跟随HF QAT的18,683-byte模板。已下载的本地CT tokenizer位于 `training/artifacts/tokenizers/gemma4-12b-ct-v2/`；远程运行须挂载同指纹文件并用 `--tokenizer` 指向该目录。tokenizer来源允许是CT目录，但CT权重仍拒绝用于训练。配置、工具schema和Jinja也应随运行一同保存。

    python training/train_sft.py --config training/configs/sft_12b_qat_nf4.yaml --dry-run

这只是无权重的配置检查。真实运行需要另行安排GPU额度和已取得的固定QAT权重；新配置的16K只是输入上限，未证明训练可容纳16K。配置只安排一个优化步骤，默认不下载。

10月13日已[冻结数据用途](../research/006/data-replay-20261013/data-usage.json)：未来首步只使用 `research/006/data-replay-20261013/outputs/encoded-12b-train-capacity.jsonl` 中的click（3368输入、519监督tokens）。pydicom单独作为select诊断，不更新参数；MarkupSafe两阶段仍为candidate。原合并文件 `encoded-12b-ct-v2.jsonl` 仅作收集归档，不能整体传入训练。本轮未据此自动启动真实12B训练。

统一模型沿用完整文本backbone、shifted selected-token loss与softcap、non-reentrant checkpoint和仅小norm转FP32的NF4准备策略。LoRA限于语言decoder的q/o与MLP；vision/audio/embedder/head均不训练。导出保留实际PEFT的 `model.language_model.layers...` 键，不伪造已在CT服务加载；当前没有验证12B vLLM运行时映射。

固定来源的恢复签名包含已核对的manifest、QAT revision/config/权重身份，并以文件指纹替代模型、tokenizer、模板等本地挂载路径。同一资产换挂载位置可恢复；更换manifest或基座时，在恢复优化器和覆盖运行元数据前拒绝。未使用新manifest的31B配方保持历史签名格式；旧12B无基座身份签名不做静默迁移。

CPU随机微型模型已实际比较原HF与selected-token的loss/LoRA梯度，并完成两步Adam、checkpoint恢复和独立导出；另以meta device读取完整12B架构，未分配或下载12B权重。这些只证明实现路径，**真实12B NF4、长序列训练、CT adapter推理和修复收益均未运行**。[实际记录](../research/006/12b-cpu-adaptation-20261013/results.md)。

## 6. 手动验证与研究状态

本仓库不再维护专用测试套件。按当前改动用一次性Shell/Python命令手动核对输入、输出和必要数值，记录结论即可，不新增测试或check脚本。

同源NF4缓存与31B四卡短序列诊断已完成；官方CT上的诊断adapter加载与工具格式兼容已验证。最新训练态6032对照中CPU offload完成前向，但反向仍CUDA OOM，尚无真实修复SFT adapter或SFT带来的修复收益。single_state四题的探索增益来自无adapter推理对照，不是训练结果。下一步见[研究计划](../docs/plan/005-trainmode-save-on-cpu.md)。

2026-10-11新增[分组注意力候选](attention.py)：默认关闭，训练入口尚未调用；仅替换32Q/4KV、head_dim=512且dropout=0的global SDPA。CPU手动比较支持输出/梯度等价，并观察到K/V共享存储减少；这不能证明CUDA峰值下降或6032训练已可用。[机制结果与下一次无需加载31B的算子研究](../research/005/grouped-attention-20261011/results.md)。

扩量改从独立外部来源开始，优先有任务ID和resolved标签的SWE-smith成功轨迹，先排除比赛全部仓库再查看内容；不把其他scaffold的工具直接改名为官方六工具。[来源许可、协议与CPU筛选说明](../research/005/data-expansion-cpu-20261011/external-source-plan.md)。
