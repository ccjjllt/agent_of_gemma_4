# P007进展：公共12题配对评测与12B真实训练首步完成

> 2026-10-18复核更正：按原始日志修正超时漏计、机制归因和部署边界。下一阶段与新正式提交57340172见[P008进展](008-reviewed-submission-and-real-sft.md)。下文记录P007实际实验，不把后续复核算成新的GPU结果。

> P009进一步更正：V6的12B文本架构override与主办方已声明的scorer设置一致，并非额外自定义部署障碍；此前复核遗漏了P006归档的官方comment3536939。下文尚未证明正式路径的表述应限于完整上下文、完整任务执行和修复收益。

2026-10-17，北京时间。按 [007计划](../plan/007-public022-and-transfer.md) 执行公共 12 题（24 次尝试）单变量配对评测与 12B QAT 真实样本训练准备。遵循最大化 GPU 利用率原则：所有依赖安装校验、任务快照与双臂编译检查、12B QAT 23.92 GB 权重下载与 NF4 量化均先在免费 CPU Kernel 完成（`0s` GPU 额度），随后再在 4×L4 GPU 上执行。未读取 heldout 或预留评测题内容，未新增测试文件。

## 1. 公共 12 题（24 次尝试）4×L4 配对评测完成：`public022_v10` vs `behavior_v1`

- **单一prompt因子对照**：基于冻结的 `public022_v10`（`flexonafft` V10 原样四文件，8 分钟 / 28 次工具调用 / 80 轮 / 240 秒单命令超时，31B CT v2，`temperature=0.2, top_p=0.95, max_output_tokens=8192, thinking_budget=4096`），构造候选 [`behavior_v1`](../../research/007/candidates/behavior_v1/agent/prompts/system.md)（三个配置文件SHA256与基线完全一致，仅修改 `prompts/system.md` 中Locate的一个bullet及Verify的三个bullet，SHA256 `079af841...`；没有独立Read小节）。
- **CPU 预检与 4×L4 滚动双配对并发**：
  - 免费 CPU 预检 `ccjjllt/p007-public12-pair-cpu-prepare` V1 `COMPLETE`（`151.03s`，`0s` GPU 额度），验证 41 个 v31 wheels、12 个 `dev_future` 任务与快照、双臂 `compile_submission` 通过。
  - 4×L4 GPU 评测 `ccjjllt/p007-public12-pair` V2 `COMPLETE`（`run-007-public12-pair-20261017T063527Z-5d0f2c69`）：单次启动共享 31B CT v2 vLLM 服务（`server_start_count = 1`，启动 `333.41s`，KV cache `57,664 tokens`），采用滚动双配对并发队列（`MAX_CONCURRENT = 4`，`--max-num-seqs 4`，峰值生成吞吐 `108.9 tokens/s`），**56.63 分钟**（墙钟 `3397.60s`，批处理 `3291.30s`）跑完全部 24 次尝试（注：V1 因 CLI 默认落到 T4 在第 16 秒被 4×L4 断言快速拦截退出）。
- **配对结果（详见 [完整评测报告](../../research/007/public12-pair-20261017/report.md) 与 [结构化数据](../../research/007/public12-pair-20261017/paired-analysis.json)）**：
  - 官方 `resolved = true`：`public022_v10` 为 **3 / 12 (25.0%)**，`behavior_v1` 为 **4 / 12 (33.3%)**。本次多过1题，只有一对独胜，不证明确定性或稳定泛化增益。
  - **含生产源码修改**且 `resolved = true`：2/12对4/12；但 `fastapi_14851` 的A空补丁也通过，不能把B的526B生产补丁单独算成新增修复收益。全部尝试的真实生产补丁为 **5/12对8/12**，非空patch为7/12对9/12，后者含scratch脚本。
  - 四格表仍为双方通过3题（`rich_3067`, `fastapi_14851`, `fastapi_14463`）、B独胜1题（`rich_3180`）、A独胜0题、均未通过8题。rich_3180中B在415.41秒首次修改生产代码，没有修复后验证或主动submit，超时后恢复patch才获得114 passed；不能称正反合同闭环。A只留下repro.py。`requests_7502`中A改生产文件、B只有repro，说明“零回退”只适用于此次resolved四格表。
  - 预算与耗时：明确agent超时 **6/12对5/12**；超时或明确工具预算错误 **6/12对6/12**。B漏计的rich_3180、fastapi_14851超时后恢复patch成功，故result.error为空。平均agent耗时364.04秒对380.25秒；含裁判耗时381.64秒对422.99秒。平均工具调用22.75对21.25。四并发共享服务的时间不能当正式串行预算预测；批墙钟3291.30秒已含333.41秒启动，不能相加。
- **新发现的两项关键环境/复现机制**：
  1. **死循环复现吞噬半场预算（`rich_4006`）**：因单命令超时设为 `240s`，双臂直接运行 `python3 -c "from rich.cells import split_graphemes; print(split_graphemes('\ufe0f'))"` 复现死循环时单条命令卡满 `240s`（消耗半场 8 分钟预算）；`behavior_v1` 仍改出 `709B` 补丁修复了 `\ufe0f` 分支，但因剩余时间不足未及修复相邻 `\ufe0e` 分支。
  2. **离线 `SubprocessManager` 对 `src/` 布局仓库的导入遮蔽**：requests_7205的B先用自造替身验证，随后实际修改源码；真实API仍呈现旧行为后，第14步查到系统requests路径，第15步显式`PYTHONPATH=/workspace/src`才观察到空凭据与正常凭据的预期区别。独立裁判仍为1 failed/205 passed/13 skipped，保留resolved=False；这支持导入失真诊断，未经修正复评不能写成“100%正确”或计为修复成功。官方Docker与Notebook subprocess安装/导入方式不同，也不能据此保证正式评分一定通过。另有两个FastAPI题缺inline_snapshot而收集失败、requests_6757配置修改被保护过滤；这些题不适合直接用作有效训练反馈。

## 2. 12B QAT CPU NF4 缓存完成与 12B GPU 训练/CT 探针完成（V6 `COMPLETE`）

- **免费 CPU 12B NF4 缓存（`ccjjllt/p007-12b-cpu-nf4-cache` V1 `COMPLETE`，`721.02s`，`0s` GPU 额度）**：
  - 校验固定源 `google/gemma-4-12B-it-qat-q4_0-unquantized` @ `b6ed86275a6a5735884e208bfed95b445a684ca2`（`model.safetensors` 23.92 GB，SHA256 `26f2cee4...`）与官方部署模型 `google/gemma-4/Other/gemma-4-12b-it-qat-w4a16-ct/2`（`chat_template.jinja` SHA256 `36e3a42e...`），生成 7.71 GB `/kaggle/working/qat-12b-nf4`（331 个 `Params4bit` 张量、4 个 `<2GB` 分片及完整 manifest）与 6 个 Python 3.12 训练 wheels（清单归档于 [`cpu-cache-output/`](../../research/007/12b-qat-cache-and-train-20261017/cpu-cache-output/cache-manifest.json)）。
- **训练器升级与免费 CPU 预检（`ccjjllt/p007-12b-train-and-ct-probe-cpu-prepare` V2 `COMPLETE`，`92.69s`，`0s` GPU 额度）**：
  - 更新 `training/train_sft.py` 与 `training/configs/sft_12b_qat_nf4.yaml` 支持12B NF4缓存来源清单与大小校验；该路径没有逐个重算全部缓存权重SHA，不能把原`local_weights_hash_checked=true`当成这一证据。在CPU预检中修复v31 `bitsandbytes-0.46.1` 与训练`0.50.2`同名冲突，43个wheels安装、dry-run与协议检查通过。报告见 [`cpu-prepare-output-v2/`](../../research/007/12b-qat-cache-and-train-20261017/cpu-prepare-output-v2/p007-12b-probe/summary-report.json)，元数据与恢复身份的后续修复见[P008训练复核](../../research/008/review-20261018/training-review.md)。
- **12B 4×L4 GPU 训练与官方 CT v2 对照探针（`ccjjllt/p007-12b-train-and-ct-probe` V6 `COMPLETE`，全程约 8 分钟墙钟 `493.7s`）**：
  - Phase 1（NF4 QLoRA 单步）：单步训练loop（含保存）`14.827s`，训练阶段PyTorch allocated峰值 `9.7362 GiB`，240/240 个 `lora_B` 非零；`click` loss `0.530146 → 0.515842`（Δ `-0.01430`），未参训 `pydicom` loss `1.654296 → 1.653286`（Δ `-0.00101`），三次运行稳定复现。
  - Phase 2（官方 12B CT v2，`--hf-overrides Gemma4ForCausalLM`，`max_model_len=8192` + `--max-num-batched-tokens 512` 解决长 prompt 全词表 `log_softmax` OOM）：启动 `195.2s`，3 模型注册，KV `182,848` token。Teacher-forced 覆盖率 100%（519 + 226 token，`missing=0`）；`zero_r8_ctrl` 与 base 差异精确 `0.0`（对照有效）。真实 adapter：`click` NLL `0.507977 → 0.503882`（Δ `-0.004095`，189 改善 / 158 退化 / 172 不变，最大单 token 变动 `1.248`），`pydicom` NLL `1.686483 → 1.684741`（Δ `-0.001742`，96 / 96 / 34）。Live 生成两案 adapter 均与 base 分叉，但尚未转化为成功修复（`edit_ok` 全 `False`）。
  - 结论边界：真实训练只有一个click样本、一个Adam步；V1–V6均从头启动，不是累计六步。数值作用存在，但未形成修复增益。V6使用官方CT权重加自定义文本后端override；普通声明式提交不会自动继承此参数，部署兼容尚待验证。Live探针只对预选内存字符串应用替换，未执行完整仓库工具或行为裁判；不能据此称whole-issue修复。V2–V5的多模态LoRA断言、OOM和字段等问题在V6诊断路径中绕过/修正，不代表正式路径均已解决。
- **外部 8 仓库候选源核验与 CPU 环境就绪检查**：
  - 已持久归档6个外部repo的完整buggy SHA及tarball；pandas/modin的完整SHA证据未在当前清单与执行汇总中找到，不能称全部8个均完成归档核实。已有6个tar的实际SHA与manifest及V5记录一致；脚本的`sha_ok=True`却未比较预期hash，gpxpy嵌入digest也少两位，原报告的自动SHA核验表述不成立。本轮只手动核实原件，没有把它扩成新的核验框架。
  - V4 CPU 44.82秒、V5 CPU 45.34秒均已完成。V5有5个目标源码import成功；gpxpy/schema/iniconfig均零用例，cookiecutter收集209项/20错误，conan301项/394错误，feedparser缺依赖；收集数不是通过数。当前0个任务完成空补丁/参考修复区分验证、0次模型配对。V6源码中的后续探查不是已执行结果。外部8repo继续整仓排除训练，先形成真实任务合同，不把import数量当模型研究进展。详见[数据复核](../../research/008/review-20261018/data-review.md)。
