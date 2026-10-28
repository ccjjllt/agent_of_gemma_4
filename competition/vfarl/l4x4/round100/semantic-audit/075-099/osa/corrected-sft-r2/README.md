# OSA修正示范：独立Gemma SFT增量

后续已取得[匹配当前固定100预算及prompt的r3重放](../corrected-sft-r3/README.md)。本r2的60分钟prompt不同于当前固定30.0分钟版本，作为独立profile的真实正确示范保留；当前prompt消费请用r3，不能只按task ID混合。

2026-10-28。仅包含固定100内的一个train任务：`luozhouyang__python-string-similarity.115acaac.func_pm_class_rm_funcs__96brcjir`，S／S4。原S600、F600及其manifest／训练数据没有重写。此前空字符串距离错误的旧示范仍保留，不能与本增量直接混合作为同一正确标签。

训练输入为[gemma-sft/sft-actions.jsonl](gemma-sft/sft-actions.jsonl)：**1题、8条助手决策（7次真实工具调用＋1条实际终止回答）、954个助手监督tokens**，最大完整动作上下文3902 tokens。动作文件SHA256为`0edabed903873a8a5b9af84a7ef92edd930d4e7d0b4784f8fbf048417986fd9b`。完整对白及loss mask留存在`sft-episodes.jsonl`，导出统计见[manifest](gemma-sft/manifest.json)。

这是Codex指定、审阅并通过真实官方六工具执行的off-policy修正示范。thinking budget为0；强制空thought控制属于输入前缀且不参与loss。没有Gemma rollout、生成思考、行为logprob、同状态偏好对或训练更新。

## 本次真实来源

所有新recipe、messages、tool-events和验收均绑定相邻`corrected-six-tools-r2/`，没有复用旧token轨迹或复制旧成功结果。原source task、revision、license及机制review继续使用固定100里的真实pin；仅将该一条catalog记录保存为[source-catalog.json](source-catalog.json)。新的原始report只记录本次真实指针及复核说明，规范化结果见[canonical report](canonical/osa-corrected-r2-replay-result.json)。

未修改原15个验收文件和18个required节点。新baseline官方结果为0（原F2P真实失败），修正后官方结果为1；实际修后run_command打印并断言`[5.0, 5.0, 1]`，对应两侧空字符串距离及相邻交换。7个recipe动作、工具调用ID、实际事件与tool messages已由原normalize_results函数逐项绑定，严格准入无缺项、无例外。

原OSA F2P只打印距离这一已知弱点仍明确保留。18节点通过不能单独证明完整距离语义，也不使该任务自动成为可靠RL奖励；本增量之所以作为修正SFT示范，另有真实生产修改及明确数值观察和人工审阅。没有把错误旧示范或故意constant-zero反例写入本导出。

## 编码与复核

复用`normalize_results.normalize`、`export_prompts.py`和`format_replays.py`，没有修改全局脚本或放宽准入。只显式传入这一个新canonical结果和本次六工具声明。新初始prompt由官方builder生成，逐字等于本次实际`agent-prompt.txt`，未加入参考补丁、验收节点或事后机制标签。

原015 tokenizer目录已清理，实际复用`research/006/data-replay-20261013/tokenizer`中的已验证31B CT缓存；其tokenizer及模板哈希与历史已验manifest完全相同：

- tokenizer：`cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`
- template：`94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413`

规范化／prompt阶段复用Python 3.12官方依赖cache，编码阶段复用`/private/tmp/vfarl-cpu-packages`中的transformers 5.13.1、tokenizers 0.22.2和jinja2 3.1.6。实际版本保存在manifest，未冒称旧4.57.1运行时。没有下载新包、模型推理、GPU或远端作业。

一次性手动核对确认：所有8条记录只有该train ID，输入token与mask长度及监督数一致，实际新messages哈希一致，thinking0／无logprob字段一致，工具观察未改写。原始来源与规范化文件哈希见[normalization-manifest.json](normalization-manifest.json)。本地macOS CPU证据仍不证明Kaggle chroot或比赛Docker等价。
