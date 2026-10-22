# P015 train 数据瓶颈交接（2026-10-22）

仅使用首轮3个train attempt与新8题的16个train draw，共19次、11个独立问题；未读dev、未读取新来源，未修改冻结的`on-policy-rft-first-v1/`。统计依据是各attempt的result/context_metrics及已生成的完整因果窗口census。

## 真实yield

- `resolved=True`：9/19次，覆盖6个独立问题。
- 严格成功（resolved、judge_valid、无agent_error）：8/19次，仍覆盖6题。oauthlib r2虽然resolved且仅23工具调用，但有session timeout，必须拒绝；选无error的r1（26工具调用）。
- 固定每题最少工具严格成功draw后，满足至少一个完整、真实成功生产edit窗口≤8192的只有3/6题（相对全部11题为3/11）。新8题贡献2/5个严格成功独立问题，另加旧boltons298。
- 最终14条action、1014个监督token、最长8165；10次src read、3次生产edit、1次submit。它是action RFT，不是完整任务SFT，未达到16–32独立问题目标。零条真实上游验证被首轮规则选中，不能把14条当14个独立修复或14次测试收益。

## 五个新成功问题的已选draw

“长度”均为完整实际请求历史＋当前完整assistant工具动作＋真实工具响应，官方31B tokenizer计数。call编号从0开始；辅助摘要请求也占call编号，不能与工具预算计数混用。下述“成功edit”指工具确实修改生产源码、所在draw最终通过host判据；不声称每个中间修改独立解决全题。

| 问题 / 固定draw | 最早成功生产edit | 已观察的历史压缩/边界 | 8K结果 |
|---|---|---|---|
| graphene1478 r1（21工具） | call10：11029；后续edit12339、12938 | census未见历史前缀压缩断点；末尾非action不纳 | 整题不纳，不能只学读取 |
| dask9212 r1（19工具） | call23真实sed源码修改：13307；call25 inline Python源码修改13859。此前edit_file均缺参数失败 | call28→29发生历史改写，call30完整窗口3084；压缩后仅查询/清理/submit，无新的成功生产edit | 整题不纳；3248-token submit不能代替生产edit |
| tldextract258 r2（24工具） | call5：7451（真实diff与最终patch同一修复） | census未见压缩前缀断点；call26终段assistant/history不一致单独排除 | 纳4条，含7944-token真实submit |
| marshmallow2715 r2（28工具） | 首次call10：12332；压缩后成功call18：7194 | call14→15发生历史改写；首个后续完整窗口call16为5401，但该edit失败，不能选；call18成功才纳 | 纳4条；官方压缩实际补回这1题 |
| oauthlib771 r1（26工具） | call21：8351 | call16→17历史改写，call18窗口3582；继续读取后edit达8351，比8192多159 | 整题不纳；不换成超时r2 |

压缩位置依据现成census中“next history rewrote/compacted”的明确断点，表示真实可见状态变化；不把tldextract末尾不一致猜成compaction。断点自身窗口已排除，之后的新请求历史原样使用，未补回旧历史。

## 下一笔投入的优先级

1. **先复用现成官方compaction状态，零新增采样GPU成本。** 这一工作已完成：marshmallow从首次edit12332降到真实后续成功edit7194，获得1题；dask压缩发生在源码修改之后，不能回溯制造短编辑样本；oauthlib即使压缩仍差159token。不能把“有compaction”推成“编辑都会落8K”。
2. **额外预算优先验证12K容量，再考虑扩样。** 若`12K=12288`可实际训练，现有已证明完整的graphene11029和oauthlib8351可恢复生产edit资格：同一6个严格成功独立问题，从3题变为潜在5题，无需新增生成。此为已有长度证据推断，未运行12K编码配方或显存/训练实验；dask13307仍不够，marshmallow首次12332也仍略超12K（但已有7194可用）。不能预报12K显存一定适配或修复收益。
3. **之后再扩独立train问题/采样。** 当前19次只得到3个8K可训独立问题，超时与动作前探索变长同时损耗yield。继续相同配置双draw并不保证解决长度问题；不能为提高yield换选更短draw、删除上下文或把失败repro动作当正向target。额外问题/采样应继续遵守freeze与来源隔离，并以独立问题数衡量增量。

当前14条已冻结，本文不改变其任何字节，也不启动GPU或新机制。完整逐action理由/SHA保存在数据目录的dataset-manifest；可追踪摘要为`on-policy-rft-first-v1-manifest.json`。
