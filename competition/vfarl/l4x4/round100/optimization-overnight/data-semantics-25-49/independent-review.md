# 25–49 澄清文本独立复核

2026-10-28。审阅13篇 proposed 文本及其映射，逐一对照归档生产代码、实际 submitted.patch 和原任务 test_patch 中的 oracle。没有改固定100身份、v4原件、旧奖励、oracle、训练入口或运行器。

确认并最小修复一处真实API错误：索引49的 `from glom import glom, Group, Min` 在该归档的 baseline/reference 均报 ImportError；包根没有导出 Group/Min。现在仅改为 `from glom import glom` 与 `from glom.grouping import Group, Min`，输入仍是 `[1, 0, 1]`，Expected仍为0。原 proposed statement SHA为 `3c9e2845dc86b2622d4fd65c8361155b3eaca88af86e3ed1f9e0c75d81c09ba2`，修订后为 `b46bf0f8fa364498acfedef45b768a48700ecdb4c395c6c844027f0793dea38f`，映射已同步，失败观察保留。

其他12篇未发现需进一步修改的文本。原样代码在原snapshot及其实际参考补丁下有界执行，均符合澄清所述差异。25/26/31只计算wait值，不执行重试sleep；29显式限制两次尝试并在两种代码状态下实际终止。32的 PostgreSQL ARRAY 和 SQL Tuple 比较/IN，39的 MarkdownRenderer 精确返回文本，41的 storage 类参数，42的直接 copy，43的两次 body 访问，以及44的 +01 输出都有真实观察支持。46保留 ValueError 要求，删除错误的精确字符串/default值假设。

没有在文本中发现参考补丁代码、私有验收节点或事后机制标签。25随机组合、32方言、43表单/JSON/multipart/大body范围均有原oracle来源支持；31“后续attempt也尊重min”延续原问题的最低等待要求，但本题原F2P的非零min断言只直接覆盖首轮，不能据此宣称后续attempt语义已被完整验收。49只澄清单个最小值后遇到较大值，未加入未证明的相等值对象身份要求。

证据：`independent-review-observations.json` 保留每篇新题面的SHA、实际生产归档/补丁SHA、原样代码和有界CPU输出（49同时保留修复前后）；`independent-review-source-bindings.json` 确认旧task/新statement映射及生产/oracle文件哈希来源。oracle只在内存中从原test_patch恢复并比对，未生成测试文件，也未运行新的全仓suite。这些行为观察不构成新官方验收、模型收益或新prompt示范重放。

运行仅使用本机Python3.12.13、既有官方CPU依赖缓存，以及已有jiyujia缓存归档中45个face/boltons纯Python源文件的临时副本；没有下载、环境覆盖、GPU或外发。
