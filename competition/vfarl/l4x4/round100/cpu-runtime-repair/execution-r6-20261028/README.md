# 当前四题 CPU R6：真实环境恢复结果

2026-10-28。私有作业 `jiyujia/vfarl-current-four-cpu-runtime-repair-r6/1` 于02:55:42 UTC提交，03:01:13 UTC读到COMPLETE。实际源SHA `16daf7dbf89cccefacb4c514618b0a548f869c8a52aa69a5381fb6cd4c3b04d2`，GPU关闭、没有模型调用和optimizer更新，平台API请求3600秒时限但回读未暴露实际值。

三组stock依赖按原固定版本在准备期真实安装，随后 `UV_OFFLINE=1` 执行原完整验收。相同四题的CPU5两端因缺依赖不能评分；本版四题均取得有效0／1，原历史null不改。

| 题目 | baseline完整用例 | 原已审修复完整用例 | 有效奖励 |
|---|---:|---:|---:|
| Moto5348（9） | 1失败、6通过 | 7通过 | 0 → 1 |
| Moto6159（10） | 1失败、68通过 | 69通过 | 0 → 1 |
| DVC7628（23） | 31失败、3通过 | 34通过 | 0 → 1 |
| Markdownify161（54） | 1失败、74通过 | 75通过 | 0 → 1 |

[聚合原件](outputs/vfarl-prevalidate/prevalidation.jsonl)SHA `fe52e2443f2879bef38f3d9117114ef4a023607e42d888f193234a3f8f92bdfe`，四行usable=true，八端verification_complete=true，LLM calls均0。八份实际JUnit、原始official verification和patch保留，见[逐题来源及XML统计](actual-results-summary.json)。实际task／snapshot／patch哈希和patch字节逐一匹配已冻结binding；54使用当前12cfb3ac新版题面，四个公开prompt实际渲染均匹配，无模型输入执行。

本次是原Linux chroot及legacy-default字节码策略的环境验收，不是新的模型修复率，也不证明isolated-prefix-no-write-v1。当前维护run.py拒绝明确未就绪环境，三组实际绑定分别为stock-moto／stock-dvc／stock-markdownify。

**可复用边界：** EXPORT_RUNTIME=False；验收后代码删除本次临时stage和material，不导出新运行时。原0a8输入仍缺这些stock profile，不得借本记录自动提升为usable。未来GPU必须实际准备或加载与此证据匹配的环境，再决定解除四题暂停。

[下载清单](outputs/download-manifest.json)保留51个小型研究输出，共859,615字节；未下载模型或大运行时。提交、源/隐私/硬件读回、真实session363022663及最终状态均在本目录，原失败CPU4/CPU5载体和记录保持。
