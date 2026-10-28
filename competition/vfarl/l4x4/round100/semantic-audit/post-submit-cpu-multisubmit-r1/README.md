# Live 提交后继续工具与再次提交

2026-10-28。pinned 官方允许第一次 submit 后继续工具操作并再次 submit；预算与 final-response 规则仍适用。当前窄修只恢复 live-original-pair 的机会，historical-three-branch 仍保留旧 late-tool guard。strict native parser、提交后不 nudge、旧 R3 奖励都不改；不宣称整套 official 等价。

修改前 `9fd1…` 源码和11条 R3 guard 命中已先冻结在 [multisubmit-source-review.json](../post-submit-terminal-scope-r1/multisubmit-source-review.json)，SHA `a1b9475fdafa6384f89b0b7a8840505263cd9621110f043a248e012e7bfdc3b1`。11条实际请求都是成功 submit 后的合法 run_command 调用形状，包括状态、diff 和清理请求；未实际执行，且原奖励全为1，不能推断恢复它们会提高修复率。

新 `run.py` SHA `3cc567d927b2905c04fa9ce613e05ddf46a19bbeed05472a65b1f8b8a95e88a7`：late-tool guard 增加 `and not live_pair`；live episode、queue execution contract 与 collection-protocol 元数据声明继续 tools 或 final、原预算以及最后一次成功显式提交。此前 sampling 异常仍 fatal、但保全已提交 patch/原验收诊断的机制未改；它的实际证明仍属于旧 `9fd1…`，没有重标成此次新源实跑。

此次两条实际 CPU 轨迹均使用固定 train index26 Tenacity wait_fixed、原74文件字节／模式、原 snapshot、当前 prompt 与 int60预算及原六个 target文件／123 required节点（13 F2P＋110 P2P，展开124 cases）。真正的官方 tools/context/manager 执行所有工具；CT tokenizer 真实编码，facade 是明确的 Codex 脚本干预，captureFalse、referenceNone、mask0、无模型或训练。

| 轨迹 | 实际行为与公开 API | 原完整验收 |
| --- | --- | --- |
| canonical-resubmit | 错误 edit 后首次非空 submit；提交后的 run_command 实际得到 fixed_delay0.5 并断言失败；继续修复得到2.5，再次 submit，最后文本结束 | baseline124/13fail；首次错误提交124/13fail（0）；第二次提交124全通过（1） |
| raw-no-resubmit | 正确 edit/API2.5 后 submit；随后把 workingtree 改为错误实现，真实 run_command 得0.5并失败；不再 submit，最后文本结束 | baseline124/13fail；最终仍验收第一次正确提交，124全通过（1） |

canonical 共8个工具调用、9个实际请求，两个 submit 均被执行；8个 canonical call ID 唯一。raw 共7个工具调用、8个请求。各自 tool/LLM 计数单调延续，预算值及类型不重置，均无提交后 nudge。raw 最终 workingtree patch 与 canonical 第一次已验收0的错误提交逐字相同，但它的 evaluated patch 仍逐字等于第一次正确提交。这直接区分了“最后已提交 patch”和“当前 workingtree”。

允许工具后，后来发生的真实 post-submit 工具错误可按既有反馈／预算规则建立合法 focal state；下一响应即使是文本，也不能仅因 patch_submittedTrue 被主观排除。现有 Snapshot 已绑定 submitted_patch、patch_submitted、workingtree 和 ledger，原 reference／restore／partner 核对未要求必须 pre-submit。本次 captureFalse，所以没有创建实际 focal pair。直接 submit成功→final 且没有新失败时，仍不会产生新的失败驱动 focal state。

后续 edit 本身不更新已保存的提交；只有再次成功 submit 才替换它。workingtree fallback 仍只用于从未明确提交的 episode。新的异常诊断保全同样评估最后一次明确提交，不把未提交改动冒充提交。

主要结果：[result.json](result.json)，SHA `fe41d9e47e01089bca2e363d471afcbaccbbbec5f693e683b0da7b5e3345a3aa`；[artifact-bindings.json](artifact-bindings.json) SHA `62e0015c4581980a9931e3dfde087bfc7f7b4e907f628f688070ce70cf0ffc3a`；窄源差异在 [live-only-production-change.patch](live-only-production-change.patch)。原始工具日志、错误 stdout、提交 patch、完整 XML、loaded-source 和一次性输入原文均保留。

没有 GPU、模型推理、下载、SFT 导出或正式训练。旧正常／异常 CPU 证明、固定100主数据、99 terminal、原 oracle 和历史奖励未改；这是合法动作机会与提交语义的机制证明，不是模型收益。
