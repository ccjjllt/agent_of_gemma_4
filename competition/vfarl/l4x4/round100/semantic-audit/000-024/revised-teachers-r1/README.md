# 八题r2新题面：真实teacher与SFT增量

2026-10-28，固定100题原索引0、1、2、3、5、6、21、22。全部按 question-revision-r2 当前task与prompt重新实际执行官方六工具；未复制旧messages冒充新示范。每题原recipe动作及生产submitted.patch字节保持相同，但获得了新的真实tool events、原始终止回答和baseline／repaired验收记录。旧teacher、原奖励、selection、task/oracle均未改。

[gemma-sft/sft-actions.jsonl](gemma-sft/sft-actions.jsonl) 可作为当前版本增量：8个episode，64条决策（56个真实工具动作及8条真实终止回答），7463个监督tokens。最大动作上下文4113，完整保留，没有裁成4096。文件SHA `75aa06822bcf3c658018168d50b109b19a115f44ddba4c7c4e763ab0d62d97fb`。CT tokenizer SHA `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`，template SHA `94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413`；实际CPU导出包版本见manifest。

[canonical-index.json](canonical-index.json) 八题均通过原normalizer严格准入，无ineligibility reasons。每个原F2P在实际baseline中失败，repaired完整原节点通过；原F2P/P2P identity、文件范围和test_patch都未放宽。下表是各题真实原验收文件范围的节点数，不是独立新研究任务数或模型收益：

| 原索引 | 原验收节点 | baseline失败 | repaired通过 | 监督tokens | 最大完整上下文 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 65 | 1 | 65 | 547 | 2785 |
| 1 | 464 | 1 | 464 | 997 | 3654 |
| 2 | 464 | 23 | 464 | 1007 | 3613 |
| 3 | 464 | 18 | 464 | 1126 | 4026 |
| 5 | 365 | 5 | 365 | 914 | 3496 |
| 6 | 365 | 1 | 365 | 870 | 3420 |
| 21 | 280 | 1 | 280 | 1033 | 4031 |
| 22 | 16 | 2 | 16 | 969 | 4113 |

[reconstruction-index.json](reconstruction-index.json) 与每题 source-reconstruction.json 记录逐字恢复原source-inventory全部文件内容、符号链接及mode，实际run source-inventory JSON与旧原件完全相同。原budget/harness的值及数值类型逐项保留，尤其Time allowance使用原整数60；调用未改的run_official.replay API，避免CLI强制float后变成60.0而改变prompt。当前task SHA、实际初始messages prompt SHA、独立official builder export SHA均匹配r2 selection对应项。每题在current-version-teacher-sft-index.json独立列出这些SHA及canonical路径，不能把原source_task SHA误当新适配task SHA。

Dominate首次preflight发现snapshot的conftest带官方post-setup hermetic头部，与原inventory不同；未启动teacher。随后从旧buggy缓存找回逐字匹配原SHA `b8f691968ba428867d7d143daab854afff289fbda78d7ec7d0f08c72d581673d` 的文件并恢复原mode，最终24文件原树完全一致。其余七题直接由snapshot恢复所有原inventory文件。

官方host固定为既有deepdiff Python3.12 cache，任务解释器为本机Python3.12.13，使用pth绑定。Flake8额外依赖从已有jiyujia归档profile-01定向取出55个纯Python／metadata成员，版本逐项匹配旧运行：pycodestyle2.12.1、pyflakes3.2.0、mccabe0.7.0及flake8的7.1.1 dist-info。临时只读symlink overlay复用官方host其余包，没有修改正在使用的profile、下载或重复复制整个703MB依赖树。原归档路径、成员SHA和profile结构见flake8-dependency-profile.json。

[source-catalog.json](source-catalog.json) 从pilot96及supplemental的同ID唯一原entry deepcopy，仅在本目录独立补全curator问题文本来源、原文／当前statement SHA、review与真实新run指向。原全局catalog未改，raw报告与独立catalog披露完全一致。第一次SFT包装调用了不存在的format_replays.main，在创建输出前停止；随后执行其未改的__main__入口成功，原因留在机器消费表，没有将失败导出当作SFT。

这是Codex指定并真实执行的off-policy材料示范，thinking0、强制空thought控制保留为prefix且不监督；没有生成模型思考、Gemma rollout、behavior logprobs、GPU或optimizer更新。macOS CPU验收和canonical SFT准入不代表Kaggle/Linux等价或新版模型独立修复收益。新的示范只适用于这8个当前版本，不追认其余revision已有新示范。
