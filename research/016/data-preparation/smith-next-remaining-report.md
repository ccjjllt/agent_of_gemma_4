# P016：缓存剩余身份的短完整轨迹扩量

2026-10-23。只使用已有固定版本 Smith 缓存，实际新增 **313 个可训完整 ≤16K 独立问题、58 仓、769,302 监督 token**。这些是数据产出，尚未训练，也没有新的修复率或比赛分数证据。

| 实际新增产物 | 独立题数 | 仓库数 | PR-ID / 命名变异 | 输入 token | 监督 token |
|---|---:|---:|---:|---:|---:|
| ≤16K 完整轨迹 | 313 | 58 | 41 / 272 | 4,201,196 | 769,302 |
| 16–32K 完整轨迹，保留待选 | 967 | 60 | 102 / 865 | 23,012,602 | 4,101,404 |

与旧 224 个短题合计，现在有 **537 个独立短题、68 仓、7,119,739 输入 token、1,291,819 监督 token**。三份原始短分片保留，并已无改写合并为独立 `research/016/data/main-sft-short-v1/`，旧 677 题 `main-sft-v2` 保持不动，没有把长轨迹截成短题。新增短分片为 `research/016/data/encoded-native/next-remaining-admitted-v2/swe-smith-native-le16k-thought-include.jsonl`，67,291,158 bytes，SHA256 `5b83947a5f467f198eea8c3de44352ccdd70aa1b18bc05e96db419397736e919`。

独立短主语料包含 `train.jsonl`、`manifest.json`、`native-tools.json`、`proposed-config.json`，已打包为 `research/016/data/gpu-bundles/smith-main-short-corpus-v1.tar.gz`；精确大小及 SHA 见[小回执](../training/main-short-corpus-package.json)，完整仓库/来源类型统计见[短主语料汇总](../encoding/corpus-summary-short-v1.json)。在既有 v2 配置上使用 16K、每卡 micro batch 1、4 ranks、accumulation 2、1 epoch、`max_steps=0`、checkpoint 每 10 步；按真实 trainer 公式是 **135 个 global microbatches、68 个 optimizer steps**。这是待预算确定的第一阶段建议，尚未启动；模型、tokenizer 路径与运行代码仍需绑定服务器环境。

## 真实生产与排除

先对已有 24,100 行身份元数据去重，排除旧两批共 1,160 个跨来源 canonical 身份、23 个预留家族、非 train 身份及来源类型未核清的纯数字 ID，然后冻结 **2,039 个候选、63 仓（316 PR-ID、1,723 命名变异）**。未先看长度换题，也没有为每个问题挑选最短的教师尝试。单一轨迹以固定 hash seed 在正文读取前确定。

- [身份冻结](smith-next-remaining-identity-freeze.json) → [精确抽取回执](smith-next-remaining-acquisition.json) → [v2 条件准入](../encoding/next-remaining-native-admission-v2.json) → [最终统计与排除明细](../encoding/next-remaining-summary-v2.json)。
- 抽取只按物理行、instance ID、trajectory ID 三重关联读取冻结 train 的 messages/model/resolved，重新核对 8 份缓存 SHA；232,384,335 bytes 正文实际抽取耗时 **2.55 秒**。
- 复用未经修改的 v2 producer，编码耗时 **378.55 秒**。2,016 条成功转换并精确计长；23 条格式失败（22 条 XML 函数块不完整/歧义，1 条未知编辑器命令）保留原件；722 条超过 32K 保留原件。
- 53 条转换记录触发既有 quality flags，其中 14 条本可装入 32K，全部整条不准入。其余已超长。没有为新样本添加 history、远程材料或撤销动作的例外。全部 313 条短轨迹均无这些既有显式 flags。
- 生产器 SHA、tokenizer、chat template、tools schema 与旧 v2 完全相同。一次性手动流式核对了输出真实 token/label 数、SHA、train 标记、全新 canonical 身份及短长两分片零重复；未新增测试文件或检查体系。
- 第一次命令传相对输出路径，已有 materializer 的 `relative_to(ROOT)` 拒绝；改绝对路径后复用相同已写原件完成。没有更改 producer 以扩大任务范围。

没有下载、GPU、上游命令重放、final patch 列读取或 heldout 正文读取。固定源 revision 仍为 `08e109b4a59eaeebf80e4675cd125d42e7ac99a4`。292 个候选复用到完整 buggy commit 与仓库许可元数据；其余保留 null，离线 SFT 仍依固定数据版本、问题/轨迹身份及 raw SHA 追溯。

## 对单位时间训练收益的含义

旧短题平均 13,029 输入 token、15.84 个动作；旧长题平均 22,965 token、23.89 个动作。若输入吞吐相同，短题每小时可覆盖的独立问题约多 **1.76 倍**。旧短/长监督密度分别为 17.903% / 17.893%，几乎相同，因此优势首先是独立问题覆盖，不能称监督 token 效率或修复率已提高。

按根任务给出的初步 300–333 输入 token/s，新增 313 个短题的一遍纯训练量约 **3.50–3.89 小时**，全部 537 个短题约 **5.94–6.59 小时**；这只是算术换算，不含加载、checkpoint 和评测，也不是授权后的运行预算。相比之下，旧 453 个长题即使保持同吞吐，纯输入处理仍需 8.68–9.63 小时。实际执行以最终 GPU 校准及当前预算为准。

应先把新增短题作为可选数据供给，在有限训练预算中提高独立问题覆盖，再保留一部分长题覆盖长观察、多步修复与恢复行为。不能把全部长题价值归零：短序列选择会偏向较容易、观察较少的问题，且数据仍是原生三工具，六工具迁移尚待实测。

新短题 **272/313 为命名变异**。剩余 PR 候选原本只来自 dask、moto、pydantic、dvc、astroid、pygments 六仓，扩量不能保持旧 50:50 采样而不大量重复。合计 537 个短题为 159 PR-ID、375 命名变异、旧批保留的 3 个纯数字身份；训练混合比例需按真实收益决定，不能让缓存余量决定最优比例。独立 issue key 去重也不等于语义近重复消除。

## 可重复命令

身份冻结已经保存，不应覆盖或重新挑题。以下命令只复用本次已固定输入：

```sh
python3 research/016/data-preparation/materialize_smith_batch.py \
  --freeze /Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/research/016/data-preparation/smith-next-remaining-identity-freeze.json \
  --output /Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/research/016/data/source-rows-next-remaining \
  --manifest /Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/research/016/data-preparation/smith-next-remaining-acquisition.json

PYTHONPATH=/private/tmp/p012libs311 /opt/homebrew/bin/python3.11 research/016/encoding/build_smith_native.py \
  --selection /Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/research/016/data-preparation/smith-next-remaining-identity-freeze.json \
  --acquisition /Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/research/016/data-preparation/smith-next-remaining-acquisition.json \
  --admission /Users/zhaohuaxiang/ccjjllt/agent_of_gemma_4/research/016/encoding/next-remaining-native-admission-v2.json \
  --admitted-only --thought-loss include \
  --out /private/tmp/p016-next-remaining-reproduction
```

复刻使用新输出目录；原始分片、源快照和 manifest 保留不动。完整生成日志位于 `/private/tmp/p016-next-remaining-native-v2.log`；最终 manifest 与汇总记录包含真实结果，不依赖临时日志才能读取结论。
