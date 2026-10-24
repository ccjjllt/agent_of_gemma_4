# P017 新确认面板：身份准备

2026-10-24。已从本地身份投影冻结 **8 个候选仓库、24 个主选问题**，附每仓固定替补顺序；随后按执行授权完成 24 题任务/判据来源获取和 24 份精确提交快照。尚未运行模型或环境复现；不能把它称为已验证可运行的 24 题面板。

冻结文件：[confirmation-identity-freeze.json](confirmation-identity-freeze.json)。采用已有 `research/016/data/metadata-lego-real.json`：`Lego-X/SWE-Lego-Real-Data`，revision `879c96b91dfa321b4d1c621d149a9200f8a1d118`，`resolved` split，原始文件 SHA256 `d9719b0f4cde7a57233cced8de37408b5578e8c8826771a8c5b9303de27f139b`。本地 5009 行投影仅有身份、仓库、base commit、上游 resolved 标签、license、image_name；没有题目或轨迹正文。

| 候选仓库 | 原始独立问题 | 不同 base commit | 主选 PR 编号，按冻结顺序 |
|---|---:|---:|---|
| networkx/networkx | 16 | 16 | 5444、7024、6908 |
| hgrecco/pint | 20 | 20 | 2089、1248、1039 |
| omni-us/jsonargparse | 26 | 26 | 560、424、242 |
| rdflib/rdflib | 13 | 12 | 1022、2084、1456 |
| facelessuser/soupsieve | 10 | 10 | 50、119、263 |
| hynek/structlog | 9 | 9 | 586、117、198 |
| fatal1ty/mashumaro | 9 | 8 | 128、185、159 |
| reata/sqllineage | 20 | 20 | 45、38、19 |

仓库按不同 namespace、有至少 9 条本地 PR 身份且具有完整 commit、许可及上游镜像标识选定，属于有目的的准备样本。各仓内按 `SHA256("p017-confirmation-metadata-v1-20261024:" + instance_id)` 排序；同仓同 base commit 只保留排序第一条，共排除 2 条。剩余 121 个身份中前 3 条/仓为主选，97 条为预先排序的替补。运行次序按 8 仓轮转，先各仓第 1 条、再第 2 条、第 3 条。未根据题目、补丁、长度或本次模型结果选题。

本地来源本身是教师已解决的 `resolved` split，这会带来难度与教师能力筛选偏差；没有额外按候选模型结果筛选，但仍不能把该面板当作所有真实 issue 或隐藏比赛分布的无偏样本。

重合核对采用现有身份记录：P016 主训练三个组成 manifest 的 SHA 均吻合，按 `native_sft_admitted=true` 且 `bucket=le16k` 重建了 537 题、68 仓；另核对 P016 70 仓训练白名单、历史 71 仓（包括额外的 `pudo/dataset`）、旧 SFT split 元数据、六工具种子的 6 个来源、P015 train/dev、P011 train 使用记录、P012/P014 旧外部评估以及官方 split 元数据。8 仓与这些集合均无仓库重合，因此不存在相同仓库/issue 或相同仓库/base commit 对。来源文件及 SHA 保存在冻结文件，不修改旧计划与 split。

**家族核对：** 冻结后读取了 8 仓 GitHub 身份元数据，均报告 `fork=false`，新旧名称均与已有训练/保留集合不重合。`omni-us/jsonargparse` 已迁到 `mauvilsa/jsonargparse`，GitHub repository ID 为 `184562685`；来源身份保留旧名，快照从新名定位同一提交。证据在 [repository-identities.json](repository-identities.json)。这是当前公开身份/fork关系核对，不能扩张成排除所有历史复制代码的证明。所有 8 仓在本阶段起仅作评估准备，不进入后续 train 数据。

后续 CPU 准备按以下顺序执行：

1. 核对仓库 fork/改名关系，再按固定 revision 提取冻结问题；问题、参考 patch 和判据只用于评估，保存与训练材料分离。
2. 核对上游镜像可获取性，或按固定依赖构建相应 Linux 环境。`image_name` 非空仅是可定位来源，尚不证明镜像可拉取、Python 兼容或判据有效。
3. 复用来源既有 F2P/P2P 判据，取得 buggy F2P 失败、reference F2P 通过及适用 P2P 的实际证据；不新增专用测试套件。保留环境失败与原始结果。
4. **在任何 C 阶段候选模型结果产生之前**完成材料接纳与替补：只因源不可获取、环境失败或前后判据无区分替换，使用同仓下一个固定身份，并记录原失败。整仓不可用时报告覆盖减少，不临时补选新仓或按模型输赢补题。
5. 冻结可运行面板和材料 SHA 后进入 Base/Step-68 A/B。若达不到 24 题，报告实际数量和原因。

身份冻结阶段的新增网络下载、题目/轨迹正文读取、GPU 时间与模型调用均为 0；未读取 heldout 正文。只用一次性 Python 命令核对已存在的元数据并生成冻结，没有新增检查框架或测试脚本。

执行阶段的 CPU 材料入口为 [prepare_material.py](prepare_material.py)：

```sh
python3 research/017/panel/prepare_material.py fetch
python3 research/017/panel/prepare_material.py snapshots
```

`fetch` 已完成，24 份 `host-source/` JSON 总计 625,871 bytes，投影传输 32,649,179 bytes；文件 SHA 见 [source-acquisition.json](source-acquisition.json)。它复用 P015 的 HTTP range 传输，从固定 Parquet 仅投影任务、参考 patch 和来源判据字段，**没有请求 `messages`、`meta` 或教师 hints**。Parquet 列页会携带相邻行的共享存储字节，但只对冻结 row_index 解码/导出；不会把相邻题或教师轨迹交给模型。Agent 可见的说明单列于 `agent-tasks.json`，参考和判据留在 host 侧。

`snapshots` 获取原始 `base_commit` 的浅 Git checkout，不执行仓库代码；保留真实 HEAD 并拒绝打包有本地修改的 checkout。`workspaces/` 不进入版本库，快照及源文件各有获取清单。它不负责安装依赖或声称 CPU 判据已通过；后续 Linux CPU 预验证由 `cpu-preparation/` 承接。

来源 `RDFLib__rdflib-1456` 和 `networkx__networkx-6908` 的 `PASS_TO_PASS` 为空。保留原身份，不因空 P2P 替换或造判据：报告为 F2P-only、回归未覆盖，P2P 结果用 null。确认报告同时给全部有效配对和按 P2P 来源元数据预先划定的覆盖子集，不把 null 计为通过。
