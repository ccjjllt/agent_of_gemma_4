# P016 公开轨迹首批供给与长度调查

2026-10-22；CPU 数据准备，无 GPU、模型采样、训练或提交。已实际取得 **160 个去重身份、53 个仓库家族、28,247,871 bytes 原始候选记录**。它们的用途全部预先冻结为 train；尚不能把上游成功标签称为本地整题成功，也不能把下表的原表示长度称为已可训练的六工具轨迹。

## 已得到什么

| 来源 | 固定 revision | 真实首批 | 原表示≤16K | 原表示≤32K | 中位 token | 最大 token |
|---|---|---:|---:|---:|---:|---:|
| SWE-Lego Real | `879c96b91dfa321b4d1c621d149a9200f8a1d118` | 48题/23仓 | 0 | 0 | 47,696 | 114,541 |
| SWE-Lego Synthetic | `3e745545b07516f8772d2b7ecc5d3bbee84c012c` | 48题/23仓 | 0 | 7 | 41,267 | 98,618 |
| SWE-smith trajectories / tool | `08e109b4a59eaeebf80e4675cd125d42e7ac99a4` | 64题/33仓 | 21 | 46 | 20,842 | 81,777 |

原件为 ignored `research/016/data/source-rows/<source>/<instance_id>.json`，逐题路径、SHA256、物理行号、教师身份和来源在 [source-acquisition.json](source-acquisition.json)。逐题长度、工具族和行为统计在 [source-profile.json](source-profile.json)。

**长度口径**：已取得的全部来源消息放入官方 31B CT v2 tokenizer/template；Lego 保留 XML，Smith 原生 `tool_calls` 序列化为 JSON 文本，工具响应标明 source response；加既有六工具 schema 的共同开销，仅用于来源比较。没有把 XML/JSON 当作目标 Agent 的合法动作。这不是实际结构化 Gemma 工具序列的长度，也不是可训练 token 数。实际结构化序列、因果性、标签掩码和监督 token 以 `research/016/encoding/` 的后续产物为准。不得由 53 条原表示≤32K 推出有 53 条可训练完整轨迹。

样本限历史 train 家族，按仓轮询，不代表整个公开数据集分布。Real 复用此前32题再补16题；Synthetic取48题；Smith以32个PR-ID和32个非PR-ID分层。冻结文件中的“synthetic32”为原先简称，实际32个非PR-ID包括27个命名变异和5个来源类型尚未核清的数字后缀，不能仅凭非PR就称全部合成。未因长度更换任何题。

## 身份隔离与真实获取

[train-family-policy.json](train-family-policy.json) 合并P015、P011和早期split注册表：允许70个历史train家族，整仓排除23个dev/select/heldout/预留家族，并保留非train问题及快照身份。未读取这些非train正文。来源元数据投影先于 [source-sample-freeze.json](source-sample-freeze.json) 冻结；正文只按已冻结身份获取。

- Real复用P015的5,009行身份投影与32条原始回执，只新增16条精确行。
- Synthetic的9,127行resolved身份投影实际读取463,305 bytes；完整源文件482,274,423 bytes，没有下载完整语料。
- Smith复用已有8份parquet，共24,100条元数据；只按物理行＋instance_id＋traj_id三重条件读取64条messages，不读取patch列。旧实验已发现patch列错配，所以不能用该列给新样本背书。
- 来源间用规范化repo＋PR编号或合成问题身份去重，160个canonical key唯一；这不是语义近重复或fork家族自动发现。
- HF `/rows` 每次仅取已分配train的offset/length=1，前后核对main等于固定revision。正文不含任何新生成内容；原始gold patch等字段仅在原件里归档，不应进入模型输入。

获取中真实发生：默认沙箱DNS不可达，经正常公开只读网络授权后继续；旧Python3.9 SSL EOF，复用已有Python3.11解决；补存HTTP `truncated_cells`回执时遇429。按根负责人指示停止额外外网，离线合并为 **Lego 78条空截断标记已确认、18条初次未保留标记**；18条不能仅凭本调查声称完整轨迹。Smith64条是本地Parquet完整行。逐条事实在acquisition的transport中；未来producer在首次下载时保存标记，避免重复请求。

## 行为覆盖与转换优先级

| 来源 | 含编辑器动作的题 | 含明确测试runner调用的题 | 含finish/submit的题 |
|---|---:|---:|---:|
| Lego Real | 48 | 45 | 46 |
| Lego Synthetic | 48 | 44 | 48 |
| Smith | 64 | 19 | 64 |

这些是语法识别，不代表动作成功或判据通过。测试runner仅统计真实命令调用pytest/unittest/tox等，未把普通Python复现脚本或工具返回里的“pytest”计入，因此不是全部验证行为数量。Real `tornadoweb__tornado-2058`、`theskumar__python-dotenv-260`未见finish，即使resolved=true也需单列终局完整性。

[六条来源长度分解](lego-length-decomposition.json) 选各Lego源最短、中位、最长已读train记录。system均3,929token，额外六工具schema仅332token；Real后续观察为18,967/28,131/82,105token，Synthetic为15,420/25,595/65,590token。长记录主要来自原交互观察与动作，不是额外schema；去掉额外schema后Real最短仍36,238token。但这仍是原XML表示，结构化编码后的长度另算，不推断为不可适配。

Smith的tool分区混有两种表示：本批31题含原生tool_calls/role=tool，其余用XML。原生tool回应的关联字段为tool_call_ids列表；assistant的content/thought/action可能重复，不可双计。Lego含turn_mask；抽查system/user为True、普通assistant为False，编码前须按源定义确认True=mask，不能反过来监督。

| 原工具 | 固定六工具的候选映射 | 主要剩余工作 |
|---|---|---|
| bash / execute_bash | run_command | 保留原命令；核对cwd、超时、输出截断与状态。仅换名后保留旧输出不能声称目标工具已执行 |
| str_replace_editor view | read_file | 解析范围和行号，处理目录查看与文件查看差异；观测格式须明确来源 |
| str_replace_editor str_replace | edit_file | old_str/new_str可对应；核对路径、当前状态、锚点唯一性，不能从最终patch倒推状态 |
| create / insert / undo | write_file或独立处置 | create可先核文件状态；insert/undo依赖编辑器状态，不能无损假定等价 |
| finish / submit | submit_patch及终局文本 | 原结束与目标提交过程不同；不得把原finish内容伪造成实际目标提交结果 |
| think | reasoning/text（单独定义） | 原source-only工具不是固定工具，需正确保留思考边界，不混进不存在的目标调用 |

本批Lego均含think；Smith集中于bash、str_replace_editor、submit，且短完整轨迹较多。**下一步最高产率是先把Smith短完整轨迹正确编码，再扩大同一缓存中按元数据冻结的独立题**；不要为了优先Real而先强行摘要48条超长轨迹。Lego Synthetic作为后续编辑与验证覆盖补充，Real保留真实问题价值但需以结构化长度和真实状态窗口再判断。首批pyupio/safety-343的repo许可字段为空；保留数据卡许可和已存在的源码许可证依据，缺字段如实标注，不把逐仓法律判断设为本轮离线SFT新增门槛。

后续 [smith-next-1000-identity-freeze.json](smith-next-1000-identity-freeze.json) 先仅用元数据准备：**1,000个新增独立身份、70仓，500个PR-ID＋500个命名变异**；排除当前160的所有canonical身份及来源类型未核清的纯数字ID。候选余量为816个PR、2,223个命名非PR。根负责人随后授权CPU抽取，现已实际完成1,000条正文，详见[后继批次报告](smith-next-1000-report.md)。没有因首批短轨迹分布预报该批实际可训产率。

## 可复用生产命令

初次元数据投影使用已有库，无需安装：

```sh
PYTHONPATH=/private/tmp/p012libs311:/private/tmp/p015-data-deps /opt/homebrew/bin/python3.11 research/016/data-preparation/profile_public_sources.py metadata
python3 research/016/data-preparation/profile_public_sources.py freeze
python3 research/016/data-preparation/profile_public_sources.py acquire --source swe-smith
PYTHONPATH=/private/tmp/p012libs311 /opt/homebrew/bin/python3.11 research/016/data-preparation/profile_public_sources.py acquire
PYTHONPATH=/private/tmp/p012libs311 /opt/homebrew/bin/python3.11 research/016/data-preparation/profile_public_sources.py receipts
PYTHONPATH=/private/tmp/p012libs311 /opt/homebrew/bin/python3.11 research/016/data-preparation/profile_public_sources.py census
```

当前版本已经冻结，`freeze`会拒绝覆盖；扩量应新建后继冻结清单。`acquire`与`receipts`复用现有缓存；其中外网步骤需正常网络授权。Python3.9只用于已有duckdb缓存精确抽取；Gemma tokenizer依赖复用3.11环境。未运行上游代码或新增测试文件。
