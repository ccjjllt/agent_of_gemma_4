# P015：SWE-Lego train 身份、实际长度与局部工具重放

2026-10-22。仅 CPU，未训练、未启动 GPU、未购买教师、未读取 heldout。数据主体在全局 ignored 的 `research/015/data/`；本目录记录身份、生产方法与真实结果。

来源是 `Lego-X/SWE-Lego-Real-Data` resolved，固定 revision `879c96b91dfa321b4d1c621d149a9200f8a1d118`。首先只投影 instance_id/repo/base_commit/resolved/license_name/image_name：源文件330,420,435 bytes，投影实际634,690 bytes，5009行。然后仅按冻结 row_index、length=1 请求指定 train 行，逐条核对 instance_id/base_commit；请求前后确认 dataset main 尚匹配固定 revision。源 JSON 原件与 SHA 可在 ignored 目录复查。filter 服务502后改 /rows；中途429后续取缓存，不丢弃失败事实。

`train-identity-freeze.json` 先按历史已分配 train 仓内元数据确定24题；首批19条局部交易重放、15条完整交易能容纳8K，于读取新内容前按原顺序追加8题，总上限32。未读其他仓正文。天然PR开发候选仅元数据给 evaluation：arrow、voluptuous、sqlglot、pyupgrade及faker备选整仓排除训练，原cloudpickle/parso/funcy也排除。

`source-length-census.json` 是完整 teacher-native OpenHands XML内容放入官方 **31B CT v2** 完整 chat template 的实际 tokenizer 计数。它诊断来源长度，不是当前六工具协议的可训交易，不把 XML 改名当协议转换。首24题最短37,164、最长114,541 tokens，无一≤8K。

`prepare_lego.py` 实际下载每题固定 base_commit 的生产Python文件，用原 v31 official read_file/edit_file/submit_patch 及已有 macOS local transport 重放来源结构化 str_replace。每个成功编辑必须由本次read展示完整精确锚点；失败编辑保留上下文、不给正向loss；后续read/编辑/提交响应均重新产生。源码语法解析及非空官方 submit是真实局部证据，**不代表整题正确、上游测试通过、自主定位或完整原轨迹重放**。来源shell/create/insert/测试不执行，旧思考/工具输出/未来消息不拼回新交易。read位置由来源答案定位，属有答案定位的action SFT。完整因果任务SFT主目标仍需同协议采样或更强行为重放，不能用本批数量冒充完成。

实际恢复样本 `sqlfluff__sqlfluff-4151`：本次read→edit返回3处锚点歧义→本次重读→扩大锚点的新edit成功→非空submit，完整交易2,686tokens。首次失败edit在上下文中且不监督；仅成功edit和submit被监督。它证明编辑协议恢复，不证明CLI语义修复。

编码来自独立取回的官方31B CTv2 tokenizer（`research/015/tokenizer-assets/official-31b-ct-v2-tokenizer`），不是把12B产物改标签。tokenizer SHA256 `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`，chat template SHA256 `94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413`。每题一条完整新交易，超8K整条排除，不裁工具JSON、错误上下文或多计切片。使用现有 `training/encoding/gemma_tool_chat.py`；结束边界处理以训练工程后续实际改动为准，需改动后整体重编。

精确数量、长度、失败原因见 `replay-results.json`；训练输入是 ignored 的 `research/015/data/encoded-lego-31b.jsonl`，用途字段 `meta.group.split=train`，协议 `p005-harness-tools-v1`。不要把来源resolved=true当本地resolved，交易内 task_resolved=null、behavior_verified=false。

最终32身份/18仓：25题实际局部重放成功，21题/16仓完整交易≤8K，共54,852输入tokens、13,292监督tokens；具体实数以dataset-manifest.json为准，7题失败原因及4题超长整条排除在逐题结果中。

训练工程修正普通assistant文本结束边界后，已离线整批重编为 `p015-full-transaction-v3`。本批所有交易以实际submit_patch工具响应结束，没有补写不存在的Done/handover；工具动作不追加EOS。人工一次性核对恢复样本仅ordinal3、4受监督，ordinal1失败动作仍masked；21个issue唯一，全部split=train且未截断。更新指纹见manifest。

## 同协议真实采样的 CPU 起点

**采样快照修正（本节旧方案已作废）**：生成期只能使用原buggy源码，不能把来源新增criterion/test_patch交给Agent。现唯一有效入口是 `research/015/data/train-material/agent-tasks.json` + `snapshots/<iid>.tar.gz`；快照从真实base_commit浅fetch，含原Git对象、HEAD与原base_commit一致，原有tests保持原版本。criterion/reference在host-criteria中指向host侧patch，judge才应用。下文记录含上游test_patch的旧sampling-buggy归档只为保留CPU前后证据，不再作为Agent输入；train-sampling-candidates.json的当前sampling_archive已更新到新原始快照。

根负责人要求不再扩短编辑数量；现有21条只供容量诊断。从已冻结train的32身份中准备3个轻量真实PR，清单 `train-sampling-candidates.json`：xmltodict81优先，boltons338/298备用。完整原始issue、base_commit、来源F2P/P2P、test_patch、依赖均在 ignored 的 `data/train-sampling/<iid>/`。`sampling-buggy.tar.gz` 只含固定buggy工作树及来源自带上游测试补丁，**不含参考修复、teacher后半程或源trajectory**；Agent只应取得该归档及原始issue，参考目录/patch.txt只用于裁判侧。

xmltodict81的完整CPU前后事实见 `xmltodict81-cpu-result.json`：用标准库unittest执行来源已有test_dicttoxml.py共18项，buggy恰好两个来源F2P错误、16个P2P通过；reference18项通过。分别核实导入当前工作树的xmltodict.py，不依赖已安装库。Python3.12.13，与来源推荐3.9不同；运行时仅stdlib，无额外依赖，完整source归档14,211bytes。

初次 `git apply` 在父仓的ignored子目录静默跳过补丁，导致xmltodict初轮15项全部通过；那次不是有效F2P证据。已用 `patch -p1` 确实应用来源原有补丁并重新运行，最终18项结果以上述文件为准；`train-sampling-environments.json` 保留最初失败环境事实。没有因此改写或新增项目自有测试。

boltons两题的源码和上游test_patch/reference_patch均已准备，但现有pytest9.1.1拒绝其旧conftest的pytest_ignore_collect(path,config)参数。安装来源pytest7.2.0的临时CPU环境遭自动审批拒绝，理由是把运行上游测试解释为违反不建设测试套件约定；未绕过拒绝，没有将其报告为已验证环境。后续如需要运行，应先按用户关于手动上游判据的真实授权解决审批，不能把环境失败记为模型失败。

根负责人确认复用已有环境可行后，evaluation提供 `research/015/evaluation/deps/sqlglot/bin/python`（只有pytest8.3.4及pytest依赖，并未安装sqlglot）。train工作树仍完全独立。用该已有安装手动运行上游既有判据，boltons338 buggy为1F2P失败+25P2P通过、reference26通过；boltons298 buggy为5F2P失败+119P2P通过、reference124通过。原新安装拒绝无需再申请，也未通过其他安装绕过。原始完整输出见 `boltons-cpu-results.json`。

三题的F2P/P2P已分别给出可直接在buggy工作树使用的命令，见 `train-sampling-candidates.json`。XML使用stdlibunittest模块名，boltons用精确pytest节点，不漏掉同参数化函数中的P2P。采样只能给原始issue、buggy归档及必要运行命令；不可提供参考目录、patch.txt、teacher轨迹、编码局部答案或本目录的判定日志。两仓都是历史train家族，并非P015外部评测四仓。

最终独立train-material与evaluation材料格式相同，3任务split=train；每题环境/wheelhouse已准备（boltons复用已有pytest8.3.4，xmltodict仅stdlib）。合并CPU验证和同次CT采样仍须保持train/dev用途与输出分开。快照身份、文件指纹、criterion可应用性见train-material-manifest.json。
