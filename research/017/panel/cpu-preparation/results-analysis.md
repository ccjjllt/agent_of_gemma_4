# P017 C 面板首轮 CPU 结果诊断

2026-10-24。依据已完整下载的 [summary.json](downloaded-output/p017-confirmation-preparation/summary.json)、[prevalidation-results.json](downloaded-output/p017-confirmation-preparation/prevalidation-results.json) 和同目录原始日志。本轮是环境与来源判据执行，模型调用为 0；尚未观察新的 Base/LoRA 结果。本分析不改变原 24 题身份冻结，也未启动新执行。

**24 题全部实际尝试，11 题满足本次前后判据，13 题尚未有效。** 13 题中 7 题在依赖构建/安装阶段异常，6 题执行了判据但不满足接纳条件；无步骤超时。11 题覆盖 5 仓：networkx 3、mashumaro 3、sqllineage 3、rdflib 1、structlog 1。9 题具有实际 P2P 前后通过证据；`networkx__networkx-6908`、`RDFLib__rdflib-1456` 为来源无 P2P 的 F2P-only，回归仍记未覆盖/null。

下载材料 `prepared-confirmation-material.bin` 实际为 **201,086,917 bytes**，实算 SHA256 为 `edea1171b00b009348b6424c81da541cda8edc461e6034450d011fb8c19ed428`，与 summary 一致。材料保留全部原题及 CPU 失败，不是已接纳的 24 题 GPU 输入；执行 C 前仍需冻结最终可运行集合。

| 未有效问题 | 实际证据与归类 | 不改变任务本质的下一步 |
|---|---|---|
| `hynek__structlog-586`、`facelessuser__soupsieve-263` | editable 安装明确 `ModuleNotFoundError: hatchling`。各快照 `pyproject.toml` 声明 hatchling；structlog 还声明 hatch-vcs、hatch-fancy-pypi-readme | 补齐原项目声明的 build backend 及其依赖、固定实际 wheel，再重跑这两题原判据。不能只补 hatchling 就声称其余 backend 依赖已满足 |
| `facelessuser__soupsieve-50`、`facelessuser__soupsieve-119` | metadata-generation-failed；日志没有暴露内部 traceback。两份实际 `setup.py` 均 `import imp`，与当前 Python 3.12 环境不兼容，是可定位的安装阻点 | 优先检查能否直接从固定源码/PYTHONPATH 导入运行全部原判据，避免不必要的 editable 安装；必须保留 module-origin 核对。若确需安装则采用兼容任务解释器。不要改生产代码或测试来兼容安装；尚不能宣称绕过安装后必然通过 |
| `hgrecco__pint-1248`、`hgrecco__pint-1039`、`hynek__structlog-117` | 分别卡在 numpy 1.24.4 构建、numpy 1.21.6 不支持当前 Python、旧 greenlet 构建。并非模型修复失败；其中两份构建日志未含底层编译 traceback | 先评估独立任务环境能否使用来源兼容 Python，复用已有依赖；不要同时升级一组数值/运行依赖来“跑绿”。若需要的移植工作超出本轮预算，按已冻结同仓候补处理并保留失败原因 |
| `omni-us__jsonargparse-560/424/242` | 三题 reference F2P 均失败，报 `ArgumentParser._parse_known_args()` 缺少 `intermixed`。运行版本为 Python 3.12.13；560/424 还存在来源 P2P node 被截断 | 先恢复兼容的任务 Python/依赖环境，再处理来源 node 映射；不要把对 argparse 的兼容代码加入待修仓库，也不能只修 P2P 名称便接纳。需重新取得原 F2P buggy 失败/reference 通过 |
| `hgrecco__pint-2089` | F2P 已真实 2 fail → 2 pass；两态 P2P 均 exit 4。原 host-source 列表已有截断的参数化 node，非本次 argv 引号错误 | 按实际 collection 对照补全原有 node，保留原字符串和一对一映射记录，再跑完整来源 P2P。此项改判据定位，不新写测试，也不能删掉找不到的节点。具体映射由 collection 审查负责 |
| `RDFLib__rdflib-1022` | 来源 nose 导入因 Python 3.12 删除 imp 而 collection 失败；源测试还访问 DBpedia | 同仓候补优先。换解释器仅能解除首个障碍，不能解除离线任务的外网依赖；不为保留题数造 shim/mock |
| `RDFLib__rdflib-2084` | reference 唯一 F2P `test_service_with_bound_solutions` 仍 `assert 0 == 2`，前后多个 P2P 也失败。原 test patch 的该 F2P 实际查询 DBpedia 的 John Lilburne/subject 数据，依赖外部服务内容 | 在当前离线评测目标下，视为未能复现的来源环境，按固定同仓候补；不改远端数据断言，不用自造 mock 替代原判据。日志本身不能区分远端数据漂移和服务行为变化，不能断言源修复错误 |

exit 4 的原 node 截断/collection 详情由并行审查补充；上表没有把相关错误归咎于模型。空 P2P 两题继续保留，不与 node 错误混淆。

建议先完成一轮有界环境修复：补构建 backend、处理不必要的旧 setup 安装、修 Pint2089 的确定性 node 定位；可望增加覆盖，但**在新的实际结果出现前仍只能报 11 有效**。三个 jsonargparse 和三个旧二进制依赖任务需要解释器兼容性决策；两个外网依赖 rdflib 任务优先走冻结候补。所有替补应在任何 C 模型结果前确定，不能按模型输赢重选。

目前 **11 题/5 仓不足以兑现原定 24 题/6–8 仓的确认范围**。若预算只容许这些题，应明确缩小范围，不能把 13 个环境问题悄悄排除后仍称“完整 24 题确认”。已有有效题无需因其他题失败重复全量准备；只对改变过的环境或判据定位重新执行必要前后验证。

当前 CPU runner 的 venv 使用 `system_site_packages=True`，因此本次有效性也依赖该 Kaggle Python 3.12.13 系统环境；201MB wheel 材料并不证明全部系统依赖已自包含。C 载体接入时应保持对应环境并记录差异，不将该 CPU 结果扩大成任意运行环境下均可复现。
