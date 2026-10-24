# C 面板 CPU v3：五题有界修复

2026-10-24。本轮仅准备 CPU 载体，尚未上传或执行，不启动 GPU。原 24 题身份冻结、首轮 11 有效/13 未有效记录不变。已有 11 个有效题不重跑；本轮若产生新有效题，需与首轮按身份明确合并，不能把沿用记录计作新的执行。

**交给 root 运行的目录是 `carrier-v3-ready/`。** 较早的 `carrier-v3/` 是本地构建稿，title 与远端 slug 未对齐，不用于上传。v1/v2 的冻结 kernel 文件 SHA 仍与各自 manifest 一致。

| 项目 | 最终值 |
|---|---|
| Dataset 目录 | `research/017/panel/cpu-preparation/carrier-v3-ready/dataset` |
| Dataset ID / title | `jiyujia/p017-confirmation-cpu-repair-v3-d7d6753a-sources` / `P017 Confirmation CPU Repair V3 D7D6753A Sources` |
| Kernel 目录 | `research/017/panel/cpu-preparation/carrier-v3-ready/kernel` |
| Kernel ID / title | `jiyujia/p017-confirmation-cpu-repair-v3-d7d6753a` / `P017 Confirmation CPU Repair V3 D7D6753A` |
| 建议平台秒数 / 内部 wall | 3000 / 2700 |
| GPU / TPU / Internet | 全为 false |
| 输入字节 | 166,516,347 |
| 输入 SHA256 | `743063a14a406b2f64e7daec6d946d13d5ae06bdffe60b4cea5219703d6b9304` |
| `kernel/run.py` SHA256 | `244b2632406b0e45eb08855263f46af06bcf8911e92ddb04c682ac111a29b233` |
| 当前 runner SHA256 | `9cb59320545b7b5dc2f03dac6690a6f7c904f8dc1b6e7f411a235811d833a8e4` |

两个 title 规范化后的 slug 已与对应 ID 逐一核对。SHA 与全部输入成员见 [ready input manifest](carrier-v3-ready/input-manifest.json)；具体改动及原缓存证据见 [repair-plan-v3.json](repair-plan-v3.json)。

本轮只重跑以下五题，仍按原冻结顺序：

1. `hgrecco__pint-2089`：来源中两条 P2P node 均在 `[cm` 处截断。源码原参数确为 `cm centimeter cm centimeter`；runner 在 buggy/reference 两态对原函数实际 `--collect-only`，每个旧 prefix 必须唯一对应一个完整 node，两态 mapping 必须一致。然后运行原有完整 P2P；记录旧字符串、完整 node 和 collection 输出。没有改写或删去测试。
2. `facelessuser__soupsieve-50`：不调用含 `imp` 的旧 setup，而从原快照/PYTHONPATH 导入源码；保留 module-origin 检查和全部原判据。
3. `hynek__structlog-586`：补原 `pyproject.toml` 声明的 hatchling、hatch-vcs、hatch-fancy-pypi-readme 及依赖。
4. `facelessuser__soupsieve-119`：与 soupsieve50 相同，仅绕过不必要的 editable 打包步骤。
5. `facelessuser__soupsieve-263`：补原声明的 hatchling 及依赖，保留原源码与判据。

复用首轮材料的 191 个 Linux wheel，新增 backend 依赖中 7 个新文件；另一个 pathspec wheel 已在缓存，内容相同，所以最终缓存 198 个 wheel。下载及 wheel METADATA/SHA 保存在 [backend-cache/manifest.json](backend-cache/manifest.json)。原 `setuptools==75.8.0`、`packaging==24.2`、`pluggy==1.5.0` 和任务库版本保持不变；解析器曾额外下载的 pluggy1.7 未打包/安装。builder 已核对五题每一个 requirement 都有对应缓存文件，runner 用 `--no-index`，因此此轮无需外网获取依赖。

本地仅有 macOS Python 3.11.15、3.12.13 及 CommandLineTools 默认 Python3.9；Docker CLI 存在，但只读查询确认 daemon 未运行。没有可直接证明等价的本地 Linux 任务环境，因此**Pint 完整 collection 与五题修复成功均尚未声称完成**；由本次 CPU 作业产生真实证据。v3 继续使用已固定 Kaggle Linux/Python3.12.13；旧 numpy/greenlet 与 jsonargparse 的兼容解释器工作不塞入本轮。

RDFLib 两题另按原冻结候补顺序保留身份：[replacement-identity-freeze-v1.json](../replacement-identity-freeze-v1.json)。`1022 → 2474`（原候补 rank4）、`2084 → 1117`（rank5），均在任何模型结果之前确定。这里只登记候补身份，未读取候补正文、未取得其快照/判据，也未将其冒称已接纳替换；原 24 题身份冻结仍不变。

CPU 作业结束后，应分别读取五题原始日志、前后判据和 P2P 覆盖，再决定哪些新增接纳。source_import_only 或 backend 安装成功都不等于修复判据有效。若仍失败，保留失败而不通过更改生产源码、删测试、添加 mock 或伪造回包来凑题数。
