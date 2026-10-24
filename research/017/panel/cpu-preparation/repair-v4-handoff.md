# C 面板 CPU4 交接：两个原题与九个固定候补

2026-10-24。本地载体已构建；未上传、未启动作业。CPU4 组装时已经看到 D 面板首个结果，但尚无 C 模型结果；本轮选择严格沿此前已冻结的候补映射，与 D 输赢无关。原 24 题身份冻结及每仓 3 题目标不改。当前跨两轮的 13 个有效题不重跑。

**使用 `carrier-v4-ready/`，全作业无网、无 GPU/TPU。** 本地缺失依赖核对后，仅新增下载 `isodate==0.6.1`（41,722 bytes，通用 wheel）；其余使用 CPU3 的 198 个实际缓存 wheel。最终 199 个 wheel 覆盖所有 11 题的精确 requirements，未盲目升级任何任务库或原 3.12 工具链。

| 项目 | 值 |
|---|---|
| Dataset 目录 | `research/017/panel/cpu-preparation/carrier-v4-ready/dataset` |
| Dataset ID / title | `jiyujia/p017-confirmation-cpu-repair-v4-ac51dacf-sources` / `P017 Confirmation CPU Repair V4 AC51DACF Sources` |
| Kernel 目录 | `research/017/panel/cpu-preparation/carrier-v4-ready/kernel` |
| Kernel ID / title | `jiyujia/p017-confirmation-cpu-repair-v4-ac51dacf` / `P017 Confirmation CPU Repair V4 AC51DACF` |
| 内部 wall / 建议平台秒数 | 5400 / 5700 |
| 输入字节 | 184,089,318 |
| 输入 SHA256 | `5decd66659b47af89df4855a1d5340c459dcd7d080bc022467a8664cf3ea959d` |
| `kernel/run.py` SHA256 | `f70bd541dc77290fd5c17c3a1ac2d018c771e0947a17f8410d8c77adea10f7df` |
| runner SHA256 | `9cb59320545b7b5dc2f03dac6690a6f7c904f8dc1b6e7f411a235811d833a8e4` |
| 新 wheel SHA256 | `0751eece944162659049d35f4f549ed815792b38793f07cf73381c1c87cbed96` |

两个 title 的规范化 slug 已和对应 ID 精确核对。预算按 11 个独立环境及原判据保留余量：每题依赖步骤/buggy/reference 判据仍有各自超时，5400 秒内部 deadline 留出最终材料保存时间；不是承诺 11 题必然在该时间全部通过。

本次执行身份按原槽位顺序为：

| 原槽位 | CPU4 实际身份 | 处理 |
|---|---|---|
| jsonargparse560 | jsonargparse564 | 固定同仓候补 |
| rdflib1022 | rdflib2474 | 固定候补；从原源码导入，避免非必要 Poetry 打包 |
| soupsieve50 | soupsieve88 | 固定同仓候补；不改旧源码兼容性 |
| structlog586 | structlog586 | 保留 CPU3 所有依赖，只跳过 editable，运行原全部判据 |
| pint1248 | pint877 | 固定同仓候补 |
| jsonargparse424 | jsonargparse592 | 固定同仓候补 |
| rdflib2084 | rdflib1117 | 固定候补；原判据调用 CLI，仍保留 editable；来源 P2P 为空，记 null |
| structlog117 | structlog669 | 固定候补；补原 pyproject 声明的 backend，复用 CPU3 wheel |
| pint1039 | pint1471 | 固定候补；补原声明的 setuptools-scm 依赖，复用 CPU3 wheel |
| jsonargparse242 | jsonargparse248 | 固定同仓候补 |
| soupsieve263 | soupsieve263 | 保留 CPU3 所有依赖，只跳过 editable，运行原全部判据 |

新候补来源、SHA、快照与环境计划来自 [replacement-material-index-v2.json](../replacement-material-index-v2.json)。组装的 [cpu4-input/confirmation-identity-freeze.json](cpu4-input/confirmation-identity-freeze.json) 只是 **11 题 CPU 执行输入清单**，通过 SHA 引用原 24 题冻结和 replacement-v2；不是将原确认面板改成 11 题。全部环境改动与复用缓存见 [repair-plan-v4.json](repair-plan-v4.json)，最终精确成员见 [ready input manifest](carrier-v4-ready/input-manifest.json)。

Pint1471 原 F2P/P2P 字符串存在截断且部分前缀有歧义。本轮保留原字符串，不套用 Pint2089 的唯一前缀映射，更不删掉找不到的测试。旧 Python 源码、argparse 接口、打包和判据仍可能失败；这些都会留下实际结果，不能预报本轮即可补足 24 题。

builder 增加了显式 `--panel-dir` 以读取独立的已冻结执行输入，并接受来源环境计划；默认原面板路径不变。runner 沿用 CPU3，部署包设置 `enable_internet=false`，依赖准备使用 `--no-index`，因此安装及判据也无外网权限。没有 socket monkeypatch、新测试、源码兼容补丁或伪造元数据。

CPU 输出继续区分 agent 可见说明/快照与 host 判据；最终 GPU 材料不打包 reference patch 或 host-source 原始行。CPU4 完成后只按实际前后判据接纳，并与原 13 个有效题按身份合并；保留旧失败和替补缘由，不把复用记录计作新执行。
