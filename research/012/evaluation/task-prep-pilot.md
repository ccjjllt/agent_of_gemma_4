# P012 任务环境试点报告：2/12 已确认区分性（CPU，可复核）

日期：2026-10-19。对应 `plan-48.md §2`。状态：schema pr_331 与 cookiecutter pr_1669
两试点完成修前失败 / 参考修复通过的区分性确认；其余 10 题待放量。
前任子代理中途因模型区域限制中断，我直接接手并对全部关键结论做了独立重跑验证；
报告中凡写“独立重跑”均为本轮直接执行命令所得。

## 1. 试点对象（元数据冻结延续）

| 试点 | source_file | row_index | traj_id | buggy_commit（`git ls-remote` 实测） |
|---|---|---|---|---|
| keleshev/schema pr_331 | shards/tool-00007-of-00008.parquet | 1035 | …pr_331.dqjv4czx | `fead5cb2…ba0142e` |
| cookiecutter pr_1669 | shards/tool-00002-of-00008.parquet | 2529 | …pr_1669.z8j2pnxm | `856e08cd…352aa20c` |

`ls-remote` 输出原文与 commit 已记入各自 `provenance.json`
（`git_ls_remote_stdout` 字段）。此后任何步骤均未读取 heldout/预留题；
准备产物只在 `evaluation/task-env/` 下，未写入训练目录。

## 2. schema pr_331：JSON schema 对 None 用 `"type": null`（应为 `"const": null`）

- 轨迹 43 消息：2 处 `str_replace`（`schema/__init__.py`，OR 单值分支 + COMPARABLE 分支），
  均经核对 `old_string` 逐字存在于 buggy 源码；另有 `reproduce.py` + 3 个 edge case 脚本
  修前/修后执行记录。
- `reference.patch`：由轨迹 2 处生产编辑重建（**不是** `patch` 列——该列是 3MB 的
  site-packages 垃圾，已逐字节比对确认不同）。`patch -p1 --dry-run` 通过。
- 独立重跑 PRE（存储的 buggy 源码，未改动）：
  `Schema({'test': None}).json_schema('my-id')` →
  `{'test': {'type': None}}`。PRE-CONFIRMED（buggy `type-is-None`）。
- 独立重跑 POST（scratch 拷贝 + reference.patch 应用）：
  → `{'test': {'const': None}}`。POST-CONFIRMED。
- 判据：同一复现表达式修前/修后输出区分；LICENSE 缺失如实记录
  （buggy commit 无顶层 LICENSE 文件，`setup.py license=MIT`，见 provenance）。

## 3. cookiecutter pr_1669：新增 `--keep-project-on-failure`（hook 失败保留产物）

- 轨迹 41 消息：9 处 `str_replace`，其中 8 处生产文件
 （`cli.py`×3、`main.py`×2、`generate.py`×3），1 处测试文件（`test_hook_failure.py`，
  准备参考时排除）。8 处生产编辑的文件/区域与 `reference.patch` 的 3 个文件 hunks 一一对应；
  内容语义一致。注意一处如实记录的差异：轨迹 msg22 的 `old_str` 为 12 空格缩进，
  buggy 文件对应调用点为 16 空格缩进——逐字重放该单条会锚点失败；
  `reference.patch` 按文件实际缩进重建后可干净应用。这属于教师轨迹转写差异，
  不影响任务本身的修前/修后区分性（任务裁判用参考修复行为，不用轨迹工具回放）。
- `reference.patch` 同样不是 `patch` 列（该列同为 3MB 垃圾，已比对确认不同）。
- 存储源码 SHA 与 buggy commit 直接下载逐字节一致
  （`main.py a8bd2752…`）；PRE 在完整镜像克隆确认：
  `main.cookiecutter` 与 `generate_files` 签名均无该 kwarg。
- 独立重跑 POST（完整克隆 + reference.patch + 最小模板 + 失败 post hook）： 
  默认 `FailedHookException` 且产物目录被删；`keep_project_on_failure=True` 时
  同样抛 `FailedHookException` 但目录与文件均保留。POST-CONFIRMED。
- 首次自建模板把 hooks 误放进 `{{cookiecutter.proj}}/hooks`（被当成普通内容拷贝，
  hook 未执行），已纠正为模板根 `hooks/post_gen_project.py` 后复测通过；
  弯路如实记录。

## 4. 结论与放量估计

- 两试点任务环境均可用，修前失败与参考修复通过有明确区分，与 `criteria.log` 一致；
  凡报告结论均经本轮独立命令重跑，非转述。
- 剩余 10 题（cookiecutter×3、feedparser×4、conan×3）按同一流水线放量：
  原子取行 → `ls-remote` 定 commit → 下载生产文件子集 → 全镜像 scratch 复现 PRE →
  重建参考补丁 → POST 确认。每题预计 10–30 分钟人工/CPU（conan 为大仓库，
  全克隆与依赖安装是主要风险，建议先做 `pr_16559` 评估体量，必要时只取子目录 +
  延迟到 GPU 作业内装依赖）。
- 10 题全部就绪后方可启动 48 次运行；旧 Rich/FastAPI/Requests 三题不进入主结果。
