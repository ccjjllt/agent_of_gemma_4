# requests_7205：已有补丁的 CPU 导入路径重放

执行日期：2026-10-18。没有重新运行模型、GPU、Kaggle kernel 或 submission；未修改 P007 原结果。只使用已暴露的 requests_7205 官方快照、原官方 test_patch 和 P007 两臂已生成的相同 503B agent patch，没有读取 gold patch、heldout、预留题或外部 8 仓库任务正文。没有编写测试文件；官方既有测试仅在临时解压目录应用并运行，真实 API 对照通过一次性 Python 命令执行。

**结论：修正源码导入后，同一旧 agent patch 通过原官方测试文件。** 基线为 `1 failed, 205 passed, 13 skipped`；补丁后为 `206 passed, 13 skipped`。保持补丁工作树不变、故意让同一快照的未修副本遮蔽导入，失败又恢复。由此获得实际 false-negative 纠正证据；P007 原记录的 resolved=false 保持原样，另记本次复评通过。

## 实际执行结果

全部阶段采用同一份官方 `tests/test_utils.py`（含原 test_patch），没有改断言。`real_api` 直接调用真正的 `requests.utils.get_netrc_auth`，使用临时真实 `.netrc`；没有 mock、替身函数或网络请求。

| 阶段 | 实际 requests 导入 | 空 default 凭据 / 正常凭据 | 官方目标用例 | 原官方整个 test_utils.py |
|---|---|---|---|---|
| baseline_source | 官方快照 `src/requests`，2.32.5，未修 | `('', '')` / `('user','pass')` | 1 failed | **1 failed, 205 passed, 13 skipped** |
| patched_source | 同一目录，同一版本，应用旧 503B patch | `None` / `('user','pass')` | 1 passed | **206 passed, 13 skipped** |
| patched_worktree_shadowed_by_identical_base | 工作树保留补丁；导入同快照未修副本 2.32.5 | `('', '')` / `('user','pass')` | 1 failed | **1 failed, 205 passed, 13 skipped** |

匹配的遮蔽对照是主动构造的因果实验：复制同一公开快照的 Requests package，恢复其原始 utils.py，再放在依赖导入路径上。它不是对旧 Linux 镜像的逐字节重建，但保持源码/测试/补丁一致，单独改变运行时导入的是修前还是修后代码。

另记录了一次本机已有包导入对照：本机项目环境安装的是 Requests **2.34.2**，该版本目标用例已经通过，但整文件出现四个 `_parse_content_type_header` 不兼容失败。这不能用来复现原 Kaggle 镜像的旧包故障，故不拿它当核心 fail→pass 证据；原始输出仍保留以明确环境差异。

## 来源和身份

- 官方单文件下载：[snapshots/requests_7205.tgz](https://www.kaggle.com/competitions/gemma-4-developer-agent/data?select=snapshots%2Frequests_7205.tgz)，使用 `kaggle competitions download gemma-4-developer-agent -f snapshots/requests_7205.tgz`，没有下载完整 tasks。
- 快照 **37,402,474 bytes**；SHA256 `42dc3d360f842a5c7a66d0008ac05447fcd807dc33ee8cc18d2f2d4876283106`，原件在 `raw/requests_7205.tgz`。
- 修前 `src/requests/utils.py` Git blob **4d3039b200a7aaf14cc6af4a1bc5927068dafa47**，与 P007 agent patch 的 preimage index 精确一致；修后 blob **d113a6ff3ebee409dc4466a1481a9cf1a628250a**。
- P007 两臂 `17-requests_7205-r1-public022_v10`、`18-requests_7205-r1-behavior_v1` patch 逐字节一致，本次只应用其中一份，没有重写实现。
- 原官方 test_patch 来自 P007 已冻结的 12 题子集，只选择 requests_7205。原字符串缺最后换行，首次 `git apply --check` 报 corrupt；**仅补一个末尾换行**后正常应用，未改任何测试逻辑。最终官方测试文件 SHA256 **749c36bc574e3e796ddb78bbdc91b0d92fccf703a39c6d21f5afbfa7b28f50f8**。
- 下载快照的 Git HEAD 为 `443f0b73558781b7dac0e304ad268a30ee35aaeb`，任务 metadata 的 base_commit 为 `1b40fdd004bfc8ba5301bcf8a6908264e9b6b877`。没有按 metadata reset；目标源码 preimage 与旧补丁精确匹配，测试失败/通过计数与旧评测结构吻合。没有据此宣称整个快照与历史运行逐字节相同。

完整命令、解释器、实际输出、耗时及身份哈希在 [requests7205-replay-results.json](requests7205-replay-results.json)；各 stdout/stderr 保存在 `raw/*-real_api.log`、`raw/*-official_target.log` 和 `raw/*-official_test_utils.log`。临时目录位置在 [local-replay-context.json](local-replay-context.json)，不依赖这些目录永久存在才能审查结果。

## 环境与复用方式

本次为 **macOS / Python 3.11.15 / pytest 8.4.2**，不是原 Kaggle Linux/Python3.12 环境。pytest 只安装在临时 venv；Requests 的通用依赖复用项目已有 site-packages，项目环境没有改动。初始临时 venv 没有继承项目 venv 的依赖，出现 urllib3/requests import error；在 PYTHONPATH 尾部显式加入现有依赖目录后完成上述实际实验。该准备问题不计模型失败或修复成功。

运行时路径顺序应为：

```text
<actual sandbox workspace>/src  （存在时）
<actual sandbox workspace>
已有依赖目录/环境默认路径
```

v31 `swegemma/sandbox/subprocess.py` 的 `exec_run` 在约 485–493 行每次重建 `extra_env`，原实现只将 `str(workspace_dir)` 放到 PYTHONPATH。因此**只设置父进程 PYTHONPATH 不够**，会被覆盖；只在 agent 命令写 `PYTHONPATH=/workspace/src` 也不能修正后续独立 verification sandbox。

本轮研究载体可在该单点把存在的 `workspace_dir / 'src'` 前置，再保留 `workspace_dir`，交给同一个 `build_sanitized_env`。这样 agent、校验和所有新创建的 subprocess sandbox 统一采用本地源码。使用真实 workspace_dir 对象生成路径，不硬编码 `/workspace` 到宿主环境。执行前记录导入路径即可确认命中了本地包；不用构建独立 gate/check 框架。

本次没有修改这份官方 wheel 或实验载体，修复交由主代理的部署分支实施；没有把本机路径写进候选 agent。

## 下一训练行为目标与诊断题选择

可训练目标应具体到行为：**源码改动后若结果仍是旧行为，先确认模块 `__file__` 是否来自当前 checkout，再通过同一个真实 API 做精确值和正常兼容输入对照。不要验证自造 helper，也不要因错误环境结果反向撤销正确补丁。** 本次旧 trace 的 import 定位可作为动作示范；原 false 反馈不能直接作为“补丁错误”的奖励标签。

没有新增模型成功轨迹，也没有宣称训练收益。本次只纠正验证环境反馈。按主代理在新模型结果出现前的决定，下一诊断题固定为 rich_3067、fastapi_14463、requests_7205。没有继续下载 inline_snapshot，也没有扩展读取其他题；fastapi_14482 的缺依赖问题不在本轮三题中。
