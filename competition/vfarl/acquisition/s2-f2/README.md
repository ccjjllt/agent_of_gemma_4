这批材料提供 **40 个完整的公开训练任务定义：S 20、F 20，每个子机制 4 题**。最终清单是 `reviewed-sf40.json` 和 `reviewed-sf40.jsonl`，不是按轨迹数凑题数。每个独立修复族有 `repair_family_id`，同一代码快照通过 `snapshot_group_id` 统一留在 train 分区。

其中 38 题来自固定 revision 的 `SWE-bench/SWE-smith-py/train`，有配对的原生 `SWE-smith-trajectories/tool` 记录；另外两题是 `SWE-rebench-V2/train` 的 Laspy 坐标缩放和 PennyLane QNode 所有权。Smith 任务的 `patch` 是**引入错误**的补丁，原生轨迹的 patch 列不作为解答来源。所有任务原文、环境 image、F2P/P2P 和来源哈希都保存在对应 `.task.json`；大文件通过本目录 `.gitignore` 留在工作区，清单和审阅说明可版本管理。

逐题机制审阅依据原问题及精确生产补丁，记录触发输入、原错误、负责符号、因果修复、区分性观察和来源限制。`candidate-decisions.json` 保存被替换的错误分类及验证契约不一致，例如 pptx 题实际重复 `@property`，Pudo 题实际是 Alembic 调用签名，不能按标题硬标坐标或缓存任务。旧 Nebius 的 13 个 S/F 轨迹题没有精确环境 join，均由本批完整任务替换。

原生轨迹是待重新采集的材料。PDFminer 纵排题本身保留，但其教师按文件名硬编码结果、导入 tests，正向动作全部隔离。其他轨迹也没有获得自动正向 loss span。来源 resolved 标签的真实计数见清单，不能当成当前官方六工具成功率。

本采集器没有执行任务命令，没有启动 GPU，也没有生成同状态偏好对。官方 `read_file/edit_file/write_file/run_command/get_status/submit_patch` 的重放及官方 verifier 证据由主任务单独产生并关联。38 个 Smith bug branch 已通过本轮只读 git ls-remote 解析为完整 40 位 buggy commit；2 个 V2 任务保留完整 base commit。共享的 8 位 repo snapshot 前缀单独保存，绝不当作执行 commit。运行中环境和 image digest 仍须在实际准备环境时记录。S/F 的旧 source-only 种子不会冒充已重放任务。

来源数据集许可与实际固定代码 revision 的上游许可分别保存；没有用数据集 MIT 覆盖上游代码许可。Yamllint 为 GPLv3，其他选中代码的 MIT/BSD/Apache 文本和 Trio 的两份授权文本都在目录内保留。

复现源采集可参考 `fetch_selected_tasks.py`、`fetch_remainder.py` 及固定选择记录；`build_inventory.py` 仅打包手工审阅后的数据，不运行测试或工具，不产生验证奖励。

本轮官方 CPU 重放发现 Freezegun 的 required `test_maya_datetimes` 在 Python 3.12 环境会 skip。S3 的 pr530 和 F4 的 stop 题已转入 `freezegun-environment-reserve.json`，原 required 列表完整保留；最终40题改用 DeepHash 显式有序/无序模式、Tornado Future 回调引用释放两个独立族。替换题完整执行 commit 和许可文本均重新按固定 SHA 获取，尚未把替换题标记为已重放成功。
