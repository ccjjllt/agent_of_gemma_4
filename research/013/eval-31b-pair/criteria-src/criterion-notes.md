# P013 step-2 criterion 编写与 teacher 重放规格说明（CPU 产出，校准待 GPU）

日期：2026-10-20。状态：**CPU 产出完成，GPU V2/V3 校准未执行**。
纪律：只读 13 个 DEV issue 正文（开发集任务准备）；未读 heldout/保留题；
未读 gold patch；未抄 teacher 测试（criterion 断言只源自 issue 正文 + 快照公开 API）；
未推送 GPU；一次性手动命令；未建 pytest 套件（单个文件即交付物）。

## 1. 产出清单（`research/013/eval-31b-pair/criteria-src/`）

- 13 个 criterion 文件 `test_p013_*.py`：每题单个 test 函数（P012 手写单节点先例）。
- `replay/<instance_id>.json` × 13：teacher edit 有序序列（只搬运 `str_replace_editor`
  的 `create`/`str_replace` 源码动作；`view` 与 `/testbed/*.py` scratch 文件已排除并计数）。
- `criteria-index.json`：每题 task_id / instance_id / test 文件 / F2P 节点 / P2P（空）/
  teacher traj_id + 来源分片行号 / 预计依赖。

## 2. teacher traj 选择规则（确定性，可复算）

- 同 issue 有多个含 edit 动作的 resolved traj 时，优先 claude-3-7-sonnet 的 traj；
  无 3-7 edit-traj 时取唯一含 edit 的 traj（t01/t02/t04/t05/t06/t07/t08/t10/s01 均唯一）。
- 无任何 edit 动作的 resolved traj（如各题 `y1wve1wy` 系列、`1cm1w3ln`、`mz3vj24e`）
  未被选为 replay 来源（其修复信息在 `patch` 列，本轮 replay 只搬运消息内编辑动作）。
- 工具约定：全部分片实际使用 `str_replace_editor(command=view|create|str_replace)`，
  未见 `edit_file`/`write_file` 工具名（任务描述中的名称与实际分片格式的差异如实记录）。

## 3. criterion 覆盖边界（逐题）

| 题 | 覆盖 | 未覆盖/边界 |
|---|---|---|
| t01 click 2604 | 4 个后缀剥离 | 显式 `name=` 透传、`ctx.invoke` 链路未断言（issue 提及但属同一代码路径） |
| t02 ms 402 | import 成功 + 基本 escape | 只覆盖 `re` 缺失这一表象；若参考修复还动了别处，本单节点不感知 |
| t03 ms 418 | 注释被完全移除（`'x'`） | 多注释/注释含标签等 edge case 未断言 |
| t04 ms 379 | lower/upper/strip/replace 四法 | issue 列出的其余 14 个方法（title/translate/zfill 等）未逐一断言 |
| t05 ms 400 | XSS 例 escape 全串相等 | 非 str 输入、quote 选项等未断言 |
| t06 pydicom 1938 | 空 SQ 的 VM==1 且 is_empty | 非空 SQ、非 SQ 的 VM 语义未断言；is_empty 的 SQ 特判语义以 V3 为准（见 §4） |
| t07 pydicom 2069 | UN→SQ（VR 与 Sequence 类型）+ PatientPosition=='HFS' | BulkDataURI、大数值集等 JSON 分支未断言；`'x' in ds` 关键字断言在 buggy 下也可能通过，判别力来自 VR/值断言 |
| t08 pydicom 1867 | 不存在文件 → ArgumentTypeError 且消息含文件名（无 IndexError） | `pydicom::` 前缀、charset 命中等分支未断言；依赖 `get_testdata_file` 离线不下载（该文件名不在 url map，无网络） |
| t09 pydicom 2163 | LUTDescriptor 元素 Dataset pickle 往返 | 多进程 DataLoader 全链路未断言（机制与 pickle 同源）；未用外部 CT_small.dcm（避免下载依赖） |
| t10 pydicom 1859 | 三个 HTJ2K UID 可导入且为 UID 实例 | 具体 UID 取值、pixels 解码器联动未断言 |
| t11 pydicom 1995 | 未映射 SRT 比较返回 False（无 KeyError） | SRT→SCT 正常映射相等语义未断言（属旧有通过行为，非本 issue 核心） |
| t12 pydicom 1987 | `repr(elem)==str(elem)` 且含 `300A` | **最不确定**：issue 只给讨论方向无期望输出；若参考补丁保留 `__repr__` 但换格式，本节点在 V3 会失败 → 走 R1/R2（见 §4） |
| s01 pydicom 1920 | 绝对路径 → ValueError（testdata + charset） | palette/dicondir 变体、CLI 端到端未断言 |

## 4. 不确定项与 GPU 校准清单（V2 修前失败 / V3 参考通过）

1. **t12（pr_1987）R1/R2 第一候选**：criterion 形式（`repr==str`）来自 teacher 终态
   （删除 `__repr__`）与 issue 讨论方向的交集；若官方参考补丁采用不同 repr 格式，
   V3 会失败 → 按第 7 节 R1（同仓下一 rank）处理；同仓无替补（pydicom 替补 s01 已占用为浮动），
   实为 R2 消耗 s01 或冻结调整，主智能体在 GPU 阶段裁决。
2. **t06 is_empty 语义**：buggy 下 `is_empty == (VM==0)` 已为 True；criterion 同时断言
   `VM==1` 与 `is_empty is True`，两者联立只在“VM 修 1 且 is_empty 加 SQ 特判”时通过。
   若参考补丁只修 VM 不碰 is_empty，V3 失败 → 届时放宽为只断言 VM（记录为适配行为）。
3. **replay 应用漂移**：pr_402/pr_400 的 op0 `old_str` 行首多一个空行（teacher 环境文件
   与 staged 快照的空白差异，已逐个验证 `lstrip` 后匹配；`apply_warning` 已写入对应 replay）。
   GPU 回放：先精确匹配，失败则做首尾空行容忍匹配；仍失败则该题 replay 不可用，
   改用官方 loader + 参考补丁做 V3（replay 只是校准手段，非门控）。
4. **pr_2069（16 ops，含 dataset.py 173→915→173 的写入又回滚）**：顺序回放终态自洽，
   但中间态依赖 teacher 环境；GPU 若某步 old 不匹配，从该步起改用“终态 diff 语义”核对
   （以 V3 参考通过为准，不纠结 replay 逐字复刻）。
5. **pydicom 全题 CPU 未执行**：本机 Python 3.9 vs 快照 `requires-python >= 3.10`
  （`X | Y` 注解求值失败），8 个 pydicom criterion 在 CPU 只能做 `py_compile`；
   V2/V3 必须在 GPU Linux + py≥3.10 + numpy 全量 env 实测。click/t01 与 markupsafe
   t02/t03/t05 已在 CPU 对 buggy 快照手动验证失败方向（见 §5）。
6. **t07 在线数据**：criterion 只用内联 JSON（InlineBinary），不触网；但
   `filespec_parser`（t08）与 `get_testdata_file` 在 GPU 离线环境若尝试下载会挂起——
   t08 用名不在 url map（`data_path_with_download` 不触发），s01 传绝对路径在 glob 前即
   被 ValueError 拦截，均无下载路径；GPU 仍建议断网或 `--offline` 复核一次。

## 5. CPU 已验证（一次性手动命令，非套件）

- `py_compile`：13 个 criterion 文件全通过。
- buggy 快照失败方向实测：t01（`init-data-command` 保留后缀）、t03（`'x '` 多一空格）、
  t05（`NameError: escape`）均如预期失败；t02/t04 快照 import 即 NameError（与 issue 一致）。
- replay：13 个 JSON 共 52 个源码 ops；每题 op0（及无依赖 op）在快照中精确或
  首空行容忍匹配，路径映射 `/testbed/...` → 快照相对路径已核对。

## 6. 主要风险

- t12 若 V3 不通过且无同仓替补，12 题冻结可能被迫动用 s01 顶替（配额变为 1/3/8），
  或触发重冻结讨论——这是本轮 criterion 唯一的单点风险，主智能体需在 GPU 阶段优先验证 t12。
- teacher- resolved 偏差（P007 已承认）同样适用于 replay：replay 只是“教师曾走通的编辑”，
  不是正确性证明；唯一裁判是 GPU V2/V3。
