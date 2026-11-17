# 012 run 中 adapter 保留（in-run retention）

状态：2026-11-17。实现与本地合成目录验证；未启动 GPU，未修改服务器上的真实 run。

## 1. 问题

SPO 每个优化步发布一个不可变 adapter（rank 128 实测约 1.94 GB），导出器还会在
`adapters/staging-vN` 留一份同样大小的候选副本；`DoubleBuffer` 发布后只推进
`registry.json` 的 active 指针，旧版本目录不会自动删除。`prune_finished_run`
只在整轮 run 结束、所有 update 均为 `CONSUMED` 后才运行，因此五步 run 在
运行期间会同时保留 v0…v5 六份 adapter 加 staging，数据盘未到收尾就可能耗尽。

更严重的是旧收尾逻辑：它删除版本目录但保留 registry 行，且不写任何“已删除”
证据。之后打开 `AdapterManager` 直接失败：

```
ValueError: registered adapter directory is missing:
  .../adapters/versions/00000000-b70fe5a72f418f8c
```

`spo-v3-5step-r128-20261117` 的真实 registry 仍是 v0–v5，收尾保留了 v3/v5，
删除了 v0/v1/v2/v4，正好落入这个状态。

## 2. 配置接口

`[run]` 下三个键，`load_config` 在启动前完成类型/符号校验：

| 键 | 类型 | 默认 | 含义 |
| --- | --- | --- | --- |
| `in_run_prune` | bool | `false` | 每个 update 提交并开始服务后执行一次清理 |
| `keep_adapter_versions` | list[int >= 0] | `[]` | 永不删除的版本；不允许重复 |
| `keep_recent_adapters` | int >= 0 | `1` | active 之前保留的最近版本数（回滚安全） |

校验规则（`training/run_spo.py:load_config`）：bool 不能当 int 用（`type(x) is
int`），列表元素必须是非负整数，不得重复，`keep_recent_adapters` 不得为负/浮点/
bool。

新配置 `configs/spo_v3_long_rank128_20261118.toml` 置为：

```toml
in_run_prune = true
keep_adapter_versions = [3, 6]
keep_recent_adapters = 1
```

已完成的 `configs/spo_v3_5step_rank128_20261117.toml` 保持原样（SHA256
`f18d9721…`），不把新键回填进历史 run 的配置记录；下面的五步表只说明语义，
不是对该历史 run 的追溯修改。

以 `keep_adapter_versions = [3, 5]`、`keep_recent_adapters = 1` 为例，五个版本的预期驻留（v0 为 genesis，保守规则始终保留）：

| active | 保留 | 本步删除 |
| --- | --- | --- |
| v1 | v0, v1 | – |
| v2 | v0, v1, v2 | – |
| v3 | v0, v2, v3 | v1 |
| v4 | v0, v3, v4 | v2 |
| v5 | v0, v3, v4, v5 | – |

运行期步间本地盘从约 11.6 GB（6 个版本）降到约 7.8 GB（4 个版本）；导出下一步
候选时会再叠加 1 个 1.94 GB staging，峰值约 9.7 GB。收尾
`prune_finished_run` 继续带 `keep_adapter_versions`，保留 v3/v5，删除 v0/v4、
checkpoint 与瞬态目录。

## 3. 清理时机与语义

Hook 位于 `run_spo.activate_and_commit()` 的 post-commit 一致性检查之后、
`return manifest.version` 之前（`spo_update` 事件之后）：

- ledger 已 `CONSUMED`，trainer、registry、DoubleBuffer、endpoint 已互相校验；
- 不在任何 `lease.training()` 内（`do_update` 和三条 resume 路径都在训练租约
  释放后调用）；
- 只做文件系统操作，不改 ledger/registry 行，不加载/卸载 GPU adapter。

删除规则（任一不满足就跳过并记录原因）：

1. 只删有“producing update 且 phase == CONSUMED”的版本；v0 没有 update 行，
   作为无法证明来源的 genesis 保留。
2. 保留 active、`keep_adapter_versions`、active 之前 `keep_recent_adapters` 个版本、
   `DoubleBuffer.ownership()` 中仍被轨迹/rollout/training slot 引用的版本。
3. 保留 registry `references > 0` 的版本，以及任何 ledger phase 非 `CONSUMED`
   的版本（EXPORTED/REGISTERED/ACTIVATED 候选）。
4. 目录内 `manifest.json` 必须与 registry 行完全一致；缺失/不匹配则跳过。
5. 删除前把 `{version, adapter_sha256, bytes, pruned_unix}` 原子写入
   `adapters/pruned_versions.json`。一次写入覆盖所有计划删除的版本，然后才
   `shutil.rmtree`。崩溃只可能留下“目录还在”或“目录没了但 marker 在”，不会
   出现无证据的缺失目录。
6. staging：`staging-genesis` 在 v0 已发布后删除；`staging-vN` 仅在其 update
   为 `CONSUMED`（含 active）时删除；pending update 的 staging 保留；无法识别
   的名字一律保留。

清理失败（包括 marker 写失败、单个 `rmtree` 失败）只写入 metrics 的
`spo_in_run_pruned`（`outcome=skipped/failed`、错误类型、skip 原因），不会把已
成功的 update 或整个 run 变成失败。

事件示例字段：`update_id, outcome, active_version, removed_count,
removed_bytes, removed_versions, removed_staging, kept_versions, skipped,
failed`。

## 4. marker 与 registry 读取

`adapters/pruned_versions.json`：

```json
{
  "schema": "gemma.spo_pruned_versions.v1",
  "versions": {
    "2": {
      "version": 2,
      "adapter_sha256": "…64 hex…",
      "bytes": 1940000000,
      "pruned_unix": 1794927029.6
    }
  }
}
```

`AdapterManager._read_registry` 对缺失版本目录的容忍顺序：

1. 新 marker 中存在同 version **且** 同 `adapter_sha256`；
2. 否则读取 run 目录的旧 `pruned_artifacts.json`
   （`gemma.spo_pruned_artifacts.v1`），要求 label 为 `adapter vN`、记录的
   `path` 解析后正好是 `versions/NNNNNNNN-<sha16>`，若记录带 sha 还必须一致。
   接受后把证据放入内存中的 pruned 表（不改写旧文件）。
3. 两者都没有、sha 不符、路径不符或 marker 损坏 → 仍然抛
   `registered adapter directory is missing` / `pruned versions marker …`。

新增只读接口：`AdapterManager.is_pruned(version)` 与
`AdapterManager.pruned_versions()`（int 键，含来源与字节数）。registry 行、
manifest、血缘继续保留；`publish`、`activate`、`acquire`、`release`、`active`
和 resume 对存活版本行为不变。`prune_finished_run` 现在也先写同一 marker 再删
目录，因此以后不再产生“删目录无证据”的 run。

另一个容易丢证据的路径也已处理：旧 run 再次执行 `prune_finished_run` 时，计划
里往往只剩 checkpoint/瞬态目录，但该函数会重写 `pruned_artifacts.json`。现在
它先把仍被旧记录覆盖、且目录确实缺失的版本迁移进新 marker，再重写旧记录；
实测二次收尾后 marker 仍含 v0/v1/v2/v4，`AdapterManager` 仍可打开。

## 5. 本地验证（合成目录，非服务器实跑）

验证均为一次性 inline Python，不新增测试文件。构造 v0…v4、ledger 全
`CONSUMED`、checkpoint 在 run 目录外：

- `active=4, keep=[1], keep_recent=0` → 删除 v2/v3，保留 v0/v1/v4，marker 写入
  v2/v3 与字节数；`keep_recent=1`（配置默认语义）则只删 v2、保留 v3。
- 重新打开 `AdapterManager`：`active()==4`，`is_pruned(2)==True`，v1/v4 路径
  存在；publish/acquire/activate v5 成功，再次打开 active 为 5。
- 无 marker 删除目录 → 抛 `registered adapter directory is missing`。
- 篡改 marker 中 v2 的 sha → 同样抛出。
- pending `EXPORTED` 的 v3：版本目录与 `staging-v3` 都保留，consumed 的
  `staging-v1/v2` 删除。
- 模拟 `rmtree` 在 v2 失败：marker 已含 v2/v3，v2 目录仍在，registry 仍可打开。
- `prune_finished_run` 删除 v0/v1 前写 marker，之后 `AdapterManager` 打开
  active=v2，`pruned_versions()` 为 {0,1}。

真实 run 结构的副本（v0–v5、仅 v3/v5 目录存在、旧 `pruned_artifacts.json` 记录
v0/v1/v2/v4）验证：`AdapterManager` 可打开，active=5，v3/v5 路径存在，
`pruned_versions()` 为 {0,1,2,4}，publish/acquire v6 成功；从记录中删掉 v4 后
重新打开则按预期抛出 missing directory。

2026-11-18 在服务器真实 run `spo-v3-5step-r128-20261117` 上只读重开
（`AdapterManager(run/adapters)`，未写 registry、未删除任何文件）：

```
OPEN_OK active 5 pruned [0, 1, 2, 4]
v3 True v5 True
v3 sha 10c1a80f3a4d71b9 v5 sha 2ce46040af3eed69
is_pruned 0,1,2,4: [True, True, True, True]
```

这证明旧收尾留下的失败状态已被兼容修复；评估 v3/v5 不再需要重建 registry。

## 6. 边界与风险

- 服务器真实 `pruned_artifacts.json` 若已被手工删除，旧 run 仍会按“无证据”
  失败——这是有意的 fail-closed。
- 被 in-run 清理的旧版本只剩 manifest/血缘，不能再次加载权重；需要历史版本
  做评测时，应把版本加入 `keep_adapter_versions`。
- 清理发生在 update 提交后的主线程；大目录 `rmtree` 会占用数秒磁盘 IO，但不
  持有训练租约，也不影响已加载的 serving adapter。
- `keep_recent_adapters` 只保证 active 之前 N 个版本可回滚；更早版本回滚需要
  显式 keep 或从共享盘备份恢复。
- 旧 `pruned_artifacts.json` 没有 sha 字段，兼容读取只能校验 label/version/
  目录名；带 sha 的新 marker 才做完整身份校验。
