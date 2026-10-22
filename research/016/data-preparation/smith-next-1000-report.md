# P016：Smith新增1000题实际供给

2026-10-22；在根负责人授权后，已从既有8份固定版本parquet实际抽取 **1,000个新增独立train问题、70仓、111,784,674 bytes**。本次生产命令墙钟 **3.0755秒**；它不含历史下载、元数据准备、编码或训练时间。零外网、零GPU、没有重放上游代码，也未读取final patch列。

身份先冻结在 [smith-next-1000-identity-freeze.json](smith-next-1000-identity-freeze.json)，再按物理行号、instance_id、traj_id三重关联只返回指定train正文。源revision为 `08e109b4a59eaeebf80e4675cd125d42e7ac99a4`；本次重新核实8分片的实际字节数和固定SHA256全部一致。

原件独立写在 ignored `research/016/data/source-rows-next1000/<instance_id>.json`，未更改原160题正文或冻结身份。逐条来源、路径、SHA、教师、上游resolved标签、确切读取行和本地元数据可用性都在 [smith-next-1000-acquisition.json](smith-next-1000-acquisition.json)。所有1,000个canonical key唯一，和原160个零重叠。500题为PR-ID、500题为命名变异；这个比例是本批分层设计，不是已证实的最优训练混合。

源消息仍是教师轨迹，**不等于固定六工具协议，不等于本地修复成功，不等于已经训练**。后继工作由 `research/016/encoding/` 生产完整原生三工具Gemma SFT分片、屏蔽明确错误动作、排除明显异常，并报告实际16K/32K独立问题和监督tokens；不使用原XML长度代替精确编码长度。

[smith-cached-task-provenance.json](smith-cached-task-provenance.json) 复用既有train研究的760个精确任务记录及许可证文件依据。当前1000题中440题能直接附完整buggy执行commit和仓库许可证据，其余560题保持null。这里的buggy commit来自以前精确任务分支的真实git ls-remote，不是把ID里的8位仓库快照扩写。**本轮离线SFT按固定dataset revision＋instance/traj＋raw SHA即可追溯；完整Git SHA在重放时再补，不作为本批新增开训门槛。** 数据卡MIT和已有许可来源保留，不建立逐仓法律门控。

可重复生产命令：

```sh
python3 research/016/data-preparation/materialize_smith_batch.py
```

入口接受 `--freeze`、`--output`、`--manifest`；现有缓存逐条比对，不覆盖不同正文。实际读入严格受显式train用途、70仓allowlist、23仓denylist、当前160身份排除、dataset revision与分片SHA限制。未新增测试文件或检查体系；仅做本轮输入身份、来源和产物数量的手动核对。
