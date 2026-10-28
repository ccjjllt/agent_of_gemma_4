# r2 curriculum 与暂停原因的独立审阅

2026-10-28。实际读取 r2 curriculum、来源元数据与当前 resident.evidence_course_order，进行了必要的一次性 CPU 函数调用；没有读取预留题正文，没有启动 GPU、模型采样或新测试文件。没有改动 curriculum、奖励或运行代码。

**结论：本轮暂停仅改变未来 RL 采集资格，固定库存和历史奖励未改。**

审阅的 curriculum SHA256 为 `fa1853a6f27f1eb7eb149ea1710c6c8ddb4ee60cbb4c612982c4d5b065e3cad9`，绑定 selection SHA256 `e665c576d29541fd65f1610157986ab4e21fa959cddbec60a1f79d8c75b80764`。

调用时 resident.py 的 SHA256 为 `247c0e75c70a6e574a1ab73a1375bbe93cf2295a26b883c771987a52f7df8dac`。

## 实际函数结果

将完整 selected 同时作为 eligible 传入真实 `evidence_course_order`，只是核对筛选/排序机制，**不表示 100 道题 CPU 通过或已可运行**。

- 返回 64 个 task ID，共同/专科各 32。相对旧 65 个新采集 ID，仅移除 OSA；没有新加入 ID。
- `training_order` 仍包含原 100 个 ID，各一次。历史 claimed 仍为 35；原 reuse candidate 和预留 ID 元数据没有变化。
- 原 runtime/scoring-unknown 四题完整保留暂停：moto-5348、moto-6159、dvc-7628、markdownify-161。加入 OSA 后暂停共五题。
- eligible 仅含五个暂停项时，实际函数返回空列表；只含一个共同和一个专科新采集项时，返回严格保留课程原相对顺序。

OSA 仍位于完整库存，`excluded_from_training=false`、`paused_scoring=true`、priority 为 paused_weak_oracle。corrected off-policy SFT manifest 仍作为独立来源引用存在，不能据它认为原 oracle 已修好。

## 版本与奖励边界

逐项比较原 course 与 r2：100 行的 observations、recovered_pair_review、resident_claimed 和历史示范长度先验均保持原值，因此其中 reward/verified 也未被改写。38 个改变 task SHA 的行全部明确记录 v4 的 historical_observation_task_input_sha256、当前 r2 problem revision 与“old observations are not new rewards”的解释；其中 16 行存在真实旧 observations。没有把这些历史过程标成新版 prompt 的示范、新版 CPU usable 或新奖励。

每行当前 task SHA、role、course、adapter、submechanism、仓库和 repair family 都与 r2 selection 相符。实际 CPU 适用性仍由独立的当前 task/snapshot/isolation 核对决定，不由上述 all-eligible 函数演练决定。

## 弱 oracle 来源与最小代码扩展

OSA 的 oracle-evidence.json 指纹 `db0011927694ab0f970f9e3917f7728b7d82d253165ea5d6ce8fa1d1004bf6e0` 与课程引用一致。已有本地证据记录 reviewed 空串语义错误实现、constant-zero 实现均在原 18 个验收用例和 unchanged required-node matcher 下通过，而语义向量分别为 `[0,0,1]`、`[0,0,0]`，正确候选为 `[5,5,1]`。该证据明确是本地 CPU，原 oracle 未改，不是 Kaggle 运行或新 rollout。这足以支持先暂停该奖励来源的未来 RL 采集，不能据此抹去旧结果或声明 SFT 也必须删除。

resident 的 paused_scoring_reasons 扩展只验证 reason 字典属于既有 paused ID、值为非空字符串，并将其写入 unscheduled 原因；实际排除仍由既有 paused 集合完成。它不重写样本、reward 或任务身份。OSA 返回的原因是 weak-original-oracle，原四题原因仍为 historical-runtime-scoring-unknown，未统一误写为模型失败。

本次对 curriculum 中显式 path/SHA 引用递归去重后，共 91 份文件实际 SHA 全部匹配，包括 parent course、历史日志/episode、38 份 revision、OSA oracle 及 corrected SFT manifest；没有递归展开预留任务内容。本次未发现需要修改的小问题。
