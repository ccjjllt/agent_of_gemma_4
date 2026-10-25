# F 流转与状态复核与补充（2026-10-26）

本批最终入口为 **catalog-v4.json**、**reviewed-f-tasks-v4.json**、**canonical-manifest-v4.json**；v1/v2/v3 和旧采集目录保留为历史证据。当前为 50 个可供 F 专科选择的独立题目，最终 F600 是否录用由全局课程清单决定，不能把本批与其他批次简单相加。

| 实际主机制 | 题数 |
| --- | ---: |
| F1 配置与参数传播 | 10 |
| F2 共享引用与分支隔离 | 3 |
| F3 缓存与重入一致性 | 1 |
| F4 资源与临时状态清理 | 3 |
| F5 事件与协议阶段 | 33 |

27 个历史身份重新核对主机制后保留，新增 23 个真实修复身份：7 个 Schedule、1 个 OAuth、4 个 Bottle、3 个 Tenacity、8 个 h11。原生任务标识、题面和全部 F2P/P2P 要求保留；50/50 完成实际官方六工具子集重放，修复前原 F2P 全部实际失败，修复后原官方匹配器全部接受。来源为发布者固定 revision 的 train：45 个 SWE-smith-py、4 个 SWE-rebench-V2、1 个 SWE-Gym。四个比赛仓库未入池，未读取 heldout 或 129 题正文。

分类以实际补丁因果为依据。SQL 构造、字符串距离、区间合并等任务不能因为变异方式是语句重排，或重现代码连续调用几次，就被称为 F5。旧 92 条记录的逐条修订与重复排除见 reviewed-existing-f-tasks-v1.json；能成立的 F1/F2/F3/F4 保留，算法和运行时协议任务移出本批 F 专科建议。

scope-audit-v4.json 逐题从真实 source-snapshot.tgz 重建 submitted.patch，解析修前/修后实际代码作用域，并核对完整修后文本与不可变 Smith upstream 归档或原发布者修复方向补丁一致（含末尾换行）。50 身份覆盖 52 个实际代码作用域，与 400 共同课程及本批其他身份没有函数重合。只比较 family 字符串不足以去重。

修复的实物问题：

- ChunkedUpdate.flush、NestedViewSetMixin.initialize_request、TornadoRetrying.__call__ 的旧示范虽然完整验收通过，但实际 edit_file 序列漏了 upstream 的末尾换行，与旧采集器的整文件一致性声称不符。已保留旧 attempt，写独立 recipe 并真实重放修正；每个身份仍只计一次。collect_f5.py 和 collect_fresh.py 现对最终实际编辑结果做精确比较，以唯一尾部锚点补齐换行差异。
- collect_fresh.py 原先用行内容集合差判断空白变更，不能区分真实重排与纯空行 hunk。改为 SequenceMatcher 的实际变化块；纯空白 hunk 才豁免作用域边界，完整文件比较仍保留。
- h11 attempt001 的两个原生回环 HTTP 项因受限宿主禁止 bind 而失败；保留该失败，在允许 127.0.0.1 临时服务的宿主权限下按原要求运行 attempt002，8 题各 78/78 原生项通过，未改测试或匹配器。

两道 async-timeout top-level 构造题虽然实际修复有效，但当前官方前缀匹配器会使 required test_timeout 与原生必跳案例冲突；timeout_at 另有原 F2P 标签与当前执行差异。均保留真实失败并排除，不增加本批题数。未借助删除、改名原生测试或缩小验证范围绕过。

示范是 Codex 编排的真实 off-policy 冷启动工具轨迹，没有 Gemma 模型采样、编造思考、logprob 或已验证同状态关键动作对。GPU 训练未启动，macOS 本地 SubprocessManager 验收不等于比赛 Docker/Linux 环境等价。原 thinking=0 冷启动前缀仍按实际执行记录保留。

独立复核 root 新增 Arrow/Soup/Schema 10 题的结果见 root-f1-cross-review-v1.json，记录了可支持的 F1 调用边及未独立隔离的 flags、异常分支等边界。.iselect/.select 是不同函数但同一调用链上的邻接 limit 传播能力，不能称作两个新机制类型。
