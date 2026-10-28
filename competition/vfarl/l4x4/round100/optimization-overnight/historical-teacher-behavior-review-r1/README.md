# supervision-r2 的53条历史 teacher 行为审阅

范围严格取固定100消费索引中 `teacher.new_replay_in_this_optimization=false` 的53条，排除了38题面 revision、8条新 Moto 和 OSA；没有扩展任务、读取 heldout、重跑这些旧 oracle、创建依赖环境或修改旧数据。

未发现旧8Moto那种“只import/compile且前后都ok”的明确候选。53条实际 commands/recipe/messages/tool-events 对得上，均有具体应用行为／相关异常的 baseline error 和 after ok。失败端 stdout 存在 `details.stdout`，顶层没有 stdout 不能被视为空观察；仅 print 而已输出明确错误值的观察也不能一概判坏。这个结论不等于每篇题面和每条契约都已完整覆盖。

三个最值得区分的窄覆盖点：

- **索引4 Bottle**：旧命令仅观察已有文件 overwrite=False，未观察题面 directory destination 主分支，而 patch 同时修改目录拼接。这是明确缺口，已按单独授权完成 `semantic-audit/000-024/bottle-upload-directory-teacher-r1` 的 fresh teacher，两分支均有前后实际观察，原 oracle／patch／prompt 保持。
- **索引40 PyPika**：旧命令真实调用 for_portion，观测状态通道及重复调用守卫；没有观察题面的 DELETE/UPDATE SQL。它有真实复现，不是无效示范。root 已独立接手该题公共 SQL／题面 API 后续，未由本审阅改写。
- **索引16 Moto4924**：旧命令真实构造 ListenerOrBalancerMissingError，baseline 正是缺 arn 的 TypeError，after 是正确验证 XML/message；没有走 boto describe_listeners()/ClientError。属于可选更外层 API 覆盖，不是 import-only，也未启动新 replay。

`evidence.json` 绑定原 task／问题 SHA、实际 recipe/messages/events SHA；逐条保留 command SHA、真实 status、stdout位置／SHA和可见行为值，并为以上三例保留完整原 command/stdout。原始消费索引与 selection 都有 SHA；这是 r2 冻结范围的审阅，后续新问题版本与新 teacher 必须单独消费，不能把旧日志改名。
