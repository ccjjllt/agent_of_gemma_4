# Bottle Route.reset no-op 候选：关闭

固定100索引5的自然过简化候选把 Route.reset 改成 pass：它保留 plugin_data，但也保留已编译 callback。实际六工具 edit_file/run_command/submit_patch 和同一原完整365节点官方 verify_task 得到：

|状态|原完整oracle|直接行为|
|---|---|---|
|原mutation|0，5失败/360通过|原cache/route状态被清空（原失败保留于baseline报告）|
|错误 no-op|0，3失败/362通过|prepare→reset 后仍有 cached call；第二次call仍使用第1次插件apply结果|
|正确 pop(call)|1，365通过|reset移除 cached call；第二次call重新apply，次数1→2，plugin_data保留|

原 test/__init__.py:13 启用 bottle.debug(True)。Bottle.reset 在DEBUG下会prepare routes，随后安装插件触发reset；因此原 TestPluginAPI.test_apply、test_callable、test_instance_method_wrapper 三个原节点能检测no-op后的陈旧callback。初看没有直接reset断言，并不意味着oracle没有覆盖其效果。此候选关闭，不新增弱oracle结论或RL暂停建议。

两条件各自从同一原source inventory逐字/逐mode恢复，实际run输出的inventory与原件完全相同。未恢复任何pyc；不同fresh workspace/工具进程/验证sandbox；两repair源码尺寸不同，直接命令断言导入workspace并打印sourceSHA。旧bytecode/cache未删除或修改。证据与原fullscope baseline/final/JUnit、actual tool events、patch SHA在 evidence.json。wrong-noop-r2中的r2仅表示补齐生产入口必需command hash后的第二次尝试；首次缺hash在任何工具/oracle前被拒，失败recipe/log仍保留。

这是Codex脚本离策略机制实验，非Gemma/human行为；producer兼容旧evidence_kind枚举但recipe另写明Codex/非训练，不做canonical或SFT准入。CLI实际budget为60.0 float，当前teacher保留int60，故这次probe promptSHA690d4a…与当前teacher4281c1…不同；原task/source/原F2P+P2P严格一致。本实验仅回答原奖励能否抓住no-op，不冒充当前prompt示范、模型收益或Linux/chroot验收。

没有修改主源码、task、oracle、teacher、历史reward、selection、当前SFT或curriculum-r5；没有运行模型/GPU/heldout或新增测试文件。
