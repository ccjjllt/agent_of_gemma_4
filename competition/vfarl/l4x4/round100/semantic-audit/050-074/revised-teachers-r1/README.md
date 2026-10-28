# 十题r2新题面的真实teacher

2026-10-28。仅固定100原索引51/52/53/54/55/66/68/69/70/74。目标task逐一使用question-revision-r2真实task-v2及其新prompt，保留原ID/family/snapshot与原F2P/P2P。这里没有改v4原件、旧teacher、旧奖励、S600/F600 manifest或全局catalog/运行脚本。

十题均实际调用官方六工具，分别在原source baseline与生产修复后完成完整原官方文件范围，取得本次0→1。所有动作、messages及独立官方结果留在各case目录。52/53/66/68/69/70/74原recipe动作原样重放。51/54/55发现旧recipe改前仅import/来源检查，故另外建立recipe-causal.json，只将首个命令替换为原改后同一真实API断言并预期失败；其余动作不变。最终这三题只选actual-run-causal，早先导入检查的真实运行仍保留但未进入当前导出。最终十题均有实际来源绑定的before API失败及after API断言通过，生产patch字节与历史相同。

原source-snapshot在harness.setup后保存，可能多出conftest.py/pytest.ini且tar extraction会改变mode。此次仅按旧source-inventory恢复每个原文件/symlink及mode，逐个SHA核对，十个source_tree_sha与原pin完全相同。原budget数值及类型也保留：六题int60，四个Funcy题float30.0；源布局按原设置（两个APISpec使用src）。实际新initial_prompt逐字SHA匹配r2 selection，并与rawmessages首user、agent-prompt原bytes及官方builder输出一致。input-bindings.json和各input-binding.json给出新旧task/prompt与来源绑定。

共享task纯依赖profile仅来自既有jiyujia本地缓存，真实版本face24.0.0/boltons24.1.0/beautifulsoup4 4.15.0/soupsieve3.0.4/whatever0.7及原dist-info，来源成员/文件hash在dependency-source-manifest.json。官方host/mac原生库使用既有deepdiff Python3.12缓存，未下载或覆盖原环境。实际新profile没有追认为删除的旧profile或Docker等价。

51首次完整官方CLI验收因缺安装的glom命令失败。原setup.py明确console_scripts glom = glom.cli:console_main；按此恢复标准入口并用run_official原有sandbox_executable映射，得到后续真实0→1。入口只调用实际workspace包，未替换实现、测试或官方标志；真实来源/内容hash见runtime-entrypoint-glom.json，首轮失败保留。

独立source-catalog只保留原十条真实source/review pin，并显式披露original/curated statement SHA及新文本。raw reports只写本次真实运行和review指针，没有复制旧成功或旧token轨迹。复用未改的normalize_results.normalize严格准入（十条eligible/reasons为空）、export_prompts和format_replays。没有新增测试文件、GPU、网络提交、Gemma采样或训练更新。原官方节点通过只说明验收事实，未声称模型独立修复收益。

实际可消费增量与逐episode来源位于independent-sft-r1；其export-bindings.json只绑定最终十条，其中51/54/55明确指向actual-run-causal。
