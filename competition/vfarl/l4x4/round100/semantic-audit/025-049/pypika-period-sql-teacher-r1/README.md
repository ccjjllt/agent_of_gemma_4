# PyPika 40：题面导入修正与真实 SQL 示范

原题把 `SYSTEM_TIME` 从 `pypika.terms` 导入；同snapshot实际只在 `pypika/__init__.py` 顶层导出实例。初次直接照题面做的 before/after 都发生 ImportError，修复生产源码也不能令该示例运行。原失败实际记录保留，不导出为有效训练材料。

`question-revision-r3` 仅将该导入改成 `from pypika import Query, Table, SYSTEM_TIME`。新题面实际运行两条公开API：baseline的DELETE和UPDATE都观察到目标AttributeError；应用与历史逐字相同的原生产补丁后，两条SQL精确等于题面Expected，同时保留原表不变与重复for_portion拒绝的行为观察。原完整40文件／1061用例oracle baseline2失败→修后全过，严格required IDs不变。

唯一消费来源是 `actual-run-current-prompt`：90源文件的内容和mode与原source-inventory完全相同，tree SHA f40ca8e8…，实际初始prompt SHA908b9b52…与新版question逐字对应。原archive额外包含不在pre-run inventory的conftest.py/pytest.ini，tar data过滤也改变group-write位；仅在私有临时提取树恢复原inventory后独立重跑，记录见source-reconstruction.json。前一个corrected-import实际0/1仍原样保存，但因60.0/60预算文本及源inventory差异不准入、不追标。

strict canonical及实际CT action-view共8决策（7工具+完整终止）、1317监督tokens、最大上下文4353，无截断。所有轨迹是Codex规定并通过真实六工具执行的离策略示范，没有Gemma行为／logprob／思维／GPU或训练收益。主100旧版本未覆盖，后续聚合须绑定r3选择及本目录export-bindings.json。Linux/chroot适用性仍需独立实际证据。
