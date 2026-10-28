# 索引95：ParseError行号的单点值oracle盲区

2026-10-28。仅固定100 train索引95，`pytest-dev__iniconfig.16793ead.func_basic__vb9u5hga`。本目录是 **oracle弱点探针，禁止训练消费**；没有teacher/SFT/canonical或奖励/名单修改。受控错误patch仅在报告指定/private/tmp路径。

原/当前题面明确：内部lineno按0起始，格式化消息应显示lineno+1。不是只要求第一行显示1，也不是新增相邻字段。原F2P `test_ParseError`有assert，但只用ParseError("filename",0,"hello")这一点；其余P2P对错误字符串只要求包含duplicate section/name等msg，不检查显示行号。源码保留path/lineno/msg，正常__str__应使用内部line+1。

三次有效的未改动官方verify_task调用使用原完整testing/test_iniconfig.py、28要求节点，参数展开42个实际用例：

|生产变体|完整42例|官方resolved|
|---|---:|---|
|原lineno−1 mutant|1失败／41通过|False|
|path/msg动态但显示行号恒1|42通过|True|
|原lineno+1 reference|42通过|True|

实际源绑定公开API观察：lineno0/7/13，坏实现都显示sample.ini:1: message；reference分别显示1/8/14，原mutant显示−1/6/12。此反例位于原行号主契约内，不依赖相邻新增需求。不是以无assert数量断定弱oracle；这里有assert，完整原oracle仍接纳了明显错误的恒行号。

有效结果及原JUnit见[runtime-r2/oracle-observations.json](runtime-r2/oracle-observations.json)。原/当前题面、源/任务/patch哈希及实际import路径、三态witness见[oracle-observations.json](oracle-observations.json)。第一轮极小依赖profile漏pytest自身py.py shim，三态都在导入前失败、0 JUnit；这些是环境失败而非奖励0，原始记录未覆盖。只从现有官方host补该真实文件，所有生产变体和oracle不变，再实际从头运行三态。

建议在新可信奖励协议形成前暂停95未来基于原oracle的RL奖励；保留正确SFT、历史真实结果与当前固定题。此处没有修改eligibility，不反向改写历史奖励。没有GPU、训练、新测试文件、网络下载或错误示范导出；macOS CPU结果不主张Kaggle/Linux等价。
