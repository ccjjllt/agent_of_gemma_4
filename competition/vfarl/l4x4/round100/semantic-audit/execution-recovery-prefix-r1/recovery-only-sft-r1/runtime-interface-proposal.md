# 最小runtime policy接口提案（未实现）

历史提案保留：后续实际实现采用另一项显式策略`isolated-prefix-no-write-v1`，以空cacheprefix＋禁写避免读取旧pyc，不执行本提案的初始缓存删除。实现、真实official manager/API证据和边界见[当前策略](../../source-bytecode-profile-r1/README.md)；本文件下文不是当前运行配置，也不改变这两条旧trace的来源。

建议collector只新增一个可选策略字段：`python_import_policy: Literal["inherit", "fresh_source_no_auto_pyc_write"] = "inherit"`。默认值保持现行为。策略对象与其有效配置/hash只进运行provenance；不改变问题、工具schema、奖励或原oracle，不把此接口加入当前训练数据入口。

`fresh_source_no_auto_pyc_write`作为原子语义包含两步：source materialize后、任何预载/import前清理workspace内对应.py的生成pycache；为所有repo subprocess（phaseA/baseline/final）在真正effective extra_env注入PYTHONDONTWRITEBYTECODE=1。必须经collector自己的manager适配层传给build_sanitized_env，不能只设host os.environ；当前官方Manager.exec没有env参数。具体适配实现交root，本文没有修改main harness或复制其实现。

名称只承诺禁自动写，并不禁读已有pyc。因此初始清理与显式env缺一不可；保持只读依赖profile、字节码fixture/无源码模块不动。它也不处理同进程已import模块/对象的reload问题。建议只记录policy版本、effective env、初始化清理范围与失败时有界诊断信息，不额外堆叠逐动作门控。当前两条真实trace继续按原运行配置解释，不能追标成使用了这个尚未实现的新接口。
