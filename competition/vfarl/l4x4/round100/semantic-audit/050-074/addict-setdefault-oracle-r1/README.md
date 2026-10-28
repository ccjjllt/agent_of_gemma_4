# Addict56: existing-key truthiness oracle gap

2026-10-28。固定100训练索引56，`mewwts__addict.75284f95.func_basic__9sg9rq7f`。本目录只保存原oracle弱点研究；[错误补丁](WRONG-ORACLE-PROBE-NOT-TRAINING.patch)明确不是训练示范，不进入canonical、SFT或RL输入。主题面、旧示范、奖励和课程均未修改。

题面明确要求Python标准dict契约：key已存在就返回原值，不存在才插入default。这按键的存在性判断，不按value真假判断。因此把正确修复的`if key in self`简化成`if self.get(key)`，会在已有None、0、False、空字符串、空列表或空字典时覆盖原值，是题内主契约错误。实际源绑定API观察中六类原值都被替换成`'replacement'`，正确参考保持各原对象；stdout与实际production SHA已保存。

原完整`test_addict.py`的setdefault调用只有两个函数（原test_patch行487–501），分别继承给Dict与ChildDict。simple重复读取已存在的truthy2，并首次插入空列表后append；nested首次插入空列表，没有对已有falsey值再次setdefault。因此静态阅读只形成候选，以下完整实际对照才确认缺口。

| Production variant | 原完整128项 | 未改official verify_task |
|---|---|---|
| 原mutant | 4fail /124pass | 0 |
| 原正确reviewed repair | 128pass | 1 |
| 错误truthiness简化 | 128pass | 1 |

三次均无error或skip；4 F2P+124 P2P共128个required节点原样保留，运行目标为完整原文件。使用实际未改的`verify_task`、`SubprocessManager`、baseline setup与原JUnit匹配器；每次verification JSON、stdout、完整XML和[总结果](oracle-observations.json)均保留。没有新写测试、改断言、改reward或放宽节点。

[来源与契约](source-and-contract.json)绑定当前r3 task/prompt、publisher source和完整8-file source inventory：bytes/modes精确等原`16022308f5ae…`，只从归档按原清单恢复；post-setup的conftest/pytest配置由官方setup生成。publisher row本身没有test_patch字段，本文的“原oracle”精确指当前hash-pinned adapted task里、历史reviewed teacher实际使用的完整test_patch；题面与F2P/P2P同publisher原样。首次准备在错误假设该字段存在时停止，未执行oracle；该边界已如实记录。

原`.venv-sfr`已不存在。本次直接复用现有CP312官方host依赖profile（pytest8.3.3/swegemma0.2.11等实际版本与源码SHA见总结果），任务本身仅用标准库加本地addict；没有下载或新建依赖环境，不声称与已删除profile所有字节或Kaggle/Docker等价。

现有正确teacher已经更强：其原inline程序先插入identity-bearing空列表，再对同key调用另一default并要求仍返回原对象。独立重执行该原程序，correct退出0、wrong退出1；这是原inline程序的真实CPU执行，没有重新冒称整条六工具teacher。因而无需为本候选新增或修改正确SFT。该证据支持root考虑暂停未来仅靠此原oracle的56题RL；本目录不自行改变准入或课程。
