# S600 共同课程：来源与实际边界

只准备题目和真实CPU工具数据，不启动训练。固定Smith train页面已逐ID对照身份索引，缓存1,602个原始来源行（Glom523、Dotenv141、Result64、Isodate129、PyPika745）；缓存行不等于已审阅家族或可训正例。

Result首批选了21个不同生产函数契约，已准备固定commit、原题面、源码及许可。先实际执行3题后发现：原要求包含`tests/type_checking/test_result.yml`，官方验收器在同组有Python文件时只将`.py`文件传给pytest，原4项YAML节点没有执行。修后45项Python用例通过仍不能满足原49项要求。3次真实False保留；余18题未执行，不计可用数据。未删除YAML要求、改变目标范围或修改官方验收器来取得True。

Glom首轮真实验收暴露原CLI用例需要`glom` console entry point。task profile复用已有pytest/face/boltons/attrs缓存，与现代官方父环境分开。后续运行显式映射已有的真实console脚本，脚本使用sandbox的python3并调用当前源码`glom.cli.console_main`；路径和脚本SHA由运行器记录。首轮CLI缺失的负结果保留。Path文本、items、startswith三题还有原F2P与实际基线差异，暂不扩大逐题例外清单来凑数；后续只推进原F2P均实际失败的七个契约。

共同课保留真实主机制标签；不能因为提供给S使用就把参数流转或异常处理伪标为S专科。一个任务的多个动作、重放尝试和同snapshot变异不增加独立题数。

后续实际完成：PyPika47个不同生产函数、Dotenv12、Isodate10及Glom41个合格任务。原题面、来源提交、许可、完整验证文件和原required均逐题保留。source-only Result其余18题仍未执行；Glom原3个标签噪声和FILL不完整基线不靠新例外凑数。

Glom第三批20题中2次原run001因缺失原生资产的末尾无换行，在collector拼接diff时把下一文件头并入源码。已保存旧task/prep/review/run，修复collector的标准No-newline标记，用一次性git apply核实4个还原资产逐字节SHA匹配，再以显式task-v2/run002真实重放；题面/required/原文件字节未变，2题现为178case全过。详见glom-native-asset-assembly-amendment.json。这是数据装配错误，不算模型失败/关键错误偏好对。
