# 第二批七道同仓候补材料

身份首先冻结于上级`replacement-identity-freeze-v2.json`，随后才获取正文。全部来自原121身份池的预定同仓rank，按repo/base commit与原24和第一批RDF候补去重；没有新选仓库。D的首个结果在本次身份冻结之后出现，不改变这些身份或环境规则；C模型评估尚未执行。

| 原题 → 候补 | 同仓rank | 来源F2P/P2P节点数 |
|---|---:|---:|
| Pint1248 → 877 | 4 | 1 / 38 |
| Pint1039 → 1471 | 5 | 1 / 10 |
| jsonargparse560 → 564 | 4 | 1 / 10 |
| jsonargparse424 → 592 | 5 | 1 / 115 |
| jsonargparse242 → 248 | 6 | 1 / 60 |
| soupsieve50 → 88 | 4 | 2 / 34 |
| structlog117 → 669 | 4 | 1 / 64 |

沿用既有严格Parquet任务列投影和精确base Git快照流程，只改变一次性调用中的已冻结候补集合与独立输出目录；不读取teacher messages/meta/hints。`source-acquisition.json`与`snapshot-acquisition.json`保存真实来源、bytes/SHA；源checkout的HEAD精确一致且工作树干净。七份snapshot共5,586,914 bytes，原两份RDF共13,250,013 bytes。

九道候补的统一CPU装配入口为上级`replacement-material-index-v2.json`，按身份指向本目录或保持不变的`replacements-v1/`。Agent题面独立于host-source中的参考patch及原判据。`environment-plans.json`只描述拟运行环境，`valid_prepost`仍为null；未执行CPU判据、模型评估或GPU，也未上传/push。

已知静态限制：Pint1471的来源F2P及多项P2P参数ID在空格处截断，部分P2P前缀对应多个参数，不能自动补猜、删除或扩张；应保留真实CPU失败，除非另有明确来源映射证据。其pyproject要求setuptools_scm，Structlog669要求hatchling/hatch-vcs/hatch-fancy-pypi-readme；原声明已附在环境计划，CPU装配者需显式记录补充构建依赖。Soup88来源Python3.6，兼容性仍需实际运行。以上是待实测限制，不是通过或失败的预断。
