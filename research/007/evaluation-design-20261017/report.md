# P007：先选题，再看模型结果

日期：2026-10-17。只读取已有划分、轨迹文件名和外部分片的来源列，没有读取新任务正文、参考补丁、heldout或预留题内容，没有执行模型或裁判。以下是已冻结的候选身份，不是已经可运行或取得成绩的评测集。

## 公共开发12题

从`research/task_splits.json`的76个dev_future身份出发，排除28题heldout与原预留8题对应的整个repo+commit组，再排除已有归档trace文件名对应的身份。剩73个候选：FastAPI41、Rich28、Requests4。以种子`p007-20261017`的SHA256排序，每repo先取4个，得到12个问题、12个不同快照。

| 仓库 | 冻结问题ID |
|---|---|
| Textualize/rich | rich_4006、rich_3067、rich_3180、rich_3472 |
| fastapi/fastapi | fastapi_14482、fastapi_14851、fastapi_14463、fastapi_14459 |
| psf/requests | requests_7205、requests_6757、requests_7427、requests_7502 |

[身份、完整commit、选择规则与来源SHA](public-dev12-manifest.json)。只读身份不等于题目内容被用于调参；没有旧trace文件也不能证明任何人或预训练模型从未见过该问题。这个面板用于公共开发比较，不是跨仓库或私榜泛化证据。未读取tasks.jsonl来填内容，也未宣称v31下所有任务环境可运行；后续只定向物化已选择题，缺环境的原因必须与模型失败分别保存，不能按候选成败换题。

## 外部8仓库候选

复用固定SWE-smith分片`tool-00004-of-00008.parquet`，revision `08e109b4a59eaeebf80e4675cd125d42e7ac99a4`，SHA256 `e41ffbd3f9d1571917ac5b9f4d0108ded4799d9776f597a054203467135a5715`。

只投影`file_row_number, instance_id, resolved, model, traj_id`，不读取messages/patch。排除比赛四repo以及10月11日已经抽读的20个repo；在剩余来源中仅保留真实PR身份（`.pr_`）及上游resolved=true。实得**184条来源记录、183个不同PR、36个仓库**。这是来源池统计，不是184条已转换训练数据，也不是本项目183次修复。

先按同一固定种子排序repo，再在每repo排序PR/trajectory，取8个仓库各一个PR。未依据任务长度、修复内容或我们模型是否成功挑选。

| 仓库 | PR身份 |
|---|---|
| tkrajina/gpxpy | tkrajina__gpxpy.09fc46b3.pr_282 |
| kurtmckee/feedparser | kurtmckee__feedparser.cad965a3.pr_368 |
| pandas-dev/pandas | pandas-dev__pandas.95280573.pr_56522 |
| keleshev/schema | keleshev__schema.24a30457.pr_330 |
| cookiecutter/cookiecutter | cookiecutter__cookiecutter.b4451231.pr_1920 |
| conan-io/conan | conan-io__conan.86f29e13.pr_17531 |
| modin-project/modin | modin-project__modin.8c7799fd.pr_7189 |
| pytest-dev/iniconfig | pytest-dev__iniconfig.16793ead.pr_49 |

[行号、来源教师、顺序和替补规则](external-pr8-manifest.json)。八个仓库整体保留作外部选择候选，不进入后续训练；若因源码或环境无法物化，先记录原因，再按已冻结替补顺序替换，同时排除替补repo的训练数据。不得因为模型失败而换题。按上游resolved选样带有可解性选择偏差，报告中必须保留；它不是对任意私有仓库的随机样本。

`instance_id`里的短commit是profile身份，**不能直接作为buggy commit**。后续仍需按官方SWE-smith loader解析instance分支、固定完整buggy commit、核任务来源与许可证、构建真实环境。上游轨迹resolved不是我们的裁判。Pandas/Conan/Modin等环境成本也未测量，不能提前承诺八题全部能离线运行。

训练数据继续遵守P006用途：click只作容量训练，pydicom只作诊断，MarkupSafe仍candidate。本轮没有重分配它们，也没有读取新八题的正确解来制作训练目标。其他28个新PR来源repo只是扩量候选，尚未授权为自动训练输入。

## 下一轮如何使用

公共12题先比较原样0.22强参考与一个明确机制变化，模型/采样/八分钟预算保持一致。外部候选在CPU环境准备完成后再做跨仓库比较；两类结果分别报告，不合并成一个看似更大的泛化分数。原28题heldout、旧预留8题及同快照仍保留，当前没有因“扩大验证”打开它们。

本次CPU读取使用临时`/private/tmp/p007-parquet-reader`中的DuckDB1.4.3（下载13.1MiB），未改项目训练环境。筛选为一次性Shell/Python命令，未新增测试、检查脚本或GPU任务。
