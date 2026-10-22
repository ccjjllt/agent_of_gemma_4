# P006 E2：历史 exec_closure 与 Budget-Fit 五分钟参考

运行载体已冻结，CPU准备V1成功，GPU对照V1已完成12/12次：A4/6、B5/6、wall35.58分钟，结果见[实际运行记录](../../research/006/reference-pair-20261013/results.md)。它是整体方案比较，不是提示单变量：两臂均5分钟Agent预算，但工具集合、调用上限、命令时限、输出预留按各自来源保持。

| 项目 | A：历史 exec_closure | B：Budget-Fit V5 matched-five-minute |
|---|---|---|
| 来源 | 10月11日已完成12次历史实验中的冻结ZIP，SHA256 `a377164fb6dd9b3a8397c9a93d27e5fbaeaa5ac07133d1770529ccca0ee38b0e` | Wilmer E. Henao 的 [V5 / 355310852](https://www.kaggle.com/code/verracodeguacas/gemma-4-agent-budget-fit-single-agent?scriptVersionId=355310852) |
| 文件 | 15文件逐字节保持 | 原四文件，仅evaluation的8→5分钟及prompt两处8 minutes→5 minutes |
| 工具 | 六基础工具和三个skills | 九官方工具，保留原prompt禁用图工具的指令 |
| 命令/工具/Agent/turn | 60秒 / 50 / 5分钟 / 80 | 240秒 / 28 / 5分钟 / 80 |
| output / thinking | 16384 / 4096，保留thoughts | 8192 / 4096，保留thoughts |
| adapter | 无 | 无 |

原Budget-Fit V5的0.18只属于原八分钟版本，修改后的B没有成绩。V5精确页面的License区于2026-10-13实际显示Apache 2.0；来源、改动及许可证存于kernel目录和运行归档frozen_bundle旁，不注入模型prompt。

固定31B CT v2、Starter Linux Python 3.12原镜像、wheelhouse v31。两个独立Evaluator/工作区/技能状态共享一次vLLM服务，`max_num_seqs=2`。三个已暴露任务fastapi_14873、fastapi_5077、rich_3518；每臂每题r1/r2两次，共12次。r1按AB/BA/AB，r2反转为BA/AB/BA。重复不是新的独立任务，prefix-cache/并发资源竞争不保证相等。

任务文本从既有v3的四题`tasks-selected.jsonl`校验后抽取三行嵌入。运行不读取、扫描或哈希官方完整tasks.jsonl，不读取heldout/预留题。只加载嵌入三题，预先确认其参考字段齐全及对应快照存在，避免官方评测器的缺失数据fallback。

v31的`EvalConfig.verification_timeout_seconds=300`对两臂显式一致，命令超时仍按各自Agent来源。worker外层900秒容纳Agent300+验证300+准备/清理；全局期限优先。安装、编译、模型加载均计入生命周期：2400秒结束评测，2460秒强停coordinator，2535秒归档截止，目标2550秒内退出。Kaggle会话必须另设2700秒硬限，4×L4按2倍系数估算上限1.5额度小时；平台额外开销与实际账户用量另记。额度不足、超时或失败保留原件，不续开、不自动重试。

## 文件与使用

- `notebooks/builders/build_006_reference_pair.py`：仅stdlib生成；读取旧生成器源码与冻结材料，不运行其Notebook或模型。
- `006-reference-pair-single-cell.py`、`006-reference-pair.ipynb`：完整独立载体。
- `research/006/reference-pair-20261013/kernel/pair.py`：GPU实际运行入口。
- 同目录`prepare.py`：同载体CPU准备模式；在CPU会话真实安装冻结wheel、创建EvalConfig/Context/隔离sandbox并compile_submission两臂。它不启动服务、不发送模型请求、不执行裁判、不解压任务快照；tokenizer指纹与部署CT v2固定。CPU构造成功不是修复或GPU推理成功。
- `kernel-metadata.json`为GPU配置，`kernel-metadata-cpu.json`是CPU准备配置；实际保存/推送时选对元数据，CPU必须`enable_gpu=false`。本生成器不推送。

重建命令：`PYTHONDONTWRITEBYTECODE=1 python3 notebooks/builders/build_006_reference_pair.py`。运行输出在`/kaggle/working/results/006/<run_id>/`：冻结两臂与三题、v31实际wheel哈希、run manifest、逐attempt官方result/patch/trace/验证输出、GPU采样、summary及bounded ZIP。CPU模式单独标为prepared、模型请求和judge为0，不填resolved成绩。完整GPU结果始终以12次计划分母报告，部分结果不冒充完整。

本次只做必要一次性手动语法、嵌入字节、调度和接口核对；CPU构造成功与GPU实际结果分别记录在上述运行报告。没有新增测试套件或检查脚本。
