# P009 V1结果：真实多仓库SFT完成，完整Agent修复打平

2026-10-18回收并复核。`ccjjllt/p009-12b-sft-agent-pair` V1为COMPLETE；内部监督器同样正常完成、6/6任务完成，并非仅Notebook退出成功。运行ID：`run-009-12b-sft-pair-20261018T075746Z-5ad7782d`。下文原件均在`gpu-output-v1/results/009/<run_id>/`。

**结论：训练、独立导出、官方12B架构路径与LoRA路由均已真实运行；本轮没有修复净增。** Base和新adapter各1/3，adapter新增Rich修复，同时丢失FastAPI修复，Requests均未修复。本版保留为研究权重，不据此称提分模型或提交金牌候选。三题是事先选择的已暴露开发诊断，不能外推公榜0.33，也不足以判断SFT总体无效。

## 1. 真实训练与产物

| 项目 | 实际记录 |
| --- | --- |
| 训练输入 | 8个独立issue、7个repo，冻结SHA `0aca8fc273f6f58e94c641f52b7bf3e4dc039c7275a9b719010ad6300c8a72f5` |
| 更新 | 4×L4 DDP、r8/alpha16、2 epochs、4个Adam步骤；每步使用LR为2e-5/1.5e-5/1e-5/5e-6 |
| 实际累计token | 71,984输入，12,294监督；数据每轮35,992/6,147，未截断 |
| 训练循环 | 101.25秒；四步计算合计100.37秒，约122.48监督tokens/s（全局） |
| 训练阶段 | 371.01秒；训练子进程365.22秒，其中各rank模型加载约196.4秒 |
| 训练峰值allocated | 11.17 / 11.88 / 11.71 / 12.09 GiB；是PyTorch训练阶段峰值，不是进程总占用 |
| 参数一致性 | 四rank最终trainable SHA一致：`b4057b5e63674dfd288269c6557d2155c4b5bab343be33cad83be75edaddff80` |
| 独立adapter | 115,020,728 bytes，SHA `c7845c23852dbc701a267eaa58be87464af90e25c62bfffb0286ac1033945ba2` |
| checkpoint | step4的adapter、optimizer/scheduler/scaler状态和4个rank RNG已保存；本地已回收 |

末步日志的learning_rate=0是scheduler更新后的值，不代表该次没有参数更新。不同batch的loss从0.5068到0.4580不能作为收益曲线；按监督token加权，epoch0/1的在线训练loss约0.507255/0.504980，并没有独立验证loss。

独立adapter文件已按摘要SHA核对。为节省重复下载，本地checkpoint的adapter权重从相同SHA独立导出文件复制，optimizer/RNG来自实际checkpoint原件；没有声称已执行GPU恢复训练。

## 2. 真实部署与配对结果

vLLM采用官方12B CT v2、Gemma4ForCausalLM覆盖、TP4、32768上下文和maxseq4，启动280.35秒。日志明确出现`Loaded new LoRA adapter: name 'trained_adapter'`；三个base trace均指向CT基座路径，三个adapter trace均指向trained_adapter；132次chat请求均HTTP200，未发现路由失败。导出清单里旧`runtime_compatibility_verified=false`属于导出时的保守字段，不能用它否定此后的真实服务证据。

| 开发题 | Base | 新adapter | 最终生产修改与裁判 |
| --- | --- | --- | --- |
| rich_3067 | 失败，0B，超时 | **成功，655B** | adapter将URL正则加入`~`，82 passed |
| fastapi_14463 | **成功，696B** | 失败，0B，超时 | base用deepcopy替代浅拷贝，1 passed；adapter没有改源码 |
| requests_7205 | 失败，1183B，超时 | 失败，143B，超时 | 两组都只有复现/配置碎片，未修改生产Requests；均1 failed/205 passed/13 skipped |
| 合计 | **1/3** | **1/3** | 一次新增修复、一次回退、一次共同失败 |

两组各2次会话超时，生产补丁均1/3。Base平均agent时间392.18秒，新adapter454.76秒（增加62.59秒）；并发服务争用与随机采样均存在，不能将这一差值全部归因SFT。非空文件不等于生产修复，Requests的碎片不能计为有效编辑。

## 3. 改变下一步方向的机制证据

**Rich收益是真正的生产修改，但不是全面行为改善。** Base始终在markup/style等位置搜索，最终没有生产编辑；adapter第4步已读highlighter.py，第20步约158.9秒成功编辑正则。之后仍用错误API进行验证，并多次重复读文件，最终411.9秒才submit。独立裁判82 passed支持本次修复成立；它不证明adapter已普遍学会有效验证。

**FastAPI回退发生在已知原因到执行修改之间。** Adapter开头已经指出浅拷贝/深拷贝问题，之后没有edit_file；三个响应分别产生4301/4129/4423 completion tokens，约在165.2/265.8/379.4秒完成，最后仍在运行自造的浅拷贝示例，随后超时。Base则在175.3秒完成真实生产编辑，并正确通过裁判。这里优先要减少重复推理和验证替身，而不是假定再训练同8题就能解决。

**Requests这次是模型没有落地修复，不应再归因旧导入问题。** 两组都读到了正确的`src/requests/utils.py`，却反复用冒号分隔的错误netrc格式或错误调用netrc方法。Base最终留下reproduce.py/test.netrc，adapter留下.netrc；没有修改生产源码。两组统一采用src优先修正，原断言的失败与未修代码一致。旧31B正确补丁的CPU复评通过，并不等于本轮12B也写出了该补丁。

**4096思考限制确实生效，但不能当整个响应或循环上限。** 官方ADK将其发为`thinking_token_budget=4096`；按同SHA官方tokenizer重算，FastAPI上述三段thinking约4093/4094/4094 tokens。Base Rich/Requests两个总输出8192的响应也各含约4094 thinking，之后还产生约4094的正文，其中再次出现思考标记和重复内容。vLLM预算按思考段处理，总输出由max_output_tokens限制。因此只改思考预算未必能停止循环。

代码依据：[官方ADK映射](../../006/official-data-20261013/wheels-source/adk_submission/resolvers/generation.py)、[vLLM固定版本处理器](https://raw.githubusercontent.com/vllm-project/vllm/v0.19.1/vllm/v1/sample/logits_processor/builtin.py)、[总输出停止条件](https://raw.githubusercontent.com/vllm-project/vllm/v0.19.1/vllm/v1/core/sched/utils.py)。上述token重算基于parser后的文本，边界标记有2–3 token差异，不能冒充原始服务端逐token ID日志。

## 4. GPU效率与配额

完整监督器1721.87秒（28.70分钟）。评测阶段93次采样，四卡平均利用率约**95.30%**，运行请求平均2.66/峰4、排队请求峰0。首尾评测采样约927.9秒生成109,582 tokens，即约118 tokens/s，是共享服务合计且含思考，不是单题速度或每卡速度。

训练阶段平均约26.66%被加载与准备摊薄；实际四步约100秒，整体阶段371秒。不能把评测95.3%描述成全程都拉满。后续用更多独立数据/复用adapter摊薄启动，优先于重复加载做同一容量探针。

账号已用配额从23,601.973秒到27,061.109秒，差**3,459.136秒（0.96087小时）**，剩22.4830小时、reserved=0。这是共享账号读数差，接近2倍实验墙钟，不构成排他账单证据。Push约07:24 UTC，研究脚本RUN_ID开始07:57:46 UTC；两者相隔约33分钟，不能将平台排队/准备时间说成模型训练时间。

## 5. 下一轮决定

暂不把此adapter提交为提分版，也不为同8题增加epoch。先复用它研究较紧的思考/总输出预算（1024/4096），其余条件固定，并保留base对照。这是两个相配预算参数的联合处理；与V1跨轮比较仍有随机性和调度差异，只作机制诊断，不直接宣称泛化胜出。独立训练数据继续扩充，不回流开发诊断题训练。

截至08:28 UTC正式submission57340172仍PENDING，已确认公榜成绩仍是上一版0.13。本报告的1/3不是公榜分数。

结构化汇总：[result-summary-v1.json](result-summary-v1.json)。原始结果、trace和checkpoint保留，没有更改裁判标签或新增测试套件。
