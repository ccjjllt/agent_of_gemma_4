# 当前四题CPU5实际结果

私有作业`jiyujia/vfarl-current-four-cpu-runtime-repair-r5/1`于2026-10-28 UTC01:32提交，01:44已读到COMPLETE；02:30再次[远端读回](final-status-readback.json)确认同version1、private、GPU=false、COMPLETE。实际session ID为363008365，kernel ID141259348是不同字段。没有模型调用或optimizer更新；平台实际CPU规格／会话timeout未暴露，不能把请求3600秒冒称已确认平台上限。

**四题均未取得有效验收。** [聚合原件](outputs/vfarl-prevalidate/prevalidation.jsonl)SHA256为`4aae3e159bbfb17438db254ff2222205b5be23ad216fa7ccb66f8bedb7774f80`。四行全部`usable=false`，8次baseline／reviewed-repair全部`verification_complete=false`、`reward=null`、LLM calls=0。官方调用返回的`status=SUCCESS`不表示验收通过。

| 原100题索引 | 任务 | baseline／repair exit | 实际环境错误 |
|---|---|---:|---|
| 9 | getmoto__moto-5348 | 4／4 | 缺sure |
| 10 | getmoto__moto-6159 | 2／2 | 缺sure |
| 23 | iterative__dvc.1d6ea681.pr_7628 | 4／4 | 缺colorama |
| 54 | matthewwithanm__python-markdownify.6258f5c3.pr_161 | 2／2 | 缺bs4 |

源载体SHA256 `4936acd37fb5102e426b79c952e20f8f607b6e144b7afb1a2f5544e18ca0b21a`；当前四题subset selection `56f75b33b4a745787a2db53936ff808d383a46fde776aa18242e767cdf7d687c`绑定parent current selection `7a6990034f651b932b8d218c8613df7d8f435925e72ef2624d3cc238ecdb893e`。任务、源快照、原修复及公开prompt渲染核对一致；54使用新版题面task SHA `12cfb3ac480707b893f6fbebbd292103958d2d6b81d83da29a3adf48bdf33c24`，其余三题task对象不变。公开prompt只做render，实际CPU oracle仍采用冻结prevalidation预算及legacy bytecode环境，没有agent采样。

UV的真实修复成功：将机器既有`/usr/local/bin/uv`复制到缺失的stage路径，原／目标SHA相同，真实版本`uv 0.11.13 (x86_64-unknown-linux-gnu)`。但`UV_OFFLINE=1`下stock-moto首先缺attrs26.1.0缓存、stock-dvc缺aiohappyeyeballs2.7.1、stock-markdownify缺beautifulsoup4 4.15.0；tzdata2025.2缓存也缺。详见[真实准备日志](outputs/vfarl-prevalidate/stock-moto-prepare.log)及同目录其他profile日志。

四题在carrier的`actual_task_environment_profiles`均无有效绑定，`runtime_unavailable_tasks`均明确不可用。冻结wrapper只在新profile存在时覆盖原runtime字段；其内嵌旧run.py的prevalidate未拒绝`linux_environment_ready=false`，随后执行了原环境。因此这次结果不能证明新stock profile有效；任务仍是环境unknown，原历史奖励及课程暂停不变。

相同作业另完成[真实Linux子孙进程预算观察](budget-probe-live-observation.json)：control SHA `fd25b6732986acd99ecfe8cb3d0d5f192a994362d8e8739297e258eb8c9412d4`，0.75秒执行＋1.5秒cleanup、外层3秒。子进程及setsid孙进程实际被TERM／KILL并回收，退出124，cleanup confirmed、无survivors，总1.570秒。此结果仅支持预算机制，不能替代模型／任务验收。

[小输出清单](small-output-manifest.json)覆盖远端58项中的57项，共188,482字节，保留日志、JSON及JUnit。864,775,947字节的`vfarl-cpu-stage.bin`没有下载；其[远端archive记录](outputs/vfarl-prevalidate/runtime-archive.json)只证明导出，不能证明运行时就绪，未提升为后续默认依赖。最终jiyujia GPU用量24.7071532375／30小时，与执行前相同，见[额度原件](quota-final-cpu5.json)。
