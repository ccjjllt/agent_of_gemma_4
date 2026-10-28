# 四题 CPU R7：保存可复用运行时

继承R6真实四题0/1方案，仍使用相同34项内嵌源、任务/patch/oracle/pins和0a8原运行时。GPU关闭、零模型调用、零optimizer更新。

本版仅开启现有EXPORT_RUNTIME和新增显式prepare-only-selected-tasks：准备范围仍只有四题的stock-moto/stock-dvc/stock-markdownify，其他原profile按基底保留，不为导出而安装其他stock依赖或运行其他题验收。准备期联网按原pins，验收期UV_OFFLINE=1。当前四题投影及实际public prompt渲染与R6相同。

既有archive.add(stage)仅打包vfarl-cpu-stage；新源码/题目材料在独立vfarl-input中，未加入新源码forest，不引入overlay或新消费协议。runtime-archive.json记录实际bytes/SHA/基底SHA，以及本次四题验收范围和prevalidation SHA。导出文件存在不代表100题新验收或其他profile可用。

API请求CPU时限3600秒，平台实际值不可从已知metadata回读，仍须观察到终态。实际输出见../execution-r7-20261028/。后续通过现有runtime-kernel/SHA加载；只有实际消费同一环境并匹配当前task/runtime验收后，才可解除相应暂停。
