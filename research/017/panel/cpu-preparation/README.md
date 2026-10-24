# C 面板 Linux CPU 准备

复用P015的独立venv、原始snapshot、来源已有F2P/P2P及buggy/reference执行方式。只运行原有判据，不新增测试、不调用模型、不启动GPU。源来自上级目录固定24题投影，完全不包含teacher messages。root统一上传和push。

`build_carrier.py`读取真实source/snapshot acquisition并逐文件核对，再生成一个opaque `.bin` 输入、dataset元数据和可直接push的CPU kernel。输入约39MB，避免Kaggle递归自动展开快照。模板只在固定Linux/Python3.12上运行。

准备步骤：按来源pip freeze精确版本准备wheel；file://依赖按同一源environment中的conda版本恢复，来源项目自身从精确base snapshot安装；不执行来源install/eval shell。pytest及基础构建工具显式统一成模板中的Py3.12工具链，其余库保持来源版本，原配置及每项变更留档。旧Python3.6–3.9来源可能不兼容，这会形成真实准备失败，不能在结果前称24题全部可用。需要额外构建backend或旧库不支持Py3.12时按日志修复并冻结新版本。

CPU作业允许互联网准备依赖，随后使用同一批wheel字节离线安装、验证实际导入任务源码，再执行来源F2P/P2P。wheel共享缓存复用，逐题目录仅链接该题精确版本；pytest插件名从实际wheel的pytest11元数据读取。buggy/reference分别从干净快照开始，参考patch和test_patch只应用于CPU host checkout。GPU材料不包含完整host-source或参考patch。

两道原本无P2P的题按冻结身份保留：F2P有真实前后区分时可以valid_prepost，同时p2p_prepost_pass=null，单列F2P-only，绝不执行空节点列表或称回归通过。所有失败保留，输出不是自动删题后的最终确认清单；root按已有冻结/替补规则及CPU证据决定最终selection，必须在任何候选模型结果之前。

当前可交付构建命令：

```
python3 research/017/panel/cpu-preparation/build_carrier.py \
  --output-dir research/017/panel/cpu-preparation/carrier-v2 \
  --dataset-id jiyujia/p017-confirmation-source-v1 \
  --kernel-id jiyujia/p017-confirmation-cpu-v1 \
  --wall-seconds 7200
```

上传`carrier-v2/dataset/`为私有dataset，再push`carrier-v2/kernel/`。内部硬时限7200秒，额外预留300秒平台结束时间，root启动时应用 **7500秒平台上限**。每次wheel准备最多600秒、每次判据180秒；子进程组超时会停止。到总时限停止接纳新题并保留未执行身份，不能把未完成当pass。

结果保存在`/kaggle/working/p017-confirmation-preparation/`：逐题真实日志、prevalidation-results、summary及`prepared-confirmation-material.bin`。summary含有效身份、F2P-only、原有P2P覆盖子集、失败与材料SHA/size。实际准备后的wheel可能显著大于39MB，按实际输出大小报告，不将初始source大小当最终GPU材料大小。
