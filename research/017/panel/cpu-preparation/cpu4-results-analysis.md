# CPU4 实际结果

2026-10-24，作业 `jiyujia/p017-confirmation-cpu-repair-v4-ac51dacf` v1。14:39:18 北京时间回读 RUNNING，14:44:52 回读 COMPLETE；远端源码 SHA `f70bd541dc77290fd5c17c3a1ac2d018c771e0947a17f8410d8c77adea10f7df` 与冻结载体一致，GPU/TPU/Internet 均 false。

11 题全部实际尝试，**新增 3 题有效，8 题未有效**。原 13 个有效题未重跑。全部 119 份日志及输出材料已取回；材料 184,122,885 bytes，实算 SHA256 `0269e2842155632dae703774cf2297ddad41e52f76ca6df753adbe2b7ccf48a2` 与 [summary](cpu4-output/p017-confirmation-preparation/summary.json) 一致。逐题原记录见 [prevalidation-results.json](cpu4-output/p017-confirmation-preparation/prevalidation-results.json)。

| 新有效题 | 实际证据 |
|---|---|
| structlog586 | 原源码直接导入；F2P 1 fail→1 pass，P2P 两态各127 pass |
| soupsieve263 | 原源码直接导入；F2P 1 fail→1 pass，P2P 两态各5 pass |
| rdflib1117 | 保留原 editable/CLI；F2P 2 fail→2 pass；来源无P2P，记回归未覆盖/null |

| 未有效题 | 实际主要原因 |
|---|---|
| jsonargparse564、592、248 | reference F2P 仍因 `ArgumentParser._parse_known_args()` 缺 `intermixed` 失败；592另有P2P exit4。固定同仓替补没有解决解释器接口不兼容 |
| rdflib2474 | 源码模块初始化直接调用 distribution metadata；跳过打包后报 `PackageNotFoundError: rdflib`，两态 collection失败。不能伪造dist-info接纳 |
| soupsieve88 | editable metadata-generation-failed，未进入判据 |
| structlog669 | backend支持检查成功，但editable metadata-generation-failed，未进入判据；日志没有内部traceback，不猜具体打包原因 |
| pint877 | F2P真实1 fail→1 pass，但两态P2P各1 fail、36 pass、1 skip；失败来自原测试调用Python3.12已不提供的 `assertEquals`。不删除失败P2P或将skip当pass |
| pint1471 | 两态collection先因继承系统NumPy缺 `cumproduct` 失败；来源requirements没有单独锁该可选依赖。原参数化node截断/歧义仍是额外已知问题，尚未被实际执行排除 |

三个完成的 CPU 阶段共 **40 次实际尝试、33 个不同候选身份**：首轮24、CPU3重跑5、CPU4执行11。去重后16个有效、17个尚未有效。没有将沿用的13个有效题计作CPU4新执行，也没有通过更改源码/判据语义补齐数量。

目前有效覆盖：networkx3、mashumaro3、sqllineage3、rdflib2、structlog2、soupsieve2、pint1；共7仓，jsonargparse为0。13题有实际P2P前后通过计数，3题F2P-only。原每仓3题/24题目标仍有8个未接纳槽位，应按 **16题环境可复现子集** 报告，不称已完成24题准备。

本轮不再扩CPU或修改问题来凑满。已将真实有效材料合并供root决定C模型评估预算与checkpoint，见 [最终GPU材料说明](../merged-valid16/README.md)。完整失败、参考执行日志、全部原CPU JSON与替补关系留在本地host审计资料；最终GPU包仅带必要精简provenance，不带参考执行trace/tail、reference patch或host-source行。
