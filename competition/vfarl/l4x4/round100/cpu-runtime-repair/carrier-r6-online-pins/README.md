# 四题 CPU 运行时恢复 R6

2026-10-28。仅执行索引9、10、23、54的当前任务版本；复用jiyujia既有固定100题材料和0a8e2770运行时，GPU关闭、无模型调用、无optimizer更新。原任务源码、patch及验收保持，Markdownify使用当前澄清题面；四题投影和实际public prompt渲染函数与CPU5字节相同。

本版冻结维护中34项源，包含run.py 8759790a的明确环境未就绪拒绝。CPU5已实际确认原输入缺runtime/bin/uv，故维护runner从现有Linux ELF/PATH或已有SHA绑定wheel恢复该可执行文件。

只准备stock-moto、stock-dvc、stock-markdownify三组原版本依赖。此前CPU5设置UV_OFFLINE=1但新session没有持久UV缓存，真实失败；本版仅准备期允许PyPI按原pins获取缺包，HTTP超时30秒、重试1次；finally恢复UV_OFFLINE=1，再进行原任务验收。没有移除失败或降低版本。实际是否成功以新运行日志为准。

EXPORT_RUNTIME=False，四题验收后清理临时材料，不生成运行时大包；不把旧导出文件当作可用环境。API请求CPU时限3600秒，平台实际回读未暴露时限，不能声称已确认。实际提交及结果见../execution-r6-20261028/。
