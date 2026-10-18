# alive-progress PR213：完整公共 API 尝试未形成新训练事务

来源仅为已明确分配 train 的 `rsalmei__alive-progress.35853799.pr_213`，固定 buggy commit **66c8728831ccad125a2485c2ecca57683eaeebee**。本轮补齐该提交的全部 `alive_progress/` 生产包、setup.py 和 MIT LICENSE，共 **31 文件、122,511 bytes**；逐文件 Git blob 与官方 GitHub tree 核对。没有取得其他问题内容或测试文件。来源、URL、SHA256 见 [source-manifest.json](source-manifest.json)；31 文件仍保持原样。

依赖只安装到临时环境：Python3.11、`about_time==3.1.1`、`grapheme==0.6.0`。真正尝试了公开导入 `from alive_progress import alive_bar, config_handler`，随后计划调用 `config_handler(refresh_secs=0.25)` 和运行两步进度条。实际在 import 阶段就失败：

```text
alive_progress/utils/terminal/tty.py:43
BASE = get(sys.stdout)
TypeError: get() missing 1 required positional argument: 'max_cols'
```

该错误发生在配置处理之前，因此本次没有观测到公开 API 上 0.25 的修前/修后区别，也没有验证默认、正常或边界值。完整命令、实际解释器、依赖和 stderr 见 [public-api-attempt.json](public-api-attempt.json)。这是一次真实 CPU 失败记录，不是源码推测或模拟返回。

进一步只读源码显示该分支还存在缺失 bool/format 配置工厂、Config/enrich_offset 字段不一致等运行链问题。解决这些会引入与浮点刷新需求无直接关系的多处生产修复。按本轮研究决策，在取得首个真实公共 API 失败后停止，没有修改生产包、尝试补齐这些接口或生成声称成功的新事务。

**结论：原 V1 AST 组件合同仍只作为 candidate，不纳入训练。** 本轮新增资产仅为完整固定来源和公共 API 失败证据；不能称为完整 API 成功、修复收益或新增独立问题。V1 冻结数据、源代码和 manifest 均未更改，没有 GPU、远端 kernel 或新测试套件。
