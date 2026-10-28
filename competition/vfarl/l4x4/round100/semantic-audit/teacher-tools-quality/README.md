# 既有teacher工具轨迹质量：初步确证候选

2026-10-28。只读固定100的已选recipe、canonical与真实tool-events，事件文本合计774167字节。确认索引8–15的8条moto示范在改前/改后执行同一命令，16次全部exit0。该命令只导入目标模块、断言来源路径、调用py_compile，未调用变更API，无法观察这8处修复的行为差异。真实messages的assistant工具参数与事件逐条一致，问题确实存在于示范轨迹。

候选为moto 5200（secret ARN后缀长度）、5348（DynamoDB空key验证）、6159（API Gateway路径冒号）、6492（Route53 Location头）、6609（KMS签名算法）、6755（SES MAIL-FROM身份）、7061（AutoScaling tag过滤）、7111（Lambda默认包类型）。每题旧recipe、事件、messages、补丁及独立官方报告指针/哈希保存在candidates.json。原官方phase B的baseline失败/reference通过是真实独立验收；本次结论仅是训练示范的行为观察不足，不能称旧reward伪造。

其余92条至少有行为断言或pytest线索，尚未据此认定它们语义充分。没有发现edit_file/create_file修改测试路径。初筛命令写入只有索引4在TemporaryDirectory中准备FileUpload的临时输入文件，未据此认定改测试或依赖。

新teacher候选须先用原source真实API有界触发，再实际六工具执行并取得原官方baseline0/reference1，才可独立规范化或导出SFT；已另在moto-behavior-r1完成索引8/12，在moto-behavior-r2完成其余6题的真实API六工具重放、原官方0/1和独立SFT增量。后6题完整51条中13条超过4096，未截断；详见各自README。旧轨迹/奖励保持原样。
