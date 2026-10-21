# P014 runtime v2

针对正式提交57438326的12小时整体超时，冻结5min/60s command/40turns/28tools。保留31B CT v2、原采样；prompt同步预算，明确edit_file必填参数和拒绝循环退出。原v1及失败提交包均保留，没有覆盖。

包：`candidate.zip`，3300bytes；SHA256 `68dfd7240c5f3bbd765274db817d06879c072f737191adc9249fa71ceb3d88db`。逐文件指纹见[manifest](manifest.json)。具体失败证据与验证范围见[修复说明](../../submission/failure-20261021/README.md)。

原八题单臂实跑已完成8/8通过，包含官方编译与独立新工作树裁判。正式修复提交为 **57471497 / PENDING**，整场是否成功仍待官方分数。见[实际结果](../../eval-31b/runtime-v2-result.md)。
