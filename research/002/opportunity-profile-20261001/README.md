# 已暴露任务的参考修改范围

2026-10-01；仅离线描述27个已暴露任务，不分析heldout内容、不生成候选补丁。

`reference-scope.json`：18/27参考补丁只涉及一个非测试文件；19/27总增删≤20行，中位12行，最大259行。4/27题的问题文字直接包含某个参考文件basename；hints均空。

这支持优先研究小范围编辑接口，不证明题目简单，也不要求合法修复必须匹配参考补丁。非测试文件包含文档/示例；行数=增加+删除，basename是大小写敏感字面匹配，不是语义定位指标。

复算：

```bash
python3 research/002/opportunity-profile-20261001/analyze_reference_scope.py \
  --tasks /private/tmp/gemma4-research/tasks.jsonl
```

脚本先根据P003两张CSV选择已暴露ID，再解析任务内容；兼容公开数据的普通unified diff头。结果只供离线研究，不能打包成agent检索资源。27题的增删数字已将普通unified diff逐文件分块、补齐末尾换行后，用`git apply --numstat --recount`在内存输入上独立交叉检查；没有应用补丁到工作区。整份多文件普通diff直接传给git的recount会把后续文件头误计入前一文件，故没有采用该错误统计方式。
