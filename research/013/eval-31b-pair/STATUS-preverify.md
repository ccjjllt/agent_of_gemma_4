# P013 GPU预验证状态

## 2026-10-20 v1 ERROR根因：--dir-mode zip挂载为*.zip文件，非目录
- `qingyu1232/p013-gpu-preverify-v1-v4` ERROR@~1.5s：`FileNotFoundError: /kaggle/input/p013-task-material/tasks-13.jsonl`。
- 等等：tasks-13.jsonl是v1平铺上传的，应在root……实际错误显示它也不存在。说明zip模式版本可能整体替换了文件布局，或挂载点不同。无论如何，加兜底：快照/criteria的.zip解包回退（两种zip内部布局均本地验证通过）+ 若tasks-13.jsonl缺失则报错指明挂载内容。
- 配额影响：失败在1.5s，GPU会计几乎为零（重拍配额确认）。
