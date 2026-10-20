# P013 GPU预验证状态

## 2026-10-20 v1 ERROR根因：--dir-mode zip挂载为*.zip文件，非目录
- `qingyu1232/p013-gpu-preverify-v1-v4` ERROR@~1.5s：`FileNotFoundError: /kaggle/input/p013-task-material/tasks-13.jsonl`。
- 等等：tasks-13.jsonl是v1平铺上传的，应在root……实际错误显示它也不存在。说明zip模式版本可能整体替换了文件布局，或挂载点不同。无论如何，加兜底：快照/criteria的.zip解包回退（两种zip内部布局均本地验证通过）+ 若tasks-13.jsonl缺失则报错指明挂载内容。
- 配额影响：失败在1.5s，GPU会计几乎为零（重拍配额确认）。

## 2026-10-20 数据集教训与v2推送
- Kaggle自动解包.tar.gz：v2/v3的13个tarball全被展开成数千文件（含.git），tasks-13.jsonl一度不可见（分页后证实需补）。结论：.tar.gz不可作不透明载体；.zip可（P012先例）。
- 修复：数据集删后重建（31文件扁平：13 replay+13 criterion+index+tasks-13+marker+snapshots.zip 48MB）；缺失小文件用单版本补齐（版本合并）。
- preverify.py适配：SNAP/CRIT/REPLAY三目录+扁平双布局自动检测；zip解包回退保留； exploded树直接拷贝+git HEAD与base_commit核对（v1_source/head_match记入result）。
- `qingyu1232/p013-preverify-v2-flat` v1已推（--accelerator NvidiaL4，推前配额9.78h；期间共享账号被他人用掉~0.7h，非本项目支出）。
