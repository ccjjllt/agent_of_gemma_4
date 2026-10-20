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

## 2026-10-20 SFT同步推送（ccjjllt，与预验证并行）
- dataset `ccjjllt/p013-behavior-data-34` ready（3文件：manifest/marker/2MB jsonl，jsonl不透明通过）。
- kernel `ccjjllt/p013-12b-behavior-sft-34` v1已推（--accelerator NvidiaL4，machine_shape NvidiaL4，推前配额7.22h）。
- 完成判定：optimizer_steps==72且导出完成；否则只留checkpoint+日志。

## 2026-10-20 v2-flat失败根因与v3自包含方案
- v2-flat ERROR@0.8s：`/kaggle/input/p013-task-material` 整个未挂载（非文件缺失）。挂载失败，无GPU消耗。
- 数据集混乱复盘：Kaggle对.tar.gz自动解包入库（13 tarball变数千文件，.git included）；小文件（json/py/txt）不透明通过；版本为合并语义。教训：二进制快照不走Kaggle dataset。
- v3方案：kernel自包含（preverify.py+tasks-13+criteria+replay共224KB，同目录挂载）+ GPU实机`git clone --branch <iid> github mirror`（materialize已验base_commit）。`enable_internet=true`与`competition_sources`互斥（400 Bad Request），去比赛源后推送成功。
- `qingyu1232/p013-preverify-v3-net` v1已推（--accelerator NvidiaL4，推前配额9.69h）。输出dev12-freeze.json+V2/V3 log。

## 2026-10-20 数据集路线放弃，改CPU自包含clone（零GPU配额）
- 定论：Kaggle对.zip/.tar.gz一律入库解包，不透明载体不可行；数千文件展开还可能破坏挂载（v2-flat整盘未挂载）。
- 新路线：CPU kernel自带小文件（224KB）+ 实机git clone（internet=true）。CPU会话免费，无GPU形状问题，无需比赛挂载。400教训：internet与competition_sources互斥。
- `qingyu1232/p013-preverify-cpu-clone` v1已推。输出dev12-freeze.json+V2/V3 log。
- 待清理：`p013-task-material`、`p013-task-zips`两个污染dataset（重命名/删除，避后续误用）。
