# P016 首轮主训练执行记录（完整一轮，最多8小时）

2026-10-23，北京时间。用户已选择完整一轮、最多8小时、原估算约18.88美元，并随后明确要求不暂停Pod、不建立定时任务。主训练已于 **11:23:32** 启动，最晚约 **19:23:32** 结束本次训练进程；Pod保持运行。启动回执见 [launch.json](main-short-training/launch.json)。用户将在训练完成后通知，再进行结果读取和回传。

- 数据：`main-sft-short-v1`，537个独立完整train问题、68仓，7,119,739输入/1,291,819监督token，全部≤16K、无截断。159 PR身份、375命名变异、3个旧pilot数字身份；来源偏差保留在统计中，不宣称真实问题占多数。
- 计算：现有Runpod `0wietnrru530l9` / 4×A40；读取费率2.36美元/小时。原基座全新LoRA初始化，不把校准adapter当主训练起点。
- 配方：r8/alpha16、学习率2e-5、4 ranks、每卡micro batch 1、积累2、1 epoch、68次更新、每10步保存完整checkpoint。总计划不随单次wall上限改变。
- 已实测16K：4次更新全部完成，最长15,768；真实保存/退出/重载/继续/导出成功。20K及25K批次也已完成更新；32,382在反向OOM，首轮主训练暂不继续这一容量分支。
- 预计537题完整一轮纯训练约6–7小时，依据7.12M输入及16K桶实测吞吐外推，尚非主训练实测。长短轨迹问题分布不同，速度优势不等于已证明提分。

## 已确认预算与运行方式

执行完整一轮537题、68次更新；服务器启动器硬限制28800秒，提前完成即退出，不因时限改写学习率计划，不自动开启下一轮。最初API费率2.36美元/小时，启动前控制台显示2.37美元/小时，8小时约18.96美元。Pod闲置和环境准备同样计费；训练进程退出不会停止Pod。

用户后续指令覆盖先前“备份后暂停Pod”选项：**保持Pod运行，不停止、重启或删除。** 曾创建的15分钟回查任务 `p016` 已按用户要求删除；没有后台定时回查。用户通知后读取最终结果、核对并回传最终adapter、最近完整checkpoint和必要元数据。训练本身已脱离SSH会话，仍有服务器端8小时进程上限。

服务器SSH：`ssh -i /Users/zhaohuaxiang/.ssh/id_ed25519_ccjjllt -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -p 22078 root@194.68.245.193`。主启动器PID为2596，torchrun PID为2597；诊断进程身份时仍以命令与启动时间为准。日志 `/workspace/p016/runs/main-sft-short-v1.log`，输出 `/workspace/p016/runs/main-sft-short-v1`。启动观察已确认 **step 1/68完成** 并进入step 2：首步85,603输入/16,264监督tokens，273.0498秒，loss 0.56796105，监督吞吐59.5642 tokens/s；首次更新处于学习率warmup，实际使用1e-5。原始快照见 [initial-metrics.jsonl](main-short-training/initial-metrics.jsonl)。还不据此宣称整轮完成或比赛提分。

## 已备齐的实际入口

服务器代码 `/workspace/p016/smith-pilot-calibration-v5`（校准包名称仅作冻结代码容器），主数据 `/workspace/p016/main-sft-short-v1`，入口 `/workspace/p016/run_main_short.py`，环境 `/workspace/p016/venv/bin/python`，NF4缓存 `/workspace/models/p005-cpu-nf4-cache/qat-nf4`。入口SHA256为 `ab1715613178c2f461054f6ff4225b40304112b86c1cc53fe14b4e30aabdffa6`。

v5只在已GPU验证的CPU adapter加载修复上补充异常快速退出：先flush错误再以非零状态退出，让torchrun及时终止peer，不等NCCL清理超时。没有改训练数学目标。运行保留 `NCCL_P2P_DISABLE=1` 和 `PYTORCH_ALLOC_CONF=expandable_segments:True`；后者已把25K的真实OOM修复为完成更新。

实际已脱离SSH会话执行以下命令；不要重复启动。

```sh
/workspace/p016/venv/bin/python /workspace/p016/run_main_short.py \
  --bundle /workspace/p016/smith-pilot-calibration-v5 \
  --corpus /workspace/p016/main-sft-short-v1 \
  --model /workspace/models/p005-cpu-nf4-cache/qat-nf4 \
  --tokenizer /workspace/p016/smith-pilot-calibration-v5/assets/tokenizer \
  --out /workspace/p016/runs/main-sft-short-v1 \
  --wall-seconds 28800
```

68步规划、源数据及配置均冻结。本次8小时主训练获得独立授权，与此前最多2小时GPU校准分开记账。完整训练后仍需固定Agent权重对照才知道真实修复收益；本次没有新增正式评测或提交。
