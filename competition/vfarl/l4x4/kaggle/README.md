# Kaggle L4×4 实验载体

账号固定 `qingyu1232`，逐进程读取已有凭据，不修改全局账号。2026-10-26 实际读取 GPU 余额 30.00h、已用 0.00h，刷新为北京时间 2026-10-31 08:00；见 `account-readback.json`。用户授权本实验私有题目材料、Notebook 代码上传、运行与更新，不另加实验墙钟限制；平台时限和账号实际配额仍适用。GPU 启动由主代理统一协调。

训练和采样复用 `ccjjllt/p005-cpu-nf4-cache`，来自官方 QAT-unquantized v2 的 NF4 缓存。不是直接对 CT 打包权重进行 QLoRA。CT v2 tokenizer/template 固定不变。既有四 L4 FSDP 证据为 r8、最长 6,846 tokens；本轮 r32、4K/8K 容量须以实际 GPU 结果为准。

`NvidiaL4` 是既有比赛作业实际使用的四张物理 L4 shape，历史计额为墙钟的 2 倍（不是乘四）。载体实际读取四卡名称再执行，拒绝其它拓扑。只在最终 GPU push 中使用 `--accelerator NvidiaL4`。

载体顺序：

1. `build_cpu.py` 冻结 `cpu_stage.py` 及 17 个真实任务环境 profile。CPU 作业安装 Linux 依赖、复制官方 tokenizer、保留镜像 CUDA torch、只检查现有 NF4 缓存元数据。已有 cache 与训练 wheels 对本账号可读；原 zhuating tokenizer 私有作业不可读，因此直接挂载官方 CT v2 复制小文件。
2. `build.py --out ...` 打包冻结的 100 个 train task、原始源码 snapshot、实际 reviewed patch、runtime 元数据和源码。`submitted.patch`、task 的 reference/test 字段始终属于裁判侧。v1 数据包 123,217,638 bytes，SHA256 `3041a09909817d0b2ab628a7319ffc6eda496e392e436b1b56eb4fe35713b565`，私有数据集 `qingyu1232/vfarl-l4x4-train100`。实际上传回执见 `material-push.json`。
3. `build_runner.py --material-build ... --mode prevalidate --out ...` 冻结当前源码。CPU 四 worker 分片执行原始 baseline 与 reviewed repair 的完整验收，并合并 100 条真实结论。移除环境中无法发布的本机 `pipdeptree==0.1.dev1+gedbd59b41` 开发工具会单独记录，不声称原环境字节等同。
4. `--mode train` 生成 GPU 载体，不自动提交。四个单卡当前策略 collector，各分配 2/1/1/1 个 focal state，共最多五个；统一模型种子 0、每题初始一轮、每状态三条分支。全部 collector 退出后才启动四卡 FSDP。只有实际参数更新 manifest 确认一步完成，才在固定前四题再采样。没有 mixed-outcome 组时保留真实数据，不冒称发生更新。

首版 CPU 实际发现：训练依赖和官方 harness 导入成功；17 个任务环境中 16 个完成，profile-02 因本机 pipdeptree 开发版本不可下载失败。bwrap 的 user namespace 创建被 Kaggle 拒绝，因此该结果只支持 observable replay，不能充作完整状态恢复或有效 VFARL 数据。后续能力调查由同一个生产准备入口的 `capabilities` 模式记录 UID/capabilities、Landlock、seccomp、chroot 与 namespace 真实返回值。原始记录见 `cpu-evidence/observed-preparation.json`、`capabilities-evidence/`。

纯 CPU `qingyu1232/vfarl-l4x4-capabilities/1` 已完成：UID/EUID 为 0，实测 child chroot 与 no_new_privs 成功，`libseccomp.so.2` 存在；CapEff 含 KILL、SETUID、SETGID、SYS_CHROOT，不含 SYS_ADMIN。mount/PID/net unshare 均 EPERM，Landlock ABI 查询返回 ENOSYS。这支持继续研究 chroot/dropuid/seccomp 实现，尚不等于完整隔离已成立。

`qingyu1232/vfarl-l4x4-jail-probe/2` 已实际通过生产 jail 的 chroot、权限下降、seccomp、gold 隐藏、运行库只读和后台 UID 进程清理。该小型运行使用 stdlib 执行适配器；正式 100 题仍必须在官方 harness 的同一 chroot 条件下验收。首版适配器 command/args 错误保留原始结果，未算作隔离失败。见 `jail-probe-evidence-v2/jail-capabilities.json`。

原 `vfarl-l4x4-cpu-stage` 的输出长期未发布，v2 日志服务返回 HTTP500。已将相同优化源码（SHA256 `a85f91d6b95efe98edd24dc6cb95d301537e1c4c88a942f53a9ba16a51fc33d2`）移至 `qingyu1232/vfarl-l4x4-linux-runtime/1` 恢复。使用快速压缩并在归档后清除 raw venv 树，只发布归档和小型报告；新作业已真实执行到 runtime/cache/tokenizer 阶段。后续消费时明确绑定完成版本，GPU 载体还强制绑定实际 runtime archive SHA256。

新 runtime/1 已 COMPLETE，实际归档 778,757,268 bytes，17 个基础环境安装成功（详见 `linux-runtime-evidence-v1/vfarl-preparation.json`）。原始 `task_site_packages_profile=null` 的 19 题实际上继承各自 host 环境，不能解释为仅需 pytest；已从原运行 metadata 与项目依赖整理为 `stock-environment-pins.json` 的 11 个独立环境，正式 CPU 验收中全部成功安装。

`qingyu1232/vfarl-l4x4-prevalidate/1` 使用 `prevalidate-v3/` 冻结源码、固定 runtime/1、四 worker 同 chroot 验收全部 100 题。实际已出现可用与不可用题，未把不可用结果记作策略失败。`dominate-repro-v1/` 是同一已失败题的单次官方验收复现，复用已核实完全一致的四个 pytest pins，只为取得详细错误；原 100 题作业继续保留所有事实。当前任务结论以实际输出为准。

所有 Python 核对使用一次性 AST/手工命令；没有新增测试套件。任务仓库原有验收测试是实验判据，不能作为训练目标或直接展示给策略。当前 GPU 结果以正式运行日志为准，此文不是训练完成声明。
