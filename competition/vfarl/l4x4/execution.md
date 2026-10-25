# 实际执行记录

日期：2026-10-26（北京时间）。本页只记已发生的结果，不把准备或 CPU 数值验证记作模型训练。

## 已发生

- 已从 `main` 的 `ea9fa421` 创建 `codex/vfarl-l4x4-training-validation`，S600 冻结数据未修改。
- `qingyu1232` 的只读配额查询显示 GPU 已用 0.00h、剩余 30.00h。用户随后取消本轮 3 小时限制，允许推进 100 题；仍只使用该账号的实际可用配额。
- 用户明确授权了这 100 题材料上传至 Kaggle 私有数据集，并另行明确授权本分支实验代码上传、运行和更新私有 Notebook。材料上传已成功：[vfarl-l4x4-train100](https://www.kaggle.com/datasets/qingyu1232/vfarl-l4x4-train100)，约 123 MB；原始材料与代码指纹见 [材料构建/上传记录](kaggle/material-push.json)。参考修复只用于裁判和环境复现。
- 首个 [CPU 准备作业](https://www.kaggle.com/code/qingyu1232/vfarl-l4x4-cpu-stage) 已实际完成训练依赖导入和 17 个任务 profile 的安装尝试。16 个成功；profile-02 的失败来自未发布的本机开发包 `pipdeptree==0.1.dev1+...`。该包与任务运行无关，后续排除时保留修正原因。`stock` profile 也需要依据原运行环境补齐项目依赖，不能把 `task_site_packages_profile=null` 误解为只需要 pytest。
- Kaggle 能力作业实际测得：UID/EUID=0；unshare 的 mount/pid/net 均为 EPERM；Landlock 为 ENOSYS；child `chroot` 和 `no_new_privs` 成功，存在 `libseccomp.so.2`。原始记录见 [隔离能力](kaggle/capabilities-evidence/capabilities/isolation-capabilities.json)。因此 bwrap/namespace 路径不可在此环境直接使用；额外 seccomp 与完整替代隔离尚需实跑，不因库存在就宣称可用。
- 后续 [chroot v2 实测](kaggle/jail-probe-evidence-v2/jail-capabilities.json) 已返回 `effective_mode=chroot`：真实 privilege drop、capabilities 清零、NNP/seccomp、gold 排除、只读运行时与 UID 进程清理通过。探针使用标准库 executor 适配层，956 个运行时文件，源码 SHA `cb2c8601c48ae583c33347ac3593fc6cb4ea687663893368ac5f048c03d3083a`；之后的 UV Python 路径等集成修补需由正式载体重新执行同一运行时探针，不能把旧 SHA 的通过当作当前代码已验收。
- 新私有 [linux-runtime/1](https://www.kaggle.com/code/qingyu1232/vfarl-l4x4-linux-runtime) 完成 17 个基础 profile，归档 778757268 bytes，SHA `468f2abea122dfef31ed7abdfae6d8ebe51aca2451d56e41a2362e18078e845d`。CPU [prevalidate/1](https://www.kaggle.com/code/qingyu1232/vfarl-l4x4-prevalidate) 补齐 stock 11 组后实际运行全部 100 题：36 usable、64 未通过环境验收。完整记录在 `kaggle/prevalidate-evidence-v1/vfarl-prevalidate/prevalidation.jsonl`；这不是模型成绩。
- 已从原始失败日志定位：30 题受共享库路径中的 `..` 重复打包影响；另一批因任务解释器链回落系统 Python 而缺 pytest。已修复路径去重、真实 base Python 创建 venv 和实际 `error_message` 字段读取。其余源 CLI、依赖 copy-mode 和 Moto 的 Python 3.11.8 按原环境证据恢复，不修改题目验收。
- 六道 dotenv 每题的两个 PTY 相关验收在当前 chroot 中无法运行；在首次模型采样之前，已生成 [v2 替换清单](v2/substitutions.json)：保留原 94 题，用 S600 中同课程/机制的六题补足 100。替补选择只依据环境兼容性和原训练白名单；没有模型分数参与。旧失败、旧清单均保留。

## 本地必要核对

- 100 题清单的唯一性、67/33 配比、train 来源、源题面哈希和重复导出一致；100 个不同 repair family，34 仓。
- 官方 SubprocessManager 的真实 task root 同路径恢复了 984 项文件/目录条目、权限、软硬链接、Git index、临时文件、venv、上下文计数及逻辑预算。改变外部依赖记录或策略身份会拒绝复用。macOS 上明确是 observable scope，不冒充 Kaggle 完整隔离。
- 新 `TaskEnvironment` 以真实 `Knio__dominate.9082227e.pr_166` 运行了官方 read_file 和两次完整验收：原始版本 reward=0、已验收修复 reward=1；两次各保存一份原始 JUnit。该命令使用临时目录，未新增测试文件；它证明实际工具与验收接线，不是模型修复结果。
- 实际 Transformers 5.13.1、PEFT 0.21.2 的 tiny Gemma4，完成缓存续算、selected logprob、eval checkpoint 梯度及 CPU 双 rank FSDP 更新/导出。分块 logprob 与完整词表误差为 0；两 rank 梯度一致，更新后正优势动作概率上升、负优势下降。该证据属于小模型数学/结构核对，未证明 31B NF4/r32 的 L4 容量。

## 尚未取得的证据

当前尚未启动本实验 GPU 采样，尚无 S600 模型 rollout、已验证同状态奖励对或 31B VFARL 参数更新结果。正在完成 Linux 任务依赖与隔离迁移；所有真实失败保留，不将环境错误当作模型负例。
